# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder definition."""

from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn

from typeguard import check_argument_types

from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    EmbedAdapter,  # noqa: H301
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import check_short_utt
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.utils.dynamic_import import dynamic_import
from espnet2.asr.encoder.abs_encoder import AbsEncoder


class ConvConformerEncoder(AbsEncoder):
    """Conformer encoder module.
    Args:
        input_size (int): Input dimension.
        output_size (int): Dimention of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        attention_dropout_rate (float): Dropout rate in attention.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            If True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            If False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        encoder_pos_enc_layer_type (str): Encoder positional encoding layer type.
        encoder_attn_layer_type (str): Encoder attention layer type.
        activation_type (str): Encoder activation function type.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        use_cnn_module (bool): Whether to use convolution module.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.
    """

    # track the number of arguments of sequential modules for JIT disamdiguation
    num_sequential_args = 3

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks_l1: int = 2,
        num_blocks_l2: int = 8,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        conv1_filters: int = 128,
        conv2_filters: int = 512,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 31,
        padding_idx: int = -1,
        use_chunk=False,
        chunk_window=None,
        chunk_left_context=None,
        chunk_right_context=None,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        activation = get_activation(activation_type)
        pos_enc_class = RelPositionalEncoding
        self.min_subsampling_length = 8


        # first block -- conv + MHA:
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(1, conv1_filters, kernel_size=(3,7), stride=(2,2), padding=(1,3)),
            torch.nn.ReLU(),
            nn.Conv2d(conv1_filters, conv1_filters, kernel_size=(3,5), stride=(2,2), padding=(1,2)),
            torch.nn.ReLU(),
        )
        self.conv1_out = torch.nn.Sequential(
            torch.nn.Linear(conv1_filters * (((input_size+1) // 2) // 2), output_size),
            pos_enc_class(output_size, 0.1),
        )
        positionwise_layer_1 = PositionwiseFeedForward
        positionwise_layer_args = (output_size, linear_units, dropout_rate, activation)        
        encoder_selfattn_layer_1 = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (attention_heads, output_size, dropout_rate)
        convolution_layer_1 = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation)        
        self.encoders_1 = repeat(
            num_blocks_l1,
            lambda lnum: EncoderLayer(
                output_size,
                encoder_selfattn_layer_1(*encoder_selfattn_layer_args),
                positionwise_layer_1(*positionwise_layer_args),
                positionwise_layer_1(*positionwise_layer_args) if macaron_style else None,
                convolution_layer_1(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )    
        self.after_norm_1 = LayerNorm(output_size)
        
        # second block -- conv + MHA:
        self.conv2 = torch.nn.Sequential(
            nn.Conv1d(output_size, conv2_filters, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            nn.Conv1d(conv2_filters, conv2_filters, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            nn.Conv1d(conv2_filters, output_size, 1, stride=1),
            torch.nn.ReLU(),
        )
        self.posenc_2 = pos_enc_class(output_size, 0.1)
        positionwise_layer_2 = PositionwiseFeedForward
        positionwise_layer_args = (output_size, linear_units, dropout_rate, activation)        
        encoder_selfattn_layer_2 = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (attention_heads, output_size, dropout_rate)
        convolution_layer_2 = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation)        
        self.encoders_2 = repeat(
            num_blocks_l2,
            lambda lnum: EncoderLayer(
                output_size,
                encoder_selfattn_layer_1(*encoder_selfattn_layer_args),
                positionwise_layer_1(*positionwise_layer_args),
                positionwise_layer_1(*positionwise_layer_args) if macaron_style else None,
                convolution_layer_2(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )    
        self.after_norm_2 = LayerNorm(output_size)


        # chunk attention attributes:
        self.use_chunk = use_chunk
        self.chunk_window = chunk_window
        self.chunk_left_context = chunk_left_context
        self.chunk_right_context = chunk_right_context

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Calculate forward propagation.
        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.
        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if xs_pad.size(1) < self.min_subsampling_length:
            raise TooShortUttError(
                f"has {xs_pad.size(1)} frames and is too short for subsampling "
                + f"(it needs more than {self.min_subsampling_length} frames), "
                + f"return empty results",
                xs_pad.size(1),
                self.min_subsampling_length,
            )

        # first block -- conv + MHA:
        xs_pad = xs_pad.unsqueeze(1)   # (b, c, t, f)
        xs_pad = self.conv1(xs_pad)
        b, c, t, f = xs_pad.size()
        xs_pad, pos_emb = self.conv1_out(xs_pad.transpose(1, 2).contiguous().view(b, t, c * f))
        masks = masks[:, :, ::4]
        xs_pad, masks, _ = self.encoders_1(xs_pad, masks, pos_emb)
        xs_pad = self.after_norm_1(xs_pad)
        
        # second block -- conv + MHA:
        xs_pad = xs_pad.transpose(1,2)    # (b, f, t)
        xs_pad = self.conv2(xs_pad)
        xs_pad = xs_pad.transpose(1,2)    # (b, t, f)
        xs_pad, pos_emb = self.posenc_2(xs_pad)
        masks = masks[:, :, ::2]
        xs_pad, masks, _ = self.encoders_2(xs_pad, masks, pos_emb)
        xs_pad = self.after_norm_2(xs_pad) 

        if masks is not None:
            olens = masks.squeeze(1).sum(1)
        else:
            olens = None
        return xs_pad, olens, None

        # chunk-attention part
        # if self.use_chunk:
        #     batch_size = xs_pad.shape[0]
        #     seq_len = xs_pad.shape[1]
        #     encoder_mask = chunk_attention_mask(
        #         seq_len,
        #         self.chunk_window,
        #         chunk_left_context=self.chunk_left_context,
        #         chunk_right_context=self.chunk_right_context,
        #     )
        #     encoder_masks = encoder_mask.expand(batch_size, -1, -1).to(xs_pad.device)

        #     initial_masks = masks[:, :, :-2:2][:, :, :-2:2]
        #     masks = encoder_masks & masks & masks.transpose(1, 2)

    def scripting_prep(self):
        """Torch.jit stripting preparations."""
        # disambiguate MultiSequential encoders
        file_path = "espnet.nets.pytorch_backend.transformer.repeat"
        encoders_class_name = "{}:MultiSequentialArg{}".format(
            file_path,
            self.num_sequential_args,
        )
        encoders_class = dynamic_import(encoders_class_name)
        self.encoders = encoders_class(*[layer for layer in self.encoders])