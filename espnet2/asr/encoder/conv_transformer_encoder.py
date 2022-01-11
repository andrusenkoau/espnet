# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import chunk_attention_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import (
    EmbedAdapter,  # noqa: H301
    PositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
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


class ConvTransformerEncoder(AbsEncoder):
    """ConvTransformer encoder module based on https://arxiv.org/abs/2008.05750.
    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
        padding_idx: padding_idx for input_layer=embed
    """

    # track the number of arguments of sequential modules for JIT disamdiguation
    num_sequential_args = 2

    def __init__(
        self,
        input_size: int,
        model_size: Optional[str] = "full",
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        conv1_filters: int = 32,
        conv2_filters: int = 512,
        conv3_filters: int = 512,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        padding_idx: int = -1,
        use_chunk=False,
        chunk_window=None,
        chunk_left_context=None,
        chunk_right_context=None,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size
        self.model_size = model_size
        attention_dim = output_size

        assert model_size in ["full", "small"]

        if model_size == "full":
            # first block -- conv + MHA:
            self.conv1 = torch.nn.Sequential(
                nn.Conv2d(1, conv1_filters, kernel_size=(3,7), stride=(1,2), padding=(1,3)),
                torch.nn.ReLU(),
                nn.Conv2d(conv1_filters, conv1_filters, kernel_size=(3,5), stride=(2,2), padding=(1,2)),
                torch.nn.ReLU(),
            )
            self.conv1_out = torch.nn.Sequential(
                torch.nn.Linear(conv1_filters * (((input_size+1) // 2) // 2), attention_dim),
                PositionalEncoding(attention_dim, 0.1),
            )
            positionwise_layer1 = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
            self.MHA1 = repeat(
                2,
                lambda lnum: EncoderLayer(attention_dim, MultiHeadedAttention(6, attention_dim, dropout_rate),
                    positionwise_layer1(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before=True,
                    concat_after=False,
                ),
            )
            self.after_norm1 = LayerNorm(attention_dim)
            
            # second block -- conv + MHA:
            self.conv2 = torch.nn.Sequential(
                nn.Conv1d(attention_dim, conv2_filters, 3, stride=1, padding=1),
                torch.nn.ReLU(),
                nn.Conv1d(conv2_filters, conv2_filters, 3, stride=2, padding=1),
                torch.nn.ReLU(),
                nn.Conv1d(conv2_filters, attention_dim, 1, stride=1),
                torch.nn.ReLU(),
            )
            self.posenc2 = PositionalEncoding(attention_dim, 0.1)
            positionwise_layer2 = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
            self.MHA2 = repeat(
                2,
                lambda lnum: EncoderLayer(attention_dim, MultiHeadedAttention(attention_heads, attention_dim, dropout_rate),
                    positionwise_layer2(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before=True,
                    concat_after=False,
                ),
            )
            self.after_norm2 = LayerNorm(attention_dim)
            
            # third block -- conv + MHA:
            self.conv3 = torch.nn.Sequential(
                nn.Conv1d(attention_dim, 512, 3, stride=1, padding=1),
                torch.nn.ReLU(),
                nn.Conv1d(conv3_filters, conv3_filters, 3, stride=2, padding=1),
                torch.nn.ReLU(),
                nn.Conv1d(conv3_filters, attention_dim, 1, stride=1),
                torch.nn.ReLU(),
            )
            self.posenc3 = PositionalEncoding(attention_dim, 0.1)
            positionwise_layer3 = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
            self.MHA3 = repeat(
                num_blocks,
                lambda lnum: EncoderLayer(attention_dim, MultiHeadedAttention(attention_heads, attention_dim, dropout_rate),
                    positionwise_layer3(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before=True,
                    concat_after=False,
                ),
            )
            self.after_norm3 = LayerNorm(attention_dim)


        if model_size == "small":
                        # first block -- conv + MHA:
            self.conv1 = torch.nn.Sequential(
                nn.Conv2d(1, conv1_filters, kernel_size=(3,7), stride=(2,2), padding=(1,3)),
                torch.nn.ReLU(),
                nn.Conv2d(conv1_filters, conv1_filters, kernel_size=(3,5), stride=(2,2), padding=(1,2)),
                torch.nn.ReLU(),
            )
            self.conv1_out = torch.nn.Sequential(
                torch.nn.Linear(conv1_filters * (((input_size+1) // 2) // 2), attention_dim),
                PositionalEncoding(attention_dim, 0.1),
            )
            positionwise_layer1 = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
            self.MHA1 = repeat(
                3,
                lambda lnum: EncoderLayer(attention_dim, MultiHeadedAttention(attention_heads, attention_dim, dropout_rate),
                    positionwise_layer1(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before=True,
                    concat_after=False,
                ),
            )
            self.after_norm1 = LayerNorm(attention_dim)
            
            # second block -- conv + MHA:
            self.conv2 = torch.nn.Sequential(
                nn.Conv1d(attention_dim, conv2_filters, 3, stride=1, padding=1),
                torch.nn.ReLU(),
                nn.Conv1d(conv2_filters, conv2_filters, 3, stride=2, padding=1),
                torch.nn.ReLU(),
                nn.Conv1d(conv2_filters, attention_dim, 1, stride=1),
                torch.nn.ReLU(),
            )
            self.posenc2 = PositionalEncoding(attention_dim, 0.1)
            positionwise_layer2 = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
            self.MHA2 = repeat(
                num_blocks,
                lambda lnum: EncoderLayer(attention_dim, MultiHeadedAttention(attention_heads, attention_dim, dropout_rate),
                    positionwise_layer2(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before=True,
                    concat_after=False,
                ),
            )
            self.after_norm2 = LayerNorm(attention_dim)


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
        """Embed positions in tensor.
        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """

        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        # chunk-attention part
        if self.use_chunk:
            batch_size = xs_pad.shape[0]
            seq_len = xs_pad.shape[1]
            encoder_mask = chunk_attention_mask(
                seq_len,
                self.chunk_window,
                chunk_left_context=self.chunk_left_context,
                chunk_right_context=self.chunk_right_context,
            )
            encoder_masks = encoder_mask.expand(batch_size, -1, -1).to(xs_pad.device)
            initial_masks = masks[:, :, ::8]
            masks = encoder_masks & masks & masks.transpose(1, 2)
            
            

        if self.model_size == "full":
            # first block -- conv + MHA:
            x = xs_pad.unsqueeze(1)   # (b, c, t, f)
            x = self.conv1(x)
            b, c, t, f = x.size()
            x = self.conv1_out(x.transpose(1, 2).contiguous().view(b, t, c * f))
            masks = masks[:, ::2, ::2]
            x, masks = self.MHA1(x, masks)
            x = self.after_norm1(x)   # (b, t, f)
            
            # second block -- conv + MHA:
            x = x.transpose(1,2)    # (b, f, t)
            x = self.conv2(x)
            x = x.transpose(1,2)    # (b, t, f)
            x = self.posenc2(x)
            masks = masks[:, ::2, ::2]
            x, masks = self.MHA2(x, masks)
            x = self.after_norm2(x)
            
            # third block -- conv + MHA:
            x = x.transpose(1,2)
            x = self.conv3(x)
            x = x.transpose(1,2)
            x = self.posenc3(x)
            masks = masks[:, ::2, ::2]
            x, masks = self.MHA3(x, masks)
            x = self.after_norm3(x)

        if self.model_size == "small":
            # first block -- conv + MHA:
            x = xs_pad.unsqueeze(1)   # (b, c, t, f)
            x = self.conv1(x)
            b, c, t, f = x.size()
            x = self.conv1_out(x.transpose(1, 2).contiguous().view(b, t, c * f))
            masks = masks[:, ::4, ::4]
            x, masks = self.MHA1(x, masks)
            x = self.after_norm1(x)   # (b, t, f)
            
            # second block -- conv + MHA:
            x = x.transpose(1,2)    # (b, f, t)
            x = self.conv2(x)
            x = x.transpose(1,2)    # (b, t, f)
            x = self.posenc2(x)
            masks = masks[:, ::2, ::2]
            x, masks = self.MHA2(x, masks)
            x = self.after_norm2(x)

        if masks is not None:
            if self.use_chunk:
                #print(f"[DEBUG]: initial_masks.shape is {initial_masks.shape}")
                olens = initial_masks.squeeze(1).sum(1)
            else:
                olens = masks.squeeze(1).sum(1)
        else:
            olens = None

        return x, olens, None

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