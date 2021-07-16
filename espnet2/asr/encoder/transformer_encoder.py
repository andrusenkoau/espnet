# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
from typing import Optional
from typing import Tuple

import torch
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
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling2
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.utils.dynamic_import import dynamic_import
from espnet2.asr.encoder.abs_encoder import AbsEncoder


class TransformerEncoder(AbsEncoder):
    """Transformer encoder module.

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
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        conv_filters: int = 0,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        use_chunk=False,
        chunk_window: int = 0,
        chunk_left_context: int = 0,
        chunk_right_context: int = 0,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        self.min_subsampling_length = 1
        if input_layer == "linear":
            self.embed = EmbedAdapter(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, conv_filters, output_size, dropout_rate)
            self.min_subsampling_length = 7
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(input_size, output_size, dropout_rate)
            self.min_subsampling_length = 3
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(input_size, output_size, dropout_rate)
            self.min_subsampling_length = 11
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(input_size, output_size, dropout_rate)
            self.min_subsampling_length = 15
        elif input_layer == "embed":
            self.embed = EmbedAdapter(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = EmbedAdapter(
                pos_enc_class(output_size, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                MultiHeadedAttention(
                    attention_heads, output_size, attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

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
        if self.use_chunk and self.chunk_left_context > 0:
            batch_size = xs_pad.shape[0]
            seq_len = xs_pad.shape[1]
            encoder_mask = chunk_attention_mask(
                seq_len,
                self.chunk_window,
                chunk_left_context=self.chunk_left_context,
                chunk_right_context=self.chunk_right_context,
            )
            encoder_masks = encoder_mask.expand(batch_size, -1, -1).to(xs_pad.device)

            initial_masks = masks[:, :, :-2:2][:, :, :-2:2]
            masks = encoder_masks & masks & masks.transpose(1, 2)
        else:
            initial_masks = masks

        if xs_pad.size(1) < self.min_subsampling_length:
            raise TooShortUttError(
                f"has {xs_pad.size(1)} frames and is too short for subsampling "
                + f"(it needs more than {self.min_subsampling_length} frames), "
                + f"return empty results",
                xs_pad.size(1),
                self.min_subsampling_length,
            )
        xs_pad, masks = self.embed(xs_pad, masks)

        xs_pad, masks = self.encoders(xs_pad, masks)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        if masks is not None:
            if self.use_chunk:
                olens = initial_masks.squeeze(1).sum(1)
            else:
                olens = masks.squeeze(1).sum(1)
        else:
            olens = None
        return xs_pad, olens, None

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
