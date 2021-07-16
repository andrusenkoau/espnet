# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
from typing import Optional
from typing import Tuple
import copy
import logging

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
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.utils.dynamic_import import dynamic_import
from espnet2.asr.encoder.abs_encoder import AbsEncoder

from filelock import FileLock
import fairseq
from espnet2.asr.encoder.wav2vec2_encoder import download_w2v

class FairSeqWav2Vec2TransformerEncoder(AbsEncoder):
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
        w2v_url: str = "",
        w2v_dir_path: str = "./",
        use_chunk: bool = False,
        chunk_window: int = 0,
        chunk_left_context: int = 0,
        chunk_right_context: int = 0,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        self.min_subsampling_length = 1
        attention_dim = output_size

        # wav2vec part:
        self.w2v_model_path = download_w2v(w2v_url, w2v_dir_path)
        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                    [self.w2v_model_path],
                    arg_overrides={"data": w2v_dir_path},
                )
        self.feature_extractor = models[0].w2v_encoder.w2v_model.feature_extractor
        self.pretrained_params = copy.deepcopy(self.feature_extractor.state_dict())
        
        # subsampling x2 part:
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            torch.nn.ReLU(),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Linear(32 * (512 // 2), attention_dim),
            PositionalEncoding(attention_dim, 0.1),
        )
        
        # MHA part:
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
        self.MHA = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before=True,
                concat_after=False,
            ),
        )
        self.after_norm = LayerNorm(attention_dim)

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
        #print(f"[DEBUG]: {self.feature_extractor.conv_layers[0][0].weight.shape}")
        #print(f"[DEBUG]: {self.feature_extractor.conv_layers[0][0].weight}")

        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        
        with torch.no_grad():
            _x = self.feature_extractor(xs_pad)  #(batch, feat_dim, time)
        #print(x.shape[-1])
        #print(_x.shape[-1])
        subsampling = xs_pad.shape[-1] // _x.shape[-1]
        masks = masks[:, :, ::subsampling]
        if _x.shape[2] < masks.shape[2]:
            #print(f"[WARNING]: _x.shape[2] ({_x.shape[2]}) < masks.shape[2] ({masks.shape[2]})")
            masks = masks[:, :, :_x.shape[2]]
            
        assert _x.shape[2] == masks.shape[2], f"_x lenght {_x.shape[2]} must be equal to masks lenght {masks.shape[2]}"
        
        _x = _x.transpose(1,2).unsqueeze(1) # (b, c, t, f)
        _x = self.conv(_x)
        b, c, t, f = _x.size()
        _x = self.conv_out(_x.transpose(1, 2).contiguous().view(b, t, c * f))
        masks = masks[:, :, ::2]
        _x, masks = self.MHA(_x, masks)
        #print(f"[DEBUG]: masks.shape is {masks.shape}")
        xs_pad = self.after_norm(_x)

        if masks is not None:
            if self.use_chunk:
                olens = initial_masks.squeeze(1).sum(1)
            else:
                olens = masks.squeeze(1).sum(1)
        else:
            olens = None
        return xs_pad, olens, None

    def reload_pretrained_parameters(self):
        self.feature_extractor.load_state_dict(self.pretrained_params)
        logging.info("Pretrained Wav2Vec.feature_extractor model parameters reloaded!")


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
