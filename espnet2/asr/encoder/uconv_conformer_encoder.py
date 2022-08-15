# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder definition."""

from typing import Optional
from typing import Tuple
from typing import List

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



class Conv2dBlock(torch.nn.Module):
    def __init__(
        self,
        idim=80,
        filters_num=128,
        odim=360,
    ):
        super(Conv2dBlock, self).__init__()
        
        self.conv = torch.nn.Sequential(
            nn.Conv2d(1, filters_num, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            torch.nn.ReLU(),
            nn.Conv2d(filters_num, filters_num, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            torch.nn.ReLU(),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Linear(filters_num * ((idim+1) // 4), odim),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # x is (batch, time, feat_dim)
        x = x.unsqueeze(1)  # (batch, 1, time, feat_dim)
        x = self.conv(x)
        b, c, t, f = x.size()
        x= self.conv_out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if masks.shape[1] > 1:
            masks = masks[:, ::4, :]
        masks = masks[:, :, ::4]
        
        return x, masks


class Conv1d2lBlock(torch.nn.Module):
    def __init__(
        self,
        idim=360,
        filters_num=512,
        stride=1,
    ):
        super(Conv1d2lBlock, self).__init__()
        
        self.stride=stride
        self.conv = torch.nn.Sequential(
            nn.Conv1d(idim, filters_num, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(filters_num, filters_num, 3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv1d(filters_num, idim, 1, stride=1),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        x = x.transpose(1,2)    # (b, f, t)
        x = self.conv(x)
        x = x.transpose(1,2)    # (b, t, f)
        if masks.shape[1] > 1:
            masks = masks[:, ::self.stride, :]
        masks = masks[:, :, ::self.stride]
        
        return x, masks


class ConformerBlock(torch.nn.Module):
    def __init__(
        self,
        output_size=360,
        attention_heads=8,
        linear_units=1024,
        num_blocks = 2,
        dropout_rate=0.1,
        activation_type="swish",
        use_cnn_module=True,
        macaron_style=False,
        cnn_module_kernel=5,
        normalize_before=True,
        concat_after=False,
    ):
        super(ConformerBlock, self).__init__()
        
        activation = get_activation(activation_type)
        pos_enc_class = RelPositionalEncoding
        self.posenc = pos_enc_class(output_size, 0.1)
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (output_size, linear_units, dropout_rate, activation)        
        encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (attention_heads, output_size, dropout_rate)
        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation)        
        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )    
        self.after_norm = LayerNorm(output_size)
        
    def forward(self, x, masks):
        
        x, pos_emb = self.posenc(x)
        x, masks = self.encoders((x, pos_emb), masks)
        x = self.after_norm(x[0]) 

        
        return x, masks


class UConvConformerEncoder(AbsEncoder):
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
        output_size: int = 280,
        attention_heads: int = 8,
        linear_units: int = 1024,
        num_blocks_1_x4: int = 3,
        num_blocks_2_x8: int = 3,
        num_blocks_3_x16: int = 3,
        num_blocks_4_x8: int = 3,
        resudial_coef = 1.00,
        upsample_mode = "nearest",
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        conv1_filters: int = 64,
        conv2_filters: int = 512,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = True,
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 5,
        padding_idx: int = -1,
        use_interctc_loss: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size
        self.use_interctc_loss = use_interctc_loss

        self.resudial_coef = resudial_coef

        # first block (1_x4) -- conv + MHA:
        self.conv1 = Conv2dBlock(idim=input_size, filters_num=conv1_filters, odim=output_size)
        self.conformer_block1 = ConformerBlock(
                                        output_size=output_size,
                                        attention_heads=attention_heads,
                                        linear_units=linear_units,
                                        num_blocks=num_blocks_1_x4,
                                        dropout_rate=dropout_rate,
                                        activation_type=activation_type,
                                        use_cnn_module=use_cnn_module,
                                        macaron_style=macaron_style,
                                        cnn_module_kernel=cnn_module_kernel,
                                        normalize_before=normalize_before,
                                        concat_after=concat_after,
        )

        # second block (2_x8) -- conv + MHA:
        self.conv2 = Conv1d2lBlock(idim=output_size, filters_num=conv2_filters, stride=2)
        self.conformer_block2 = ConformerBlock(
                                        output_size=output_size,
                                        attention_heads=attention_heads,
                                        linear_units=linear_units,
                                        num_blocks=num_blocks_2_x8,
                                        dropout_rate=dropout_rate,
                                        activation_type=activation_type,
                                        use_cnn_module=use_cnn_module,
                                        macaron_style=macaron_style,
                                        cnn_module_kernel=cnn_module_kernel,
                                        normalize_before=normalize_before,
                                        concat_after=concat_after,
        )
        
        # third block (3_x16) -- conv + MHA:
        self.conv3 = Conv1d2lBlock(idim=output_size, filters_num=conv2_filters, stride=2)
        self.conformer_block3 = ConformerBlock(
                                        output_size=output_size,
                                        attention_heads=attention_heads,
                                        linear_units=linear_units,
                                        num_blocks=num_blocks_3_x16,
                                        dropout_rate=dropout_rate,
                                        activation_type=activation_type,
                                        use_cnn_module=use_cnn_module,
                                        macaron_style=macaron_style,
                                        cnn_module_kernel=cnn_module_kernel,
                                        normalize_before=normalize_before,
                                        concat_after=concat_after,
        )
        
        # fourth block (4_x8) -- upsample + MHA:
        self.upsampling4 = nn.Upsample(scale_factor=(2,1), mode=upsample_mode)
        self.conformer_block4 = ConformerBlock(
                                        output_size=output_size,
                                        attention_heads=attention_heads,
                                        linear_units=linear_units,
                                        num_blocks=num_blocks_4_x8,
                                        dropout_rate=dropout_rate,
                                        activation_type=activation_type,
                                        use_cnn_module=use_cnn_module,
                                        macaron_style=macaron_style,
                                        cnn_module_kernel=cnn_module_kernel,
                                        normalize_before=normalize_before,
                                        concat_after=concat_after,
        )


    def output_size(self) -> int:
        return self._output_size


    def align_tensors(self, x, y):
        min_len = min(x.shape[1], y.shape[1]) 
        return x[:,:min_len,:], y[:,:min_len,:]


    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, List[torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]:
        #Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
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
        
        x = xs_pad

        masks = (~make_pad_mask(ilens)[:, None, :]).to(x.device)

        # first block x4 -- conv + MHA:
        x, masks_x4 = self.conv1(x, masks)
        x, _ = self.conformer_block1(x, masks_x4)
        
        # second block x8 -- conv + MHA:
        x, masks_x8 = self.conv2(x, masks_x4)
        inter_x_1 = x     
        x, _ = self.conformer_block2(x, masks_x8)
        inter_x_2 = x

        # third block x16 -- conv + MHA:
        x, masks_x16 = self.conv3(x, masks_x8)       
        x, _ = self.conformer_block3(x, masks_x16)
        
        # fourth block x8 -- upsample + conv + MHA
        x = x.unsqueeze(1)
        x = self.upsampling4(x)
        x = x.squeeze(1)
        inter_x_3 = x
        if x.shape[1] != inter_x_2.shape[1]:
            x, inter_x_2 = self.align_tensors(x, inter_x_2)
        x = self.resudial_coef*inter_x_2 + x
        x, masks = self.conformer_block4(x, masks_x8)

        xs_pad = x


        olens = masks.squeeze(1).sum(1)

        if self.use_interctc_loss:
            return (xs_pad, [[1, inter_x_1], [2, inter_x_2], [3, inter_x_3]]), olens, None
        else:
            return xs_pad, olens, None
