# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch.nn as nn
from torch.nn import LayerNorm

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks import PatchEmbed
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from .DWT_Attention import *

rearrange, _ = optional_import("einops", name="rearrange")

class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """
# # UNETR
    # def __init__(
    #     self,
    #     in_channels: int,
    #     out_channels: int,
    #     img_size: Sequence[int] | int,
    #     # feature_size: int = 32,
    #     # hidden_size: int = 24,
    #     feature_size: int = 16,
    #     hidden_size: int = 768,
    #     mlp_dim: int = 3072,
    #     num_heads: int = 12,
    #     pos_embed: str = "conv",
    #     norm_name: tuple | str = "instance",
    #     conv_block: bool = True,
    #     res_block: bool = True,
    #     dropout_rate: float = 0.0,
    #     spatial_dims: int = 3,
    #     qkv_bias: bool = False,
    # ) -> None:
    #     """
    #     Args:
    #         in_channels: dimension of input channels.
    #         out_channels: dimension of output channels.
    #         img_size: dimension of input image.
    #         feature_size: dimension of network feature size.
    #         hidden_size: dimension of hidden layer.
    #         mlp_dim: dimension of feedforward layer.
    #         num_heads: number of attention heads.
    #         pos_embed: position embedding layer type.
    #         norm_name: feature normalization type and arguments.
    #         conv_block: bool argument to determine if convolutional block is used.
    #         res_block: bool argument to determine if residual block is used.
    #         dropout_rate: faction of the input units to drop.
    #         spatial_dims: number of spatial dims.
    #         qkv_bias: apply the bias term for the qkv linear layer in self attention block

    #     Examples::

    #         # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
    #         >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

    #          # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
    #         >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)

    #         # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
    #         >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

    #     """

    #     super().__init__()

    #     if not (0 <= dropout_rate <= 1):
    #         raise ValueError("dropout_rate should be between 0 and 1.")

    #     if hidden_size % num_heads != 0:
    #         raise ValueError("hidden_size should be divisible by num_heads.")

    #     self.num_layers = 12
    #     img_size = ensure_tuple_rep(img_size, spatial_dims)
    #     self.patch_size = ensure_tuple_rep(16, spatial_dims)
    #     self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
    #     self.hidden_size = hidden_size
    #     self.classification = False
    #     self.vit = ViT(
    #         in_channels=in_channels,
    #         img_size=img_size,
    #         patch_size=self.patch_size,
    #         hidden_size=hidden_size,
    #         mlp_dim=mlp_dim,
    #         num_layers=self.num_layers,
    #         num_heads=num_heads,
    #         pos_embed=pos_embed,
    #         classification=self.classification,
    #         dropout_rate=dropout_rate,
    #         spatial_dims=spatial_dims,
    #     )
    #     self.encoder1 = UnetrBasicBlock(
    #         spatial_dims=spatial_dims,
    #         in_channels=in_channels,
    #         out_channels=feature_size,
    #         kernel_size=3,
    #         stride=1,
    #         norm_name=norm_name,
    #         res_block=res_block,
    #     )
    #     self.encoder2 = UnetrPrUpBlock(
    #         spatial_dims=spatial_dims,
    #         in_channels=hidden_size,
    #         out_channels=feature_size * 2,
    #         num_layer=2,
    #         kernel_size=3,
    #         stride=1,
    #         upsample_kernel_size=2,
    #         norm_name=norm_name,
    #         conv_block=conv_block,
    #         res_block=res_block,
    #     )
    #     self.encoder3 = UnetrPrUpBlock(
    #         spatial_dims=spatial_dims,
    #         in_channels=hidden_size,
    #         out_channels=feature_size * 4,
    #         num_layer=1,
    #         kernel_size=3,
    #         stride=1,
    #         upsample_kernel_size=2,
    #         norm_name=norm_name,
    #         conv_block=conv_block,
    #         res_block=res_block,
    #     )
    #     self.encoder4 = UnetrPrUpBlock(
    #         spatial_dims=spatial_dims,
    #         in_channels=hidden_size,
    #         out_channels=feature_size * 8,
    #         num_layer=0,
    #         kernel_size=3,
    #         stride=1,
    #         upsample_kernel_size=2,
    #         norm_name=norm_name,
    #         conv_block=conv_block,
    #         res_block=res_block,
    #     )
    #     self.decoder5 = UnetrUpBlock(
    #         spatial_dims=spatial_dims,
    #         in_channels=hidden_size,
    #         out_channels=feature_size * 8,
    #         kernel_size=3,
    #         upsample_kernel_size=2,
    #         norm_name=norm_name,
    #         res_block=res_block,
    #     )
    #     self.decoder4 = UnetrUpBlock(
    #         spatial_dims=spatial_dims,
    #         in_channels=feature_size * 8,
    #         out_channels=feature_size * 4,
    #         kernel_size=3,
    #         upsample_kernel_size=2,
    #         norm_name=norm_name,
    #         res_block=res_block,
    #     )
    #     self.decoder3 = UnetrUpBlock(
    #         spatial_dims=spatial_dims,
    #         in_channels=feature_size * 4,
    #         out_channels=feature_size * 2,
    #         kernel_size=3,
    #         upsample_kernel_size=2,
    #         norm_name=norm_name,
    #         res_block=res_block,
    #     )
    #     self.decoder2 = UnetrUpBlock(
    #         spatial_dims=spatial_dims,
    #         in_channels=feature_size * 2,
    #         out_channels=feature_size,
    #         kernel_size=3,
    #         upsample_kernel_size=2,
    #         norm_name=norm_name,
    #         res_block=res_block,
    #     )
    #     self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
    #     self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
    #     self.proj_view_shape = list(self.feat_size) + [self.hidden_size]


    # def proj_feat(self, x):
    #     new_view = [x.size(0)] + self.proj_view_shape
    #     x = x.view(new_view)
    #     x = x.permute(self.proj_axes).contiguous()
    #     return x

    # def forward(self, x_in):
    #     x, hidden_states_out = self.vit(x_in)
    #     enc1 = self.encoder1(x_in)
    #     x2 = hidden_states_out[3]
    #     enc2 = self.encoder2(self.proj_feat(x2))
    #     x3 = hidden_states_out[6]
    #     enc3 = self.encoder3(self.proj_feat(x3))
    #     x4 = hidden_states_out[9]
    #     enc4 = self.encoder4(self.proj_feat(x4))
    #     dec4 = self.proj_feat(x)
    #     dec3 = self.decoder5(dec4, enc4)
    #     dec2 = self.decoder4(dec3, enc3)
    #     dec1 = self.decoder3(dec2, enc2)
    #     out = self.decoder2(dec1, enc1)
    #     return self.out(out)
# frequency UNETR
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        # feature_size: int = 32,
        # hidden_size: int = 24,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "conv",
        norm_name: tuple | str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.
            qkv_bias: apply the bias term for the qkv linear layer in self attention block

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)

            # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
        )
        # unetr架構
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.dwtatt_conv=DWT_Attention() #dwtattention
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=o_channels)
        # unetr
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]
        self.proj = nn.Linear(864, 216)
        #----MSA-----
        self.blocks = nn.ModuleList(
                [TransformerBlock(hidden_size=hidden_size, mlp_dim=mlp_dim, num_heads=num_heads, dropout_rate=dropout_rate, qkv_bias=qkv_bias,mode="SABlock") for i in range(self.num_layers)]
            )
        self.norm = nn.LayerNorm(self.hidden_size)
        #----MSA----
	    
	
    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x
    

    def forward(self, x_in):
        # unetr
        x, hidden_states_out = self.vit(x_in)
        # print("im3:{},im6:{},im9:{}".format(hidden_states_out[3].shape,hidden_states_out[6].shape,hidden_states_out[9].shape))
        enc1 = self.encoder1(x_in)
        # print("1",enc1.shape)SS
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))
        # print("2",enc2.shape)
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))
        # print("3",enc3.shape)
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))
        # print("4",enc4.shape)
        #--------------Concat MSA---------------
        x_concat=torch.concat([x, x2,x3,x4], axis=1).permute([0, 2, 1])
        x_concat=self.proj(x_concat).permute([0, 2, 1])

        #---------MSA-------------------
        
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x_concat.shape[0], -1, -1)
            x_concat = torch.cat((cls_token, x_concat), dim=1)
        # hidden_states_out = []
        for blk in self.blocks:
            x_concat = blk(x_concat)
            # hidden_states_out.append(x)
        x_concat = self.norm(x_concat)
        if hasattr(self, "classification_head"):
            x_concat = self.classification_head(x_concat[:, 0])
        # -------------MSA------------------
        enc4 = self.encoder4(self.proj_feat(x_concat))
        # print("concat",x_concat.shape)

        #--------------Concat MSA----------------

        # print("x:",x.shape)
        #------------------DWT_attention------------------
        # x_dwt_attention=self.dwtatt_conv(x)
        # x_dwt_attention=x_dwt_attention.reshape(4,216,768)
        # # print("x_dwtattn",x_dwt_attention.shape)
        # dec4 = self.proj_feat(x_dwt_attention)
        #------------------DWT_attention------------------
        dec4 = self.proj_feat(x)

        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        # print("dec4:{},dec3:{},dec2:{},dec1:{},out:{}".format(dec4.shape,dec3.shape,dec2.shape,dec1.shape,out.shape))
        return self.out(out)
#   ours  DWT_UNETR


#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         img_size: Sequence[int] | int,
#         # feature_size: int = 32,
#         # hidden_size: int = 24,
#         feature_size: int = 48,
#         hidden_size: int = 768,
#         mlp_dim: int = 3072,
#         num_heads: int = 12,
#         pos_embed: str = "conv",
#         norm_name: tuple | str = "instance",
#         conv_block: bool = True,
#         res_block: bool = True,
#         dropout_rate: float = 0.0,
#         spatial_dims: int = 3,
#         qkv_bias: bool = False,
#     ) -> None:
#         """
#         Args:
#             in_channels: dimension of input channels.
#             out_channels: dimension of output channels.
#             img_size: dimension of input image.
#             feature_size: dimension of network feature size.
#             hidden_size: dimension of hidden layer.
#             mlp_dim: dimension of feedforward layer.
#             num_heads: number of attention heads.
#             pos_embed: position embedding layer type.
#             norm_name: feature normalization type and arguments.
#             conv_block: bool argument to determine if convolutional block is used.
#             res_block: bool argument to determine if residual block is used.
#             dropout_rate: faction of the input units to drop.
#             spatial_dims: number of spatial dims.
#             qkv_bias: apply the bias term for the qkv linear layer in self attention block

#         Examples::

#             # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
#             >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

#              # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
#             >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)

#             # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
#             >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

#         """

#         super().__init__()

#         if not (0 <= dropout_rate <= 1):
#             raise ValueError("dropout_rate should be between 0 and 1.")

#         if hidden_size % num_heads != 0:
#             raise ValueError("hidden_size should be divisible by num_heads.")

#         self.num_layers = 1
#         img_size = ensure_tuple_rep(img_size, spatial_dims)
#         self.patch_size = ensure_tuple_rep(2, spatial_dims)
#         self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
#         self.hidden_size = hidden_size
#         self.classification = False

#         self.patch_embed = PatchEmbed(
#             patch_size=self.patch_size,
#             in_chans=in_channels,
#             embed_dim=48,
#             norm_layer=nn.LayerNorm,  # type: ignore
#             spatial_dims=spatial_dims,
#         )#Linear Embed Block
#         self.Patch_merge0=PatchMerging(dim=48, 
#             norm_layer= nn.LayerNorm, 
#             spatial_dims= 3)#patch merging

#         self.vit1 = ViT(
#             in_channels=96,#96
#             img_size=tuple(item//4 for item in img_size),
#             patch_size=self.patch_size,
#             hidden_size=96,
#             mlp_dim=mlp_dim,
#             num_layers=6,
#             num_heads=num_heads,
#             pos_embed=pos_embed,
#             classification=self.classification,
#             dropout_rate=dropout_rate,
#             spatial_dims=spatial_dims,
#             qkv_bias=qkv_bias,
#         )
#         self.Patch_merge1=PatchMerging(
#             dim=96,
#             norm_layer= nn.LayerNorm,
#             spatial_dims= 3
#         )
#         self.vit2 = ViT(
#             in_channels=192,
#             img_size=tuple(item//8 for item in img_size),
#             patch_size=self.patch_size,
#             hidden_size=192,
#             mlp_dim=mlp_dim,
#             num_layers=6,
#             num_heads=num_heads,
#             pos_embed=pos_embed,
#             classification=self.classification,
#             dropout_rate=dropout_rate,
#             spatial_dims=spatial_dims,
#             qkv_bias=qkv_bias,
#         )
#         self.Patch_merge2=PatchMerging(
#             dim=192,
#             norm_layer= nn.LayerNorm,
#             spatial_dims= 3
#         )
#         self.vit3 = ViT(
#             in_channels=384,
#             img_size=tuple(item//16 for item in img_size),
#             patch_size=self.patch_size,
#             hidden_size=384,
#             mlp_dim=mlp_dim,
#             num_layers=6,
#             num_heads=num_heads,
#             pos_embed=pos_embed,
#             classification=self.classification,
#             dropout_rate=dropout_rate,
#             spatial_dims=spatial_dims,
#             qkv_bias=qkv_bias,
#         )
#         self.Patch_merge3=PatchMerging(
#             dim=384,
#             norm_layer= nn.LayerNorm,
#             spatial_dims= 3
#         )

#         # skip connection
#         self.encoder1 = UnetrBasicBlock(
#             spatial_dims=spatial_dims,
#             in_channels=in_channels,
#             out_channels=feature_size,#48
#             kernel_size=3,
#             stride=1,
#             norm_name=norm_name,
#             res_block=True,
#         )

#         self.encoder2 = UnetrBasicBlock(
#             spatial_dims=spatial_dims,
#             in_channels=feature_size,#48
#             out_channels=feature_size,#48
#             kernel_size=3,
#             stride=1,
#             norm_name=norm_name,
#             res_block=True,
#         )

#         self.encoder3 = UnetrBasicBlock(
#             spatial_dims=spatial_dims,
#             in_channels=2 * feature_size,#96
#             out_channels=2 * feature_size,#96
#             kernel_size=3,
#             stride=1,
#             norm_name=norm_name,
#             res_block=True,
#         )

#         self.encoder4 = UnetrBasicBlock(
#             spatial_dims=spatial_dims,
#             in_channels=4 * feature_size,#192
#             out_channels=4 * feature_size,#192
#             kernel_size=3,
#             stride=1,
#             norm_name=norm_name,
#             res_block=True,
#         )

#         self.encoder10 = UnetrBasicBlock(
#             spatial_dims=spatial_dims,
#             in_channels=16 * feature_size,#768
#             out_channels=16 * feature_size,#768
#             kernel_size=3,
#             stride=1,
#             norm_name=norm_name,
#             res_block=True,
#         )
#         #upsample
#         self.decoder5 = UnetrUpBlock(
#             spatial_dims=spatial_dims,
#             in_channels=16 * feature_size,#768
#             out_channels=8 * feature_size,#384
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             res_block=True,
#         )

#         self.decoder4 = UnetrUpBlock(
#             spatial_dims=spatial_dims,
#             in_channels=feature_size * 8,#384
#             out_channels=feature_size * 4,#192
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             res_block=True,
#         )

#         self.decoder3 = UnetrUpBlock(
#             spatial_dims=spatial_dims,
#             in_channels=feature_size * 4,#192
#             out_channels=feature_size * 2,#96
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             res_block=True,
#         )
#         self.decoder2 = UnetrUpBlock(
#             spatial_dims=spatial_dims,
#             in_channels=feature_size * 2,#96
#             out_channels=feature_size,#48
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             res_block=True,
#         )

#         self.decoder1 = UnetrUpBlock(
#             spatial_dims=spatial_dims,
#             in_channels=feature_size,#48
#             out_channels=feature_size,#48
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             res_block=True,
#         )
#         #dwt_attention
#         self.dwtatt_conv1to2=DWT_Attention(in_feature=96,out_feature=96) #out 192
#         self.dwtatt_conv2to3=DWT_Attention(in_feature=192,out_feature=192) #out 384 
       
#         self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
#         # unetr
#         self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
#         self.proj_view_shape = list(self.feat_size) + [self.hidden_size]
	    
	
#     def proj_feat(self, x):
#         new_view = [x.size(0)] + self.proj_view_shape
#         x = x.view(new_view)
#         x = x.permute(self.proj_axes).contiguous()
#         return x
    
#     def proj_out(self, x, normalize=False):
#         if normalize:
#             x_shape = x.size()
#             if len(x_shape) == 5:
#                 n, ch, d, h, w = x_shape
#                 x = rearrange(x, "n c d h w -> n d h w c")
#                 x = F.layer_norm(x, [ch])
#                 x = rearrange(x, "n d h w c -> n c d h w")
#             elif len(x_shape) == 4:
#                 n, ch, h, w = x_shape
#                 x = rearrange(x, "n c h w -> n h w c")
#                 x = F.layer_norm(x, [ch])
#                 x = rearrange(x, "n h w c -> n c h w")
#         return x

#     def forward(self, x_in):
#         #stage1
#         x_in1=self.patch_embed(x_in)#1->48
#         # x_in2=self.Patch_merge0(self.proj_out(x_in1,True))#48->96
#         x_in2=self.Patch_merge0(x_in1)#48->96

#         #stage2
#         x1, _ = self.vit1(x_in2)#96->96
#         # x1_out=self.Patch_merge1(x1.view(4,24,24,24,96))#96->192
#         x1_out=x1.view(4,12,12,12,192)
#         #stage3
#         x2,_= self.vit2(x1)#192->192
#         # x2=x2.view(4,6,6,6,192)
#         # x2_out=self.Patch_merge2(x2.view(4,12,12,12,192))#192->384
#         x2_out=x2.view(4,6,6,6,384)
#         #stage4
#         x3,_ = self.vit3(x2_out)#384->384
#         # x3_out=self.Patch_merge3(x3.view(4,6,6,6,384))#384->768
#         x3_out=x3.view(4,3,3,3,768)

#         enc0 = self.encoder1(x_in)#1->48
#         enc1 = self.encoder2(x_in1)#48->48
#         enc2 = self.encoder3(x_in2.permute(0,4,1,2,3))#96->96
#         enc3 = self.encoder4(x1_out.permute(0,4,1,2,3))#192->192
#         dec4 = self.encoder10(x3_out.permute(0,4,1,2,3))#768->768

#         #dwt_attention
#         # x_dwt_attention1to2=self.dwtatt_conv1to2(enc2)
#         x_dwt_attention2to3=self.dwtatt_conv2to3(enc3)
#         # print("x_dwtattn2to3",x_dwt_attention2to3.shape)
#         # x_dwt_attention=x_dwt_attention.reshape(4,216,768)
        

#         dec3 = self.decoder5(dec4, x2_out.permute(0,4,1,2,3))#768->384
#         # dec2 = self.decoder4(dec3, enc3)#384->192
#         dec2 = self.decoder4(x_dwt_attention2to3, enc3)
#         dec1 = self.decoder3(dec2, enc2)#192->96
#         # dec1 = self.decoder3(x_dwt_attention1to2, enc2)#192->96
#         dec0 = self.decoder2(dec1, enc1)#96->48
#         # print("final",dec0.shape,enc0.shape)
#         out = self.decoder1(dec0, enc0)#48->48
        
#         return self.out(out)
    
# class PatchMergingV2(nn.Module):
#     """
#     Patch merging layer based on: "Liu et al.,
#     Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
#     <https://arxiv.org/abs/2103.14030>"
#     https://github.com/microsoft/Swin-Transformer
#     """

#     def __init__(self, dim: int, norm_layer: type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3) -> None:
#         """
#         Args:
#             dim: number of feature channels.
#             norm_layer: normalization layer.
#             spatial_dims: number of spatial dims.
#         """

#         super().__init__()
#         self.dim = dim
#         if spatial_dims == 3:
#             self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
#             self.norm = norm_layer(8 * dim)
#         elif spatial_dims == 2:
#             self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
#             self.norm = norm_layer(4 * dim)

#     def forward(self, x):
#         x_shape = x.size()
#         if len(x_shape) == 5:
#             b, d, h, w, c = x_shape
#             pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
#             if pad_input:
#                 x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
#             x = torch.cat(
#                 [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))], -1
#             )

#         elif len(x_shape) == 4:
#             b, h, w, c = x_shape
#             pad_input = (h % 2 == 1) or (w % 2 == 1)
#             if pad_input:
#                 x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
#             x = torch.cat([x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))], -1)

#         x = self.norm(x)
#         x = self.reduction(x)
#         return x


# class PatchMerging(PatchMergingV2):
#     """The `PatchMerging` module previously defined in v0.9.0."""

#     def forward(self, x):
#         x_shape = x.size()
#         if len(x_shape) == 4:
#             return super().forward(x)
#         if len(x_shape) != 5:
#             raise ValueError(f"expecting 5D x, got {x.shape}.")
#         b, d, h, w, c = x_shape
#         pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
#         if pad_input:
#             x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
#         x0 = x[:, 0::2, 0::2, 0::2, :]
#         x1 = x[:, 1::2, 0::2, 0::2, :]
#         x2 = x[:, 0::2, 1::2, 0::2, :]
#         x3 = x[:, 0::2, 0::2, 1::2, :]
#         x4 = x[:, 1::2, 0::2, 1::2, :]
#         x5 = x[:, 0::2, 1::2, 0::2, :]
#         x6 = x[:, 0::2, 0::2, 1::2, :]
#         x7 = x[:, 1::2, 1::2, 1::2, :]
#         x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
#         x = self.norm(x)
#         x = self.reduction(x)
#         return x


# MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}