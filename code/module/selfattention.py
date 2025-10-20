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

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.utils import optional_import
from .DWT_IDWT_layer import *
from .torch_wavelets import DWT_2D, IDWT_2D

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")


# source SABlock
# class SABlock(nn.Module):
#     """
#     A self-attention block, based on: "Dosovitskiy et al.,
#     An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
#     """


#     def __init__(
#         self,
#         hidden_size: int,
#         num_heads: int,
#         dropout_rate: float = 0.0,
#         qkv_bias: bool = False,
#         save_attn: bool = False,
#     ) -> None:
#         """
#         Args:
#             hidden_size (int): dimension of hidden layer.
#             num_heads (int): number of attention heads.
#             dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
#             qkv_bias (bool, optional): bias term for the qkv linear layer. Defaults to False.
#             save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.

#         """

#         super().__init__()

#         if not (0 <= dropout_rate <= 1):
#             raise ValueError("dropout_rate should be between 0 and 1.")

#         if hidden_size % num_heads != 0:
#             raise ValueError("hidden size should be divisible by num_heads.")

#         self.num_heads = num_heads
#         self.out_proj = nn.Linear(hidden_size, hidden_size)
#         self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
#         self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
#         self.out_rearrange = Rearrange("b h l d -> b l (h d)")
#         self.drop_output = nn.Dropout(dropout_rate)
#         self.drop_weights = nn.Dropout(dropout_rate)
#         self.head_dim = hidden_size // num_heads
#         self.scale = self.head_dim**-0.5
#         self.save_attn = save_attn
#         self.att_mat = torch.Tensor()

#     def forward(self, x):
#         output = self.input_rearrange(self.qkv(x))
#         q, k, v = output[0], output[1], output[2]
#         att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
#         if self.save_attn:
#             # no gradients and new tensor;
#             # https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
#             self.att_mat = att_mat.detach()

#         att_mat = self.drop_weights(att_mat)
#         x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
#         x = self.out_rearrange(x)
#         x = self.out_proj(x)
#         x = self.drop_output(x)
#         return x
#ours    
class SABlock(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0, qkv_bias: bool = False) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            qkv_bias: bias term for the qkv linear layer.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        # self.save_attn = save_attn
        self.att_mat = torch.Tensor()

    def forward(self, x):
        output = self.input_rearrange(self.qkv(x))
        q, k, v = output[0], output[1], output[2]
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)

        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x
class WABlock(nn.Module):
#revise 2
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0, qkv_bias: bool = False) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            qkv_bias: bias term for the qkv linear layer.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")
        self.hidden_size=hidden_size
        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        #--------------wavelet vit-----------------------
        self.dwt = DWT_3D(wavename='haar')
        self.idwt = IDWT_3D(wavename='haar')
        self.filter = nn.Sequential(
            nn.Conv3d(hidden_size*8, hidden_size*2, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm3d(hidden_size*2),
            nn.ReLU(),
        )
        self.kv = nn.Sequential(
            nn.LayerNorm(27),
            nn.Linear(27, 216)
        )
        self.proj = nn.Linear(hidden_size*2, hidden_size)
        #--------------wavelet vit-----------------------

    def forward(self, x):
        output = self.input_rearrange(self.qkv(x))
        q=output[0]
        del output

        #---------------Wavelet ViT--------------------------
        B, N, C = x.shape #4,216,768
        x = x.reshape((B,6 ,6,6, C)).permute([0, 4, 1, 2,3]) #4 768 6 6 6
        # print(x.get_device())
        LLL , LLH , LHL , LHH , HLL , HLH , HHL , HHH= self.dwt(x) #(4 768 ,6,6,6)->8 x(4 768 ,3,3,3)
        x_dwt=torch.concat([LLL , LLH , LHL , LHH , HLL , HLH , HHL , HHH],axis=1)#8 x(4 768 ,3,3,3)->(4,768*8,3,3,3)
        #IDWT
        x_idwt = self.idwt(LLL , LLH , LHL , LHH , HLL , HLH , HHL , HHH) #8*(4,768,3,3,3)->(4, 192, 6, 6, 6)
        del LLL
        del LLH 
        del LHL
        del LHH 
        del HLL 
        del HLH
        del HHL
        del HHH  #clear
        # print("x in SA idwt",x_idwt.shape)
        x_idwt = x_idwt.reshape((B, -1, x_idwt.shape[-2] * x_idwt.shape[-1]*x_idwt.shape[-3])).permute([0, 2, 1]) #(4,192,6,6,6)->(4, 216, 192)
        # print("x_idwt",x_idwt.shape)

        # print("x in SA dwt ",x_dwt.shape)
        x_dwt = self.filter(x_dwt) #((4,768*8,3,3,3)->4 768*2 ,3,3,3)
        # print("x_dwt",x_dwt.shape)
        kv=self.kv(x_dwt.reshape((4,self.hidden_size*2,27)))
        # print("kv",kv.shape)
        k, v = kv[:,0:self.hidden_size,:], kv[:,self.hidden_size:self.hidden_size*2,:]
        del kv #clear
        # print("k,v",k.shape,v.shape)
        k=k.reshape((B, N, self.num_heads, C // self.num_heads)).permute([0, 2, 1, 3])
        v=v.reshape((B, N, self.num_heads, C // self.num_heads)).permute([0, 2, 1, 3])
        # print("k,v",k.shape,v.shape)
        #---------------Wavelet ViT--------------------------

        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        # print(x.shape)
        x = self.out_rearrange(x)#4,216,768
        # print(x.shape)
        # x = self.out_proj(x)
        x = self.proj(torch.concat([x, x_idwt], axis=-1))
        x = self.drop_output(x)
        return x