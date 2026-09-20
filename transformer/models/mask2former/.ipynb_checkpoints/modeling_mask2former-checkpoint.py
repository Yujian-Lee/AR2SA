# coding=utf-8
# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Mask2Former model."""
from thop import profile
import matplotlib.pyplot as plt
import math
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F
import numpy as np
import torch
from torch import Tensor, nn
import math
from ... import AutoBackbone, SwinConfig
from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scipy_available,
    replace_return_docstrings,
    requires_backends,
)
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_mask2former import Mask2FormerConfig
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence
import math
if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

logger = logging.get_logger(__name__)


_CONFIG_FOR_DOC = "Mask2FormerConfig"
_CHECKPOINT_FOR_DOC = "facebook/mask2former-swin-small-coco-instance"
_IMAGE_PROCESSOR_FOR_DOC = "Mask2FormerImageProcessor"

MASK2FORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/mask2former-swin-small-coco-instance",
    # See all mask2former models at https://huggingface.co/models?filter=mask2former
]
class Semantic_flow(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(Semantic_flow, self).__init__()
        #inchannel为低分辨率图片通道数，outchannel为高分辨率图片通道数
        self.down_h=nn.Conv2d( inchannel, outchannel,1,bias=False)
        self.down_l=nn.Conv2d( outchannel, outchannel,1,bias=False)
        self.flow_make=nn.Conv2d( outchannel*2,2,kernel_size=3,padding=1,bias=False)
        #self.conv=nn.Conv2d(inchannel*2,outchannel,1)
    def forward(self,h_feature, low_feature):
        B, C, HW_h = h_feature.shape
        B, C, HW_l = low_feature.shape
    
        Hh = int(HW_h ** 0.5)
        Hl = int(HW_l ** 0.5)
    
        # reshape 回 feature map
        h_feature = h_feature.view(B, C, Hh, Hh)
        low_feature = low_feature.view(B, C, Hl, Hl)
    
        h_feature_origin = h_feature
    
        size = Hl
        # 将high-level 和 low-level feature分别通过两个1x1卷积进行压缩
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        # 将high-level feature进行双线性上采样
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=False)
        # 预测语义流场 === 其实就是输入一个3x3的卷积
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        # 将Flow Field warp 到当前的 high-level feature中
        h_feature = self.flow_warp(h_feature_origin, flow, size=size)
        #h_feature=self.conv(h_feature)
        B, C, H, W = h_feature.shape
        h_feature = h_feature.view(B, C, H * W)
        return h_feature

    @staticmethod
    def flow_warp(inputs, flow, size):
        out_h = size  # 对应高分辨率的low-level feature的特征图尺寸
        out_w = size
        n, c, h, w = inputs.size()  # 对应低分辨率的high-level feature的4个输入维度


        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(inputs).to(inputs.device)
        # 从-1到1等距离生成out_h个点，每一行重复out_w个点，最终生成(out_h, out_w)的像素点
        # 生成w的转置矩阵
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, out_h),
            torch.linspace(-1.0, 1.0, out_w),
            indexing="ij"
        )
        # 展开后进行合并
        grid = torch.stack((grid_x, grid_y), 2)
        grid = grid.unsqueeze(0).repeat(n, 1, 1, 1).to(inputs.device)
        grid = grid + flow.permute(0,2,3,1) / norm
        #print(grid.size())
        # grid指定由input空间维度归一化的采样像素位置，其大部分值应该在[ -1, 1]的范围内
        # 如x=-1,y=-1是input的左上角像素，x=1,y=1是input的右下角像素。
        # 具体可以参考《Spatial Transformer Networks》，下方参考文献[2]
        output = F.grid_sample(inputs, grid,align_corners=False)
        #print(output.size())
        return output
class TokenConv(nn.Conv2d):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        groups = kwargs['groups'] if 'groups' in kwargs.keys() else 1
        self.skip = nn.Conv1d(in_channels=kwargs['in_channels'],
                              out_channels=kwargs['out_channels'],
                              kernel_size=1, bias=False,
                              groups=groups)

    def get_grid_index(self,init_size, map_size, device):
        """For each initial grid, get its index in the feature map.
        Returns:
            idx (LongTensor[B, N_init]): index in flattened feature map.

        Args:
            init_grid_size(list[int] or tuple[int]): initial grid resolution in
                format [H_init, W_init].
            map_size(list[int] or tuple[int]): feature map resolution in format
                [H, W].
            device: the device of output
        """
        H_init, W_init = init_size
        H, W = map_size
        idx = torch.arange(H * W, device=device).reshape(1, 1, H, W)
        idx = F.interpolate(idx.float(), [H_init, W_init], mode='nearest').long()
        return idx.flatten()

    def token2map(self,token_dict):
        """Transform vision tokens to feature map. This function only
        works when the resolution of the feature map is not higher than
        the initial grid structure.
        Returns:
            x_out (Tensor[B, C, H, W]): feature map.

        Args:
            token_dict (dict): dict for token information.
        """

        x = token_dict['x']
        H, W = token_dict['map_size']
        H_init, W_init = token_dict['init_grid_size']
        idx_token = token_dict['idx_token']
        B, N, C = x.shape
        N_init = H_init * W_init
        device = x.device

        if N_init == N and N == H * W:
            # for the initial tokens with grid structure, just reshape
            return x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # for each initial grid, get the corresponding index in
        # the flattened feature map.
        idx_hw = self.get_grid_index(
            [H_init, W_init], [H, W], device=device)[None, :].expand(B, -1)
        idx_batch = torch.arange(B, device=device)[:, None].expand(B, N_init)
        value = x.new_ones(B * N_init)

        # choose the way with fewer flops.
        if N_init < N * H * W:
            # use sparse matrix multiplication
            # Flops: B * N_init * (C+2)
            idx_hw = idx_hw + idx_batch * H * W
            idx_tokens = idx_token + idx_batch * N
            coor = torch.stack([idx_hw, idx_tokens], dim=0).reshape(2, B * N_init)

            # torch.sparse do not support fp16
            with torch.cuda.amp.autocast(enabled=False):
                # torch.sparse do not support gradient for
                # sparse tensor, so we detach it
                value = value.detach().float()

                # build a sparse matrix with the shape [B * H * W, B * N]
                A = torch.sparse.FloatTensor(coor, value, torch.Size([B * H * W, B * N]))

                # normalize the weight for each row
                all_weight = A @ x.new_ones(B * N, 1).type(torch.float32) + 1e-6
                value = value / all_weight[idx_hw.reshape(-1), 0]

                # update the matrix with normalize weight
                A = torch.sparse.FloatTensor(coor, value, torch.Size([B * H * W, B * N]))

                # sparse matrix multiplication
                x_out = A @ x.reshape(B * N, C).type(torch.float32)  # [B*H*W, C]

        else:
            # use dense matrix multiplication
            # Flops: B * N * H * W * (C+2)
            coor = torch.stack([idx_batch, idx_hw, idx_token], dim=0).reshape(3, B * N_init)

            # build a matrix with shape [B, H*W, N]
            A = torch.sparse.FloatTensor(coor, value, torch.Size([B, H * W, N])).to_dense()
            # normalize the weight
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

            x_out = A @ x  # [B, H*W, C]

        x_out = x_out.type(x.dtype)
        x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x_out

    def map2token(self,feature_map, token_dict):
        idx_token = token_dict['idx_token']
        N = token_dict['token_num']
        H_init, W_init = token_dict['init_grid_size']
        N_init = H_init * W_init

        # agg_weight = token_dict['agg_weight'] if 'agg_weight' in token_dict.keys() else None
        agg_weight = None  # we do not use the weight value here

        B, C, H, W = feature_map.shape
        device = feature_map.device

        if N_init == N and N == H * W:
            # for the initial tokens with grid structure, just reshape
            return feature_map.flatten(2).permute(0, 2, 1).contiguous()

        idx_hw = self.get_grid_index(
            [H_init, W_init], [H, W], device=device)[None, :].expand(B, -1)

        idx_batch = torch.arange(B, device=device)[:, None].expand(B, N_init)
        if agg_weight is None:
            value = feature_map.new_ones(B * N_init)
        else:
            value = agg_weight.reshape(B * N_init).type(feature_map.dtype)

        # choose the way with fewer flops.
        if N_init < N * H * W:
            # use sparse matrix multiplication
            # Flops: B * N_init * (C+2)
            idx_token = idx_token + idx_batch * N
            idx_hw = idx_hw + idx_batch * H * W
            indices = torch.stack([idx_token, idx_hw], dim=0).reshape(2, -1)

            # torch.sparse do not support fp16
            with torch.cuda.amp.autocast(enabled=False):
                # sparse mm do not support gradient for sparse matrix
                value = value.detach().float()
                # build a sparse matrix with shape [B*N, B*H*W]
                A = torch.sparse_coo_tensor(indices, value, (B * N, B * H * W))
                # normalize the matrix
                all_weight = A @ torch.ones(
                    [B * H * W, 1], device=device, dtype=torch.float32) + 1e-6
                value = value / all_weight[idx_token.reshape(-1), 0]

                A = torch.sparse_coo_tensor(indices, value, (B * N, B * H * W))
                # out: [B*N, C]
                out = A @ feature_map. \
                    permute(0, 2, 3, 1).contiguous().reshape(B * H * W, C).float()
        else:
            # use dense matrix multiplication
            # Flops: B * N * H * W * (C+2)
            indices = torch.stack([idx_batch, idx_token, idx_hw], dim=0).reshape(3, -1)
            value = value.detach()  # To reduce the training time, we detach here.
            A = torch.sparse_coo_tensor(indices, value, (B, N, H * W)).to_dense()
            # normalize the matrix
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

            out = A @ feature_map.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()

        out = out.type(feature_map.dtype)
        out = out.reshape(B, N, C)
        return out

    def forward(self, token_dict):
        x = token_dict['x']
        x = self.skip(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_map = self.token2map(token_dict)
        x_map = super().forward(x_map)
        x = x + self.map2token(x_map, token_dict)
        return x





class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        
    def drop_path(self, x, drop_prob: float = 0., training: bool = False):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
        """
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# Mlp for dynamic tokens
class TCMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = TokenConv(in_channels=hidden_features,
                                out_channels=hidden_features,
                                kernel_size=3, padding=1, stride=1,
                                bias=True,
                                groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
    def _no_grad_trunc_normal_(self, tensor, mean, std, a, b):
        # Cut & paste from PyTorch official master until it's in a few official releases - RW
        # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
        def norm_cdf(x):
            # Computes standard normal cumulative distribution function
            return (1. + math.erf(x / math.sqrt(2.))) / 2.
    
        if (mean < a - 2 * std) or (mean > b + 2 * std):
            warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                          "The distribution of values may be incorrect.",
                          stacklevel=2)
    
        with torch.no_grad():
            # Values are generated by using a truncated uniform distribution and
            # then using the inverse CDF for the normal distribution.
            # Get upper and lower cdf values
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
    
            # Uniformly fill tensor with values from [l, u], then translate to
            # [2l-1, 2u-1].
            tensor.uniform_(2 * l - 1, 2 * u - 1)
    
            # Use inverse cdf transform for normal distribution to get truncated
            # standard normal
            tensor.erfinv_()
    
            # Transform to proper mean, std
            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)
    
            # Clamp to ensure it's in the proper range
            tensor.clamp_(min=a, max=b)
            return tensor


    def trunc_normal_(self, tensor, mean=0., std=1., a=-2., b=2.):
        # type: (Tensor, float, float, float, float) -> Tensor
        r"""Fills the input Tensor with values drawn from a truncated
        normal distribution. The values are effectively drawn from the
        normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
        with values outside :math:`[a, b]` redrawn until they are within
        the bounds. The method used for generating the random values works
        best when :math:`a \leq \text{mean} \leq b`.
        Args:
            tensor: an n-dimensional `torch.Tensor`
            mean: the mean of the normal distribution
            std: the standard deviation of the normal distribution
            a: the minimum cutoff value
            b: the maximum cutoff value
        Examples:
            >>> w = torch.empty(3, 5)
            >>> nn.init.trunc_normal_(w)
        """
        return self._no_grad_trunc_normal_(tensor, mean, std, a, b)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, token_dict):
        token_dict['x'] = self.fc1(token_dict['x'])
        x = self.dwconv(token_dict)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Attention for dynamic tokens
class TCAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, use_sr_layer=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.use_sr_layer = use_sr_layer
        if sr_ratio > 1:
            if self.use_sr_layer:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _no_grad_trunc_normal_(self, tensor, mean, std, a, b):
        # Cut & paste from PyTorch official master until it's in a few official releases - RW
        # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
        def norm_cdf(x):
            # Computes standard normal cumulative distribution function
            return (1. + math.erf(x / math.sqrt(2.))) / 2.
    
        if (mean < a - 2 * std) or (mean > b + 2 * std):
            warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                          "The distribution of values may be incorrect.",
                          stacklevel=2)
    
        with torch.no_grad():
            # Values are generated by using a truncated uniform distribution and
            # then using the inverse CDF for the normal distribution.
            # Get upper and lower cdf values
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
    
            # Uniformly fill tensor with values from [l, u], then translate to
            # [2l-1, 2u-1].
            tensor.uniform_(2 * l - 1, 2 * u - 1)
    
            # Use inverse cdf transform for normal distribution to get truncated
            # standard normal
            tensor.erfinv_()
    
            # Transform to proper mean, std
            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)
    
            # Clamp to ensure it's in the proper range
            tensor.clamp_(min=a, max=b)
            return tensor


    def trunc_normal_(self, tensor, mean=0., std=1., a=-2., b=2.):
        # type: (Tensor, float, float, float, float) -> Tensor
        r"""Fills the input Tensor with values drawn from a truncated
        normal distribution. The values are effectively drawn from the
        normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
        with values outside :math:`[a, b]` redrawn until they are within
        the bounds. The method used for generating the random values works
        best when :math:`a \leq \text{mean} \leq b`.
        Args:
            tensor: an n-dimensional `torch.Tensor`
            mean: the mean of the normal distribution
            std: the standard deviation of the normal distribution
            a: the minimum cutoff value
            b: the maximum cutoff value
        Examples:
            >>> w = torch.empty(3, 5)
            >>> nn.init.trunc_normal_(w)
        """
        return self._no_grad_trunc_normal_(tensor, mean, std, a, b)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def get_grid_index(self,init_size, map_size, device):
        """For each initial grid, get its index in the feature map.
        Returns:
            idx (LongTensor[B, N_init]): index in flattened feature map.

        Args:
            init_grid_size(list[int] or tuple[int]): initial grid resolution in
                format [H_init, W_init].
            map_size(list[int] or tuple[int]): feature map resolution in format
                [H, W].
            device: the device of output
        """
        H_init, W_init = init_size
        H, W = map_size
        idx = torch.arange(H * W, device=device).reshape(1, 1, H, W)
        idx = F.interpolate(idx.float(), [H_init, W_init], mode='nearest').long()
        return idx.flatten()
        
    def token2map(self,token_dict):
        """Transform vision tokens to feature map. This function only
        works when the resolution of the feature map is not higher than
        the initial grid structure.
        Returns:
            x_out (Tensor[B, C, H, W]): feature map.

        Args:
            token_dict (dict): dict for token information.
        """

        x = token_dict['x']
        H, W = token_dict['map_size']
        H_init, W_init = token_dict['init_grid_size']
        idx_token = token_dict['idx_token']
        B, N, C = x.shape
        N_init = H_init * W_init
        device = x.device

        if N_init == N and N == H * W:
            # for the initial tokens with grid structure, just reshape
            return x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # for each initial grid, get the corresponding index in
        # the flattened feature map.
        idx_hw = self.get_grid_index(
            [H_init, W_init], [H, W], device=device)[None, :].expand(B, -1)
        idx_batch = torch.arange(B, device=device)[:, None].expand(B, N_init)
        value = x.new_ones(B * N_init)

        # choose the way with fewer flops.
        if N_init < N * H * W:
            # use sparse matrix multiplication
            # Flops: B * N_init * (C+2)
            idx_hw = idx_hw + idx_batch * H * W
            idx_tokens = idx_token + idx_batch * N
            coor = torch.stack([idx_hw, idx_tokens], dim=0).reshape(2, B * N_init)

            # torch.sparse do not support fp16
            with torch.cuda.amp.autocast(enabled=False):
                # torch.sparse do not support gradient for
                # sparse tensor, so we detach it
                value = value.detach().float()

                # build a sparse matrix with the shape [B * H * W, B * N]
                A = torch.sparse.FloatTensor(coor, value, torch.Size([B * H * W, B * N]))

                # normalize the weight for each row
                all_weight = A @ x.new_ones(B * N, 1).type(torch.float32) + 1e-6
                value = value / all_weight[idx_hw.reshape(-1), 0]

                # update the matrix with normalize weight
                A = torch.sparse.FloatTensor(coor, value, torch.Size([B * H * W, B * N]))

                # sparse matrix multiplication
                x_out = A @ x.reshape(B * N, C).type(torch.float32)  # [B*H*W, C]

        else:
            # use dense matrix multiplication
            # Flops: B * N * H * W * (C+2)
            coor = torch.stack([idx_batch, idx_hw, idx_token], dim=0).reshape(3, B * N_init)

            # build a matrix with shape [B, H*W, N]
            A = torch.sparse.FloatTensor(coor, value, torch.Size([B, H * W, N])).to_dense()
            # normalize the weight
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

            x_out = A @ x  # [B, H*W, C]

        x_out = x_out.type(x.dtype)
        x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x_out

    def forward(self, q_dict, kv_dict):
        q = q_dict['x']
        kv = kv_dict['x']
        B, Nq, C = q.shape
        Nkv = kv.shape[1]
        conf_kv = kv_dict['token_score'] if 'token_score' in kv_dict.keys() else kv.new_zeros(B, Nkv, 1)

        q = self.q(q).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            tmp = torch.cat([kv, conf_kv], dim=-1)
            tmp_dict = kv_dict.copy()
            tmp_dict['x'] = tmp
            tmp_dict['map_size'] = q_dict['map_size']
            tmp = self.token2map(tmp_dict)

            kv = tmp[:, :C]
            conf_kv = tmp[:, C:]

            if self.use_sr_layer:
                kv = self.sr(kv)
                _, _, h, w = kv.shape
                kv = kv.reshape(B, C, -1).permute(0, 2, 1).contiguous()
                kv = self.norm(kv)
            else:
                kv = F.avg_pool2d(kv, kernel_size=self.sr_ratio, stride=self.sr_ratio)
                kv = kv.reshape(B, C, -1).permute(0, 2, 1).contiguous()

            conf_kv = F.avg_pool2d(conf_kv, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            conf_kv = conf_kv.reshape(B, 1, -1).permute(0, 2, 1).contiguous()

        kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q * self.scale) @ k.transpose(-2, -1)

        conf_kv = conf_kv.squeeze(-1)[:, None, None, :]
        attn = attn + conf_kv
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Transformer block for dynamic tokens
class TCBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, use_sr_layer=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TCAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, use_sr_layer=use_sr_layer)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TCMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)
    def _no_grad_trunc_normal_(self, tensor, mean, std, a, b):
        # Cut & paste from PyTorch official master until it's in a few official releases - RW
        # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
        def norm_cdf(x):
            # Computes standard normal cumulative distribution function
            return (1. + math.erf(x / math.sqrt(2.))) / 2.
    
        if (mean < a - 2 * std) or (mean > b + 2 * std):
            warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                          "The distribution of values may be incorrect.",
                          stacklevel=2)
    
        with torch.no_grad():
            # Values are generated by using a truncated uniform distribution and
            # then using the inverse CDF for the normal distribution.
            # Get upper and lower cdf values
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
    
            # Uniformly fill tensor with values from [l, u], then translate to
            # [2l-1, 2u-1].
            tensor.uniform_(2 * l - 1, 2 * u - 1)
    
            # Use inverse cdf transform for normal distribution to get truncated
            # standard normal
            tensor.erfinv_()
    
            # Transform to proper mean, std
            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)
    
            # Clamp to ensure it's in the proper range
            tensor.clamp_(min=a, max=b)
            return tensor


    def trunc_normal_(self, tensor, mean=0., std=1., a=-2., b=2.):
        # type: (Tensor, float, float, float, float) -> Tensor
        r"""Fills the input Tensor with values drawn from a truncated
        normal distribution. The values are effectively drawn from the
        normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
        with values outside :math:`[a, b]` redrawn until they are within
        the bounds. The method used for generating the random values works
        best when :math:`a \leq \text{mean} \leq b`.
        Args:
            tensor: an n-dimensional `torch.Tensor`
            mean: the mean of the normal distribution
            std: the standard deviation of the normal distribution
            a: the minimum cutoff value
            b: the maximum cutoff value
        Examples:
            >>> w = torch.empty(3, 5)
            >>> nn.init.trunc_normal_(w)
        """
        return self._no_grad_trunc_normal_(tensor, mean, std, a, b)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, inputs):
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            q_dict, kv_dict = inputs
        else:
            q_dict, kv_dict = inputs, None

        x = q_dict['x']
        # norm1
        q_dict['x'] = self.norm1(q_dict['x'])
        if kv_dict is None:
            kv_dict = q_dict
        else:
            kv_dict['x'] = self.norm1(kv_dict['x'])

        # attn
        x = x + self.drop_path(self.attn(q_dict, kv_dict))

        # mlp
        q_dict['x'] = self.norm2(x)
        x = x + self.drop_path(self.mlp(q_dict))
        q_dict['x'] = x

        return q_dict


class CTM(nn.Module):
    def __init__(self, sample_ratio, embed_dim, dim_out, k):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out
        self.conv = TokenConv(in_channels=embed_dim, out_channels=dim_out, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(self.dim_out)
        self.score = nn.Linear(self.dim_out, 1)
        self.k = k
        
    def index_points(self,points, idx):
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points
        
    def merge_tokens(self, token_dict, idx_cluster, cluster_num, token_weight=None):
        x = token_dict['x']
        idx_token = token_dict['idx_token']
        agg_weight = None  # token_dict['agg_weight']

        B, N, C = x.shape
        if token_weight is None:
            token_weight = x.new_ones(B, N, 1)

        idx_batch = torch.arange(B, device=x.device)[:, None]
        idx = idx_cluster + idx_batch * cluster_num

        all_weight = token_weight.new_zeros(B * cluster_num, 1)
        all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                              source=token_weight.reshape(B * N, 1))
        all_weight = all_weight + 1e-6
        norm_weight = token_weight / all_weight[idx]

        # average token features
        x_merged = x.new_zeros(B * cluster_num, C)
        source = x * norm_weight
        x_merged.index_add_(dim=0, index=idx.reshape(B * N),
                            source=source.reshape(B * N, C).type(x.dtype))
        x_merged = x_merged.reshape(B, cluster_num, C)

        idx_token_new = self.index_points(idx_cluster[..., None], idx_token).squeeze(-1)
        weight_t = self.index_points(norm_weight, idx_token)
        # agg_weight_new = agg_weight * weight_t
        # agg_weight_new / agg_weight_new.max(dim=1, keepdim=True)[0]

        out_dict = {}
        out_dict['x'] = x_merged
        out_dict['token_num'] = cluster_num
        out_dict['map_size'] = token_dict['map_size']
        out_dict['init_grid_size'] = token_dict['init_grid_size']
        out_dict['idx_token'] = idx_token_new
        out_dict['agg_weight'] = None  # agg_weight_new
        return out_dict
        

    def cluster_dpc_knn(self, token_dict, cluster_num, token_mask=None):
        
        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape
    
            dist_matrix = torch.cdist(x, x) / (C ** 0.5)
    
            if token_mask is not None:
                token_mask = token_mask > 0
                # in order to not affect the local density, the distance between empty tokens
                # and any other tokens should be the maximal distance.
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                              (dist_matrix.max() + 1) * (~token_mask[:, None, :])
            dist_nearest, index_nearest = torch.topk(dist_matrix, k=self.k, dim=-1, largest=False)
    
            density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
            # add a little noise to ensure no tokens have the same density.
            density = density + torch.rand(
                density.shape, device=density.device, dtype=density.dtype) * 1e-6
    
            if token_mask is not None:
                # the density of empty token should be 0
                density = density * token_mask
            mask = density[:, None, :] > density[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)
            score = dist * density
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)
            dist_matrix = self.index_points(dist_matrix, index_down)
            idx_cluster = dist_matrix.argmin(dim=1)
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
    
        return idx_cluster, cluster_num
    def forward(self, token_dict):
        token_dict = token_dict.copy()
        # 1. 特征演进与权重计算
        x = self.conv(token_dict)
        x = self.norm(x)
        token_score = self.score(x)
        token_weight = token_score.exp()

        token_dict['x'] = x
        B, N, C = x.shape
        token_dict['token_score'] = token_score

        # 2. DPC-KNN 聚类下采样
        cluster_num = max(math.ceil(N * self.sample_ratio), 1)
        idx_cluster, cluster_num = self.cluster_dpc_knn(token_dict, cluster_num)

        # 3. 合并特征
        down_dict = self.merge_tokens(token_dict, idx_cluster, cluster_num, token_weight)

        # 4. 更新地图尺寸
        H, W = token_dict['map_size']
        down_dict['map_size'] = [math.floor((H - 1) / 2 + 1), math.floor((W - 1) / 2 + 1)]

        return down_dict, token_dict


class SemanticMerging(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)
        # self.ln = nn.LayerNorm(embed_dim)
    def forward(self, x_tokens, idx_cluster, cluster_num, relevance_score, select):
        """
        Args:
            x_tokens: CTM 输出后的视觉特征 [B, N, C]
            idx_cluster: 每个 token 所属的簇索引 [B, N]
            cluster_num: 簇的总数 P
            relevance_score: 音频引导产生的权重 [B, P, 1] (通常是 MHA 的 weight 经过处理)
        """
        B, N, C = x_tokens.shape
        P = cluster_num
        device = x_tokens.device

        # 1. 计算每个组 (Group) 的质心特征
        # 使用 scatter 方式高效求每组的平均值
        group_sum = torch.zeros(B, P, C, device=device)
        group_count = torch.zeros(B, P, 1, device=device)

        idx_expand = idx_cluster.unsqueeze(-1).expand(B, N, C)
        group_sum.scatter_add_(1, idx_expand, x_tokens)

        ones = torch.ones(B, N, 1, device=device)
        group_count.scatter_add_(1, idx_cluster.unsqueeze(-1), ones)

        # 组质心 [B, P, C]
        group_features = group_sum / (group_count + 1e-6)
        
        # # 2. 语义合并 (Semantic Merging): 应用相关性得分 (Relevance Score)
        # # 根据图片逻辑，这里是点乘 w_gp
        # print(group_features.shape)
        # print(relevance_score.shape)
        # torch.Size([5, 36, 256])
        # torch.Size([5, 1, 36])
        if select == "pos":
            semantic_query = group_features * relevance_score.transpose(1,2)  # [B, P, C]
        else:
            # torch.Size([5, 36, 256])
            # torch.Size([5, 36, 100])
            semantic_query = torch.einsum("bqc, bqn -> bnc", group_features,relevance_score)
            # print(semantic_query.shape)
        return self.proj(semantic_query)
####################################################################################################################################################################

@dataclass
class Mask2FormerPixelDecoderOutput(ModelOutput):
    """
    Mask2Former's pixel decoder module output, practically a Multi-Scale Deformable Attention based decoder. It returns
    the mask features and the multiscale features.

    Args:
        multi_scale_features (`tuple(torch.FloatTensor)`):
            Tuple of multi-scale features of scales [1/8, 1/16, 1/32] and shape `(batch_size, num_channels, height,
            width)`from the Multi-Scale Deformable Attenntion based Pixel Decoder.
        mask_features (`torch.FloatTensor`):
            Tensor of shape `(batch_size, num_channels, height, width)`, 1/4 scale features from the last Pixel Decoder
            Layer.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights from pixel decoder. Returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`
    """

    multi_scale_features: Tuple[torch.FloatTensor] = None
    mask_features: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    pixel_value_out= None


@dataclass
class Mask2FormerMaskedAttentionDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    Base class for outputs of the Transformer decoder. This class adds two attributes to
    BaseModelOutputWithCrossAttentions for mask predictions logits and a tuple of intermediate decoder activations,
    i.e. the output of each decoder layer, each of them gone through a layernorm.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs. Returned when `output_hidden_states=True`.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads. Returned when `output_attentions=True`.
        masks_queries_logits (`tuple(torch.FloatTensor)` of shape `(batch_size, num_queries, height, width)`):
            Tuple of mask predictions from all layers of the transformer decoder.
        intermediate_hidden_states (`tuple(torch.FloatTensor)` of shape `(num_queries, 1, hidden_size)`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[torch.FloatTensor] = None
    masks_queries_logits: Tuple[torch.FloatTensor] = None
    intermediate_hidden_states: Tuple[torch.FloatTensor] = None


@dataclass
class Mask2FormerPixelLevelModuleOutput(ModelOutput):
    """
    Mask2Former's pixel level module output. It returns the output of the encoder (optional) and all hidden states
    (multi-scale features) from the `decoder`. By default, the `encoder` is a Swin Backbone and the `decoder` is a
    Multi-Scale Deformable Attention based decoder.

    The `decoder_last_hidden_state` are the **per-pixel embeddings** while `decoder_hidden_states` refer to multi-scale
    feature maps produced using **multi-scaling strategy** defined in the paper.

    Args:
        encoder_last_hidden_state (`torch.FloatTensor`):
            Last hidden states (final feature map of shape `(batch_size, num_channels, height, width)`) of the last
            stage of the encoder.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`. Hidden states (also
            called feature maps) of the model at the output of each stage. Returned if output_hidden_states is set to
            True.
        decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)):
            1/4 scale features from the last Pixel Decoder Layer.
        decoder_hidden_states (`tuple(torch.FloatTensor)`):
            Tuple of `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`. Hidden states (also
            called feature maps) of the model at the output of each stage.
    """

    encoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_last_hidden_state: torch.FloatTensor = None
    decoder_hidden_states: Tuple[torch.FloatTensor] = None


@dataclass
class Mask2FormerModelOutput(ModelOutput):
    """
    Class for outputs of [`Mask2FormerModel`]. This class returns all the needed hidden states to compute the logits.

    Args:
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
            Last hidden states (final feature map) of the last stage of the encoder model (backbone). Returned when
            `output_hidden_states=True` is passed.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
            model at the output of each stage. Returned when `output_hidden_states=True` is passed.
        pixel_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
            Last hidden states (final feature map) of the last stage of the pixel decoder model.
        pixel_decoder_hidden_states (`tuple(torch.FloatTensor)`, , *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
            decoder model at the output of each stage. Returned when `output_hidden_states=True` is passed.
        transformer_decoder_last_hidden_state (`tuple(torch.FloatTensor)`):
            Final output of the transformer decoder `(batch_size, sequence_length, hidden_size)`.
        transformer_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
            transformer decoder at the output of each stage. Returned when `output_hidden_states=True` is passed.
        transformer_decoder_intermediate_states (`tuple(torch.FloatTensor)` of shape `(num_queries, 1, hidden_size)`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
        masks_queries_logits (`tuple(torch.FloatTensor)` of shape `(batch_size, num_queries, height, width)`)
            Mask Predictions from each layer in the transformer decoder.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed):
            Tuple of `tuple(torch.FloatTensor)` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Self attentions weights from transformer decoder.
    """

    encoder_last_hidden_state: torch.FloatTensor = None
    pixel_decoder_last_hidden_state: torch.FloatTensor = None
    transformer_decoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_intermediate_states: Tuple[torch.FloatTensor] = None
    masks_queries_logits: Tuple[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    pixel_value_out:Optional[Mask2FormerPixelLevelModuleOutput]=None


@dataclass
class Mask2FormerForUniversalSegmentationOutput(ModelOutput):
    """
    Class for outputs of [`Mask2FormerForUniversalSegmentationOutput`].

    This output can be directly passed to [`~Mask2FormerImageProcessor.post_process_semantic_segmentation`] or
    [`~Mask2FormerImageProcessor.post_process_instance_segmentation`] or
    [`~Mask2FormerImageProcessor.post_process_panoptic_segmentation`] to compute final segmentation maps. Please, see
    [`~Mask2FormerImageProcessor] for details regarding usage.

    Args:
        loss (`torch.Tensor`, *optional*):
            The computed loss, returned when labels are present.
        class_queries_logits (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, num_labels + 1)` representing the proposed classes for each
            query. Note the `+ 1` is needed because we incorporate the null class.
        masks_queries_logits (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
            query.
        auxiliary_logits (`List[Dict(str, torch.FloatTensor)]`, *optional*):
            List of class and mask predictions from each layer of the transformer decoder.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the encoder model (backbone).
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
            model at the output of each stage.
        pixel_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the pixel decoder model.
        pixel_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
            decoder model at the output of each stage.
        transformer_decoder_last_hidden_state (`tuple(torch.FloatTensor)`):
            Final output of the transformer decoder `(batch_size, sequence_length, hidden_size)`.
        transformer_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
            transformer decoder at the output of each stage.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tuple(torch.FloatTensor)` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Self and Cross Attentions weights from transformer decoder.
    """

    loss: Optional[torch.FloatTensor] = None
    class_queries_logits: torch.FloatTensor = None
    masks_queries_logits: torch.FloatTensor = None
    auxiliary_logits: Optional[List[Dict[str, torch.FloatTensor]]] = None
    encoder_last_hidden_state: torch.FloatTensor = None
    pixel_decoder_last_hidden_state: torch.FloatTensor = None
    transformer_decoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    pixel_value_out:Optional[Mask2FormerPixelLevelModuleOutput]=None


# Copied from transformers.models.detr.modeling_detr._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, target_len: Optional[int] = None):
    """
    Expands attention_mask from `[batch_size, seq_len]` to `[batch_size, 1, target_seq_len, source_seq_len]`.
    """
    batch_size, source_len = mask.size()
    target_len = target_len if target_len is not None else source_len

    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, target_len, source_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


# Adapted from https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/point_rend/point_features.py
def sample_point(
    input_features: torch.Tensor, point_coordinates: torch.Tensor, add_dim=False, **kwargs
) -> torch.Tensor:
    """
    A wrapper around `torch.nn.functional.grid_sample` to support 3D point_coordinates tensors.

    Args:
        input_features (`torch.Tensor` of shape (batch_size, channels, height, width)):
            A tensor that contains features map on a height * width grid
        point_coordinates (`torch.Tensor` of shape (batch_size, num_points, 2) or (batch_size, grid_height, grid_width,:
        2)):
            A tensor that contains [0, 1] * [0, 1] normalized point coordinates
        add_dim (`bool`):
            boolean value to keep track of added dimension

    Returns:
        point_features (`torch.Tensor` of shape (batch_size, channels, num_points) or (batch_size, channels,
        height_grid, width_grid):
            A tensor that contains features for points in `point_coordinates`.
    """
    if point_coordinates.dim() == 3:
        add_dim = True
        point_coordinates = point_coordinates.unsqueeze(2)

    # use nn.function.grid_sample to get features for points in `point_coordinates` via bilinear interpolation
    point_features = torch.nn.functional.grid_sample(input_features, 2.0 * point_coordinates - 1.0, **kwargs)
    if add_dim:
        point_features = point_features.squeeze(3)

    return point_features


# Copied from transformers.models.maskformer.modeling_maskformer.dice_loss
def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    r"""
    Compute the DICE loss, similar to generalized IOU for masks as follows:

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x \cap y }{x \cup y + 1}} $$

    In practice, since `labels` is a binary mask, (only 0s and 1s), dice can be computed as follow

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x * y }{x + y + 1}} $$

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_masks (`int`):
            The number of masks present in the current batch, used for normalization.

    Returns:
        `torch.Tensor`: The computed loss.
    """
    probs = inputs.sigmoid().flatten(1)
    numerator = 2 * (probs * labels).sum(-1)
    denominator = probs.sum(-1) + labels.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss.sum() / num_masks
    return loss


def sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor, num_masks: int) -> torch.Tensor:
    r"""
    Args:
        inputs (`torch.Tensor`):
            A float tensor of arbitrary shape.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        loss (`torch.Tensor`): The computed loss.
    """
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    cross_entropy_loss = criterion(inputs, labels)

    loss = cross_entropy_loss.mean(1).sum() / num_masks
    return loss


# Copied from transformers.models.maskformer.modeling_maskformer.pair_wise_dice_loss
def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    """
    A pair wise version of the dice loss, see `dice_loss` for usage.

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        `torch.Tensor`: The computed loss between each pairs.
    """
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, labels)
    # using broadcasting to get a [num_queries, NUM_CLASSES] matrix
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def pair_wise_sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    r"""
    A pair wise version of the cross entropy loss, see `sigmoid_cross_entropy_loss` for usage.

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        loss (`torch.Tensor`): The computed loss between each pairs.
    """

    height_and_width = inputs.shape[1]

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    cross_entropy_loss_pos = criterion(inputs, torch.ones_like(inputs))
    cross_entropy_loss_neg = criterion(inputs, torch.zeros_like(inputs))

    loss = torch.einsum("nc,mc->nm", cross_entropy_loss_pos, labels) + torch.einsum(
        "nc,mc->nm", cross_entropy_loss_neg, (1 - labels)
    )
    loss = loss / height_and_width
    return loss


# Adapted from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py
class Mask2FormerHungarianMatcher(nn.Module):
    """This class computes an assignment between the labels and the predictions of the network.

    For efficiency reasons, the labels don't include the no_object. Because of this, in general, there are more
    predictions than labels. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    """

    def __init__(
        self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0, num_points: int = 12544
    ):
        """Creates the matcher

        Params:
            cost_class (`float`, *optional*, defaults to 1.0):
                Relative weight of the classification error in the matching cost.
            cost_mask (`float`, *optional*,  defaults to 1.0):
                This is the relative weight of the focal loss of the binary mask in the matching cost.
            cost_dice (`float`, *optional*, defaults to 1.0):
                This is the relative weight of the dice loss of the binary mask in the matching cost.
            num_points (`int`, *optional*, defaults to 12544):
                No. of points to sample on which the mask loss will be calculated. The same set of K points are
                uniformly sampled for all prediction and ground truth masks to construct the cost matrix for bipartite
                matching.
        """
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("All costs cant be 0")

        self.num_points = num_points
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> List[Tuple[Tensor]]:
        """
        Params:
            masks_queries_logits (`torch.Tensor`):
                A tensor of dim `batch_size, num_queries, num_labels` with the classification logits.
            class_queries_logits (`torch.Tensor`):
                A tensor of dim `batch_size, num_queries, height, width` with the predicted masks.
            class_labels (`torch.Tensor`):
                A tensor of dim `num_target_boxes` (where num_target_boxes is the number of ground-truth objects in the
                target) containing the class labels.
            mask_labels (`torch.Tensor`):
                A tensor of dim `num_target_boxes, height, width` containing the target masks.

        Returns:
            matched_indices (`List[Tuple[Tensor]]`): A list of size batch_size, containing tuples of (index_i, index_j)
            where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected labels (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes).
        """
        indices: List[Tuple[np.array]] = []

        # iterate through batch size
        batch_size = masks_queries_logits.shape[0]
        for i in range(batch_size):
            pred_probs = class_queries_logits[i].softmax(-1)
            pred_mask = masks_queries_logits[i]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL, but approximate it in 1 - proba[target class]. The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -pred_probs[:, class_labels[i]]
            target_mask = mask_labels[i].to(pred_mask)
            target_mask = target_mask[:, None]
            pred_mask = pred_mask[:, None]

            # Sample ground truth and predicted masks
            point_coordinates = torch.rand(1, self.num_points, 2, device=pred_mask.device)

            target_coordinates = point_coordinates.repeat(target_mask.shape[0], 1, 1)
            target_mask = sample_point(target_mask, target_coordinates, align_corners=False).squeeze(1)

            pred_coordinates = point_coordinates.repeat(pred_mask.shape[0], 1, 1)
            pred_mask = sample_point(pred_mask, pred_coordinates, align_corners=False).squeeze(1)

            # compute the cross entropy loss between each mask pairs -> shape (num_queries, num_labels)
            cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
            # Compute the dice loss betwen each mask pairs -> shape (num_queries, num_labels)
            cost_dice = pair_wise_dice_loss(pred_mask, target_mask)
            # final cost matrix
            cost_matrix = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            # do the assigmented using the hungarian algorithm in scipy
            assigned_indices: Tuple[np.array] = linear_sum_assignment(cost_matrix.cpu())
            indices.append(assigned_indices)

        # It could be stacked in one tensor
        matched_indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices
        ]
        return matched_indices


# Adapted from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py
class Mask2FormerLoss(nn.Module):
    def __init__(self, config: Mask2FormerConfig, weight_dict: Dict[str, float]):
        """
        The Mask2Former Loss. The loss is computed very similar to DETR. The process happens in two steps: 1) we
        compute hungarian assignment between ground truth masks and the outputs of the model 2) we supervise each pair
        of matched ground-truth / prediction (supervise class and mask)

        Args:
            config (`Mask2FormerConfig`):
                The configuration for Mask2Former model also containing loss calculation specific parameters.
            weight_dict (`Dict[str, float]`):
                A dictionary of weights to be applied to the different losses.
        """
        super().__init__()
        requires_backends(self, ["scipy"])
        self.num_labels = config.num_labels
        self.weight_dict = weight_dict

        # Weight to apply to the null class
        self.eos_coef = config.no_object_weight
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = config.train_num_points
        self.oversample_ratio = config.oversample_ratio
        self.importance_sample_ratio = config.importance_sample_ratio

        self.matcher = Mask2FormerHungarianMatcher(
            cost_class=1.0,
            cost_dice=config.dice_weight,
            cost_mask=config.mask_weight,
            num_points=self.num_points,
        )

    def _max_by_axis(self, sizes: List[List[int]]) -> List[int]:
        maxes = sizes[0]
        for sublist in sizes[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    # Adapted from nested_tensor_from_tensor_list() in original implementation
    def _pad_images_to_max_in_batch(self, tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
        # get the maximum size in the batch
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        # compute final size
        batch_shape = [len(tensors)] + max_size
        batch_size, _, height, width = batch_shape
        dtype = tensors[0].dtype
        device = tensors[0].device
        padded_tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
        padding_masks = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        # pad the tensors to the size of the biggest one
        for tensor, padded_tensor, padding_mask in zip(tensors, padded_tensors, padding_masks):
            padded_tensor[: tensor.shape[0], : tensor.shape[1], : tensor.shape[2]].copy_(tensor)
            padding_mask[: tensor.shape[1], : tensor.shape[2]] = False

        return padded_tensors, padding_masks

    def loss_labels(
        self, class_queries_logits: Tensor, class_labels: List[Tensor], indices: Tuple[np.array]
    ) -> Dict[str, Tensor]:
        """Compute the losses related to the labels using cross entropy.

        Args:
            class_queries_logits (`torch.Tensor`):
                A tensor of shape `batch_size, num_queries, num_labels`
            class_labels (`List[torch.Tensor]`):
                List of class labels of shape `(labels)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.

        Returns:
            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing the following key:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
        """

        pred_logits = class_queries_logits
        batch_size, num_queries, _ = pred_logits.shape
        criterion = nn.CrossEntropyLoss(weight=self.empty_weight)
        idx = self._get_predictions_permutation_indices(indices)  # shape of (batch_size, num_queries)
        target_classes_o = torch.cat(
            [target[j] for target, (_, j) in zip(class_labels, indices)]
        )  # shape of (batch_size, num_queries)
        target_classes = torch.full(
            (batch_size, num_queries), fill_value=self.num_labels, dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o
        # Permute target_classes (batch_size, num_queries, num_labels) -> (batch_size, num_labels, num_queries)
        pred_logits_transposed = pred_logits.transpose(1, 2)
        loss_ce = criterion(pred_logits_transposed, target_classes)
        losses = {"loss_cross_entropy": loss_ce}
        return losses

    def loss_masks(
        self,
        masks_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        indices: Tuple[np.array],
        num_masks: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute the losses related to the masks using sigmoid_cross_entropy_loss and dice loss.

        Args:
            masks_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, height, width)`.
            mask_labels (`torch.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.
            num_masks (`int)`:
                The number of masks, used for normalization.

        Returns:
            losses (`Dict[str, Tensor]`): A dict of `torch.Tensor` containing two keys:
            - **loss_mask** -- The loss computed using sigmoid cross entropy loss on the predicted and ground truth.
              masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth,
              masks.
        """
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        # shape (batch_size * num_queries, height, width)
        pred_masks = masks_queries_logits[src_idx]
        # shape (batch_size, num_queries, height, width)
        # pad all and stack the targets to the num_labels dimension
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]
        # No need to upsample predictions as we are using normalized coordinates
        pred_masks = pred_masks[:, None]#num_masks*1*96*96
        target_masks = target_masks[:, None]#num_masks*1*384*384
        # Sample point coordinates
        with torch.no_grad():
            point_coordinates = self.sample_points_using_uncertainty(
                target_masks,
                pred_masks,
                lambda logits: self.calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )

            point_labels = sample_point(target_masks, point_coordinates, align_corners=False).squeeze(1)

        point_logits = sample_point(pred_masks, point_coordinates, align_corners=False).squeeze(1)

        losses = {
            "loss_mask": sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

        del pred_masks
        del target_masks
        return losses

    def _get_predictions_permutation_indices(self, indices):
        # Permute predictions following indices
        batch_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        predictions_indices = torch.cat([src for (src, _) in indices])
        return batch_indices, predictions_indices

    def _get_targets_permutation_indices(self, indices):
        # Permute labels following indices
        batch_indices = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        target_indices = torch.cat([tgt for (_, tgt) in indices])
        return batch_indices, target_indices

    def calculate_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """
        In Mask2Former paper, uncertainty is estimated as L1 distance between 0.0 and the logit prediction in 'logits'
        for the foreground class in `classes`.

        Args:
            logits (`torch.Tensor`):
            A tensor of shape (R, 1, ...) for class-specific or class-agnostic, where R is the total number of predicted masks in all images and C is:
            the number of foreground classes. The values are logits.

        Returns:
            scores (`torch.Tensor`): A tensor of shape (R, 1, ...) that contains uncertainty scores with the most
            uncertain locations having the highest uncertainty score.
        """
        uncertainty_scores = -(torch.abs(logits))
        return uncertainty_scores

    def sample_points_using_uncertainty(
        self,
        gtmask,
        logits: torch.Tensor,
        uncertainty_function,
        num_points: int,
        oversample_ratio: int,
        importance_sample_ratio: float,
    ) -> torch.Tensor:
        """
        This function is meant for sampling points in [0, 1] * [0, 1] coordinate space based on their uncertainty. The
        uncertainty is calculated for each point using the passed `uncertainty function` that takes points logit
        prediction as input.

        Args:
            logits (`float`):
                Logit predictions for P points.
            uncertainty_function:
                A function that takes logit predictions for P points and returns their uncertainties.
            num_points (`int`):
                The number of points P to sample.
            oversample_ratio (`int`):
                Oversampling parameter.
            importance_sample_ratio (`float`):
                Ratio of points that are sampled via importance sampling.

        Returns:
            point_coordinates (`torch.Tensor`):
                Coordinates for P sampled points.
        """

        num_boxes = logits.shape[0]
        num_points_sampled = int(num_points * oversample_ratio)

        # Get random point coordinates
        point_coordinates = torch.rand(num_boxes, num_points_sampled, 2, device=logits.device)
        # Get sampled prediction value for the point coordinates
        #point_coordinates=torch.zeros([num_boxes,num_points_sampled,2], dtype=torch.long, device=logits.device)
        #for i in range(num_boxes):
           # if all==1:
               # point_coordinates[i,:,:] = torch.rand( num_points_sampled, 2, device=logits.device)
            #else:
               # indices=torch.nonzero(gtmask[i,0], as_tuple=False)
               # point_coordinates[i,:,:]=indices[random.choices(range(indices.shape[0]), k=num_points_sampled)]/384
            
        point_logits = sample_point(logits, point_coordinates, align_corners=False)
        # Calculate the uncertainties based on the sampled prediction values of the points
        point_uncertainties = uncertainty_function(point_logits)

        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points

        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_points_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
        idx += shift[:, None]
        point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)

        if num_random_points > 0:
            point_coordinates = torch.cat(
                [point_coordinates, torch.rand(num_boxes, num_random_points, 2, device=logits.device)],
                dim=1,
            )
        return point_coordinates

    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        class_labels: List[torch.Tensor],
        auxiliary_predictions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        This performs the loss computation.

        Args:
            masks_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, height, width)`.
            class_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, num_labels)`.
            mask_labels (`torch.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            class_labels (`List[torch.Tensor]`):
                List of class labels of shape `(labels)`.
            auxiliary_predictions (`Dict[str, torch.Tensor]`, *optional*):
                if `use_auxiliary_loss` was set to `true` in [`Mask2FormerConfig`], then it contains the logits from
                the inner layers of the Mask2FormerMaskedAttentionDecoder.

        Returns:
            losses (`Dict[str, Tensor]`): A dict of `torch.Tensor` containing three keys:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
            - **loss_mask** -- The loss computed using sigmoid cross_entropy loss on the predicted and ground truth
              masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth
              masks.
            if `use_auxiliary_loss` was set to `true` in [`Mask2FormerConfig`], the dictionary contains additional
            losses for each auxiliary predictions.
        """

        # retrieve the matching between the outputs of the last layer and the labels

        indices = self.matcher(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
        # compute the average number of target masks for normalization purposes
        num_masks = self.get_num_masks(class_labels, device=class_labels[0].device)
        # get all the losses
        losses: Dict[str, Tensor] = {
            **self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks),
            **self.loss_labels(class_queries_logits, class_labels, indices),
        }
        # in case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if auxiliary_predictions is not None:
            for idx, aux_outputs in enumerate(auxiliary_predictions):
                masks_queries_logits = aux_outputs["masks_queries_logits"]
                class_queries_logits = aux_outputs["class_queries_logits"]
                loss_dict = self.forward(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
                loss_dict = {f"{key}_{idx}": value for key, value in loss_dict.items()}
                losses.update(loss_dict)

        return losses

    def get_num_masks(self, class_labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Computes the average number of target masks across the batch, for normalization purposes.
        """
        num_masks = sum([len(classes) for classes in class_labels])
        num_masks_pt = torch.as_tensor([num_masks], dtype=torch.float, device=device)
        return num_masks_pt


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.multi_scale_deformable_attention
def multi_scale_deformable_attention(
    value: Tensor, value_spatial_shapes: Tensor, sampling_locations: Tensor, attention_weights: Tensor
) -> Tensor:
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([height.item() * width.item() for height, width in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # batch_size, height*width, num_heads, hidden_dim
        # -> batch_size, height*width, num_heads*hidden_dim
        # -> batch_size, num_heads*hidden_dim, height*width
        # -> batch_size*num_heads, hidden_dim, height, width
        value_l_ = (
            value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        # batch_size, num_queries, num_heads, num_points, 2
        # -> batch_size, num_heads, num_queries, num_points, 2
        # -> batch_size*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        # batch_size*num_heads, hidden_dim, num_queries, num_points
        sampling_value_l_ = nn.functional.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (batch_size, num_queries, num_heads, num_levels, num_points)
    # -> (batch_size, num_heads, num_queries, num_levels, num_points)
    # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


# Copied from transformers.models.maskformer.modeling_maskformer.MaskFormerSinePositionEmbedding with MaskFormer->Mask2Former
class Mask2FormerSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None
    ):
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# Modified from transformers.models.detr.modeling_deformable_detr.DeformableDetrMultiscaleDeformableAttention
class Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def __init__(self, embed_dim: int, num_heads: int, n_levels: int, n_points: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {embed_dim} and {num_heads}"
            )
        dim_per_head = embed_dim // num_heads
        # check if dim_per_head is power of 2
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in DeformableDetrMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        self.im2col_step = 128

        self.d_model = embed_dim
        self.n_levels = n_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * n_levels * n_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self.avs_adapt = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//4),
            nn.ReLU(),
            nn.Linear(embed_dim//4, embed_dim),
        )

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != sequence_length:
            raise ValueError(
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
            )

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            # we invert the attention_mask
            value = value.masked_fill(attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = nn.functional.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )
        # batch_size, num_queries, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        output = multi_scale_deformable_attention(value, spatial_shapes, sampling_locations, attention_weights)
        # output = self.output_proj(output)
        
        output = self.avs_adapt(output)

        return output, attention_weights


class Mask2FormerPixelDecoderEncoderLayer(nn.Module):
    def __init__(self, config: Mask2FormerConfig):
        super().__init__()
        self.embed_dim = config.feature_size
        self.self_attn = Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            n_levels=3,
            n_points=4,
        )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = nn.functional.relu
        self.activation_dropout = config.dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_feedforward_dim)
        self.fc2 = nn.Linear(config.encoder_feedforward_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input to the layer.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Attention mask.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings, to be added to `hidden_states`.
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.
            spatial_shapes (`torch.LongTensor`, *optional*):
                Spatial shapes of the backbone feature maps.
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Apply Multi-scale Deformable Attention Module on the multi-scale feature maps.
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights.transpose(1, 0),)

        return outputs


# Modified from from transformers.models.detr.modeling_deformable_detr.DeformableDetrEncoder with DeformableDetrEncoder->Mask2FormerPixelDecoderEncoderOnly
class Mask2FormerPixelDecoderEncoderOnly(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a
    [`Mask2FormerPixelDecoderEncoderLayer`]. The encoder updates the flattened multi-scale feature maps through
    multiple deformable attention layers.

    Args:
        config: Mask2FormerConfig
    """

    def __init__(self, config: Mask2FormerConfig):
        super().__init__()

        self.config = config
        self.dropout = config.dropout
        self.layers = nn.ModuleList(
            [Mask2FormerPixelDecoderEncoderLayer(config) for _ in range(config.encoder_layers)]
        )

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        Get reference points for each feature map. Used in decoder.

        Args:
            spatial_shapes (`torch.LongTensor`):
                Spatial shapes of each feature map, has shape of `(num_feature_levels, 2)`.
            valid_ratios (`torch.FloatTensor`):
                Valid ratios of each feature map, has shape of `(batch_size, num_feature_levels, 2)`.
            device (`torch.device`):
                Device on which to create the tensors.
        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
        """
        reference_points_list = []
        for lvl, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)

        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]

        return reference_points

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        position_embeddings=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            position_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`):
                Starting index of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Ratio of valid area in each feature level.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = inputs_embeds
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=inputs_embeds.device)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states.transpose(1, 0),)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                position_embeddings=position_embeddings,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states.transpose(1, 0),)

        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


# Modified from from transformers.models.detr.modeling_deformable_detr.DeformableDetrModel with DeformableDetrModel->Mask2FormerPixelDecoder
class Mask2FormerPixelDecoder(nn.Module):
    def __init__(self, config: Mask2FormerConfig, feature_channels):
        super().__init__()

        self.config = config

        feature_dim = config.feature_size
        mask_dim = config.mask_feature_size
        num_pos_features = feature_dim // 2

        self.position_embedding = Mask2FormerSinePositionEmbedding(num_pos_feats=num_pos_features, normalize=True)
        self.num_feature_levels = 3
        transformer_in_channels = feature_channels[-self.num_feature_levels :]

        self.transformer_feature_strides = config.feature_strides[-self.num_feature_levels :]
        self.feature_channels = feature_channels
        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, feature_dim))

        # Create input projection layers
        if self.num_feature_levels > 1:
            input_projections_list = []
            for in_channels in transformer_in_channels[::-1]:
                input_projections_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, feature_dim, kernel_size=1),
                        nn.GroupNorm(32, feature_dim),
                    )
                )
            self.input_projections = nn.ModuleList(input_projections_list)
        else:
            self.input_projections = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(transformer_in_channels[-1], feature_dim, kernel_size=1),
                        nn.GroupNorm(32, feature_dim),
                    )
                ]
            )

        self.encoder = Mask2FormerPixelDecoderEncoderOnly(config)
        self.mask_projection = nn.Conv2d(feature_dim, mask_dim, kernel_size=1, stride=1, padding=0)

        # Extra FPN levels
        stride = min(self.transformer_feature_strides)
        self.common_stride = config.common_stride
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        for idx, in_channels in enumerate(self.feature_channels[: self.num_fpn_levels]):
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_channels, feature_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, feature_dim),
            )

            output_conv = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(32, feature_dim),
                nn.ReLU(),
            )
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        # Order convolutional layers from low to high resolution
        self.lateral_convolutions = lateral_convs[::-1]
        self.output_convolutions = output_convs[::-1]

    def get_valid_ratio(self, mask):
        """Get the valid ratio of all feature maps."""

        _, height, width = mask.shape
        valid_height = torch.sum(~mask[:, :, 0], 1)
        valid_width = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_heigth = valid_height.float() / height
        valid_ratio_width = valid_width.float() / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        return valid_ratio

    def forward(
        self,
        features,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        input_embeds = []
        position_embeddings = []
        for level, x in enumerate(features[::-1][: self.num_feature_levels]):
            input_embeds.append(self.input_projections[level](x.float()))
            position_embeddings.append(self.position_embedding(x.float()))

        masks = [
            torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in input_embeds
        ]

        # Prepare encoder inputs (by flattening)
        spatial_shapes = [(embed.shape[2], embed.shape[3]) for embed in input_embeds]
        input_embeds_flat = torch.cat([embed.flatten(2).transpose(1, 2) for embed in input_embeds], 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=input_embeds_flat.device)
        masks_flat = torch.cat([mask.flatten(1) for mask in masks], 1)

        position_embeddings = [embed.flatten(2).transpose(1, 2) for embed in position_embeddings]
        level_pos_embed_flat = [x + self.level_embed[i].view(1, 1, -1) for i, x in enumerate(position_embeddings)]
        level_pos_embed_flat = torch.cat(level_pos_embed_flat, 1)

        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(mask) for mask in masks], 1)

        # Send input_embeds_flat + masks_flat + level_pos_embed_flat (backbone + proj layer output) through encoder
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=input_embeds_flat,
                attention_mask=masks_flat,
                position_embeddings=level_pos_embed_flat,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        last_hidden_state = encoder_outputs.last_hidden_state
        batch_size = last_hidden_state.shape[0]

        split_sizes = [None] * self.num_feature_levels
        for i in range(self.num_feature_levels):
            if i < self.num_feature_levels - 1:
                split_sizes[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_sizes[i] = last_hidden_state.shape[1] - level_start_index[i]

        encoder_output = torch.split(last_hidden_state, [size.item() for size in split_sizes], dim=1)

        # Compute final features
        outputs = [
            x.transpose(1, 2).view(batch_size, -1, spatial_shapes[i][0], spatial_shapes[i][1])
            for i, x in enumerate(encoder_output)
        ]

        # Append extra FPN levels to outputs, ordered from low to high resolution
        for idx, feature in enumerate(features[: self.num_fpn_levels][::-1]):
            lateral_conv = self.lateral_convolutions[idx]
            output_conv = self.output_convolutions[idx]
            current_fpn = lateral_conv(feature.float())

            # Following FPN implementation, we use nearest upsampling here
            out = current_fpn + nn.functional.interpolate(
                outputs[-1], size=current_fpn.shape[-2:], mode="bilinear", align_corners=False
            )
            out = output_conv(out)
            outputs.append(out)

        num_cur_levels = 0
        multi_scale_features = []

        for out in outputs:
            if num_cur_levels < self.num_feature_levels:
                multi_scale_features.append(out)
                num_cur_levels += 1

        return Mask2FormerPixelDecoderOutput(
            mask_features=self.mask_projection(outputs[-1]),
            multi_scale_features=tuple(multi_scale_features),
            attentions=encoder_outputs.attentions,
        )

from peft import LoraConfig, get_peft_model
class Mask2FormerPixelLevelModule(nn.Module):
    def __init__(self, config: Mask2FormerConfig):
        """
        Pixel Level Module proposed in [Masked-attention Mask Transformer for Universal Image
        Segmentation](https://arxiv.org/abs/2112.01527). It runs the input image through a backbone and a pixel
        decoder, generating multi-scale feature maps and pixel embeddings.

        Args:
            config ([`Mask2FormerConfig`]):
                The configuration used to instantiate this model.
        """
        super().__init__()

        backbone_config_dict = config.backbone_config.to_dict()
        backbone_config = SwinConfig.from_dict(backbone_config_dict)
        # print(type(backbone_config)) 'transformers.models.swin.configuration_swin.SwinConfig' 'models.transformers.models.swin.configuration_swin.SwinConfig'
        self.encoder = AutoBackbone.from_config(backbone_config)
        self.decoder = Mask2FormerPixelDecoder(config, feature_channels=self.encoder.channels)
        
        
        
    def forward(self, pixel_values: Tensor, audio_boost_mask, output_hidden_states: bool = False) -> Mask2FormerPixelLevelModuleOutput:
        backbone_features = self.encoder(pixel_values).feature_maps
        # print(len(backbone_features)) 4
        # print(backbone_features[0].shape) torch.Size([5, 128, 96, 96])
        # print(backbone_features[1].shape) torch.Size([5, 256, 48, 48])
        # print(backbone_features[2].shape) torch.Size([5, 512, 24, 24])
        # print(backbone_features[3].shape) torch.Size([5, 1024, 12, 12])
        
        # backbone_features = list(backbone_features)
        # audio_list = [audio_ori.view(5,1,128),audio.view(5,1,256)]
        # backbone_features_0_1 = [backbone_features[0], backbone_features[1]]
        # backbone_features_0_1 = self.dpcknn(backbone_features_0_1, audio_list)
        # backbone_features[0] = backbone_features_0_1[0]
        # backbone_features[1] = backbone_features_0_1[1]
        # backbone_features = list(backbone_features)
        # for i in range(3):
        #         backbone_features[i+1] = backbone_features[i+1] + torch.nn.functional.interpolate(
        #         audio_boost_mask.unsqueeze(1), size=backbone_features[i+1].shape[-2:],
        #         mode="bilinear")
 
        decoder_output = self.decoder(backbone_features, output_hidden_states=output_hidden_states)

        return Mask2FormerPixelLevelModuleOutput(
            encoder_last_hidden_state=backbone_features[-1],
            encoder_hidden_states=tuple(backbone_features) if output_hidden_states else None,
            decoder_last_hidden_state=decoder_output.mask_features,
            decoder_hidden_states=decoder_output.multi_scale_features,
        )


# Modified from transformers.models.detr.modeling_detr.DetrAttention with Detr->Mask2Former
class Mask2FormerAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Here, we add position embeddings to the queries and
    keys (as explained in the DETR paper).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.avs_adapt = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//4),
            nn.ReLU(),
            nn.Linear(embed_dim//4, embed_dim),
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        key_value_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        hidden_states = hidden_states.permute(1, 0, 2) if hidden_states is not None else None
        position_embeddings = position_embeddings.permute(1, 0, 2) if position_embeddings is not None else None
        key_value_states = key_value_states.permute(1, 0, 2) if key_value_states is not None else None
        key_value_position_embeddings = (
            key_value_position_embeddings.permute(1, 0, 2) if key_value_position_embeddings is not None else None
        )

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        batch_size, target_len, embed_dim = hidden_states.size()

        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        # add key-value position embeddings to the key value states
        if key_value_position_embeddings is not None:
            key_value_states_original = key_value_states
            key_value_states = self.with_pos_embed(key_value_states, key_value_position_embeddings)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.v_proj(key_value_states_original), -1, batch_size)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
            value_states = self._shape(self.v_proj(hidden_states_original), -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        source_len = key_states.size(1)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (batch_size * self.num_heads, target_len, source_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, target_len, source_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size * self.num_heads, target_len, source_len):
                raise ValueError(
                    f"Attention mask should be of size {(target_len, batch_size * self.num_heads, source_len)}, but is"
                    f" {attention_mask.size()}"
                )
            attn_weights += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, target_len, source_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (batch_size * self.num_heads, target_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, target_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)
        # attn_output = self.out_proj(attn_output)
        attn_output = self.avs_adapt(attn_output).permute(1, 0, 2)
        return attn_output, attn_weights_reshaped


class Mask2FormerMaskedAttentionDecoderLayer(nn.Module):
    """
    The Mask2FormerMaskedAttentionDecoderLayer is made up of self-attention, cross (masked) attention as well as FFN
    blocks. The cross attention block used as part of `Mask2FormerMaskedAttentionDecoderLayer` is actually a `masked
    attention` block that restricts the attention to localized features centered around predicted segments which leads
    to faster convergence and improved performance. The order of self and cross (i.e. masked) attention blocks have
    also been swapped in Mask2FormerMaskedAttentionDecoder compared to a standard DetrDecoder as an optimization
    improvement.

    Args:
        config (`Mask2FormerConfig`):
            The configuration used to initialize the Mask2FormerMaskedAttentionDecoder.
    """

    def __init__(self, config: Mask2FormerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = self.config.hidden_dim
        self.pre_norm = self.config.pre_norm
        self.self_attn = Mask2FormerAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            is_decoder=True,
        )

        self.dropout = self.config.dropout
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.activation_dropout = self.config.dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, self.config.num_attention_heads, self.config.dropout)
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.config.dim_feedforward)
        self.fc2 = nn.Linear(self.config.dim_feedforward, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        hidden_states: torch.Tensor,
        level_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        # Masked(Cross)-Attention Block
        cross_attn_weights = None
        self_attn_weights = None

        residual = hidden_states

        hidden_states, cross_attn_weights = self.cross_attn(
            query=self.with_pos_embed(hidden_states, query_position_embeddings),
            key=self.with_pos_embed(encoder_hidden_states[level_index], position_embeddings[level_index]),
            value=encoder_hidden_states[level_index],
            attn_mask=encoder_attention_mask,
            key_padding_mask=None,
        )
        # print(hidden_states.shape) torch.Size([100, 5, 256])
        # print(cross_attn_weights.shape) torch.Size([5, 100, 144])
        
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)
        
        # Self Attention Block
        residual = hidden_states

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=query_position_embeddings,
            attention_mask=None,
            output_attentions=True,
        )
        # print(hidden_states.shape) torch.Size([100, 5, 256])
        # print(self_attn_weights.shape) torch.Size([5, 8, 100, 100])
        # print("***")
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        # print(outputs[0].shape)
        return outputs

    def forward_pre(
        self,
        hidden_states: torch.Tensor,
        level_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        # Masked(Cross)-Attention Block
        cross_attn_weights = None
        self_attn_weights = None

        residual = hidden_states

        hidden_states = self.cross_attn_layer_norm(hidden_states)

        hidden_states, cross_attn_weights = self.cross_attn(
            query=self.with_pos_embed(hidden_states, query_position_embeddings),
            key=self.with_pos_embed(encoder_hidden_states[level_index], position_embeddings[level_index]),
            value=encoder_hidden_states[level_index],
            attn_mask=encoder_attention_mask,
            key_padding_mask=None,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Self Attention Block
        residual = hidden_states

        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=query_position_embeddings,
            attention_mask=None,
            output_attentions=True,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        level_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(seq_len, batch, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(1, seq_len, tgt_len, src_len)`.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the keys in the masked-attention layer.
            query_position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the queries and keys in the self-attention layer.
            encoder_hidden_states (`torch.FloatTensor`):
                Cross attention input to the layer of shape `(seq_len, batch, embed_dim)`.
            encoder_attention_mask (`torch.FloatTensor`):
                Encoder attention mask of size`(1, seq_len, tgt_len, src_len)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """

        if self.pre_norm:
            outputs = self.forward_pre(
                hidden_states=hidden_states,
                level_index=level_index,
                position_embeddings=position_embeddings,
                query_position_embeddings=query_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
        else:
            # 用的这里
            outputs = self.forward_post(
                hidden_states=hidden_states,
                level_index=level_index,
                position_embeddings=position_embeddings,
                query_position_embeddings=query_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )

        return outputs

class Mask2FormerMaskedAttentionDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a
    [`Mask2FormerMaskedAttentionDecoderLayer`]. The decoder updates the query embeddings through multiple cross
    (masked) and self-attention layers. The decoder uses a new **masked attention** mechanism instead of the standard
    cross-attention, which extracts localized features by constraining cross-attention to within the foreground region
    of the predicted mask for each query, instead of attending to the full feature map.

    Args:
        config (`Mask2FormerConfig`):
            Configuration used to instantiate Mask2FormerMaskedAttentionDecoder.
    """

    def __init__(self, config: Mask2FormerConfig):
        super().__init__()
        
        self.config = config
        self.mask_feature_size = config.mask_feature_size
        self.dropout = config.dropout
        self.layerdrop = config.dropout
        self.num_feature_levels = 3  # level embedding (3 scales)
        self.decoder_layers = config.decoder_layers - 1

        self.num_heads=config.num_attention_heads
        self.layers = nn.ModuleList(
            [Mask2FormerMaskedAttentionDecoderLayer(self.config) for _ in range(self.decoder_layers)]
        )
        # print(range(self.decoder_layers)) (0,9)
        self.layernorm = nn.LayerNorm(config.hidden_dim)

        self.mask_predictor = Mask2FormerMaskPredictor(
            hidden_size=config.hidden_dim,
            num_heads=self.num_heads,
            mask_feature_size=self.mask_feature_size,
        )

        self.gradient_checkpointing = False

        #################################################

        # self.k_values_feats = [3,7,13]
        # self.ratios_feats = [1,1,1] # 或者调整为 [0.5, 0.25, 0.15]
        # # self.sr_ratios = [4,2,1]
        # self.ctm_layer_feats = nn.ModuleList([
        #     CTM(sample_ratio=r, embed_dim=256, dim_out=256, k=k) 
        #     for r, k in zip(self.ratios_feats, self.k_values_feats)
        # ])

        # self.tcblock = nn.ModuleList([TCBlock(
        #         dim=256, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None,
        #         drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
        #         sr_ratio=sr)for sr in self.sr_ratios])

        # self.pre_mask_predictor = Mask2FormerMaskPredictor(
        #     hidden_size=config.hidden_dim,
        #     num_heads=self.num_heads,
        #     mask_feature_size=self.mask_feature_size,
        # )
        # self.pre_layers = nn.ModuleList(
        #     [Mask2FormerMaskedAttentionDecoderLayer(self.config) for _ in range(self.decoder_layers)]
        # )


        
    # def dpcknn(self, VIS_FEATS):
    #     VIS_FEATS_CTM = []
    #     # all_hidden_states = ()
    #     ratio = [1,1,1]
    #     for i in range(len(VIS_FEATS)):
    #         ctm = self.ctm_layer_feats[i]
    #         # blk = self.tcblock[i]
    #         visual_features = VIS_FEATS[i]
    #         K, B, C = visual_features.shape
    #         H = W = int(math.sqrt(K))
    #         # x = visual_features.permute(1,0,2)
    #         # B, N, _ = x.shape
    #         # device = x.device
    #         # idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
    #         # agg_weight = x.new_ones(B, N, 1)
    #         # token_dict = {'x': x,
    #         #               'token_num': N,
    #         #               'map_size': [H, W],
    #         #               'init_grid_size': [H, W],
    #         #               'idx_token': idx_token,
    #         #               'agg_weight': agg_weight}
    #         # token_dict = ctm(token_dict)  # down sample
    #         # token_dict = blk(token_dict)
    #         # token_dict['x'] = self.layernorm(token_dict['x'])
    #         visual_features = visual_features.unsqueeze(-1).reshape(B,C,int(math.sqrt(K)),int(math.sqrt(K)))
            
    #         B,C,H,W = visual_features.shape
    #         N_original = H * W  # 2304
    #         initial_token_dict = {
    #             'x': visual_features.flatten(2).transpose(1, 2).contiguous(),
    #             'map_size': [H, W],
    #             'init_grid_size': [H, W],
    #             'token_num': N_original,
    #             'idx_token': torch.arange(N_original)[None, :].expand(B, -1).cuda()
    #         }
    #         token_dict, info_dict = ctm(initial_token_dict)

    #         VIS_FEATS_CTM.append(token_dict['x'].permute(1,0,2))

    #     return VIS_FEATS_CTM
        
    def forward(
        self,
        inputs_embeds: torch.Tensor = None,
        multi_stage_positional_embeddings: torch.Tensor = None,
        anchor_multi_stage_positional_embeddings: torch.Tensor = None,
        pixel_embeddings: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        anchor_encoder_hidden_states: torch.Tensor = None,
        query_position_embeddings: torch.Tensor = None,
        feature_size_list: List = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        audio_boost_mask: Optional[torch.Tensor] = None,
        anchor_pixel_embeddings: Optional[torch.Tensor] = None,
        anchor_audio_boost_mask: Optional[torch.Tensor] = None,
        pre_intermediate_mask_predictions: Optional[tuple] = None,
        pre_intermediate: Optional[tuple] = None,
        pre_anchor_intermediate_mask_predictions: Optional[tuple] = None,
        pre_anchor_intermediate: Optional[tuple] = None,
        aux_mask_features: Optional[Tensor]=None,
        aux_multi_stage_features: Optional[Tensor]=None,
        aux_multi_stage_positional_embeddings: Optional[Tensor]=None,
        aux_audio_boost_mask: Optional[torch.Tensor] = None,
           
        # pre_inputs_embeds: torch.Tensor = None,
        # pre_all_hidden_states: Optional[tuple] = None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(num_queries, batch_size, hidden_size)`):
                The query embeddings that are passed into the decoder.
            multi_stage_positional_embeddings (`torch.FloatTensor` of shape `(height*width, batch_size, num_channels)`):
                Position embeddings that are added to the keys in each cross(masked)-attention layer.
            pixel_embeddings (`torch.FloatTensor`):
                Tensor of shape `(batch_size, num_channels, height, width)`, 1/4 scale features from the last Pixel
                Decoder.
            query_position_embeddings (`torch.FloatTensor` of shape `(num_queries, batch_size, hidden_size)`):
                , *optional*): Position embeddings that are added to the queries and keys in each self-attention layer.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the
                cross(masked)-attention of the decoder.
            feature_size_list (`List[torch.Size]` ):
                This is a list containing shapes (height & width) of multi-scale features from the Pixel Decoder.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        ###############################################################################################################################################
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
            # pre_hidden_states = inputs_embeds

        # intermediate hidden states with layernorm applied - required for predicting class logits
        intermediate = ()
        anchor_intermediate = ()
        # pre_intermediate_mask_predictions, pre_intermediate = (),()
        # decoder layers
        pre_all_hidden_states,all_hidden_states = (),() if output_hidden_states else None
        pre_attentions,attentions = (),() if output_attentions else None

        # intermediate mask predictions from transformer decoder layers
        intermediate_mask_predictions = ()
        anchor_intermediate_mask_predictions = ()
        

        intermediate_hidden_states = self.layernorm(inputs_embeds)
        intermediate += (intermediate_hidden_states,)

        # pre_intermediate += (intermediate_hidden_states,)
        
        predicted_mask, attention_mask = self.mask_predictor(
            intermediate_hidden_states, pixel_embeddings, feature_size_list[0]
        )         

        # pre_predicted_mask, pre_attention_mask = self.pre_mask_predictor(intermediate_hidden_states,pixel_embeddings,
        #                                                             feature_size_list[0])

        if audio_boost_mask is not None:
            audio_boost_mask = nn.functional.interpolate(
                audio_boost_mask.unsqueeze(1), size=feature_size_list[0], mode="bilinear"
            ).unsqueeze(1).repeat(1, self.num_heads, inputs_embeds.shape[0], 1, 1).flatten(0, 1).flatten(2)  # (b*nh)*q*hw
            attention_mask[torch.where(audio_boost_mask <0.5)] = True
            attention_mask[torch.where(audio_boost_mask >=0.5)] = False
            # if aux_audio_boost_mask is not None:
            #     aux_audio_boost_mask = nn.functional.interpolate(
            #     aux_audio_boost_mask.unsqueeze(1), size=feature_size_list[0], mode="bilinear"
            # ).unsqueeze(1).repeat(1, self.num_heads, inputs_embeds.shape[0], 1, 1).flatten(0, 1).flatten(2)  # (b*nh)*q*hw
            #     audio_boost_mask = torch.sigmoid(audio_boost_mask + aux_audio_boost_mask)
            #     attention_mask[torch.where(audio_boost_mask <0.5)] = True
            #     attention_mask[torch.where(audio_boost_mask >=0.5)] = False

        # if audio_boost_mask is not None:

        #     audio_boost_mask = nn.functional.interpolate(
        #         audio_boost_mask.unsqueeze(1),
        #         size=feature_size_list[0],
        #         mode="bilinear"
        #     ).unsqueeze(1).repeat(1, self.num_heads, inputs_embeds.shape[0], 1, 1)\
        #      .flatten(0,1).flatten(2)
        
        #     if aux_audio_boost_mask is not None:
        #         aux_audio_boost_mask = nn.functional.interpolate(
        #             aux_audio_boost_mask.unsqueeze(1),
        #             size=feature_size_list[0],
        #             mode="bilinear"
        #         ).unsqueeze(1).repeat(1, self.num_heads, inputs_embeds.shape[0], 1, 1)\
        #          .flatten(0,1).flatten(2)
        
        #         # 融合两个mask
        #         final_mask = torch.maximum(audio_boost_mask, aux_audio_boost_mask)
        
        #     else:
        #         final_mask = audio_boost_mask
        
        # attention_mask[torch.where(final_mask < 0.5)] = True
        # attention_mask[torch.where(final_mask >=0.5)] = False
                
            
            

            # pre_attention_mask[torch.where(audio_boost_mask < 0.5)] = True
            # pre_attention_mask[torch.where(audio_boost_mask >= 0.5)] = False
        
        intermediate_mask_predictions += (predicted_mask,)
        # pre_intermediate_mask_predictions += (pre_predicted_mask,)
        #########################################################
        # aux_multi_stage_features = self.dpcknn(encoder_hidden_states)
        #########################################################
        # print(feature_size_list)
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                # pre_all_hidden_states += (pre_hidden_states,)

            dropout_probability = random.uniform(0, 1)

            if self.training and (dropout_probability < self.layerdrop):
                continue

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    None,
                    None,
                )
            
            else:
                level_index = idx % self.num_feature_levels
                
                attention_mask[torch.where(attention_mask.sum(-1) == attention_mask.shape[-1])] = False

                layer_outputs = decoder_layer(
                    hidden_states,
                    level_index=level_index,
                    position_embeddings=multi_stage_positional_embeddings,
                    query_position_embeddings=query_position_embeddings,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )

                intermediate_hidden_states = self.layernorm(layer_outputs[0])

                predicted_mask, attention_mask = self.mask_predictor(
                    intermediate_hidden_states,
                    pixel_embeddings,
                    feature_size_list[(idx + 1) % self.num_feature_levels],
                )

                intermediate_mask_predictions += (predicted_mask,)

                # add intermediate hidden states with layer norm applied which will be used for predicting class logits
                intermediate += (intermediate_hidden_states,)



                ################################################################################
                # pre_attention_mask[torch.where(pre_attention_mask.sum(-1) == pre_attention_mask.shape[-1])] = False

                # pre_layer_outputs = self.pre_layers[idx](
                #     pre_hidden_states,
                #     level_index=level_index,
                #     position_embeddings=multi_stage_positional_embeddings,
                #     query_position_embeddings=query_position_embeddings,
                #     encoder_hidden_states=aux_multi_stage_features,
                #     encoder_attention_mask=pre_attention_mask,
                #     output_attentions=output_attentions,
                # )

                # pre_intermediate_hidden_states = self.layernorm(pre_layer_outputs[0])

                # pre_predicted_mask, pre_attention_mask = self.pre_mask_predictor(
                #     pre_intermediate_hidden_states,
                #     pixel_embeddings,
                #     feature_size_list[(idx + 1) % self.num_feature_levels],
                # )

                # pre_intermediate_mask_predictions += (pre_predicted_mask,)
                # pre_intermediate += (pre_intermediate_hidden_states,)
                ################################################################################

            hidden_states = layer_outputs[0]
            # pre_hidden_states = pre_layer_outputs[0]
            # print(output_attentions) # FALSE
            if output_attentions:
                attentions += (layer_outputs[1],)
                # pre_attentions += (pre_layer_outputs[1],)
                
        
        # combined_intermediate = tuple(a + b for a, b in zip(pre_intermediate, intermediate))
        # combined_intermediate_mask_predictions = tuple(a + b for a, b in zip(pre_intermediate_mask_predictions, intermediate_mask_predictions))
        ##############################################################################################################################################
        # if anchor_pixel_embeddings is not None: 
        #     with torch.no_grad():
        #         anchor_intermediate += (self.layernorm(inputs_embeds),)
        #         anchor_predicted_mask, anchor_attention_mask = self.mask_predictor(
        #             self.layernorm(inputs_embeds), anchor_pixel_embeddings, feature_size_list[0]
        #         )         
        #         anchor_audio_boost_mask = nn.functional.interpolate(
        #                 anchor_audio_boost_mask.unsqueeze(1), size=feature_size_list[0], mode="bilinear"
        #             ).unsqueeze(1).repeat(1, self.num_heads, inputs_embeds.shape[0], 1, 1).flatten(0, 1).flatten(2)  # (b*nh)*q*hw
        #         anchor_attention_mask[torch.where(audio_boost_mask <0.5)] = True
        #         anchor_attention_mask[torch.where(audio_boost_mask >=0.5)] = False
                
        #         anchor_intermediate_mask_predictions += (anchor_predicted_mask,)
        #         for idx, decoder_layer in enumerate(self.layers):
        #             level_index = idx % self.num_feature_levels
        #             anchor_attention_mask[torch.where(anchor_attention_mask.sum(-1) == anchor_attention_mask.shape[-1])] = False
    
        #             anchor_layer_outputs = decoder_layer(
        #                 hidden_states,
        #                 level_index=level_index,
        #                 position_embeddings=anchor_multi_stage_positional_embeddings,
        #                 query_position_embeddings=query_position_embeddings,
        #                 encoder_hidden_states=anchor_encoder_hidden_states,
        #                 encoder_attention_mask=anchor_attention_mask,
        #                 output_attentions=output_attentions,
        #             )
    
        #             anchor_intermediate_hidden_states = self.layernorm(anchor_layer_outputs[0])
    
        #             anchor_predicted_mask, anchor_attention_mask = self.mask_predictor(
        #                 anchor_intermediate_hidden_states,
        #                 anchor_pixel_embeddings,
        #                 feature_size_list[(idx + 1) % self.num_feature_levels],
        #             )
    
        #             anchor_intermediate_mask_predictions += (anchor_predicted_mask,)
    
        #             # add intermediate hidden states with layer norm applied which will be used for predicting class logits
        #             anchor_intermediate += (anchor_intermediate_hidden_states,)

        #     combined_anchor_intermediate = tuple(a + b for a, b in zip(pre_anchor_intermediate, anchor_intermediate))
        #     combined_anchor_intermediate_mask_predictions = tuple(a + b for a, b in zip(pre_anchor_intermediate_mask_predictions, anchor_intermediate_mask_predictions))
            
        
        if output_hidden_states:#True
            all_hidden_states += (hidden_states,)
            # pre_all_hidden_states += (pre_hidden_states,)

        hidden_states = hidden_states.transpose(1, 0)
        if not return_dict:
            outputs = [hidden_states, all_hidden_states, attentions, intermediate, intermediate_mask_predictions]
            return tuple(v for v in outputs if v is not None)
            
        
        # hidden_states = pre_hidden_states.transpose(1,0)+hidden_states
        # all_hidden_states = tuple(a + b for a, b in zip(pre_all_hidden_states,all_hidden_states))
        # attentions = tuple(a + b for a, b in zip(attentions,pre_attentions))
        return Mask2FormerMaskedAttentionDecoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=attentions,
            intermediate_hidden_states=intermediate,
            masks_queries_logits=intermediate_mask_predictions,
        ), 0#total_loss#intermediate_loss+intermediate_mask_predictions_loss


# Copied from transformers.models.maskformer.modeling_maskformer.PredictionBlock with MaskFormer->Mask2Former
class Mask2FormerPredictionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module) -> None:
        super().__init__()
        self.layers = [nn.Linear(in_dim, out_dim), activation]
        # Maintain submodule indexing as if part of a Sequential block
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class Mask2FormerMLPPredictionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        """
        A classic Multi Layer Perceptron (MLP).

        Args:
            input_dim (`int`):
                The input dimensions.
            hidden_dim (`int`):
                The hidden dimensions.
            output_dim (`int`):
                The output dimensions.
            num_layers (int, *optional*, defaults to 3):
                The number of layers.
        """
        super().__init__()
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]

        self.layers = []
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            activation = nn.ReLU() if i < num_layers - 1 else nn.Identity()
            layer = Mask2FormerPredictionBlock(in_dim, out_dim, activation=activation)
            self.layers.append(layer)
            # Provide backwards compatibility from when the class inherited from nn.Sequential
            # In nn.Sequential subclasses, the name given to the layer is its index in the sequence.
            # In nn.Module subclasses they derived from the instance attribute they are assigned to e.g.
            # self.my_layer_name = Layer()
            # We can't give instance attributes integer names i.e. self.0 is not permitted and so need to register
            # explicitly
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class Mask2FormerMaskPredictor(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mask_feature_size: torch.Tensor):
        """
        This class is used to get the predicted mask for a given Mask2FormerMaskedAttentionDecoder layer. It also
        generates the binarized attention mask associated with the given predicted mask. The attention mask obtained
        using predicted mask of the (l-1)th decoder layer is fed to the cross(masked)-attention block of the next
        decoder layer as input.

        Args:
            hidden_size (`int`):
                The feature dimension of the Mask2FormerMaskedAttentionDecoder
            num_heads (`int`):
                The number of heads used in the Mask2FormerMaskedAttentionDecoder
            mask_feature_size (`torch.Tensor`):
                one of the output dimensions of the predicted masks for each query
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.mask_embedder = Mask2FormerMLPPredictionHead(self.hidden_size, self.hidden_size, mask_feature_size)

    def forward(self, outputs: torch.Tensor, pixel_embeddings: torch.Tensor, attention_mask_target_size: int = None):
        mask_embeddings = self.mask_embedder(outputs.transpose(0, 1))
        outputs_mask = torch.einsum("bqc,   bchw -> bqhw", mask_embeddings, pixel_embeddings)
        # print('output mask:', outputs_mask.shape)  # output mask: torch.Size([1, 100, 96, 96])
        attention_mask = nn.functional.interpolate(
            outputs_mask, size=attention_mask_target_size, mode="bilinear", align_corners=False
        )
        attention_mask = attention_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attention_mask = (attention_mask.flatten(0, 1) < 0.5).bool()
        attention_mask = attention_mask.detach()
        
        return outputs_mask, attention_mask


######################################################################################################################################################################


######################################################################################################################################################################
class REF_Forward(nn.Module):


    def __init__(self, config: Mask2FormerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = self.config.hidden_dim
        self.pre_norm = self.config.pre_norm
        self.self_attn = Mask2FormerAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            is_decoder=True,
        )

        self.dropout = self.config.dropout
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.activation_dropout = self.config.dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, self.config.num_attention_heads, self.config.dropout)
        # self.mha = REF_AVS_Transformer()
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.config.dim_feedforward)
        self.fc2 = nn.Linear(self.config.dim_feedforward, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_pre(
            self,
            hidden_states: torch.Tensor,
            level_index: int = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_embeddings: Optional[torch.Tensor] = None,
            query_position_embeddings: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
    ):
        # Masked(Cross)-Attention Block
        cross_attn_weights = None
        self_attn_weights = None

        residual = hidden_states

        hidden_states = self.cross_attn_layer_norm(hidden_states)

        hidden_states, cross_attn_weights = self.cross_attn(
            query=self.with_pos_embed(hidden_states, query_position_embeddings),
            key=self.with_pos_embed(encoder_hidden_states[level_index], position_embeddings[level_index]),
            value=encoder_hidden_states[level_index],
            attn_mask=encoder_attention_mask,
            key_padding_mask=None,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Self Attention Block
        residual = hidden_states

        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=query_position_embeddings,
            attention_mask=None,
            output_attentions=True,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        level_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(seq_len, batch, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(1, seq_len, tgt_len, src_len)`.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the keys in the masked-attention layer.
            query_position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the queries and keys in the self-attention layer.
            encoder_hidden_states (`torch.FloatTensor`):
                Cross attention input to the layer of shape `(seq_len, batch, embed_dim)`.
            encoder_attention_mask (`torch.FloatTensor`):
                Encoder attention mask of size`(1, seq_len, tgt_len, src_len)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        outputs = self.forward_pre(
            hidden_states=hidden_states,
            level_index=level_index,
            position_embeddings=position_embeddings,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
      
        return outputs
        
class REF_AVS_Transformer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super(REF_AVS_Transformer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 256
        self.scaling = self.head_dim ** -0.5

        self.query_embedding = nn.Linear(embed_dim, embed_dim)
        self.key_embedding = nn.Linear(embed_dim, embed_dim)
        self.value_embedding = nn.Linear(embed_dim, embed_dim)

        self.out_projection = nn.Linear(embed_dim, embed_dim)

        self.beta_source_pool = nn.Parameter(torch.ones([1]))
        self.beta_source_attn = nn.Parameter(torch.ones([1]))

    def forward(self, target, source):
        seq_len_tgt, bsz, dim = target.size()

        _, seq_len_src, _ = source.size()
        seq_len_q = seq_len_tgt
        seq_len_kv = seq_len_src

        q = self.query_embedding(target.permute(1, 0, 2)) 
        k = self.key_embedding(source) 
        v = self.value_embedding(source) 

        q = q.view(bsz, seq_len_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  
        k = k.view(bsz, seq_len_kv, self.num_heads, self.head_dim).permute(0, 2, 3, 1)  
        v = v.view(bsz, seq_len_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)   

        scores = torch.matmul(q, k) * self.scaling 
        attention_weights = F.softmax(scores, dim=-1) 
        attended_values = torch.matmul(attention_weights, v)  
        attended_values = attended_values.permute(0, 2, 1, 3).reshape(bsz, seq_len_q, -1) 

        output = self.out_projection(attended_values)  
        
        output = nn.Sigmoid()(self.beta_source_attn) * output
        source_pool = nn.Sigmoid()(self.beta_source_pool) * torch.mean(source, dim=1).view(1, bsz, self.embed_dim)
 
        return target + output.permute(1, 0, 2) + source_pool


class Mask2FormerTransformerModule(nn.Module):
    """
    The Mask2Former's transformer module.
    """

    def __init__(self, in_features: int, config: Mask2FormerConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        self.hidden_dim=hidden_dim
        self.num_feature_levels = 3
        # 100
        # print(config.num_queries)
        self.position_embedder = Mask2FormerSinePositionEmbedding(num_pos_feats=hidden_dim // 2, normalize=True)
        self.queries_embedder = nn.Embedding(config.num_queries, hidden_dim)
        self.queries_features = nn.Embedding(config.num_queries, hidden_dim)

        self.input_projections = []

        for _ in range(self.num_feature_levels):
            if in_features != hidden_dim or config.enforce_input_projection:
                self.input_projections.append(nn.Conv2d(in_features, hidden_dim, kernel_size=1))
            else:
                self.input_projections.append(nn.Sequential())

        self.decoder = Mask2FormerMaskedAttentionDecoder(config=config)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        # self.audio_embed_0=nn.Embedding(1, hidden_dim)
        # self.uncertain_embed_0=nn.Embedding(1, hidden_dim)
        # self.silence_embed_0=nn.Embedding(1, hidden_dim)
        # self.audio_embed_1=nn.Embedding(1, hidden_dim)
        # self.uncertain_embed_1=nn.Embedding(1, hidden_dim)
        # self.silence_embed_1=nn.Embedding(1, hidden_dim)
        # self.audio_embed_2=nn.Embedding(1, hidden_dim)
        # self.uncertain_embed_2=nn.Embedding(1, hidden_dim)
        # self.silence_embed_2=nn.Embedding(1, hidden_dim)
    
        self.audio_embed=nn.Embedding(1, hidden_dim)
        # self.uncertain_embed=nn.Embedding(1, hidden_dim)
        self.silence_embed=nn.Embedding(1, hidden_dim)

        self.learn_masks = nn.ParameterList([
            nn.Parameter(torch.zeros(5, 144)),
            nn.Parameter(torch.zeros(5, 576)),
            nn.Parameter(torch.zeros(5, 2304))
        ])
        for mask in self.learn_masks:
            nn.init.normal_(mask, std=0.01)

        # self.auxaudio_emb = nn.Embedding(1,hidden_dim)
        # self.auxsilence_emb = nn.Embedding(1,hidden_dim)
        # self.aux_learn_masks = nn.ParameterList([
        #     nn.Parameter(torch.zeros(5, 144)),
        #     nn.Parameter(torch.zeros(5, 576)),
        #     nn.Parameter(torch.zeros(5, 2304))
        # ])
        # for mask in self.aux_learn_masks:
        #     nn.init.normal_(mask, std=0.01)

        # self.maskparam = nn.ParameterList([
        #     nn.Parameter(torch.tensor(1.0)),
        #     nn.Parameter(torch.tensor(1.0)),
        #     nn.Parameter(torch.tensor(1.0))
        # ])

    
        # self.uncertain_audio_embed=nn.Embedding(1, hidden_dim)
        # self.uncertain_silence_embed=nn.Embedding(1, hidden_dim)
        # nn.init.normal_(self.audio_embed.weight, std=0.02)
        # nn.init.normal_(self.uncertain_embed.weight, std=0.02)
    #     nn.init.normal_(self.silence_embed.weight, std=0.02)
    #     self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    #     self.learn_masks = nn.ParameterList([
    #     nn.Parameter(torch.zeros(5, 144)),
    #     nn.Parameter(torch.zeros(5, 576)),
    #     nn.Parameter(torch.zeros(5, 2304))
    # ])
    #     for mask in self.learn_masks:
    #         nn.init.normal_(mask, std=0.01)
        
        # nn.init.normal_(self.uncertain_audio_embed.weight, std=0.02)
        # nn.init.normal_(self.uncertain_silence_embed.weight, std=0.02)

        # self.highcon_audio_embed=nn.Embedding(1, hidden_dim)
        # self.highcon_silence_embed=nn.Embedding(1, hidden_dim)
        # self.higcon_uncertain_embed=nn.Embedding(1,hidden_dim)
        
        # self.ref_avs_attn = REF_AVS_Transformer()


        # self.firstlevel_embed = nn.Parameter(torch.empty(5, 256, 12,12))
        # nn.init.kaiming_uniform_(self.firstlevel_embed, a=math.sqrt(5))
        # self.secondlevel_embed = nn.Parameter(torch.empty(5, 256, 24,24))
        # nn.init.kaiming_uniform_(self.secondlevel_embed, a=math.sqrt(5))
        # self.thirdlevel_embed = nn.Parameter(torch.empty(5, 256, 48,48))
        # nn.init.kaiming_uniform_(self.thirdlevel_embed, a=math.sqrt(5))

        # self.ref_avs_0 = REF_AVS_Transformer()
        # self.ref_avs_1 = REF_AVS_Transformer()
        # self.ref_avs_2 = REF_AVS_Transformer()
        # self.k_values = [2, 2, 2]
        # self.ratios = [0.25, 0.25, 0.25] # 或者调整为 [0.5, 0.25, 0.15]
        
        # self.ctm_layer = nn.ModuleList([
        #     CTM(sample_ratio=r, embed_dim=256, dim_out=256, k=k) 
        #     for r, k in zip(self.ratios, self.k_values)
        # ])


        ##############################################################
    
        # self.k_values_feats = [3,7,13]
        # self.ratios_feats = [1,1,1] # 或者调整为 [0.5, 0.25, 0.15]
        # # self.sr_ratios = [4,2,1]
        # self.ctm_layer_feats = nn.ModuleList([
        #     CTM(sample_ratio=r, embed_dim=256, dim_out=256, k=k) 
        #     for r, k in zip(self.ratios_feats, self.k_values_feats)
        # ])

        # # self.tcblock = nn.ModuleList([TCBlock(
        # #         dim=256, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None,
        # #         drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
        # #         sr_ratio=sr)for sr in self.sr_ratios])

        # self.prelayers = nn.ModuleList(
        #     [REF_Forward(self.config) for _ in range(9)]
        # )
        # self.premask_predictor = Mask2FormerMaskPredictor(
        #     hidden_size=config.hidden_dim,
        #     num_heads=8,
        #     mask_feature_size=config.mask_feature_size,
        # )
        # self.layernorm = nn.LayerNorm(config.hidden_dim)      


        self.upsample_24_TO_12 = Semantic_flow(256,256)
        self.upsample_48_TO_24 = Semantic_flow(256,256)
    
        ##############################################################


    # def token_downup(self, target_dict, source_dict):
    #     """Transform token features between different distribution.
    
    #     Returns:
    #         x_out (Tensor[B, N, C]): token features.
    
    #     Args:
    #         target_dict (dict): dict for target token information
    #         source_dict (dict): dict for source token information.
    #     """
    
    #     x_s = source_dict['x']
    #     idx_token_s = source_dict['idx_token']
    #     idx_token_t = target_dict['idx_token']
    #     T = target_dict['token_num']
    #     B, S, C = x_s.shape
    #     N_init = idx_token_s.shape[1]
    
    #     weight = target_dict['agg_weight'] if 'agg_weight' in target_dict.keys() else None
    #     if weight is None:
    #         weight = x_s.new_ones(B, N_init, 1)
    #     weight = weight.reshape(-1)
    
    #     # choose the way with fewer flops.
    #     if N_init < T * S:
    #         # use sparse matrix multiplication
    #         # Flops: B * N_init * (C+2)
    #         idx_token_t = idx_token_t + torch.arange(B, device=x_s.device)[:, None] * T
    #         idx_token_s = idx_token_s + torch.arange(B, device=x_s.device)[:, None] * S
    #         coor = torch.stack([idx_token_t, idx_token_s], dim=0).reshape(2, B * N_init)
    
    #         # torch.sparse.spmm does not support fp16
    #         with torch.cuda.amp.autocast(enabled=False):
    #             # torch.sparse does not support grad for sparse matrix
    #             weight = weight.float().detach()
    #             # build a matrix with shape [B*T, B*S]
    #             A = torch.sparse.FloatTensor(coor, weight, torch.Size([B * T, B * S]))
    #             # normalize the matrix
    #             all_weight = A.type(torch.float32) @ x_s.new_ones(B * S, 1).type(torch.float32) + 1e-6
    #             weight = weight / all_weight[(idx_token_t).reshape(-1), 0]
    #             A = torch.sparse.FloatTensor(coor, weight, torch.Size([B * T, B * S]))
    #             # sparse matmul
    #             x_out = A.type(torch.float32) @ x_s.reshape(B * S, C).type(torch.float32)
    #     else:
    #         # use dense matrix multiplication
    #         # Flops: B * T * S * (C+2)
    #         idx_batch = torch.arange(B, device=x_s.device)[:, None].expand(B, N_init)
    #         coor = torch.stack([idx_batch, idx_token_t, idx_token_s], dim=0).reshape(3, B * N_init)
    #         weight = weight.detach()  # detach to reduce training time
    #         # build a matrix with shape [B, T, S]
    #         A = torch.sparse.FloatTensor(coor, weight, torch.Size([B, T, S])).to_dense()
    #         # normalize the matrix
    #         A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
    #         # dense matmul
    #         x_out = A @ x_s
    
    #     x_out = x_out.reshape(B, T, C).type(x_s.dtype)
    #     return x_out
    # def aggregate_pos_by_cluster(self, pos, idx_cluster, cluster_num):
    #     """
    #     Args:
    #         pos: [N, B, C] 原始位置编码
    #         idx_cluster: [B, N] 每个 token 属于哪个簇的索引
    #         cluster_num: P (下采样后的簇数量)
    #     """
    #     B, N, C = pos.shape
    #     device = pos.device

    #     sum_pos = torch.zeros(B, cluster_num, C, device=device)
    #     count = torch.zeros(B, cluster_num, 1, device=device)
    #     idx_expand = idx_cluster.unsqueeze(-1).expand(B, N, C)
    #     sum_pos.scatter_add_(1, idx_expand, pos)
    #     ones = torch.ones(B, N, 1, device=device)
    #     count.scatter_add_(1, idx_cluster.unsqueeze(-1), ones)
    #     downsampled_pos = sum_pos / (count + 1e-6)
    #     return downsampled_pos.permute(1, 0, 2)      

    def dpcknn(self, query_features, query_embeddings, VIS_FEATS, VIS_FEATS_pos, attention_mask, mask_features, size_list, intermediate_mask_predictions,  
               intermediate, select):
        VIS_FEATS_CTM = []
        VIS_FEATS_CTM_pos = []
        # all_hidden_states = ()
        ratio = [1,1,1]
        for i in range(len(VIS_FEATS)):
            ctm = self.ctm_layer_feats[i]
            # blk = self.tcblock[i]
            
            visual_features = VIS_FEATS[i]
            K, B, C = visual_features.shape
            H = W = int(math.sqrt(K))
            # x = visual_features.permute(1,0,2)

            visual_features = visual_features.permute(1, 2, 0).reshape(B, C, H, W)
            B,C,H,W = visual_features.shape
            N_original = H * W  # 2304
            P = int(H*W*ratio[i])  # 采样后的组数 (2304 * 0.25)
            initial_token_dict = {
                'x': visual_features.flatten(2).transpose(1, 2).contiguous(),
                'map_size': [H, W],
                'init_grid_size': [H, W],
                'token_num': N_original,
                'idx_token': torch.arange(N_original)[None, :].expand(B, -1).cuda()
            }
            token_dict, info_dict = ctm(initial_token_dict)


            # B, N, _ = x.shape
            # device = x.device
            # idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
            # agg_weight = x.new_ones(B, N, 1)
            # token_dict = {'x': x,
            #               'token_num': N,
            #               'map_size': [H, W],
            #               'init_grid_size': [H, W],
            #               'idx_token': idx_token,
            #               'agg_weight': agg_weight}


            # token_dict = ctm(token_dict)  # down sample
            # token_dict = blk(token_dict)
            
            # token_dict['x'] = self.layernorm(token_dict['x'])

            # print(token_dict['x'].shape)  # print(token_dict['x'].shape) torch.Size([5, 36, 256]) torch.Size([5, 144, 256]) torch.Size([5, 576, 256])

            # blk = self.tcblock[i]

            
            VIS_FEATS_CTM.append(token_dict['x'].permute(1,0,2))
            VIS_FEATS_CTM_pos=VIS_FEATS_pos#.append(VIS_FEATS_pos)

        for i in range(9):
            # all_hidden_states += (query_features, )
            level_index = i % 3
            attention_mask[torch.where(attention_mask.sum(-1) == attention_mask.shape[-1])] = False

            outputs = self.prelayers[i].forward_pre(
                hidden_states=query_features,
                level_index=level_index,
                position_embeddings=VIS_FEATS_CTM_pos,
                query_position_embeddings=query_embeddings,
                encoder_hidden_states=VIS_FEATS_CTM,
                encoder_attention_mask=attention_mask,
                output_attentions=True,
            )
            intermediate_hidden_states = self.layernorm(outputs[0])

            predicted_mask, attention_mask = self.premask_predictor(
                intermediate_hidden_states,
                mask_features,
                size_list[(i + 1) % 3],
            )
            query_features = outputs[0]

            intermediate_mask_predictions += (predicted_mask,)
            intermediate += (intermediate_hidden_states,)
        # all_hidden_states += (query_features,)
        return intermediate_mask_predictions, intermediate#, query_features, all_hidden_states
    
    def forward(
            self,
            multi_scale_features: List[Tensor],
            dpcknn_scale_features: List[Tensor],
            mask_features: Tensor,
            prompt_features_projected: Tensor = None,
            output_hidden_states: bool = False,
            output_attentions: bool = False,
            visualClip: Optional[bool]=None,
            audio_boost_mask: Optional[Tensor] = None,
            anchor_mask_features: Optional[Tensor] = None,
            anchor_audio_boost_mask: Optional[Tensor] = None,
            CLIP_intersection: Optional[Tensor] = None,
            aux_multi_scale_features: Optional[Tensor]=None,
            aux_mask_features: Optional[Tensor]=None,
            anchor_pixel_level_module_output_decoder: Optional[List]=None,
            iter: Optional[int] = None,
            aux_audio_boost_mask: Optional[Tensor] = None,
            batch_id: Optional[str]=None,
            
        
    ) -> Mask2FormerMaskedAttentionDecoderOutput:
        
        anchor_multi_stage_features, anchor_multi_stage_positional_embeddings = [],[]
        aux_multi_stage_features,aux_multi_stage_positional_embeddings = [],[]
        multi_stage_features, multi_stage_positional_embeddings = [],[]
        size_list,entropy_map_list = [],[]
        anchor_L_audio = 0
        anchor_L_feat = 0.0
        anchor_L_pos  = 0.0
        # print(audio_boost_mask.shape) [5,384,384]
        loss_bce, loss_conf,loss_feat_dist, loss_mmd,multiSales_diceloss =0.0,0.0,0.0,0.0,0.0
        low = 0.4
        high = 0.6      

        if audio_boost_mask is not None:
            for i in range(self.num_feature_levels):
                size_list.append(multi_scale_features[i].shape[-2:])
                eps = 1e-8
                H, W = size_list[-1]
                position_embed = self.position_embedder(multi_scale_features[i], None).flatten(2).permute(0, 2, 1) # [B, HW, C]
                audio_prompt = nn.functional.interpolate(audio_boost_mask.unsqueeze(1), size=size_list[-1], mode="bilinear").flatten(2).squeeze(1) # [B, HW]
                # anchor_audio_prompt = nn.functional.interpolate(anchor_audio_boost_mask.unsqueeze(1), size=size_list[-1], mode="bilinear").flatten(2).squeeze(1)
                # pred = (audio_prompt*self.learn_masks[i]).sigmoid()
                # target = anchor_audio_prompt
                # intersection = (pred * target).sum(dim=1)
                # union = pred.sum(dim=1) + target.sum(dim=1)
                # multi_diceloss = 1 - (2 * intersection + eps) / (union + eps)
                # multiSales_diceloss += multi_diceloss.mean()
                # ------------------ Trick 核心开始 ------------------
                # 2. 全局统计量计算 (利用 Batch 中所有非黑帧来稳健协方差)
                all_mask_s = audio_prompt <= low
                all_mask_a = audio_prompt >= high

                # 收集整个 Batch 的特征用于计算协方差
                all_s_feat = position_embed[all_mask_s]
                all_a_feat = position_embed[all_mask_a]
                
                # 默认 inv_cov (防止整个 batch 全黑导致报错)
                inv_cov = torch.eye(position_embed.shape[-1], device=position_embed.device)
                
                if all_a_feat.shape[0] > 10: # 如果 batch 内有足够的物体像素
                    feat_combined = torch.cat([all_s_feat, all_a_feat], dim=0)
                    mean_global = feat_combined.mean(0, keepdim=True)
                    feat_centered = feat_combined - mean_global
                    cov = (feat_centered.T @ feat_centered) / (feat_centered.shape[0] - 1)
                    cov = cov + 1e-5 * torch.eye(cov.shape[0], device=cov.device)
                    inv_cov = torch.inverse(cov)
                entropy_map = -(audio_prompt * torch.log(audio_prompt + eps) + (1 - audio_prompt) * torch.log(1 - audio_prompt + eps))
                entropy_map = entropy_map / torch.log(torch.tensor(2.0, device=audio_prompt.device))
                uncertain_mask = (audio_prompt > low) & (audio_prompt < high)
                penalty_map = torch.zeros_like(audio_prompt)
                penalty_map[uncertain_mask] = 1.0 - 4.0 * (audio_prompt[uncertain_mask] - 0.5) ** 2
                
                sample_uncertainty = entropy_map.mean(dim=-1) + 5.0 * penalty_map.mean(dim=-1)
                sample_weights = F.softmax(sample_uncertainty.detach(), dim=0) * 5
                loss_conf += (sample_uncertainty * sample_weights).mean()

                m = audio_prompt.reshape(-1,1,H,W)
                u = uncertain_mask.reshape(-1,1,H,W).float()
                diff_x = torch.abs(m[:,:,:,:-1]-m[:,:,:,1:])
                diff_y = torch.abs(m[:,:,:-1,:]-m[:,:,1:,:])
                loss_conf += (2*((diff_x*u[:,:,:,:-1]).mean()+(diff_y*u[:,:,:-1,:]).mean()))

                ########可视化########
                from sklearn.manifold import TSNE
                from sklearn.decomposition import PCA
                # from matplotlib.patches import Ellipse
                def show(x, title):
                    plt.imshow(x, cmap='viridis')
                    plt.colorbar()
                    plt.title(title)
                    plt.axis('off')
            
                # for kkk in range(min(5, audio_prompt.shape[0])):
                #     prompt = audio_prompt[kkk].reshape(H, W).detach().cpu()
                #     entropy = entropy_map[kkk].reshape(H, W).detach().cpu()
                #     penalty = penalty_map[kkk].reshape(H, W).detach().cpu()
                #     uncertain = uncertain_mask[kkk].reshape(H, W).float().detach().cpu()
                #     plt.figure(figsize=(16,4))
                #     plt.subplot(1,4,1)
                #     show(prompt, "Audio Prompt")
                #     plt.subplot(1,4,2)
                #     show(entropy, "Entropy")
                #     plt.subplot(1,4,3)
                #     show(penalty, "Penalty")
                #     plt.subplot(1,4,4)
                #     show(uncertain, "Uncertain Region")
                #     plt.savefig(f'./VIS_new/{batch_id}_uncertainty_{kkk}.png')
                #     plt.close()


                    # feat = position_embed[kkk].reshape(-1, position_embed.shape[-1]).detach().cpu().numpy()
                    
                    # pca = PCA(n_components=2)
                    # feat_2d = pca.fit_transform(feat)
                    # features=feat_2d
                    # cov = np.cov(features.T)
                    # mean = features.mean(axis=0)
                
                    # eigvals, eigvecs = np.linalg.eigh(cov)
                    # angle = np.degrees(np.arctan2(*eigvecs[:,1][::-1]))
                
                    # width, height = 2 * np.sqrt(eigvals)
                
                    # fig, ax = plt.subplots()
                    # ax.scatter(features[:,0], features[:,1], s=5)
                
                    # ellipse = Ellipse(mean, width, height, angle=angle, fill=False)
                    # ax.add_patch(ellipse)
                
                    # plt.title("RD: Covariance Structure")
                    # plt.savefig(f'./VIS_RD/{batch_id}_RD_{kkk}.png')

                        # ===== 1. reshape =====
                    # feat = position_embed[kkk].reshape(-1, position_embed.shape[-1]).detach().cpu().numpy()
                    # prompt = audio_prompt[kkk].reshape(-1).detach().cpu().numpy()
                
                    # # ===== 2. 三种 mask =====
                    # mask_s = prompt <= low
                    # mask_a = prompt >= high
                    # mask_u = (prompt > low) & (prompt < high)
                
                    # # ===== 3. t-SNE =====
                    # tsne = TSNE(n_components=2, perplexity=30, random_state=0)
                    # feat_2d = tsne.fit_transform(feat)
                
                    # # ===== 4. 开始画 =====
                    # plt.figure(figsize=(18,5))
                
                    # # -----------------------------------
                    # # 🔵 1. 原始（不区分 uncertain）
                    # # -----------------------------------
                    # plt.subplot(1,3,1)
                    # plt.scatter(feat_2d[:,0], feat_2d[:,1], s=2, alpha=0.5)
                    # plt.title("Raw Feature Distribution")
                    # plt.xticks([])
                    # plt.yticks([])
                
                    # # -----------------------------------
                    # # 🟢 2. RD + ODS（两类拉开）
                    # # -----------------------------------
                    # plt.subplot(1,3,2)
                    # plt.scatter(feat_2d[mask_s,0], feat_2d[mask_s,1], s=2, alpha=0.5, label="Silence")
                    # plt.scatter(feat_2d[mask_a,0], feat_2d[mask_a,1], s=2, alpha=0.5, label="Active")
                    # plt.title("After RD + ODS")
                    # plt.legend()
                    # plt.xticks([])
                    # plt.yticks([])
                
                    # # -----------------------------------
                    # # 🔴 3. HARD（uncertain参与）
                    # # -----------------------------------
                    # plt.subplot(1,3,3)
                    # plt.scatter(feat_2d[mask_s,0], feat_2d[mask_s,1], s=2, alpha=0.4, label="Silence")
                    # plt.scatter(feat_2d[mask_a,0], feat_2d[mask_a,1], s=2, alpha=0.4, label="Active")
                    # plt.scatter(feat_2d[mask_u,0], feat_2d[mask_u,1], s=3, alpha=0.8, label="Uncertain")
                    # plt.title("After HARD (Final)")
                    # plt.legend()
                    # plt.xticks([])
                    # plt.yticks([])
                
                    # plt.tight_layout()
                    # plt.savefig(f'./VIS_RD/{batch_id}_RD_{kkk}.png')
                    # plt.close()

                ########可视化########


                
                batch_size = position_embed.shape[0]
                for b in range(batch_size):
                    b_prompt = audio_prompt[b] # (HW,)
                    b_feat = position_embed[b] # (HW, C)
                    
                    b_mask_a = b_prompt >= high
                    b_mask_s = b_prompt <= low
                    b_mask_u = (b_prompt > low) & (b_prompt < high)
                    
                    
                    # --- Trick: 处理全黑样本 ---
                    if not b_mask_a.any():
                        # 如果是全黑帧（无物体），强制所有特征向“静音中心”靠拢
                        # 这样能净化背景特征，防止背景里出现类似物体的噪声
                        s_center_fixed = self.silence_embed.weight
                        dist_to_silence = torch.norm(b_feat - s_center_fixed, dim=-1)
                        loss_mmd += 0.2 * dist_to_silence.mean() / batch_size
                        continue 
            
                    # --- 处理含物体样本 ---
                    s_Feat = b_feat[b_mask_s]
                    a_Feat = b_feat[b_mask_a]
                    u_Feat = b_feat[b_mask_u]

                    
                    
                    if s_Feat.shape[0] < 5: 
                        continue # 安全检查
            
                    s_center = s_Feat.mean(0, keepdim=True)
                    a_center = a_Feat.mean(0, keepdim=True)
                    # 计算不确定像素的拉力与推力
                    if u_Feat.shape[0] > 0:
                        x_norm = torch.clamp(torch.sum(u_Feat*u_Feat, dim=-1), max=1-1e-5)
                        y_norm_s = torch.clamp(torch.sum(s_center*s_center, dim=-1), max=1-1e-5)
                        diff_s = torch.sum((u_Feat-s_center)**2, dim=-1)
                        denom_s = (1-x_norm)*(1-y_norm_s)
                        z_s = 1 + 2*diff_s/denom_s
                        d_to_s = torch.acosh(z_s)

                        
                        y_norm_a = torch.clamp(torch.sum(a_center*a_center, dim=-1), max=1-1e-5)
                        diff_a = torch.sum((u_Feat-a_center)**2, dim=-1)
                        denom_a = (1-x_norm)*(1-y_norm_a)
                        z_a = 1 + 2*diff_a/denom_a
                        d_to_a = torch.acosh(z_a)
                        pull_dist = torch.min(d_to_s, d_to_a)
                        push_dist = torch.max(d_to_s, d_to_a)
                        loss_mmd += (pull_dist / (push_dist + 1e-6)).mean() / batch_size
                        # 把 pull_dist和 push_dist改成以下  -> 不行 上面的是（72.5 81.3）
                        # pos = anchor_audio_prompt
                        # neg = 1 - anchor_audio_prompt
                        # pull = pos * d_to_a + neg * d_to_s
                        # push = pos * d_to_s + neg * d_to_a
                        
                        # loss_mmd += (pull / (push + 1e-6)).mean()
                    

                        # loss_mmd += F.softplus(d_to_s - d_to_a).mean() / batch_size
                        
                        # 使用全局 inv_cov 计算马氏距离 （72.3 82.8）
                        # diff_s = u_Feat - s_center
                        # diff_a = u_Feat - a_center
                        # d_to_s = torch.sqrt(torch.sum((diff_s @ inv_cov) * diff_s, dim=-1))
                        # d_to_a = torch.sqrt(torch.sum((diff_a @ inv_cov) * diff_a, dim=-1
            
                    # 类间距惩罚 (让物体中心与背景中心拉开) 欧式距离
                    diff_centers = s_center - a_center
                    d_cluster = torch.sqrt(torch.sum((diff_centers @ inv_cov) * diff_centers, dim=-1)).mean()
                    loss_mmd += (-0.05 * d_cluster / batch_size)#-0.1->0.05
                    # 正交分解
                    s_norm = F.normalize(s_center, dim=-1)
                    a_norm = F.normalize(a_center, dim=-1)
                    orth_loss = ((s_norm * a_norm).sum(-1)) ** 2
                    loss_mmd += 0.5 * orth_loss.mean() / batch_size
            
                # 将位置编码反馈回特征中 (根据音频 Mask 修正)
                position_embed[audio_prompt <= low] += self.silence_embed.weight
                position_embed[audio_prompt >= high] += self.audio_embed.weight

                multi_stage_positional_embeddings.append(position_embed.permute(1,0,2))
                multi_stage_features.append(
                    self.input_projections[i](multi_scale_features[i]).flatten(2)
                    + self.level_embed.weight[i][None, :, None]
                )
                multi_stage_features[-1] = multi_stage_features[-1].permute(2, 0, 1)
                entropy_map_list.append(entropy_map)
                # print(multi_stage_features[0].shape)             

                # multi_feat = (self.input_projections[i](multi_scale_features[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
                # multi_stage_features.append(multi_feat)
                # proj_feat = multi_feat.permute(0,2,1)  + position_embed
                # B, HW, C = proj_feat.shape
                # obj_mask = (audio_prompt >= high).float().unsqueeze(-1).detach() # [B, HW, 1]
                # bg_mask = (audio_prompt <= low).float().unsqueeze(-1).detach()   # [B, HW, 1]
                # obj_count = obj_mask.sum(dim=1) # [B, 1]
                # bg_count = bg_mask.sum(dim=1)   # [B, 1]
                # feat_obj_center = (proj_feat * obj_mask).sum(dim=1) / (obj_count + 1e-6) # [B, C]
                # feat_bg_center = (proj_feat * bg_mask).sum(dim=1) / (bg_count + 1e-6)   # [B, C]
                # # if audio_emb.shape[-1] != C:
                # #     # 如果维度不一致，这里通常需要一个小的 projector 或 linear 层
                # #     return torch.tensor(0.0, device=proj_feat.device)
                # tmp = prompt_features_projected.squeeze(0)
                # loss_align = F.cosine_similarity(feat_obj_center, tmp, dim=-1)
                # loss_sep = F.cosine_similarity(feat_obj_center, feat_bg_center, dim=-1)
                # valid_mask = (obj_count.squeeze(-1) > 5).float()
                # final_loss = ((-loss_align + 0.1 * loss_sep) * valid_mask).sum() / (valid_mask.sum() + 1e-6)

                
                
                
                
        else:
            for i in range(self.num_feature_levels):
                size_list.append(multi_scale_features[i].shape[-2:])
                multi_stage_positional_embeddings.append(self.position_embedder(multi_scale_features[i], None).flatten(2))
                multi_stage_features.append(
                    self.input_projections[i](multi_scale_features[i]).flatten(2)
                    + self.level_embed.weight[i][None, :, None]
                )
    
                # Flatten (batch_size, num_channels, height, width) -> (height*width, batch_size, num_channels)
                multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(2, 0, 1)
                multi_stage_features[-1] = multi_stage_features[-1].permute(2, 0, 1)

        # print(multi_stage_features[0].shape) C B HW
        # stage_tokens = [f.permute(1,0,2) for f in multi_stage_features]  # C B HW → B,C,HW?
        # #归一化到 Poincare ball
        # stage_tokens = [F.normalize(f, dim=-1) * 0.9 for f in stage_tokens]
        
        
        # fine,mid,coarse = stage_tokens[0],stage_tokens[1],stage_tokens[2]
        # coarse_node = coarse.mean(1, keepdim=True)
        # mid_node = mid.mean(1, keepdim=True)
        # x2 = torch.sum(mid * mid, dim=-1, keepdim=True)
        # y2 = torch.sum(coarse_node  * coarse_node , dim=-1, keepdim=True)
        # xy = torch.sum(mid * coarse_node, dim=-1, keepdim=True)
        # num = (1 + 2 * xy + y2) * mid + (1 - x2) * coarse_node
        # denom = 1 + 2 * xy + x2 * y2 + eps
        # mid_reason = num / denom

        # x2 = torch.sum(fine * fine, dim=-1, keepdim=True)
        # y2 = torch.sum(mid_node * mid_node, dim=-1, keepdim=True)
        # xy = torch.sum(fine * mid_node, dim=-1, keepdim=True)
        # num = (1 + 2 * xy + y2) * fine + (1 - x2) * mid_node
        # denom = 1 + 2 * xy + x2 * y2 + eps
        # fine_reason = num / denom

        # entropy_fine = entropy_map_list[0].unsqueeze(-1)
        # entropy_mid = entropy_map_list[1].unsqueeze(-1)
    
        # entropy_fine = entropy_fine / (entropy_fine.mean(dim=1,keepdim=True)+1e-6)
        # entropy_mid = entropy_mid / (entropy_mid.mean(dim=1,keepdim=True)+1e-6)
    
        # entropy_fine = torch.clamp(entropy_fine,0,2)
        # entropy_mid = torch.clamp(entropy_mid,0,2)
    
        # mid = mid + entropy_mid * mid_reason
        # fine = fine + entropy_fine * fine_reason
    
        # stage_tokens[0] = fine
        # stage_tokens[1] = mid
    
        # # B,HW,C → B,C,HW
        # stage_features = [stage_tokens[i].permute(1,0,2)for i in range(len(stage_tokens))]
        # multi_stage_features = stage_features

        # print(multi_stage_positional_embeddings[0].shape) 144,5,256
        # print(multi_stage_features[0].shape) torch.Size([144, 5, 256])

        multi_stage_features[0] = entropy_map_list[0].permute(1,0).unsqueeze(-1) * multi_stage_features[0] + 0.1 * self.upsample_24_TO_12(multi_stage_positional_embeddings[1].permute(1,2,0),multi_stage_features[0].permute(1,2,0)).permute(2,0,1)
        multi_stage_features[1] = entropy_map_list[1].permute(1,0).unsqueeze(-1) * multi_stage_features[1] + 0.1 * self.upsample_48_TO_24(multi_stage_positional_embeddings[2].permute(1,2,0),multi_stage_features[1].permute(1,2,0)).permute(2,0,1)


        feat = multi_stage_features[0]  # [HW, B, C]
        feat = feat.permute(1, 2, 0).reshape(5, 256, 12, 12)
        
        feat_mean = feat.mean(dim=1)  # [B, H, W]
        
        # for kkk in range(5):
        #     fmap = feat[kkk].reshape(256, -1).permute(1,0).detach().cpu().numpy()  # [HW, C]
        
        #     pca = PCA(n_components=1)
        #     fmap_pca = pca.fit_transform(fmap)  # [HW,1]
        #     fmap_pca = fmap_pca.reshape(12,12)
        
        #     plt.figure(figsize=(5,5))
        #     show(fmap_pca, "Feature PCA")
        #     plt.savefig(f'./VIS_feat/{batch_id}_feat_pca_{kkk}.png')
        #     plt.close()
        


                
                
        # print(anchor_L_audio)
        _, batch_size, _ = multi_stage_features[0].shape

        # [num_queries, batch_size, num_channels]
        query_embeddings = self.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        query_features = self.queries_features.weight.unsqueeze(1).repeat(1, batch_size, 1)
        sim = torch.cosine_similarity(self.queries_embedder.weight.unsqueeze(1), prompt_features_projected, dim=-1)

        # print(sim.shape) torch.Size([100, 5])
        # print(query_features.shape) torch.Size([100, 5, 256])
        # print(prompt_features_projected.shape) torch.Size([1, 5, 256])

        if prompt_features_projected is not None:
            query_features = query_features + torch.einsum("qbn, nbc -> qbc", sim.unsqueeze(-1),prompt_features_projected)

       
        pre_intermediate_mask_predictions, pre_intermediate, anchor_pre_intermediate_mask_predictions, anchor_pre_intermediate = (), (), (), ()

        decoder_output,anchor_L = self.decoder(
            inputs_embeds=query_features,
            multi_stage_positional_embeddings=multi_stage_positional_embeddings,
            anchor_multi_stage_positional_embeddings=anchor_multi_stage_positional_embeddings,
            pixel_embeddings=mask_features,
            anchor_pixel_embeddings=anchor_mask_features,
            encoder_hidden_states=multi_stage_features,
            anchor_encoder_hidden_states=anchor_multi_stage_features,
            query_position_embeddings=query_embeddings,
            feature_size_list=size_list,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
            audio_boost_mask=audio_boost_mask,
            anchor_audio_boost_mask=anchor_audio_boost_mask,
            pre_intermediate_mask_predictions=pre_intermediate_mask_predictions,
            pre_intermediate=pre_intermediate,
            aux_audio_boost_mask=aux_audio_boost_mask,
            # pre_anchor_intermediate_mask_predictions=anchor_pre_intermediate_mask_predictions,
            # pre_anchor_intermediate=anchor_pre_intermediate,
            # aux_multi_stage_features=aux_multi_stage_features,
            # aux_multi_stage_positional_embeddings=aux_multi_stage_positional_embeddings,
            # aux_mask_features=aux_mask_features,
            # pre_inputs_embeds=pre_query_features,
            # pre_all_hidden_states=pre_all_hidden_states,

        )
        # print(anchor_L_audio)
        loss_audio_text = loss_conf + loss_mmd + multiSales_diceloss# + final_loss#+ loss_feat_dist + loss_tv

        return decoder_output,loss_audio_text


MASK2FORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Mask2FormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MASK2FORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoFeatureExtractor`]. See
            [`AutoFeatureExtractor.__call__`] for details.
        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).

            [What are attention masks?](../glossary#attention-mask)
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of Detr's decoder attention layers.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~Mask2FormerModelOutput`] instead of a plain tuple.
"""


class Mask2FormerPreTrainedModel(PreTrainedModel):
    config_class = Mask2FormerConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"

    def _init_weights(self, module: nn.Module):
        xavier_std = self.config.init_xavier_std
        std = self.config.init_std

        if isinstance(module, Mask2FormerTransformerModule):
            if module.input_projections is not None:
                for input_projection in module.input_projections:
                    if not isinstance(input_projection, nn.Sequential):
                        nn.init.xavier_uniform_(input_projection.weight, gain=xavier_std)
                        nn.init.constant_(input_projection.bias, 0)

        elif isinstance(module, Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention):
            nn.init.constant_(module.sampling_offsets.weight.data, 0.0)
            thetas = torch.arange(module.n_heads, dtype=torch.float32) * (2.0 * math.pi / module.n_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(module.n_heads, 1, 1, 2)
                .repeat(1, module.n_levels, module.n_points, 1)
            )
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1
            with torch.no_grad():
                module.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

            nn.init.constant_(module.attention_weights.weight.data, 0.0)
            nn.init.constant_(module.attention_weights.bias.data, 0.0)
            nn.init.xavier_uniform_(module.value_proj.weight.data)
            nn.init.constant_(module.value_proj.bias.data, 0.0)
            nn.init.xavier_uniform_(module.output_proj.weight.data)
            nn.init.constant_(module.output_proj.bias.data, 0.0)

        elif isinstance(module, Mask2FormerMaskedAttentionDecoderLayer):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=xavier_std)

        elif isinstance(module, Mask2FormerPixelLevelModule):
            for submodule in module.modules():
                if isinstance(submodule, (nn.Conv2d, nn.Linear)):
                    submodule.weight.data.normal_(mean=0.0, std=std)
                    if submodule.bias is not None:
                        submodule.bias.data.zero_()

        elif isinstance(module, Mask2FormerPixelDecoder):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.normal_(module.level_embed, std=0)

        elif isinstance(module, Mask2FormerPixelDecoderEncoderOnly):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        if hasattr(module, "reference_points"):
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            nn.init.constant_(module.reference_points.bias.data, 0.0)


@add_start_docstrings(
    "The bare Mask2Former Model outputting raw hidden-states without any specific head on top.",
    MASK2FORMER_START_DOCSTRING,
)
class Mask2FormerModel(Mask2FormerPreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)
        # print(type(config))
        # <class 'transformers.models.mask2former.configuration_mask2former.Mask2FormerConfig'>
        self.pixel_level_module = Mask2FormerPixelLevelModule(config)
        self.transformer_module = Mask2FormerTransformerModule(in_features=config.feature_size, config=config)
        
        self.post_init()
        # self.maskparam = nn.Parameter(torch.tensor(1.0))

    @add_start_docstrings_to_model_forward(MASK2FORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Mask2FormerModelOutput, config_class=_CONFIG_FOR_DOC)
    
    def forward(
        self,
        pixel_values: Tensor,
        pixel_mask: Optional[Tensor] = None,
        prompt_features_projected: Optional[Tensor] = None,
        prompt_features_ori: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        visualClip: Optional[bool]=None,
        audio_boost_mask: Optional[Tensor] = None,
        anchor_audio_boost_mask: Optional[Tensor] = None,
        CLIP_intersection: Optional[Tensor] = None,
        aux_pixel_values_audio: Optional[Tensor] = None,
        aux_pixel_values_bg: Optional[Tensor] = None,
        anchor_aux_pixel_values_audio: Optional[Tensor] = None,
        anchor_aux_pixel_values_bg: Optional[Tensor] = None,
        iter: Optional[int]=None,
        aux_audio_boost_mask: Optional[Tensor] = None,
        batch_id: Optional[str]=None,
    ) -> Mask2FormerModelOutput:
        r"""
        Returns:
            `Mask2FormerModelOutput`

        Examples:
        ```python
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoImageProcessor, Mask2FormerModel

        >>> # load image
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> # load image preprocessor and Mask2FormerModel trained on COCO instance segmentation dataset
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")
        >>> model = Mask2FormerModel.from_pretrained("facebook/mask2former-swin-small-coco-instance")
        >>> inputs = image_processor(image, return_tensors="pt")

        >>> # forward pass
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # model outputs last hidden states of shape (batch_size, num_queries, hidden_size)
        >>> print(outputs.transformer_decoder_last_hidden_state.shape)
        torch.Size([1, 100, 256])
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print('pix:', pixel_values.shape)  # [1, 3, 384, 384]
        batch_size, _, height, width = pixel_values.shape

        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=pixel_values.device)

        pixel_level_module_output = self.pixel_level_module(
            pixel_values=pixel_values, audio_boost_mask=audio_boost_mask, output_hidden_states=output_hidden_states
        )
        # pixel_level_module_output_audio = self.pixel_level_module(
        #     pixel_values=aux_pixel_values_audio, audio_boost_mask=audio_boost_mask, output_hidden_states=output_hidden_states
        # )
        # pixel_level_module_output_bg = self.pixel_level_module(
        #     pixel_values=aux_pixel_values_bg, audio_boost_mask=audio_boost_mask, output_hidden_states=output_hidden_states
        # )
        # anchor_pixel_level_module_output_audio = self.pixel_level_module(
        #     pixel_values=anchor_aux_pixel_values_audio, audio_boost_mask=audio_boost_mask, output_hidden_states=output_hidden_states
        # )
        # anchor_pixel_level_module_output_bg = self.pixel_level_module(
        #     pixel_values=anchor_aux_pixel_values_bg, audio_boost_mask=audio_boost_mask, output_hidden_states=output_hidden_states
        # )
        # print(len(pixel_level_module_output.encoder_hidden_states))
        # print(len(pixel_level_module_output.decoder_hidden_states))
        # anchor_pixel_level_module_output_encoder, anchor_pixel_level_module_output_decoder = [], []
        # pixel_level_module_output.encoder_hidden_states,  pixel_level_module_output_audio.encoder_hidden_states, pixel_level_module_output_bg.encoder_hidden_states, anchor_pixel_level_module_output_audio.encoder_hidden_states, anchor_pixel_level_module_output_bg.encoder_hidden_states = list(pixel_level_module_output.encoder_hidden_states), list(pixel_level_module_output_audio.encoder_hidden_states), list(pixel_level_module_output_bg.encoder_hidden_states), list(anchor_pixel_level_module_output_audio.encoder_hidden_states), list(anchor_pixel_level_module_output_bg.encoder_hidden_states)

        # pixel_level_module_output.decoder_hidden_states,  pixel_level_module_output_audio.decoder_hidden_states, pixel_level_module_output_bg.decoder_hidden_states, anchor_pixel_level_module_output_audio.decoder_hidden_states, anchor_pixel_level_module_output_bg.decoder_hidden_states = list(pixel_level_module_output.decoder_hidden_states), list(pixel_level_module_output_audio.decoder_hidden_states), list(pixel_level_module_output_bg.decoder_hidden_states), list(anchor_pixel_level_module_output_audio.decoder_hidden_states), list(anchor_pixel_level_module_output_bg.decoder_hidden_states)

        encoder_L = 0
        decoder_L = 0
        # for i in range(4):
        #     pixel_level_module_output.encoder_hidden_states[i] = pixel_level_module_output.encoder_hidden_states[i] - pixel_level_module_output_bg.encoder_hidden_states[i] + pixel_level_module_output_audio.encoder_hidden_states[i]
        #     anchor_pixel_level_module_output_encoder_i = pixel_level_module_output.encoder_hidden_states[i] - anchor_pixel_level_module_output_bg.encoder_hidden_states[i] + anchor_pixel_level_module_output_audio.encoder_hidden_states[i]
        #     encoder_L += F.mse_loss(pixel_level_module_output.encoder_hidden_states[i], anchor_pixel_level_module_output_encoder_i)
        # for i in range(3):
        #     pixel_level_module_output.decoder_hidden_states[i] = pixel_level_module_output.decoder_hidden_states[i] - pixel_level_module_output_bg.decoder_hidden_states[i] + pixel_level_module_output_audio.decoder_hidden_states[i]
        #     anchor_pixel_level_module_output_decoder_i = pixel_level_module_output.decoder_hidden_states[i] - anchor_pixel_level_module_output_bg.decoder_hidden_states[i] + anchor_pixel_level_module_output_audio.decoder_hidden_states[i]
        #     decoder_L += F.mse_loss(pixel_level_module_output.decoder_hidden_states[i], anchor_pixel_level_module_output_decoder_i)

        
            
        # pixel_level_module_output.decoder_last_hidden_state = pixel_level_module_output.decoder_last_hidden_state - pixel_level_module_output_bg.decoder_last_hidden_state + pixel_level_module_output_audio.decoder_last_hidden_state
        # pixel_level_module_output.encoder_hidden_states, pixel_level_module_output.decoder_hidden_states = tuple(pixel_level_module_output.encoder_hidden_states), tuple(pixel_level_module_output.decoder_hidden_states)
        # aux_pixel_level_module_output = self.pixel_level_module(
        #     pixel_values=aux_pixel_values, output_hidden_states=output_hidden_states
        # )

        


        ###########
        # VIS_FEAT = list(pixel_level_module_output.decoder_hidden_states)
        # VIS_FEAT.append(pixel_level_module_output.decoder_last_hidden_state)
        
        # visualfeat = self.dpcknn(pixel_level_module_output.decoder_hidden_states,prompt_features_projected)
        # hidden_states_list = list(pixel_level_module_output.decoder_hidden_states)
        # hidden_states_list[0] = visualfeat[0] 
        # hidden_states_list[1] = visualfeat[1] 
        # hidden_states_list[2] = visualfeat[2]
        # dpcknn_scale_features = hidden_states_list
        
        # pixel_level_module_output.decoder_last_hidden_state = visualfeat[3]
        # print(pixel_level_module_output.encoder_hidden_states[0].shape) torch.Size([5, 128, 96, 96])
        # print(pixel_level_module_output.encoder_hidden_states[1].shape) torch.Size([5, 256, 48, 48])
        # print(pixel_level_module_output.encoder_hidden_states[2].shape) torch.Size([5, 512, 24, 24])
        ###########
        # pixel_level_module_output.decoder_last_hidden_state = self.dpcknn(pixel_level_module_output.decoder_last_hidden_state,prompt_features_projected)
        mask_feat_L = 0
        pre_decoder_L = 0
        
        # pixel_level_module_output.decoder_hidden_states = list(pixel_level_module_output.decoder_hidden_states)
        ############改！！！！！！！！
        # if audio_boost_mask is not None:
        #     mask_features = pixel_level_module_output.decoder_last_hidden_state + torch.nn.functional.interpolate(
        #         audio_boost_mask.unsqueeze(1), size=pixel_level_module_output.decoder_last_hidden_state.shape[-2:],
        #         mode="bilinear")
        #     if aux_audio_boost_mask is not None:
        #         mask_features = mask_features + torch.nn.functional.interpolate(
        #             aux_audio_boost_mask.unsqueeze(1), size=pixel_level_module_output.decoder_last_hidden_state.shape[-2:],
        #             mode="bilinear")
        #     if anchor_audio_boost_mask is not None:
        #         anchor_mask_features = pixel_level_module_output.decoder_last_hidden_state + torch.nn.functional.interpolate(
        #             anchor_audio_boost_mask.unsqueeze(1), size=pixel_level_module_output.decoder_last_hidden_state.shape[-2:],
        #             mode="bilinear")
        #         mask_feat_L = F.mse_loss(mask_features, anchor_mask_features)
        
        #     else:
        #         anchor_mask_features = None
        ############改！！！！！！！！
        if audio_boost_mask is not None:
            # if aux_audio_boost_mask is not None:
            #     mask_features_audio = torch.nn.functional.interpolate(audio_boost_mask.unsqueeze(1), size=pixel_level_module_output.decoder_last_hidden_state.shape[-2:],
            #     mode="bilinear")
            #     mask_features_aux_audio = torch.nn.functional.interpolate(aux_audio_boost_mask.unsqueeze(1), size=pixel_level_module_output.decoder_last_hidden_state.shape[-2:],
            #         mode="bilinear")
            #     mask_features = self.maskparam * mask_features_audio + (1-self.maskparam) * mask_features_aux_audio

            #     mask_features = pixel_level_module_output.decoder_last_hidden_state * (1 + mask_features)
            # else:
            mask_features = pixel_level_module_output.decoder_last_hidden_state + torch.nn.functional.interpolate(audio_boost_mask.unsqueeze(1), size=pixel_level_module_output.decoder_last_hidden_state.shape[-2:],
                mode="bilinear")
                
            if anchor_audio_boost_mask is not None:
                anchor_mask_features = pixel_level_module_output.decoder_last_hidden_state + torch.nn.functional.interpolate(
                    anchor_audio_boost_mask.unsqueeze(1), size=pixel_level_module_output.decoder_last_hidden_state.shape[-2:],
                    mode="bilinear")
                mask_feat_L = F.mse_loss(mask_features, anchor_mask_features)
        
            else:
                anchor_mask_features = None


        else:
            mask_features = pixel_level_module_output.decoder_last_hidden_state
            anchor_mask_features = None

        # pixel_level_module_output.decoder_hidden_states = tuple(pixel_level_module_output.decoder_hidden_states)
        # print(mask_features.shape) torch.Size([5, 256, 96, 96])    
            
        # print(pixel_level_module_output.decoder_hidden_states[0].shape)
        
        #有编码器的最后一层特征图，解码器的最后一层特征图（B*256*96*96），�?output_hidden_states为True，还会返回编码器和解码器的�?�尺度特征图，前两项其实就是后两项的[-1]
        transformer_module_output,loss_audio_text = self.transformer_module(
            prompt_features_projected=prompt_features_projected,
            multi_scale_features=pixel_level_module_output.decoder_hidden_states,#解码器的多尺度特征图
            mask_features=mask_features,#pixel_level_module_output.decoder_last_hidden_state,
            anchor_mask_features=anchor_mask_features,
            output_hidden_states=True,
            output_attentions=output_attentions,
            visualClip=visualClip,
            audio_boost_mask=audio_boost_mask,
            anchor_audio_boost_mask = anchor_audio_boost_mask,
            CLIP_intersection=CLIP_intersection,
            dpcknn_scale_features=None,
            anchor_pixel_level_module_output_decoder=None,
            # aux_multi_scale_features = aux_pixel_level_module_output.decoder_hidden_states,
            # aux_mask_features=aux_mask_features,
            iter = iter,
            aux_audio_boost_mask=aux_audio_boost_mask,
            batch_id=batch_id
        )

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None
        transformer_decoder_intermediate_states = None

        if output_hidden_states:
            encoder_hidden_states = pixel_level_module_output.encoder_hidden_states
            pixel_decoder_hidden_states = pixel_level_module_output.decoder_hidden_states
            transformer_decoder_hidden_states = transformer_module_output.hidden_states
            transformer_decoder_intermediate_states = transformer_module_output.intermediate_hidden_states

        output = Mask2FormerModelOutput(
            encoder_last_hidden_state=pixel_level_module_output.encoder_last_hidden_state,
            pixel_decoder_last_hidden_state=pixel_level_module_output.decoder_last_hidden_state,
            transformer_decoder_last_hidden_state=transformer_module_output.last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            transformer_decoder_intermediate_states=transformer_decoder_intermediate_states,
            attentions=transformer_module_output.attentions,
            masks_queries_logits=transformer_module_output.masks_queries_logits,
            pixel_value_out=pixel_level_module_output,
        )
        # print(transformer_module_output.masks_queries_logits[0].shape) torch.Size([5, 100, 96, 96])

        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)
        
        return output,mask_feat_L+encoder_L+decoder_L+loss_audio_text


@add_start_docstrings(
    "The Mask2Former Model with heads on top for instance/semantic/panoptic segmentation.",
    MASK2FORMER_START_DOCSTRING,
)
class Mask2FormerForUniversalSegmentation(Mask2FormerPreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)
        self.model = Mask2FormerModel(config)

        self.weight_dict: Dict[str, float] = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }
        # print(config.num_labels)
        self.class_predictor = nn.Linear(config.hidden_dim, config.num_labels + 1)

        self.criterion = Mask2FormerLoss(config=config, weight_dict=self.weight_dict)
        self.post_init()
        
    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        auxiliary_predictions: Dict[str, Tensor],
        prompt_features_projected: Tensor = None, 
    ) -> Dict[str, Tensor]:
        loss_dict: Dict[str, Tensor] = self.criterion(
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            mask_labels=mask_labels,
            class_labels=class_labels,
            auxiliary_predictions=auxiliary_predictions,
        )

        # weight each loss by `self.weight_dict[<LOSS_NAME>]` including auxiliary losses
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight

        return loss_dict

    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        return sum(loss_dict.values())

    def get_auxiliary_logits(self, classes: torch.Tensor, output_masks: torch.Tensor):
        auxiliary_logits: List[Dict(str, Tensor)] = []

        for aux_binary_masks, aux_classes in zip(output_masks[:-1], classes[:-1]):
            # print(aux_classes.shape)
            auxiliary_logits.append({"masks_queries_logits": aux_binary_masks, "class_queries_logits": aux_classes})

        return auxiliary_logits

    @add_start_docstrings_to_model_forward(MASK2FORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Mask2FormerForUniversalSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Tensor,
        prompt_features_projected: Optional[Tensor] = None,
        prompt_features_ori: Optional[Tensor] = None,
        mask_labels: Optional[List[Tensor]] = None,
        class_labels: Optional[List[Tensor]] = None,
        pixel_mask: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_auxiliary_logits: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        visualClip: Optional[bool]=None,
        audio_boost_mask: Optional[Tensor] = None,
        anchor_audio_boost_mask: Optional[Tensor] = None,
        label_post_mask: Optional[Tensor] = None,
        CLIP_intersection: Optional[Tensor] = None,
        aux_pixel_values_audio: Optional[Tensor] = None,
        aux_pixel_values_bg: Optional[Tensor] = None,
        anchor_aux_pixel_values_audio: Optional[Tensor] = None,
        anchor_aux_pixel_values_bg: Optional[Tensor] = None,
        iter: Optional[int]=None,
        aux_audio_boost_mask: Optional[Tensor] = None,
        batch_id: Optional[str]=None,
    ) -> Mask2FormerForUniversalSegmentationOutput:
        r"""
        mask_labels (`List[torch.Tensor]`, *optional*):
            List of mask labels of shape `(num_labels, height, width)` to be fed to a model
        class_labels (`List[torch.LongTensor]`, *optional*):
            list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the
            labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.

        Returns:
            `Mask2FormerUniversalSegmentationOutput`

        Examples:

        Instance segmentation example:

        ```python
        >>> from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        >>> from PIL import Image
        >>> import requests
        >>> import torch

        >>> # Load Mask2Former trained on COCO instance segmentation dataset
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")
        >>> model = Mask2FormerForUniversalSegmentation.from_pretrained(
        ...     "facebook/mask2former-swin-small-coco-instance"
        ... )

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = image_processor(image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # Model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # Perform post-processing to get instance segmentation map
        >>> pred_instance_map = image_processor.post_process_semantic_segmentation(
        ...     outputs, target_sizes=[image.size[::-1]]
        ... )[0]
        >>> print(pred_instance_map.shape)
        torch.Size([480, 640])
        ```

        Semantic segmentation example:
        ```python
        >>> from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        >>> from PIL import Image
        >>> import requests
        >>> import torch

        >>> # Load Mask2Former trained on ADE20k semantic segmentation dataset
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
        >>> model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic")

        >>> url = (
        ...     "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
        ... )
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = image_processor(image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # Model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # Perform post-processing to get semantic segmentation map
        >>> pred_semantic_map = image_processor.post_process_semantic_segmentation(
        ...     outputs, target_sizes=[image.size[::-1]]
        ... )[0]
        >>> 
        (pred_semantic_map.shape)
        torch.Size([512, 683])
        ```

        Panoptic segmentation example:

        ```python
        >>> from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        >>> from PIL import Image
        >>> import requests
        >>> import torch

        >>> # Load Mask2Former trained on CityScapes panoptic segmentation dataset
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-panoptic")
        >>> model = Mask2FormerForUniversalSegmentation.from_pretrained(
        ...     "facebook/mask2former-swin-small-cityscapes-panoptic"
        ... )

        >>> url = "https://cdn-media.huggingface.co/Inference-API/Sample-results-on-the-Cityscapes-dataset-The-above-images-show-how-our-method-can-handle.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = image_processor(image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # Model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # Perform post-processing to get panoptic segmentation map
        >>> pred_panoptic_map = image_processor.post_process_panoptic_segmentation(
        ...     outputs, target_sizes=[image.size[::-1]]
        ... )[0]["segmentation"]
        >>> print(pred_panoptic_map.shape)
        torch.Size([338, 676])
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs,loss_audio_text = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            prompt_features_projected=prompt_features_projected,
            prompt_features_ori=prompt_features_ori,
            output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss,
            output_attentions=output_attentions,
            return_dict=True,
            visualClip=visualClip,
            audio_boost_mask=audio_boost_mask,
            anchor_audio_boost_mask=anchor_audio_boost_mask,
            CLIP_intersection=CLIP_intersection,
            aux_pixel_values_audio=aux_pixel_values_audio,
            aux_pixel_values_bg=aux_pixel_values_bg,
            anchor_aux_pixel_values_audio=anchor_aux_pixel_values_audio,
            anchor_aux_pixel_values_bg=anchor_aux_pixel_values_bg,
            iter=iter,
            aux_audio_boost_mask=aux_audio_boost_mask,
            batch_id=batch_id
        )
        loss, loss_dict, auxiliary_logits = None, None, None
        class_queries_logits = ()

        for decoder_output in outputs.transformer_decoder_intermediate_states:
            class_prediction = self.class_predictor(decoder_output.transpose(0, 1))
            class_queries_logits += (class_prediction,)

        masks_queries_logits = outputs.masks_queries_logits

        auxiliary_logits = self.get_auxiliary_logits(class_queries_logits, masks_queries_logits)

        # print(loss_audio_text)
        # intersection = []
        # for i in range(5):
        #     # intersection.append(pixel_mask[i].to(torch.int) & audio_boost_mask[i].to(torch.int))

        #     intersection.append(torch.where((mask_labels[i].to(torch.int) & audio_boost_mask[i:i+1,:,:].to(torch.int)).bool(), mask_labels[i].to(torch.int) & audio_boost_mask[i:i+1,:,:].to(torch.int), mask_labels[i].to(torch.int) & audio_boost_mask[i:i+1,:,:].to(torch.int)).float())
        # # mask_labels = mask_labels & audio_boost_mask
        
        
        # for i in range(5):
        #     for j in range(len(mask_labels[i])):
        #         print(mask_labels[i][j].sum())
        #这里为什么会加起来是384*384 1 0 都有
        # 因为有background类的存在 所以1 0 都有
        
        # print(masks_queries_logits[-1].shape)
        
        # print(extraloss['loss_mask'])
        # print(len(mask_labels))
        # print(extraloss)

        # print(mask_labels[0][0].sum()) 104149 background
        # print(mask_labels[0][1].sum()) 43307 object
        # print(class_labels[0][0]) 0
        # print(class_labels[0][1]) 1
        # print(masks_queries_logits[-1].shape) torch.Size([5, 100, 96, 96])
        # print(class_queries_logits[-1].shape) torch.Size([5, 100, 72])
        if mask_labels is not None and class_labels is not None:
            # print(class_queries_logits[-1].shape,len(mask_labels),mask_labels[0].shape,class_labels)
            loss_dict = self.get_loss_dict(
                masks_queries_logits=masks_queries_logits[-1],
                class_queries_logits=class_queries_logits[-1],
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=auxiliary_logits,
            )
            # print(loss_dict.keys())
        # if label_post_mask is not None:
        #     extraloss_dict = self.get_loss_dict(
        #         masks_queries_logits=masks_queries_logits[-1],
        #         class_queries_logits=class_queries_logits[-1],
        #         mask_labels=label_post_mask,
        #         class_labels=class_labels,
        #         auxiliary_predictions=auxiliary_logits
        #     )
        #     # print(extraloss_dict.keys())
        #     # loss_dict['loss_mask'] += extraloss_dict['loss_mask']
        #     loss = self.get_loss(loss_dict)  + self.get_loss(extraloss_dict)
        # else:
        loss = self.get_loss(loss_dict)
    
            # loss_dict['loss_mask'] += extraloss_dict['loss_mask']
        loss = loss
        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None
        
        # print(outputs.pixel_decoder_hidden_states[-1].shape)
        # input('hidden')

            

        
        if output_hidden_states:
            encoder_hidden_states = outputs.encoder_hidden_states
            pixel_decoder_hidden_states = outputs.pixel_decoder_hidden_states
            transformer_decoder_hidden_states = outputs.transformer_decoder_hidden_states

        output_auxiliary_logits = (
            self.config.output_auxiliary_logits if output_auxiliary_logits is None else output_auxiliary_logits
        )
        if not output_auxiliary_logits:
            auxiliary_logits = None

        output = Mask2FormerForUniversalSegmentationOutput(
            loss=loss,
            class_queries_logits=class_queries_logits[-1],
            masks_queries_logits=masks_queries_logits[-1],
            auxiliary_logits=auxiliary_logits,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            pixel_decoder_last_hidden_state=outputs.pixel_decoder_last_hidden_state,
            transformer_decoder_last_hidden_state=outputs.transformer_decoder_last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            attentions=outputs.attentions,
            pixel_value_out=outputs.pixel_value_out
        )

        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)
            if loss is not None:
                output = ((loss)) + output
        return output,loss_audio_text