""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in:
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929
`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270
The official jax code is released and available at https://github.com/google-research/vision_transformer
DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2021 Ross Wightman

Modified by Botao Ye
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.mplt_track.base_backbone import BaseBackbone
from lib.models.mplt_track.utils import combine_tokens, recover_tokens
from lib.models.mplt_track.utils import token2feature, feature2token



class Fovea(nn.Module):

    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        '''
            x: [batch_size, features, k]
        '''
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h*w)

        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        output = mask * x
        output = output.contiguous().view(b, c, h, w)

        return output


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x


class Prompt_block(nn.Module, ):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False):
        super(Prompt_block, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_2 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.fovea = Fovea(smooth=smooth)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C/3), :, :].contiguous()
        x0 = self.conv0_0(x0)
        x1 = x[:, int(C/3):int(2*C/3), :, :].contiguous()
        x1 = self.conv0_1(x1)
        x2 = x[:, int(2*C / 3):, :, :].contiguous()
        x2 = self.conv0_2(x2)
        x0 = self.fovea(x0) + x1 + x2

        return self.conv1x1(x0)

class Prompt_block_init(nn.Module, ):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False):
        super(Prompt_block_init, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.fovea = Fovea(smooth=smooth)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C/2), :, :].contiguous()
        x0 = self.conv0_0(x0)
        x1 = x[:, int(C/2):, :, :].contiguous()
        x1 = self.conv0_1(x1)
        x0 = self.fovea(x0) + x1
        return self.conv1x1(x0)


class SpatialAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        """
        第一层全连接层神经元个数较少，因此需要一个比例系数ratio进行缩放
        """
        super(SpatialAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        """
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False)
        )
        """
        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class TokenAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(TokenAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Prompt_cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7, smooth=True):
        super(Prompt_cbam_block, self).__init__()
        self.spatialattention0 = SpatialAttention(channel, ratio=ratio)
        self.tokenattention0 = TokenAttention(kernel_size=kernel_size)
        self.spatialattention1 = SpatialAttention(channel, ratio=ratio)
        self.tokenattention1 = TokenAttention(kernel_size=kernel_size)
        self.spatialattention2 = SpatialAttention(channel, ratio=ratio)
        self.tokenattention2 = TokenAttention(kernel_size=kernel_size)
        self.fovea = Fovea(smooth=smooth)

    def forward(self, x):
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C / 3), :, :].contiguous()
        x1 = x[:, int(C / 3):int(2*C / 3), :, :].contiguous()
        x2 = x[:, int(2*C / 3):, :, :].contiguous()
        x0 = x0 * self.spatialattention0(x0)
        x0 = x0 * self.tokenattention0(x0)
        x1 = x1 * self.spatialattention1(x1)
        x1 = x1 * self.tokenattention1(x1)
        x2 = x2 * self.spatialattention1(x2)
        x2 = x2 * self.tokenattention2(x2)
        x0 = self.fovea(x0) + x1 + x2
        return x0

class Prompt_cbam_block_init(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7, smooth=True):
        super(Prompt_cbam_block_init, self).__init__()
        self.spatialattention0 = SpatialAttention(channel, ratio=ratio)
        self.tokenattention0 = TokenAttention(kernel_size=kernel_size)
        self.spatialattention1 = SpatialAttention(channel, ratio=ratio)
        self.tokenattention1 = TokenAttention(kernel_size=kernel_size)
        self.fovea = Fovea(smooth=smooth)

    def forward(self, x):
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C / 2), :, :].contiguous()
        x1 = x[:, int(C / 2):, :, :].contiguous()
        x0 = x0 * self.spatialattention0(x0)
        x0 = x0 * self.tokenattention0(x0)
        x1 = x1 * self.spatialattention1(x1)
        x1 = x1 * self.tokenattention1(x1)
        x0 = self.fovea(x0) + x1
        return x0


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            feat, attn = self.attn(self.norm1(x), True)
            x = x + self.drop_path(feat)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class VisionTransformerMPLT(BaseBackbone):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 mplt_loc=None, mplt_drop_path=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches_template = 64
        self.num_patches_search = 256

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        prompt_blocks = []
        prompt_blocks_rev = []
        block_nums = depth
        for i in range(block_nums):
            prompt_blocks.append(Prompt_cbam_block(channel=embed_dim, ratio=8, kernel_size=7, smooth=True))
            prompt_blocks_rev.append(Prompt_cbam_block(channel=embed_dim, ratio=8, kernel_size=7, smooth=True))
        self.prompt_blocks = nn.Sequential(*prompt_blocks)
        self.prompt_blocks_rev = nn.Sequential(*prompt_blocks_rev)
        self.prompt_blocks_init = Prompt_cbam_block_init(channel=embed_dim, ratio=8, kernel_size=7, smooth=True)
        self.prompt_blocks_init_rev = Prompt_cbam_block_init(channel=embed_dim, ratio=8, kernel_size=7, smooth=True)

        prompt_norms = []
        prompt_norms_rev = []
        for i in range(block_nums):
            prompt_norms.append(norm_layer(embed_dim))
            prompt_norms_rev.append(norm_layer(embed_dim))
        self.prompt_norms = nn.Sequential(*prompt_norms)
        self.prompt_norms_rev = nn.Sequential(*prompt_norms_rev)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def forward_features(self, z, x):
        B, H, W = x[0].shape[0], x[0].shape[2], x[0].shape[3]

        x_v = self.patch_embed(x[0])
        z_v = self.patch_embed(z[0])
        x_i = self.patch_embed(x[1])
        z_i = self.patch_embed(z[1])
        z_feat_rgb = token2feature(self.prompt_norms[0](z_v))
        x_feat_rgb = token2feature(self.prompt_norms[0](x_v))
        z_dte_feat = token2feature(self.prompt_norms[0](z_i))
        x_dte_feat = token2feature(self.prompt_norms[0](x_i))
        z_feat = torch.cat([z_feat_rgb, z_dte_feat], dim=1)
        x_feat = torch.cat([x_feat_rgb, x_dte_feat], dim=1)
        z_feat_rev = torch.cat([z_dte_feat, z_feat_rgb], dim=1)
        x_feat_rev = torch.cat([x_dte_feat, x_feat_rgb], dim=1)
        z_feat = self.prompt_blocks_init(z_feat)
        x_feat = self.prompt_blocks_init(x_feat)
        x_feat_rev = self.prompt_blocks_init_rev(x_feat_rev)
        z_feat_rev = self.prompt_blocks_init_rev(z_feat_rev)
        z_tokens = feature2token(z_feat)
        x_tokens = feature2token(x_feat)
        z_tokens_rev = feature2token(z_feat_rev)
        x_tokens_rev = feature2token(x_feat_rev)
        z_prompted, x_prompted = z_tokens, x_tokens
        z_prompted_rev, x_prompted_rev = z_tokens_rev, x_tokens_rev
        z_v = z_v + z_tokens
        x_v = x_v + x_tokens
        z_i = z_i + z_tokens_rev
        x_i = x_i + x_tokens_rev
        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        # Visible and infrared data share the positional encoding and other parameters in ViT
        z_v += self.pos_embed_z
        x_v += self.pos_embed_x

        z_i += self.pos_embed_z
        x_i += self.pos_embed_x


        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x_v = combine_tokens(z_v, x_v, mode=self.cat_mode)
        x_i = combine_tokens(z_i, x_i, mode=self.cat_mode)

        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x_v = self.pos_drop(x_v)
        x_i = self.pos_drop(x_i)


        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        # lens_z = 64
        # lens_x = 256
        
        mplt_index = 0
        for i, blk in enumerate(self.blocks):
            if i >= 1:
                # prompt
                x_ori = x_v
                x_ori_rev = x_i
                x_v = self.prompt_norms[i - 1](x_v)  # todo
                x_i = self.prompt_norms_rev[i - 1](x_i)  # todo
                z_tokens = x_v[:, :lens_z, :]
                x_tokens = x_v[:, lens_z:, :]
                z_tokens_rev = x_i[:, :lens_z, :]
                x_tokens_rev = x_i[:, lens_z:, :]
                z_feat_rgb = token2feature(z_tokens)
                x_feat_rgb = token2feature(x_tokens)
                z_feat_rev = token2feature(z_tokens_rev)
                x_feat_rev = token2feature(x_tokens_rev)

                z_prompted = self.prompt_norms[i](z_prompted)
                x_prompted = self.prompt_norms[i](x_prompted)
                z_prompted_rev = self.prompt_norms_rev[i](z_prompted_rev)
                x_prompted_rev = self.prompt_norms_rev[i](x_prompted_rev)
                z_prompt_feat = token2feature(z_prompted)
                x_prompt_feat = token2feature(x_prompted)
                z_prompt_feat_rev = token2feature(z_prompted_rev)
                x_prompt_feat_rev = token2feature(x_prompted_rev)

                z_feat = torch.cat([z_feat_rgb, z_prompt_feat, z_feat_rev], dim=1)
                x_feat = torch.cat([x_feat_rgb, x_prompt_feat, x_feat_rev], dim=1)
                z_feat_rev = torch.cat([z_feat_rev, z_prompt_feat_rev, z_feat_rgb], dim=1)
                x_feat_rev = torch.cat([x_feat_rev, x_prompt_feat_rev, x_feat_rgb], dim=1)
                z_feat = self.prompt_blocks[i](z_feat)
                x_feat = self.prompt_blocks[i](x_feat)
                z_feat_rev = self.prompt_blocks_rev[i](z_feat_rev)
                x_feat_rev = self.prompt_blocks_rev[i](x_feat_rev)

                z_v = feature2token(z_feat)
                x_v = feature2token(x_feat)
                z_i = feature2token(z_feat_rev)
                x_i = feature2token(x_feat_rev)
                z_prompted, x_prompted = z_v, x_v
                z_prompted_rev, x_prompted_rev = z_i, x_i
                x_v = combine_tokens(z_v, x_v, mode=self.cat_mode)
                x_i = combine_tokens(z_i, x_i, mode=self.cat_mode)
                x_v = x_ori + x_v
                x_i = x_ori_rev + x_i
            x_v = blk(x_v)
            x_i = blk(x_i)

        x_v = recover_tokens(x_v, lens_z, lens_x, mode=self.cat_mode)
        x_i = recover_tokens(x_i, lens_z, lens_x, mode=self.cat_mode)
        x = torch.cat([x_v, x_i], dim=1)
        
        aux_dict = {"attn": None}
        return self.norm(x), aux_dict

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model: VisionTransformerMPLT, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    print('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = VisionTransformerMPLT(**kwargs)
    # tensor = ([torch.rand(1, 3, 128, 128),torch.rand(1, 3, 128, 128)],
    #           [torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256)])
    # flops = FlopCountAnalysis(model, tensor)
    # print("FLOPs: ", flops.total())
    #
    # # 分析parameters
    # print(parameter_count_table(model))
    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_patch16_224_mplt(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) with PointFlow between RGB and T search regions.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


def vit_small_patch16_224_mplt(pretrained=False, **kwargs):
    """
    ViT-Small model (ViT-S/16) with PointFlow between RGB and T search regions.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


def vit_tiny_patch16_224_mplt(pretrained=False, **kwargs):
    """
    ViT-Tiny model (ViT-S/16) with PointFlow between RGB and T search regions.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model
