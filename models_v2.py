
import os

import matplotlib.pyplot as plt
import torchvision.models as models
import torch
import torch.nn as nn
from functools import partial
import numpy as np
import cv2
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
__all__ = [
    'deit_tiny_patch16_LS', 'deit_small_patch16_LS', 'deit_medium_patch16_LS',
    'deit_base_patch16_LS', 'deit_large_patch16_LS',
    'deit_huge_patch14_LS', 'deit_huge_patch14_52_LS',
    'deit_huge_patch14_26x2_LS', 'deit_Giant_48x2_patch14_LS',
    'deit_giant_40x2_patch14_LS', 'deit_Giant_48_patch14_LS',
    'deit_giant_40_patch14_LS', 'deit_small_patch16_36_LS',
    'deit_small_patch16_36', 'deit_small_patch16_18x2_LS',
    'deit_small_patch16_18x2', 'deit_base_patch16_18x2_LS',
    'deit_base_patch16_18x2', 'deit_base_patch16_36x1_LS',
    'deit_base_patch16_36x1'
]
class Zoomer(nn.Module):
    def __init__(self):
        super(Zoomer, self).__init__()

        # 加载预训练的ResNet-18模型
        resnet18 = models.resnet18()

        # 移除最后的全连接层
        self.features = nn.Sequential(*list(resnet18.children())[:-5])
        self.pool = nn.AdaptiveAvgPool2d((28, 28))
        # 定义新的全连接层，输出14x14的特征图
        self.new_fc = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1), nn.ReLU(inplace=True))
        self.project = nn.Linear(28*28, 14*14)

    def forward(self, x):
        x = self.features(x)  # 通过ResNet特征提取部分
        x = self.pool(x)  # 输出14x14的特征图
        x = self.new_fc(x)  # 输出14x14的特征图
        x = x.view(x.size(0), -1)
        x = self.project(x)
        x = torch.sigmoid(x)
        return x

class Importance_Score(nn.Module):
    def __init__(self,patch_size):
        super(Importance_Score, self).__init__()
        self.model = Zoomer()
        ckp = torch.load('Zoomer_70.pth')
        self.model.load_state_dict(ckp)
        # set requires_grad to False
        for param in self.model.parameters():
            param.requires_grad = False
        self.patch_size = patch_size
    def forward(self, x):
        x = self.model(x)
        if self.patch_size==16:
            return x
        if self.patch_size==32:
            x = x.reshape((x.shape[0],14,14))
            # downsample to 7x7
            x = F.avg_pool2d(x,2).reshape((x.shape[0],-1))
            return x

class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        with torch.cuda.amp.autocast(enabled=False):
            q, k, v = qkv[0], qkv[1], qkv[2]
        
            q = q * self.scale

            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x 
    
class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class Layer_scale_init_Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_1_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        
    def forward(self, x):
        x = x + self.drop_path(self.gamma_1*self.attn(self.norm1(x))) + self.drop_path(self.gamma_1_1 * self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x))) + self.drop_path(self.gamma_2_1 * self.mlp1(self.norm21(x)))
        return x
        
class Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.mlp1(self.norm21(x)))
        return x
        
        
class hMLP_stem(nn.Module):
    """ hMLP_stem: https://arxiv.org/pdf/2203.09795.pdf
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, embed_dim=768,norm_layer=nn.SyncBatchNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = torch.nn.Sequential(*[nn.Conv2d(in_chans, embed_dim//4, kernel_size=4, stride=4),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim//4, embed_dim, kernel_size=2, stride=2),
                                          norm_layer(embed_dim),
                                         ])
        

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class vit_models(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers = Block,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                 Attention_block = Attention, Mlp_block=Mlp,
                dpr_constant=True,init_scale=1e-4,
                mlp_ratio_clstk = 4.0,use_zoomer=False,thres=0.03,zoom_factor=2,**kwargs):
        super().__init__()
        self.use_zoomer = use_zoomer
        self.dropout_rate = drop_rate
        self.thres = thres
        self.zoom_factor = zoom_factor
            
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        # dq
        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches_c = self.patch_embed.num_patches
        # self.patch_embed_f = Patch_layer(
        #         img_size=img_size, patch_size=int(patch_size/2), in_chans=in_chans, embed_dim=embed_dim)
        # num_patches_f = self.patch_embed_f.num_patches
        num_patches_f = 4 * num_patches_c

        self.pos_embedding_c = self.PE_absolute_sincos_embedding(num_patches_c + 1, embed_dim).cuda()
        self.pos_embedding_f = self.PE_absolute_sincos_embedding(num_patches_f + 1, embed_dim).cuda()

        self.segment_c = (torch.zeros(1, num_patches_c + 1, embed_dim)).long().cuda()
        self.segment_f = (torch.ones(1, num_patches_f + 1, embed_dim)).long().cuda()
        self.split_token = (torch.ones(1, 1) * 0.5).cuda()



        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale)
            for i in range(depth)])
        

        
            
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # trunc_normal_(self.pos_embed, std=.02)# if use dq else trunc_normal_(self.pos_embed_c, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.Importance_Score = Importance_Score(patch_size).cuda().eval()
    def PE_absolute_sincos_embedding(self,n_pos, dim):
        n_pos_vec = torch.arange(n_pos, dtype=torch.float)
        assert dim % 2 == 0, "wrong dim"
        position_embedding = torch.zeros(n_pos_vec.numel(), dim, dtype=torch.float)

        omega = torch.arange(dim // 2, dtype=torch.float)
        omega /= dim / 2.
        omega = 1. / (10000 ** omega)

        sita = n_pos_vec[:, None] @ omega[None, :]
        emb_sin = torch.sin(sita)
        emb_cos = torch.cos(sita)

        position_embedding[:, 0::2] = emb_sin
        position_embedding[:, 1::2] = emb_cos

        return position_embedding
    def Zoom_patch_embed(self, img):
        if self.zoom_factor==2:
            x_c = self.patch_embed(img)  # b, 196, 768
            x_f = self.patch_embed(img.repeat_interleave(2,dim=2).repeat_interleave(2, dim=3)) # b, 784, 768
        elif self.zoom_factor==0.5:
            x_c = self.patch_embed(F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=False)) # b, 49, 768
            x_f = self.patch_embed(img) # b, 196, 768
        else:
            x_c = self.patch_embed(img)  # b, 196, 768
            x_f = self.patch_embed(img.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3))  # b, 784, 768
        b, n_c, _ = x_c.shape
        b, n_f, _ = x_f.shape
        if self.zoom_factor==2:
            entropy_c = self.Importance_Score(img).reshape((b, int(np.sqrt(n_c)), int(np.sqrt(n_c))))
            entropy_f = entropy_c.repeat_interleave(2,dim=1).repeat_interleave(2, dim=2)
            # entropy_f = F.interpolate(entropy_c.unsqueeze(0), scale_factor=2, mode='nearest').squeeze(0)
            entropy_c = entropy_c.reshape((b, -1))
            entropy_f = entropy_f.reshape((b, -1))
        elif self.zoom_factor==0.5:
            entropy_f = self.Importance_Score(img).reshape((b, int(np.sqrt(n_f)), int(np.sqrt(n_f))))
            entropy_c = F.max_pool2d(entropy_f.unsqueeze(0),kernel_size=2,stride=2).squeeze(0)
            entropy_c = entropy_c.reshape((b, -1))
            entropy_f = entropy_f.reshape((b, -1))

        x_c.add_(self.pos_embedding_c.unsqueeze(0).repeat_interleave(b, dim=0)[:, 1:(n_c+1)])
        x_c.add_(self.segment_c[:, :(n_c)])
        x_f.add_(self.pos_embedding_f.unsqueeze(0).repeat_interleave(b, dim=0)[:, 1:(n_f+1)])
        x_f.add_(self.segment_f[:, :(n_f)])

        score_idx_c = torch.argsort(entropy_c.reshape((b, -1)), dim=1,descending=True)
        score_idx_f = torch.argsort(entropy_f.reshape((b, -1)), dim=1,descending=True)
        # sort x_c as score_id
        x_c = x_c[torch.arange(b)[:, None],score_idx_c,:]
        x_f = x_f[torch.arange(b)[:, None],score_idx_f,:]
        # sort entropy as score_id
        entropy_c = entropy_c[torch.arange(b)[:, None], score_idx_c]
        entropy_f = entropy_f[torch.arange(b)[:, None], score_idx_f]
        c_matrix = torch.where(entropy_c < self.thres, 1, torch.zeros_like(entropy_c)).reshape((b, -1))
        f_matrix = torch.where(entropy_f > self.thres, 1, torch.zeros_like(entropy_f)).reshape((b, -1))
        mask_c = torch.cat((c_matrix, self.split_token.repeat(b, 1)), dim=1)
        mask_f = torch.cat((f_matrix, self.split_token.repeat(b, 1)), dim=1)
        pad = torch.zeros((b, 1, x_c.shape[-1])).to(device=img.device)
        x_c = torch.cat((x_c, pad), dim=1)
        x_f = torch.cat((x_f, pad), dim=1)

        idx_c = torch.argsort(mask_c, dim=1,descending=True)
        idx_f = torch.argsort(mask_f, dim=1,descending=True)
        x_c = x_c[torch.arange(b)[:,None],idx_c,:]
        x_f = x_f[torch.arange(b)[:,None],idx_f,:]
        idx_idx_c = mask_c[torch.arange(b)[:,None],idx_c]
        idx_idx_f = mask_f[torch.arange(b)[:,None],idx_f]

        x_c[torch.where(idx_idx_c < 1e-6)] = 0
        x_f[torch.where(idx_idx_f < 1e-6)] = 0
        cut_c = torch.where(idx_c==n_c)
        cut_f = torch.where(idx_f==n_f)
        max_len_c = torch.max(cut_c[1])
        max_len_f = torch.max(cut_f[1])
        x_c = x_c[:,:max_len_c+1,:]
        x_f = x_f[:,:max_len_f+1,:]
        x = torch.cat((x_f, x_c), dim=1)

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):

        if self.use_zoomer:
            x = self.Zoom_patch_embed(x)
        else:
            x = self.patch_embed(x)
        B, n_c, _ = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if not self.use_zoomer:
            x += self.pos_embedding_c.unsqueeze(0).repeat_interleave(B, dim=0)[:, :(n_c + 1)]
        else:
            x[:, 0] += self.pos_embedding_c.unsqueeze(0).repeat_interleave(B, dim=0)[:, 0]

        for i , blk in enumerate(self.blocks):
            x = blk(x)
            
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):

        x = self.forward_features(x)
        
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)
        
        return x


@register_model
def deit_tiny_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,   **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,use_zoomer=True,thres=0.03,zoom_factor=2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    
    return model
    
    
@register_model
def deit_small_patch16_LS(pretrained=False, img_size=384, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,use_zoomer=True,thres=0.01,zoom_factor=2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_small_'+str(img_size)+'_'
        if pretrained_21k:
            name+='21k.pth'
        else:
            name+='1k.pth'
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])

    return model


def create_model(model_name,**kwargs):
    if model_name in __all__:
        create_fn = globals()[model_name]
        model = create_fn(**kwargs)
        model.default_cfg = _cfg()
        return model
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

if __name__ == '__main__':
    img = cv2.imread(r'G:\imagenet\train\n01440764\n01440764_334.JPEG')
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0)
    img = img.repeat(2,1,1,1)
    # im = Importance_Score(16).cuda()
    # im.eval()
    # score = im(img.cuda()).cpu().detach().numpy().reshape((14, 14))
    model = deit_tiny_patch16_LS().cuda()
    # ckp = 'deit_tiny_patch16_224-a1311bcf.pth'
    # pt = torch.load(ckp)['model']
    # model.load_state_dict(pt,strict=False)
    out = model(img.cuda())
    out = torch.softmax(out, dim=-1)
    cls = torch.argmax(out, dim=-1)
    print(cls)
    # print(model)