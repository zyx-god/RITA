import timm
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
import math

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
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
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.float()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W



class TransitionGatedFusion(nn.Module):
    def __init__(self, channels):
        super(TransitionGatedFusion, self).__init__()
        # 1x1 conv layers for gating
        self.gate_from_mask = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.gate_from_image = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        # initialize conv layers for stable gating (optional)
        nn.init.kaiming_uniform_(self.gate_from_mask.weight, a=1)
        nn.init.kaiming_uniform_(self.gate_from_image.weight, a=1)
        if self.gate_from_mask.bias is not None:
            nn.init.constant_(self.gate_from_mask.bias, 0.0)
        if self.gate_from_image.bias is not None:
            nn.init.constant_(self.gate_from_image.bias, 0.0)
        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, g):
        # x and g are both tensors of shape [B, C, H, W]
        # Compute gating coefficients
        A = self.sigmoid(self.gate_from_mask(g))  # gate for x, shape [B, C, H, W]
        B = self.sigmoid(self.gate_from_image(x))  # gate for g, shape [B, C, H, W]
        # Apply gating (element-wise multiplication)
        x_mod = x * A
        g_mod = g * B
        # Fuse features by summation (or concatenation + conv if desired)
        out = x_mod + g_mod
        return out


def copy_conv_weights(source, target):
    """
    复制 source 的卷积层权重和偏置到 target 的卷积层中。
    
    参数:
        source (OverlapPatchEmbed): 源实例。
        target (OverlapPatchEmbed): 目标实例。
    """
    # 检查 source 和 target 的卷积层是否兼容
    if (source.proj.weight.shape == target.proj.weight.shape and
        source.proj.bias.shape == target.proj.bias.shape):
        # 复制权重和偏置
        target.proj.weight.data.copy_(source.proj.weight.data)
        target.proj.bias.data.copy_(source.proj.bias.data)
    else:
        raise ValueError("卷积层的权重或偏置形状不匹配，无法复制。")

class MixVisionTransformer(nn.Module):
    def __init__(self,pretrain_path=None, img_size=512, patch_size=4, in_chans=3,embed_dims=[64, 128, 320, 512],num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])

        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
        if pretrain_path is not None:
            print("Load segformer pretrain pth.")
            self.load_state_dict(torch.load(pretrain_path),
                                strict=False)
        
        self.patch_embed5 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=1,
                                              embed_dim=embed_dims[0])
        
        self.patch_embed6 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        copy_conv_weights(self.patch_embed2 ,self.patch_embed6)
        self.patch_embed7 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        copy_conv_weights(self.patch_embed3, self.patch_embed7, )
        self.patch_embed8 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        copy_conv_weights(self.patch_embed4, self.patch_embed8, )
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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

    def forward_features(self, x, iter_embed):
        B = x.shape[0]
        gray_x = x[:,3:4,:,:]
        x = x[:,:3,:,:]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        gray_x, H, W = self.patch_embed5(gray_x)
        gray_x = gray_x + iter_embed
        x = torch.cat([x, gray_x],dim = 1)
        for i, blk in enumerate(self.block1):
            x = blk(x, H * 2, W)
        x = self.norm1(x)
        x = x.reshape(B, 2 * H, W, -1).permute(0, 3, 1, 2).contiguous()
        gray_x = x[:,:,128:,:]
        x = x[:,:,:128,:]
        outs.append(x)
        outs.append(gray_x)


        # stage 2
        x, H, W = self.patch_embed2(x)
        gray_x, H, W = self.patch_embed6(gray_x)
        x = torch.cat([x, gray_x],dim = 1)
        for i, blk in enumerate(self.block2):
            x = blk(x, 2 * H, W)
        x = self.norm2(x)
        x = x.reshape(B, 2 * H, W, -1).permute(0, 3, 1, 2).contiguous()

        gray_x = x[:,:,64:,:]
        x = x[:,:,:64,:]
        
        outs.append(x)
        outs.append(gray_x)


        # stage 3
        x, H, W = self.patch_embed3(x)
        gray_x, H, W = self.patch_embed7(gray_x)
        x = torch.cat([x, gray_x],dim = 1)
        for i, blk in enumerate(self.block3):
            x = blk(x, 2 * H, W)
        x = self.norm3(x)
        x = x.reshape(B, 2 * H, W, -1).permute(0, 3, 1, 2).contiguous()
        gray_x = x[:,:,32:,:]
        x = x[:,:,:32,:]
        outs.append(x)
        outs.append(gray_x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        gray_x, H, W = self.patch_embed8(gray_x)
        x = torch.cat([x, gray_x],dim = 1)
        for i, blk in enumerate(self.block4):
            x = blk(x, 2 * H, W)
        x = self.norm4(x)
        x = x.reshape(B, 2 * H, W, -1).permute(0, 3, 1, 2).contiguous()
        gray_x = x[:,:,16:,:]
        x = x[:,:,:16,:]
        outs.append(x)
        outs.append(gray_x)
        
        return x, gray_x, outs

    def forward(self, x, iter_embed):
        x, gray_x, outs = self.forward_features(x, iter_embed)
        return x, gray_x, outs 


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class UpsampleConcatSegformer(nn.Module):
    def __init__(self):
        super(UpsampleConcatSegformer, self).__init__()
        # 192到96的上采样，单次上采样
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        # 384到96的上采样，两次上采样，逐步降低通道数
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )

        # 768到96的上采样，三次上采样，逐步降低通道数
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(512, 320, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )

        self.upsample4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        # 384到96的上采样，两次上采样，逐步降低通道数
        self.upsample5 = nn.Sequential(
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )

        # 768到96的上采样，三次上采样，逐步降低通道数
        self.upsample6 = nn.Sequential(
            nn.ConvTranspose2d(512, 320, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )
        self.cgf1 = TransitionGatedFusion(64)   # 对应 x1 和 g1
        self.cgf2 = TransitionGatedFusion(128)  # 对应 x2 和 g2
        self.cgf3 = TransitionGatedFusion(320)  # 对应 x3 和 g3
        self.cgf4 = TransitionGatedFusion(512)  # 对应 x4 和 g4


    def forward(self, inputs):
        # 上采样
        x1,g1,x2,g2,x3,g3,x4,g4 = inputs

        fused1 = self.cgf1(x1, g1)
        # 融合特征后再上采样
        fused2 = self.cgf2(x2, g2)
        fused3 = self.cgf3(x3, g3)
        fused4 = self.cgf4(x4, g4)

        up2 = self.upsample1(fused2)
        up3 = self.upsample2(fused3)
        up4 = self.upsample3(fused4)
        gup2 = self.upsample4(g2)
        gup3 = self.upsample5(g3)
        gup4 = self.upsample6(g4)
        x = torch.cat([fused1, up2, up3, up4, g1, gup2, gup3, gup4], dim=1)
        return x


class RITA(nn.Module):
    def __init__(self, num_classes=4):
        super(RITA, self).__init__()
        self.segformer = MixVisionTransformer(pretrain_path=None)
        self.loss_fn = nn.CrossEntropyLoss()
        self.updown = UpsampleConcatSegformer()
        self.conv1 = nn.Conv2d(512, num_classes, 1)
        self.iter_embed = nn.Embedding(256, 64) 
        self.num_classes = num_classes
        self.eos_token_id = self.num_classes - 2
        self.resize = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
    def forward(self, x):
        iter_values = x[:, 4:, 0, 0].squeeze(1).long()
        iter_embed = self.iter_embed(iter_values)  # (B, embed_dim)
        iter_embed = iter_embed.unsqueeze(1)  # (B, embed_dim, 1, 1)
        inputs = x[:,:4,:,:]
        x, gray_x, outs = self.segformer(inputs,iter_embed)
        pred_mask = self.updown(outs)
        pred_mask = self.conv1(pred_mask)
        pred_mask = self.resize(pred_mask)
        return pred_mask