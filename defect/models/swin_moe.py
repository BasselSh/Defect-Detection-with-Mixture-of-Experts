from mmdet.models.backbones.swin import SwinTransformer, SwinBlockSequence, SwinBlock, build_norm_layer, ShiftWindowMSA
from .ffn_moe import SwitchMoEFFN
from mmengine.model import BaseModule, ModuleList
from copy import deepcopy
from mmdet.registry import MODELS
import torch
import torch.nn as nn
from mmdet.models.layers.transformer import PatchMerging


class SwinBlockSequenceMOE(BaseModule):
    """Implements one stage in Swin Transformer.
        alternate_moe (int): Indicator of the sequence of moe block in the moe
            layer. -1 means no moe block will be added.0 means the blocks with
            even number will be moe blocks. 1 is means the blocks with odd
            number will be moe blocks. 2 means all blocks are moe blocks
        Default: -1
        num_experts (int): Number of experts in every moe block.
            Default: 4.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None,
                 alternate_moe=-1,
                 num_experts=4,
                 use_aux_loss=True,
                 ):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            if alternate_moe!=-1:
                if alternate_moe==2:
                    block.ffn = SwitchMoEFFN(embed_dims,embed_dims,num_experts=num_experts, use_aux_loss=use_aux_loss)
                else:
                    if i%2==alternate_moe:
                        block.ffn = SwitchMoEFFN(embed_dims,embed_dims,num_experts=num_experts, use_aux_loss=use_aux_loss)

            self.blocks.append(block)
        self.downsample = downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape

@MODELS.register_module()
class SwinTransformerMOE(SwinTransformer):
    """ Swin Transformer With Mixture-of-Experts
        moe_layer (int): The layer in which the moe blocks will be added.
            Default: -1 (-1 means no moe blocks are added)
        alternate_moe (int): Indicator of the sequence of moe block in the moe
            layer. -1 means no moe block will be added.0 means the blocks with
            even number will be moe blocks. 1 is means the blocks with odd
            number will be moe blocks. 2 means all blocks are moe blocks
        Default: -1
        num_experts (int): Number of experts in every moe block.
            Default: 4.
    """
    
    def __init__(self,
                 moe_stage = -1,
                 alternate_moe=-1,
                 num_experts=4,
                 use_aux_loss=True,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 init_cfg=None):
        super().__init__(pretrain_img_size=pretrain_img_size,
                 in_channels=3,
                 embed_dims=embed_dims,
                 patch_size=patch_size,
                 window_size=window_size,
                 mlp_ratio=mlp_ratio,
                 depths=depths,
                 num_heads=num_heads,
                 strides=strides,
                 out_indices=out_indices,
                 qkv_bias=qkv_bias,
                 qk_scale=qk_scale,
                 patch_norm=patch_norm,
                 drop_rate=drop_rate,
                 attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate,
                 use_abs_pos_embed=use_abs_pos_embed,
                 act_cfg=act_cfg,
                 norm_cfg=norm_cfg,
                 with_cp=with_cp,
                 pretrained=pretrained,
                 convert_weights=convert_weights,
                 frozen_stages=frozen_stages,
                 init_cfg=init_cfg)

        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]
        self.stages = ModuleList()
        in_channels = embed_dims
        num_layers = len(depths)
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None
            params = dict(embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            
            if i == (moe_stage-1):
                stage = SwinBlockSequenceMOE(alternate_moe=alternate_moe,
                                             num_experts=num_experts,
                                             use_aux_loss=use_aux_loss,
                                             **params)
            else:
                stage = SwinBlockSequence(**params)
                
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels
        print(self.stages)
