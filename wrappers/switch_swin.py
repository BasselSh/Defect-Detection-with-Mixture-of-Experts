from mmdet.models.backbones.swin import SwinTransformer, SwinBlockSequence, SwinBlock, build_norm_layer, ShiftWindowMSA
from .switch import SwitchMoE
from mmengine.model import BaseModule, ModuleList
from copy import deepcopy
from mmdet.registry import MODELS
import torch
import torch.nn as nn
from mmdet.models.layers.transformer import PatchMerging

class SwinBlockSequenceMOE(BaseModule):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
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
                    block.ffn = SwitchMoE(embed_dims,embed_dims,embed_dims,num_experts=num_experts)
                else:
                    if i%2==alternate_moe:
                        block.ffn = SwitchMoE(embed_dims,embed_dims,embed_dims,num_experts=num_experts)

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
    A PyTorch implement of : `Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows`  -
        https://arxiv.org/abs/2103.14030

    Inspiration from
    https://github.com/microsoft/Swin-Transformer

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dims (int): The feature dimension. Default: 96.
        patch_size (int | tuple[int]): Patch size. Default: 4.
        window_size (int): Window size. Default: 7.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin
            Transformer stage. Default: (3, 6, 12, 24).
        strides (tuple[int]): The patch merging or patch embedding stride of
            each Swin Transformer stage. (In swin, we set kernel size equal to
            stride.) Default: (4, 2, 2, 2).
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch
            merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: False.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer at
            output of backone. Defaults: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            Default: -1 (-1 means not freezing any parameters).
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
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
                 moe_layer = -1,
                 alternate_moe=-1,
                 num_experts=4,
                 **kwargs):
        super().__init__(**kwargs)

        total_depth = sum(kwargs['depths'])
        dpr = [
            x.item() for x in torch.linspace(0, kwargs['drop_path_rate'], total_depth)
        ]
        stages = nn.ModuleList()
        in_channels = kwargs['embed_dims']
        num_layers = len(kwargs['depths'])
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=kwargs['strides'][i + 1],
                    norm_cfg=kwargs['norm_cfg'] if kwargs['patch_norm'] else None,
                    init_cfg=None)
            else:
                downsample = None
            params = dict(embed_dims=in_channels,
                num_heads=kwargs['num_heads'][i],
                feedforward_channels=kwargs['mlp_ratio'] * in_channels,
                depth=kwargs['depths'][i],
                window_size=kwargs['window_size'],
                qkv_bias=kwargs['qkv_bias'],
                qk_scale=kwargs['qk_scale'],
                drop_rate=kwargs['drop_rate'],
                attn_drop_rate=kwargs['attn_drop_rate'],
                drop_path_rate=dpr[sum(kwargs['depths'][:i]):sum(kwargs['depths'][:i + 1])],
                downsample=downsample,
                act_cfg=kwargs['act_cfg'],
                norm_cfg=kwargs['norm_cfg'],
                with_cp=kwargs['with_cp'],
                init_cfg=None)
            
            if i == (moe_layer-1):
                stage = SwinBlockSequenceMOE(alternate_moe=alternate_moe,
                                             num_experts=num_experts,
                                             **params)
            else:
                stage = SwinBlockSequence(**params)
                
            stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels
