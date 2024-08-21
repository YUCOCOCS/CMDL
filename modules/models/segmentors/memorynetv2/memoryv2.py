'''
Function:
    Implementation of FeaturesMemoryV2
Author:
    Zhenchao Jin
'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from ..base import SelfAttentionBlock
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


'''FeaturesMemoryV2'''
class FeaturesMemoryV2(nn.Module):
    def __init__(self, num_classes, feats_channels, transform_channels, out_channels, use_hard_aggregate=False, 
                 downsample_before_sa=False, norm_cfg=None, act_cfg=None, align_corners=False):
        super(FeaturesMemoryV2, self).__init__()
        # set attributes
        self.align_corners = align_corners
        self.num_classes = num_classes
        self.feats_channels = feats_channels # 512维度
        self.transform_channels = transform_channels # 256维度
        self.out_channels = out_channels # 512维度
        self.use_hard_aggregate = use_hard_aggregate
        if downsample_before_sa: # False
            self.downsample_before_sa = nn.Sequential(
                nn.Conv2d(feats_channels, feats_channels, kernel_size=3, stride=2, padding=1, bias=False),
                BuildNormalization(constructnormcfg(placeholder=feats_channels, norm_cfg=norm_cfg)),
                BuildActivation(act_cfg),
            )

        # init memory
        self.memory = nn.Parameter(torch.cat([
            torch.zeros(num_classes, 1, dtype=torch.float), torch.ones(num_classes, 1, dtype=torch.float),
        ], dim=1), requires_grad=False) #一个保存均值，一个保存方差

        # define self_attention module
        self.self_attention = SelfAttentionBlock(
            key_in_channels=feats_channels,
            query_in_channels=feats_channels,
            transform_channels=transform_channels,
            out_channels=feats_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=2,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out_project=True,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        # bottleneck used to fuse feats and selected_memory
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feats_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=out_channels, norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
    '''forward'''
    def forward(self, feats, preds=None): # 输入的特征是
        batch_size, num_channels, h, w = feats.size()
        # extract the history features
        # --(B, num_classes, H, W) --> (B*H*W, num_classes)
        weight_cls = preds.permute(0, 2, 3, 1).contiguous()
        weight_cls = weight_cls.reshape(-1, self.num_classes)
        weight_cls = F.softmax(weight_cls, dim=-1) #每个类的权重
        if self.use_hard_aggregate:
            labels = weight_cls.argmax(-1).reshape(-1, 1)
            onehot = torch.zeros_like(weight_cls).scatter_(1, labels.long(), 1)
            weight_cls = onehot
        # --(num_classes, C)
        memory_means = self.memory.data[:, 0]
        memory_stds = self.memory.data[:, 1]
        memory = []

        # 根据高斯分布来获得相关特征
        for idx in range(self.num_classes):
            torch.manual_seed(idx)
            cls_memory = torch.normal( #告诉向量的均值和方差,产生对应的特征向量,维度是1* 512维度
                mean=torch.full((1, self.feats_channels), memory_means[idx]), # 给定的均值
                std=torch.full((1, self.feats_channels), memory_stds[idx]) # 给定的方差
            )
            memory.append(cls_memory)


        memory = torch.cat(memory, dim=0).type_as(weight_cls)
        # --(B*H*W, num_classes) * (num_classes, C) --> (B*H*W, C)
        selected_memory = torch.matmul(weight_cls, memory)
        # calculate selected_memory
        # --(B*H*W, C) --> (B, H, W, C)
        selected_memory = selected_memory.view(batch_size, h, w, num_channels)
        # --(B, H, W, C) --> (B, C, H, W)
        selected_memory = selected_memory.permute(0, 3, 1, 2).contiguous()
        # --feed into the self attention module
        if hasattr(self, 'downsample_before_sa'):
            feats_in, selected_memory_in = self.downsample_before_sa(feats), self.downsample_before_sa(selected_memory)
        else:
            feats_in, selected_memory_in = feats, selected_memory #第一个是ResNet产生的最后一维度的特征，第二个是选择的内存库中的特征

        selected_memory = self.self_attention(feats_in, selected_memory_in) #进行一个自注意力机制


        if hasattr(self, 'downsample_before_sa'):
            selected_memory = F.interpolate(selected_memory, size=feats.size()[2:], mode='bilinear', align_corners=self.align_corners)

        # return
        memory_output = self.bottleneck(torch.cat([feats, selected_memory], dim=1)) #最后融合的是基本特征和产生的数据集级别的类别语义特征
        return memory.data, memory_output


        
    '''update'''
    def update(self, features, segmentation, ignore_index=255, momentum_cfg=None, learning_rate=None):
        batch_size, num_channels, h, w = features.size()
        momentum = momentum_cfg['base_momentum']  # 基本设置为0.1
        if momentum_cfg['adjust_by_learning_rate']: # False
            momentum = momentum_cfg['base_momentum'] / momentum_cfg['base_lr'] * learning_rate
        # use features to update memory
        segmentation = segmentation.long()
        features = features.permute(0, 2, 3, 1).contiguous()
        features = features.view(batch_size * h * w, num_channels)
        clsids = segmentation.unique()
        for clsid in clsids:
            if clsid == ignore_index: continue
            # --(B, H, W) --> (B*H*W,)
            seg_cls = segmentation.view(-1)
            # --extract the corresponding feats: (K, C)
            feats_cls = features[seg_cls == clsid]
            # --update memory
            feats_cls = feats_cls.mean(0)
            mean, std = feats_cls.mean(), feats_cls.std()
            self.memory[clsid][0] = (1 - momentum) * self.memory[clsid][0].data + momentum * mean
            self.memory[clsid][1] = (1 - momentum) * self.memory[clsid][1].data + momentum * std
        # syn the memory
        if dist.is_available() and dist.is_initialized():
            memory = self.memory.data.clone()
            dist.all_reduce(memory.div_(dist.get_world_size()))
            self.memory = nn.Parameter(memory, requires_grad=False)