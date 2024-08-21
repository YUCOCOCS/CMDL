import torch
import torch.nn as nn
from ..base import BaseSegmentor
from mmcv.ops import CrissCrossAttention
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg

class CCNet_unit(nn.Module):
    def __init__(self, in_channels, out_channels,  align_corners=False, norm_cfg=None, act_cfg=None):
        super(CCNet_unit, self).__init__()
        self.in_channels = in_channels
        self.feats_channels = out_channels
        self.align_corners = align_corners
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        ## build criss-cross attention
        self.conv_before_cca = nn.Sequential(
            nn.Conv2d(self.in_channels,self.feats_channels,kernel_size=3,stride=1,padding=1,bias=False),
            BuildNormalization(constructnormcfg(placeholder=self.feats_channels,norm_cfg=self.norm_cfg)),
            BuildActivation(self.act_cfg),
        )

        self.cca = CrissCrossAttention(self.feats_channels)
        self.conv_after_cca = nn.Sequential(
            nn.Conv2d(self.feats_channels,self.feats_channels,kernel_size=3,stride=1,padding=1,bias=False),
            BuildNormalization(constructnormcfg(placeholder=self.feats_channels,norm_cfg=self.norm_cfg)),
            BuildActivation(self.act_cfg),
        )
    
    def forward(self,x):
        feats = self.conv_before_cca(x)
        for _ in range(2):
            feats = self.cca(feats)
        
        feats = self.conv_after_cca(feats)
        return feats