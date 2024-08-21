import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseSegmentor
from ..pspnet import PyramidPoolingModule
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg

class upper(nn.Module):
    def __init__(self, in_channels, out_channels, in_channels_list,feats_channels, pool_scales,align_corners,dropout, norm_cfg=None, act_cfg=None):
        super(upper, self).__init__()
        self.align_corners = align_corners
        # build pyramid pooling module
        ppm_cfg = {
            'in_channels': in_channels_list[-1],
            'out_channels': feats_channels,
            'pool_scales': pool_scales,
            'align_corners': align_corners,
            'norm_cfg': copy.deepcopy(norm_cfg),
            'act_cfg': copy.deepcopy(act_cfg),
        }
        self.ppm_net = PyramidPoolingModule(**ppm_cfg)
        # build lateral convs
        act_cfg_copy = copy.deepcopy(act_cfg)
        if 'inplace' in act_cfg_copy: act_cfg_copy['inplace'] = False
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list[:-1]:
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, feats_channels, kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(constructnormcfg(feats_channels, norm_cfg=norm_cfg)),
                BuildActivation(act_cfg_copy),
            ))
        # build fpn convs
        self.fpn_convs = nn.ModuleList()
        for in_channels in [feats_channels,] * len(self.lateral_convs):
            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, feats_channels, kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(constructnormcfg(placeholder=feats_channels, norm_cfg=norm_cfg)),
                BuildActivation(act_cfg_copy),
            ))
        
        # build decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(feats_channels* len(in_channels_list), feats_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=feats_channels, norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        
    
    def forward(self, backbone_outputs):
        ppm_out = self.ppm_net(backbone_outputs[-1])
        # ppm_out = self.ppm_net(backbone_outputs)
         # apply fpn
        inputs = backbone_outputs[:-1]
        lateral_outputs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        lateral_outputs.append(ppm_out)
        for i in range(len(lateral_outputs) - 1, 0, -1):
            prev_shape = lateral_outputs[i - 1].shape[2:]
            lateral_outputs[i - 1] = lateral_outputs[i - 1] + F.interpolate(lateral_outputs[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners)
        fpn_outputs = [self.fpn_convs[i](lateral_outputs[i]) for i in range(len(lateral_outputs) - 1)]
        fpn_outputs.append(lateral_outputs[-1])
        fpn_outputs = [F.interpolate(out, size=fpn_outputs[0].size()[2:], mode='bilinear', align_corners=self.align_corners) for out in fpn_outputs]
        fpn_out = torch.cat(fpn_outputs, dim=1)
        feats = self.decoder(fpn_out)
        return feats

