'''
Function:
    Implementation of ISNet
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseSegmentor
from .imagelevel import ImageLevelContext
from .semanticlevel import SemanticLevelContext
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


'''ISNet'''
class ISNet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(ISNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        # build image-level context module
        ilc_cfg = {
            'feats_channels': head_cfg['feats_channels'],
            'transform_channels': head_cfg['transform_channels'],
            'concat_input': head_cfg['concat_input'],
            'norm_cfg': copy.deepcopy(norm_cfg),
            'act_cfg': copy.deepcopy(act_cfg),
            'align_corners': align_corners,
        }
        self.ilc_net = ImageLevelContext(**ilc_cfg)
        # build semantic-level context module
        slc_cfg = {
            'feats_channels': head_cfg['feats_channels'],
            'transform_channels': head_cfg['transform_channels'],
            'concat_input': head_cfg['concat_input'],
            'norm_cfg': copy.deepcopy(norm_cfg),
            'act_cfg': copy.deepcopy(act_cfg),
        }
        self.slc_net = SemanticLevelContext(**slc_cfg)
        # build decoder
        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        if head_cfg['shortcut']['is_on']:
            self.shortcut = nn.Sequential(
                nn.Conv2d(head_cfg['shortcut']['in_channels'], head_cfg['shortcut']['feats_channels'], kernel_size=1, stride=1, padding=0),
                BuildNormalization(constructnormcfg(placeholder=head_cfg['shortcut']['feats_channels'], norm_cfg=norm_cfg)),
                BuildActivation(act_cfg),
            )
            self.decoder_stage2 = nn.Sequential(
                nn.Conv2d(head_cfg['feats_channels'] + head_cfg['shortcut']['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
                BuildActivation(act_cfg),
                nn.Dropout2d(head_cfg['dropout']),
                nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
            )
        else:
            self.decoder_stage2 = nn.Sequential(
                nn.Dropout2d(head_cfg['dropout']),
                nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
            )
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
        # layer names for training tricks
        self.layer_names = ['backbone_net', 'bottleneck', 'ilc_net', 'slc_net', 'shortcut', 'decoder_stage1', 'decoder_stage2', 'auxiliary_decoder']
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to bottleneck
        feats = self.bottleneck(backbone_outputs[-1])
        # feed to image-level context module
        feats_il = self.ilc_net(feats)
        # feed to decoder stage1
        preds_stage1 = self.decoder_stage1(feats)
        # feed to semantic-level context module
        preds = preds_stage1
        if preds_stage1.size()[2:] != feats.size()[2:]:
            preds = F.interpolate(preds_stage1, size=feats.size()[2:], mode='bilinear', align_corners=self.align_corners)
        feats_sl = self.slc_net(feats, preds, feats_il)
        # feed to decoder stage2
        if hasattr(self, 'shortcut'):
            shortcut_out = self.shortcut(backbone_outputs[0])
            feats_sl = F.interpolate(feats_sl, size=shortcut_out.shape[2:], mode='bilinear', align_corners=self.align_corners)
            feats_sl = torch.cat([feats_sl, shortcut_out], dim=1)
        preds_stage2 = self.decoder_stage2(feats_sl)
        # return according to the mode
        if self.mode == 'TRAIN':
            outputs_dict = self.forwardtrain(
                predictions=preds_stage2,
                targets=targets,
                backbone_outputs=backbone_outputs,
                losses_cfg=losses_cfg,
                img_size=img_size,
                compute_loss=False,
            )
            preds_stage2 = outputs_dict.pop('loss_cls')
            preds_stage1 = F.interpolate(preds_stage1, size=img_size, mode='bilinear', align_corners=self.align_corners)
            outputs_dict.update({'loss_cls_stage1': preds_stage1, 'loss_cls_stage2': preds_stage2})
            return self.calculatelosses(
                predictions=outputs_dict, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return preds_stage2