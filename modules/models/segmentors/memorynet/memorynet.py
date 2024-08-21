'''
Function:
    Implementation of "Mining Contextual Information Beyond Image for Semantic Segmentation"
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from ..deeplabv3 import ASPP
from ..base import BaseSegmentor
from .memory import FeaturesMemory
from ..pspnet import PyramidPoolingModule
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg
from .projection import ProjectionHead
from sklearn.mixture import GaussianMixture
from einops import rearrange, repeat
from ..ccnet import CCNet_unit
from ..upernet import upper
#from ....utils.helpers import label_onehot
# from ......ssseg.modules.utils.
'''MemoryNet'''
class MemoryNet(BaseSegmentor):
    def __init__(self, cfg, mode,logger_handle):
        super(MemoryNet, self).__init__(cfg, mode,logger_handle)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        self.head_cfg = head_cfg
        # build norm layer
        if 'norm_cfg' in head_cfg: # False
            self.norm_layers = nn.ModuleList()
            for in_channels in head_cfg['norm_cfg']['in_channels_list']:
                norm_cfg_copy = head_cfg['norm_cfg'].copy()
                norm_cfg_copy.pop('in_channels_list')
                norm_layer = BuildNormalization(constructnormcfg(placeholder=in_channels, norm_cfg=norm_cfg_copy))
                self.norm_layers.append(norm_layer)
        # build memory
        if head_cfg['downsample_backbone']['stride'] > 1: # False
            self.downsample_backbone = nn.Sequential(
                nn.Conv2d(head_cfg['in_channels'], head_cfg['in_channels'], **head_cfg['downsample_backbone']),
                BuildNormalization(constructnormcfg(placeholder=head_cfg['in_channels'], norm_cfg=norm_cfg)),
                BuildActivation(act_cfg),
            )
        context_within_image_cfg = head_cfg['context_within_image'] # 是否使用ASPP或者PSPNet网络
        if context_within_image_cfg['is_on']: # False
            cwi_cfg = context_within_image_cfg['cfg']
            cwi_cfg.update({
                'in_channels': head_cfg['in_channels'],
                'out_channels': head_cfg['feats_channels'],
                'align_corners': align_corners,
                'norm_cfg': copy.deepcopy(norm_cfg),
                'act_cfg': copy.deepcopy(act_cfg),
            })
            supported_context_modules = {
                'aspp': ASPP,
                'ppm': PyramidPoolingModule,
                'ccnet':CCNet_unit,
                'upernet':upper
            }
            self.context_within_image_module = supported_context_modules[context_within_image_cfg['type']](**cwi_cfg)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        
        self.memory_module = FeaturesMemory( # 内存库模块
            num_classes=cfg['num_classes'], 
            feats_channels=head_cfg['feats_channels'], 
            transform_channels=head_cfg['transform_channels'],
            num_feats_per_cls=head_cfg['num_feats_per_cls'],
            anchor_pixels = cfg['anchor_pixels'],
            negative_pixels = cfg['negative_pixels'],
            out_channels=head_cfg['out_channels'],
            use_context_within_image=context_within_image_cfg['is_on'],
            use_hard_aggregate=head_cfg['use_hard_aggregate'],
            norm_cfg=copy.deepcopy(norm_cfg),
            act_cfg=copy.deepcopy(act_cfg),
        )
        # build decoder
        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0),
        )
        self.decoder_stage2 = nn.Sequential(
            nn.Conv2d(head_cfg['out_channels'], head_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['out_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        self.feat_norm = nn.LayerNorm(head_cfg['feats_channels'])
        self.mask_norm = nn.LayerNorm(cfg['num_classes']) # 对每个样本的内部做一次标准化操作
        self.proj_head = ProjectionHead(head_cfg['feats_channels'], head_cfg['feats_channels']) # 创建表示头
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
        # layer names for training tricks
        self.layer_names = [
            'backbone_net', 'bottleneck', 'memory_module', 'decoder_stage1', 'decoder_stage2', 'norm_layers',
            'downsample_backbone', 'context_within_image_module', 'auxiliary_decoder'
        ]
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None,prototype=None, **kwargs):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        if hasattr(self, 'norm_layers'):
            assert len(backbone_outputs) == len(self.norm_layers)
            for idx in range(len(backbone_outputs)):
                backbone_outputs[idx] = self.norm(backbone_outputs[idx], self.norm_layers[idx])
        if self.cfg['head']['downsample_backbone']['stride'] > 1:
            for idx in range(len(backbone_outputs)):
                backbone_outputs[idx] = self.downsample_backbone(backbone_outputs[idx])
        # feed to context within image module
        feats_ms = self.context_within_image_module(backbone_outputs[-1]) if hasattr(self, 'context_within_image_module') else None
        # feed to memory
        memory_input = self.bottleneck(backbone_outputs[-1])
        preds_stage1 = self.decoder_stage1(memory_input)
        memory_output = self.memory_module(memory_input, preds_stage1, feats_ms)

        #memory_output = self.memory_module(memory_input, preds_stage1, feats_ms)
        # feed to decoder
        preds_stage2 = self.decoder_stage2(memory_output)
        # forward according to the mode
        if self.mode == 'TRAIN':
            c = self.proj_head(memory_input)
            # = F.normalize(memory_input,p=2, dim=1)
            _c = rearrange(c,'b c h w ->(b h w) c')
            _c = self.feat_norm(_c)
            _c = F.normalize(_c,p=2,dim=-1)
            mm = F.normalize(self.memory_module.memory,p=2,dim=-1)
            self.memory_module.memory.data.copy_(mm)
            masks = torch.einsum('nd,kmd->nmk',_c,self.memory_module.memory)
            out_seg = torch.amax(masks,dim=1)
            out_seg = self.mask_norm(out_seg)
            out_seg = rearrange(out_seg,'(b h w) k -> b k h w',b=memory_input.shape[0],h=memory_input.shape[2])

            gt_seg = F.interpolate(targets.unsqueeze(1).float(),size=memory_input.size()[2:],mode='nearest').view(-1)
            loss_ce,loss_ppc,min_distance = self.memory_module.prototype_learning(_c, out_seg, gt_seg, masks,targets)

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

            loss, losses_log_dict = self.calculatelosses(
                predictions=outputs_dict, 
                loss_ce = loss_ce,
                loss_ppc =loss_ppc,
                multiply_distance=min_distance,
                loss_distance_weight = self.head_cfg['loss_distance'],
                loss_ce_weight = self.head_cfg['loss_ce_weight'],
                loss_ppc_weight = self.head_cfg['loss_ppc_weight'],
                #loss_ppd = loss_ppd,
                targets=targets, 
                losses_cfg=losses_cfg
            )

            return loss, losses_log_dict
        return preds_stage2
    '''norm layer'''
    def norm(self, x, norm_layer):
        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()
        x = norm_layer(x)
        x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()
        return x
    
    def gather_togetherr(self,data):
        dist.barrier()

        world_size = dist.get_world_size()
        gather_data = [None for _ in range(world_size)]
        dist.all_gather_object(gather_data, data)

        return gather_data
