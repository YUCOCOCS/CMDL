'''
Function:
    Implementation of MemoryNetV2 - "MCIBI++: Soft Mining Contextual Information Beyond Image for Semantic Segmentation"
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..deeplabv3 import ASPP
from ..base import BaseSegmentor
from ..base import SelfAttentionBlock
from .memoryv2 import FeaturesMemoryV2
from ..pspnet import PyramidPoolingModule
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


'''MemoryNetV2'''
class MemoryNetV2(BaseSegmentor):
    def __init__(self, cfg, mode,logger_handle):
        super(MemoryNetV2, self).__init__(cfg, mode,logger_handle)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build memory
        context_within_image_cfg = head_cfg['context_within_image'] #True
        if context_within_image_cfg['is_on']: #True
            cwi_cfg = context_within_image_cfg['cfg']
            cwi_cfg.update({
                'in_channels': head_cfg['in_channels'], # 输入是2048
                'out_channels': head_cfg['feats_channels'], # 输出是512维度
                'align_corners': align_corners, #False
                'norm_cfg': copy.deepcopy(norm_cfg), #使用同步批处理
                'act_cfg': copy.deepcopy(act_cfg),  # 使用的是relu激活
            })
            supported_context_modules = {
                'aspp': ASPP, # 对应的是deeplabv3的结构
                'ppm': PyramidPoolingModule,  #对应的是PSPNet的结构
            }
            if context_within_image_cfg['type'] == 'aspp':
                cwi_cfg.pop('pool_scales')
            elif context_within_image_cfg['type'] == 'ppm':
                cwi_cfg.pop('dilations')
            self.context_within_image_module = supported_context_modules[context_within_image_cfg['type']](**cwi_cfg) #这里面选用的是ASPP或者是PPM
            if context_within_image_cfg.get('use_self_attention', True): #这个里面使用了自注意力机制
                self.self_attention = SelfAttentionBlock(key_in_channels=head_cfg['feats_channels'],  # 512
                                                        query_in_channels=head_cfg['feats_channels'], # 512
                                                        transform_channels=head_cfg['feats_channels']//2,  # 256维度
                                                        out_channels=head_cfg['feats_channels'],  # 512维度
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
                                                        )  #这个里面使用了自注意力机制
        self.bottleneck = nn.Sequential(  #进来的维度是2048维度，出来的维度是512维度
            nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        self.memory_module = FeaturesMemoryV2(
            num_classes=cfg['num_classes'],  # 19个类别
            feats_channels=head_cfg['feats_channels'],  # 512维度
            transform_channels=head_cfg['transform_channels'], # 256维度
            out_channels=head_cfg['out_channels'],    # 512维度
            use_hard_aggregate=head_cfg['use_hard_aggregate'],  # False
            downsample_before_sa=head_cfg['downsample_before_sa'], # False
            norm_cfg=copy.deepcopy(norm_cfg),
            act_cfg=copy.deepcopy(act_cfg),
            align_corners=align_corners,
        )
        # build fpn  deeplabv3 和psp是没有 fpn
        if head_cfg.get('fpn', None) is not None:
            act_cfg_copy = copy.deepcopy(act_cfg)
            if 'inplace' in act_cfg_copy: act_cfg_copy['inplace'] = False
            self.lateral_convs = nn.ModuleList()
            for in_channels in head_cfg['fpn']['in_channels_list'][:-1]:
                self.lateral_convs.append(nn.Sequential(
                    nn.Conv2d(in_channels, head_cfg['fpn']['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                    BuildNormalization(constructnormcfg(placeholder=head_cfg['fpn']['feats_channels'], norm_cfg=norm_cfg)),
                    BuildActivation(act_cfg_copy),
                ))
            self.fpn_convs = nn.ModuleList()
            for in_channels in [head_cfg['fpn']['feats_channels'],] * len(self.lateral_convs):
                self.fpn_convs.append(nn.Sequential(
                    nn.Conv2d(in_channels, head_cfg['fpn']['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                    BuildNormalization(constructnormcfg(placeholder=head_cfg['fpn']['out_channels'], norm_cfg=norm_cfg)),
                    BuildActivation(act_cfg_copy),
                ))

        # build decoder
        for key, value in head_cfg['decoder'].items():  #这里面有三个解码器
            if key == 'cwi' and (not context_within_image_cfg['is_on']): continue
            setattr(self, f'decoder_{key}', nn.Sequential())
            decoder = getattr(self, f'decoder_{key}')
            decoder.add_module('conv1', nn.Conv2d(value['in_channels'], value['out_channels'], kernel_size=value.get('kernel_size', 1), stride=1, padding=value.get('padding', 0), bias=False))
            decoder.add_module('bn1', BuildNormalization(constructnormcfg(placeholder=value['out_channels'], norm_cfg=norm_cfg)))
            decoder.add_module('act1', BuildActivation(act_cfg))
            decoder.add_module('dropout', nn.Dropout2d(value['dropout']))
            decoder.add_module('conv2', nn.Conv2d(value['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))

        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])  #这里面包含一个辅助解码器


        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()


        # layer names for training tricks
        self.layer_names = [
            'backbone_net', 'bottleneck', 'memory_module', 'decoder_cls', 'decoder_cwi', 'lateral_convs', 'fpn_convs',
            'self_attention', 'context_within_image_module', 'auxiliary_decoder'
        ]


    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None, **kwargs):
        img_size = x.size(2), x.size(3)

        # feed to backbone network 产生ResNet的四层特征
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))

        # feed to context within image module
        if hasattr(self, 'context_within_image_module'):
            feats_cwi = self.context_within_image_module(backbone_outputs[-1])
            if hasattr(self, 'decoder_cwi'): 
                preds_cwi = self.decoder_cwi(feats_cwi)



        # feed to memory
        pixel_representations = self.bottleneck(backbone_outputs[-1])
        preds_pr = self.decoder_pr(pixel_representations)
        if self.cfg['head'].get('force_use_preds_pr', False): #False
            memory_gather_logits = preds_pr
        else:
            memory_gather_logits = preds_cwi if (hasattr(self, 'context_within_image_module') and hasattr(self, 'decoder_cwi')) else preds_pr

        memory_input = pixel_representations
        assert memory_input.shape[2:] == memory_gather_logits.shape[2:]
        if (self.mode == 'TRAIN') and (kwargs['epoch'] < self.cfg['head'].get('warmup_epoch', 0)):
            with torch.no_grad():
                gt = targets['segmentation']
                gt = F.interpolate(gt.unsqueeze(1), size=memory_gather_logits.shape[2:], mode='nearest')[:, 0, :, :]
                assert len(gt.shape) == 3, 'segmentation format error'
                preds_gt = gt.new_zeros(memory_gather_logits.shape).type_as(memory_gather_logits)
                valid_mask = (gt >= 0) & (gt < self.cfg['num_classes'])
                idxs = torch.nonzero(valid_mask, as_tuple=True)
                if idxs[0].numel() > 0:
                    preds_gt[idxs[0], gt[valid_mask].long(), idxs[1], idxs[2]] = 1
            stored_memory, memory_output = self.memory_module(memory_input, preds_gt.detach())
        else:
            if 'memory_gather_logits' in kwargs: 
                memory_gather_logits_aux = F.interpolate(kwargs['memory_gather_logits'], size=memory_gather_logits.shape[2:], mode='bilinear', align_corners=self.align_corners)
                weights = kwargs.get('memory_gather_logits_weights', [2, 1.5])
                memory_gather_logits = (memory_gather_logits * weights[0] + memory_gather_logits_aux * weights[1]) / (sum(weights) - 1)
            
            #这里面输出的stored_memory是产生的数据集级别的语义特征，而memory_output指的是经过融合之后的特征，维度是512维度
            stored_memory, memory_output = self.memory_module(memory_input, memory_gather_logits)  #输入的是最后一维度的特征和cwi的预测


        # feed to fpn & decoder
        if hasattr(self, 'context_within_image_module'):
            if hasattr(self, 'self_attention'): 
                memory_output = self.self_attention(feats_cwi, memory_output) # 产生的图像级别的特征和之前融合之后的特征之间的融合
            if hasattr(self, 'fpn_convs'):
                inputs = backbone_outputs[:-1]
                lateral_outputs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
                if self.cfg['head'].get('fuse_memory_cwi_before_fpn', True):
                    lateral_outputs.append(torch.cat([memory_output, feats_cwi], dim=1))
                else:
                    lateral_outputs.append(feats_cwi)
                for i in range(len(lateral_outputs) - 1, 0, -1):
                    prev_shape = lateral_outputs[i - 1].shape[2:]
                    lateral_outputs[i - 1] = lateral_outputs[i - 1] + F.interpolate(lateral_outputs[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners)
                fpn_outputs = [self.fpn_convs[i](lateral_outputs[i]) for i in range(len(lateral_outputs) - 1)]
                fpn_outputs.append(lateral_outputs[-1])
                fpn_outputs = [F.interpolate(out, size=fpn_outputs[0].size()[2:], mode='bilinear', align_corners=self.align_corners) for out in fpn_outputs]
                if not self.cfg['head'].get('fuse_memory_cwi_before_fpn', True): 
                    fpn_outputs.append(F.interpolate(memory_output, size=fpn_outputs[0].shape[2:], mode='bilinear', align_corners=self.align_corners))
                memory_output = torch.cat(fpn_outputs, dim=1)
            else:
                memory_output = torch.cat([memory_output, feats_cwi], dim=1)
        preds_cls = self.decoder_cls(memory_output) #这是最后分类的结果

        # 融合的总的过程为：
        '''
            由backbone产生的四个基本特征,将最后一维度的特征输入到Context Module Within Image得到预测的概率分布,将Cwi的类概率分布和backbone产生的高维特征进行数据集级别的特征提取,
            产生select_memory_first与基本特征进行注意力机制融合,得到select_memory,同时将feat_Cwi与聚合之后的select_memory进行注意力机制的融合,最后拼在一起。
            进行分类
        '''
        # forward according to the mode
        if self.mode == 'TRAIN':
            outputs_dict = self.forwardtrain( #对ResNet产生的高维特征的一个辅助分类器
                predictions=preds_cls,
                targets=targets,
                backbone_outputs=backbone_outputs,
                losses_cfg=losses_cfg,
                img_size=img_size,
                compute_loss=False,
            )
            preds_cls = outputs_dict.pop('loss_cls')
            preds_pr = F.interpolate(preds_pr, size=img_size, mode='bilinear', align_corners=self.align_corners)
            outputs_dict.update({
                'loss_pr': preds_pr, 
                'loss_cls': preds_cls,
            })
            if hasattr(self, 'context_within_image_module') and hasattr(self, 'decoder_cwi'): 
                preds_cwi = F.interpolate(preds_cwi, size=img_size, mode='bilinear', align_corners=self.align_corners)
                outputs_dict.update({'loss_cwi': preds_cwi})
            with torch.no_grad():
                self.memory_module.update( # 对当前批处理的类别的数据集级别的特征进行更新
                    features=F.interpolate(pixel_representations, size=img_size, mode='bilinear', align_corners=self.align_corners), 
                    #segmentation=targets['segmentation'],
                    segmentation=targets,
                    learning_rate=kwargs['learning_rate'],
                    **self.cfg['head']['update_cfg']
                )
            loss, losses_log_dict = self.calculatelosses( #计算总的损失 和 每一项的损失 都是celoss
                predictions=outputs_dict, 
                targets=targets, 
                losses_cfg=losses_cfg,
                loss_ce=None,
                loss_ce_weight=None,
                loss_ppc=None,
                loss_ppc_weight=None,
            )
            return loss, losses_log_dict  #返回的是总损失和每一项损失的大小
        return preds_cls