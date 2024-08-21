'''memorynet_deeplabv3_resnet50os8_ade20k'''
import os
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG.update({
    'type': 'ade20k',
    'rootdir': '/home/yjj/datasets/ADE20K/ADE20k',
})
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
DATALOADER_CFG['train']['batch_size'] = 16
# modify optimizer config
OPTIMIZER_CFG = OPTIMIZER_CFG.copy()
# modify scheduler config
SCHEDULER_CFG = SCHEDULER_CFG.copy()
SCHEDULER_CFG.update({
    'max_epochs': 140
})
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update({
    'num_classes': 150,
    'backbone': {
        'type': 'resnet50',
        'series': 'resnet',
        'pretrained': True,
        'outstride': 8,
        'use_stem': True,
        'selected_indices': (0, 1, 2, 3),
    },
})
SEGMENTOR_CFG['head']['context_within_image']['is_on']=False
SEGMENTOR_CFG['head']['use_loss'] = False # 这里面是否需要进行
SEGMENTOR_CFG['head']['update_cfg']['momentum_cfg']['base_lr'] = 0.001 * 0.9
INFERENCE_CFG = INFERENCE_CFG.copy()
# INFERENCE_CFG = {
#     'mode': 'whole',
#     'opts': {}, 
#     'tricks': {
#         'multiscale': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
#         'flip': True,
#         'use_probs_before_resize': True
#     }
# }
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = '/home/yjj/MDRL/MDRL/ADE20K/FCN_50'
COMMON_CFG['logfilepath'] = '/home/yjj/MDRL/MDRL/ADE20K/FCN_50/log_9_ade20k_PSPNet_50.log'
COMMON_CFG['resultsavepath'] = '/home/yjj/MDRL/MDRL/ADE20K/FCN_50/memorynet_PSPNet_resnet50os8_ade20k_results.pkl'