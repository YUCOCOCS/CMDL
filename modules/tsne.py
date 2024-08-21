import os
import cv2
import copy
import torch
import warnings
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from configs import BuildConfig
from train_logger import logger_config
import pandas as pd
from modules import (
    BuildDataset, BuildDistributedDataloader, BuildDistributedModel, BuildOptimizer, BuildScheduler,
    BuildLoss, BuildBackbone, BuildSegmentor, BuildPixelSampler, Logger, setRandomSeed, BuildPalette, checkdir, loadcheckpoints, savecheckpoints
)
warnings.filterwarnings('ignore')


'''parse arguments in command line'''
def parseArgs():
    parser = argparse.ArgumentParser(description='SSSegmentation is an open source supervised semantic segmentation toolbox based on PyTorch')
    parser.add_argument('--imagedir', dest='imagedir', default='/home/y212202015/SSEG/pre/sssegmentation-main/ssseg/test_picture/s', type=str)
    parser.add_argument('--imagepath', dest='imagepath', default='/home/yinjianjian/YJJ/Cityscapes', type=str)
    parser.add_argument('--outputfilename', dest='outputfilename', default='result_multi', type=str)
    parser.add_argument('--cfgfilepath', dest='cfgfilepath', default='/home/y212202015/SSEG/sseg/configs/ours/ADE20K/PSPNet_ResNet101.py', type=str, required=False)
    parser.add_argument('--checkpointspath', dest='checkpointspath', default='/home/y212202015/SSEG/sseg/done/memorynet_PSPNet_resnet101os8_ade20k/45.29.pth', type=str, required=False)
    args = parser.parse_args()
    return args


'''Demo'''
class Demo():
    def __init__(self):
        self.cmd_args = parseArgs()
        self.cfg, self.cfg_file_path = BuildConfig(self.cmd_args.cfgfilepath)
        logger_handle = logger_config(log_path="/home/y212202015/SSEG/pre/sssegmentation-main/ssseg/out101.log",logging_name="train.log")
        self.logger_handle = logger_handle
        assert self.cmd_args.imagepath or self.cmd_args.imagedir, 'imagepath or imagedir should be specified'
    '''start'''
    def start(self):
        cmd_args, cfg, cfg_file_path = self.cmd_args, self.cfg, self.cfg_file_path
        # check work dir
        checkdir(cfg.COMMON_CFG['work_dir'])
        # cuda detect
        use_cuda = torch.cuda.is_available()
        # initialize logger_handle
        logger_handle = Logger(cfg.COMMON_CFG['logfilepath'])
        # build segmentor
        cfg.SEGMENTOR_CFG['backbone']['pretrained'] = False
        segmentor = BuildSegmentor(segmentor_cfg=copy.deepcopy(cfg.SEGMENTOR_CFG), mode='TEST',logger_handle=logger_handle)
        if use_cuda: segmentor = segmentor.cuda()
        # build dataset
        dataset_cfg = copy.deepcopy(cfg.DATASET_CFG)
        dataset_cfg['type'] = 'base'
        dataset = BuildDataset(mode='TEST', logger_handle=logger_handle, dataset_cfg=copy.deepcopy(cfg.DATASET_CFG))
        # build palette
        palette = BuildPalette(dataset_type=cfg.DATASET_CFG['type'], num_classes=cfg.SEGMENTOR_CFG['num_classes'])
        # load checkpoints
        cmd_args.local_rank = 0
        checkpoints = loadcheckpoints(cmd_args.checkpointspath, logger_handle=logger_handle, cmd_args=cmd_args)
        try:
            segmentor.load_state_dict(checkpoints['model'])
            print("11")
    
        except Exception as e:
            logger_handle.warning(str(e) + '\n' + 'Try to load checkpoints by using strict=False')
            segmentor.load_state_dict(checkpoints['model'], strict=False)
        # set eval
        segmentor.eval()
        # start to test
        inference_cfg = copy.deepcopy(cfg.INFERENCE_CFG)
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        rootdir = "/home/y212202015/SSEG/pre/sssegmentation-main/ssseg/test_picture/s"
        image_dir =  rootdir
        #image_dir = os.path.join(rootdir, 'leftImg8bit', "val")
        #ann_dir = os.path.join(rootdir, 'gtFine', "val")
        ann_dir  = rootdir
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, "val"+'.txt'), names=['imageids'])
        imageids = df['imageids'].values
        imageids = [str(_id) for _id in imageids]

        # if not cmd_args.imagedir:
        #     imagepaths = [cmd_args.imagepath]
        # else:
        #     imagenames = os.listdir(cmd_args.imagedir)
        #     imagepaths = [os.path.join(cmd_args.imagedir, name) for name in imagenames]
        pbar = tqdm(range(len(imageids)))
        overall_embeddings = []
        overall_labels = []
        for idxx in pbar:
            if idxx==88:
                break
            imageid = imageids[idxx]
            imagepath = os.path.join(image_dir, imageid+'.jpg')
            #annpath = os.path.join(ann_dir, imageid.replace('leftImg8bit', 'gtFine_labelIds')+'.png')
            annpath = os.path.join(image_dir, imageid+'.png')
            if imagepath.split('.')[-1] not in ['jpg', 'jpeg', 'png']: 
                continue
            pbar.set_description('Processing %s' % imagepath)
            sample = dataset.read(imagepath, annpath, True)
            image = sample['image']
            label = sample['segmentation']
            sample = dataset.synctransform(sample, 'all')
            image_tensor = sample['image'].unsqueeze(0).type(FloatTensor)
            label_tensor = sample['segmentation'].unsqueeze(0).type(FloatTensor)
            with torch.no_grad():
                outputs = segmentor(image_tensor,features=True)
            embeddings = outputs # 对应的特征图
            B, C, H, W= embeddings.size()
            print(embeddings.size())
            embeddings = embeddings.permute(0, 2, 3, 1) # B * H * W *C 
            embeddings = embeddings.contiguous().view(-1, embeddings.shape[-1])#尺寸为(B*H*W)*C

            labels = label_tensor
            labels = F.interpolate(labels.unsqueeze(1), (H, W), mode='nearest')
            labels = labels.permute(0, 2, 3, 1)
            labels = labels.contiguous().view(-1, 1)
            
            index_1 =(~(labels == 255)).squeeze(-1) # 筛选出不等于255
            #index_1 =((labels == 6)).squeeze(-1)
            embeddings = embeddings[index_1]
            labels = labels[index_1]
            print(labels.unique())
            overall_embeddings.append(embeddings) # 存储所有的不等于255的特征图
            overall_labels.append(labels)
            del image,label,image_tensor,label_tensor,sample,outputs,labels,embeddings
            
        overall_embeddings = torch.cat(overall_embeddings, dim=0)
        overall_labels = torch.cat(overall_labels, dim=0)

        print('overall_embeddings', overall_embeddings.size())
        print('overall_labels', overall_labels.size())

        overall_embeddings = overall_embeddings.cpu().numpy()
        overall_labels = overall_labels.cpu().numpy()
            
        import numpy as np
        np.save('/home/y212202015/SSEG/pre/sssegmentation-main/ssseg/test_picture/s/seg_embeddings_1.npy', overall_embeddings)
        np.save('/home/y212202015/SSEG/pre/sssegmentation-main/ssseg/test_picture/s/seg_labels_1.npy', overall_labels)

           


'''debug'''
if __name__ == '__main__':
    print("222")
    with torch.no_grad():
        client = Demo() # 初始化一个类别 
        print("11")
        client.start()  # 开始进行处理