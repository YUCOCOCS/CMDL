import os
import pandas as pd
from .base import BaseDataset


'''CamVidDataset'''
class CamVidDataset(BaseDataset):
    num_classes = 11
    classnames = ['sky', 'building', 'pole', 'road', 'pavement','tree', 'signsymbol', 'fence', 'car','pedestrian', 'bicyclist']
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(CamVidDataset, self).__init__(mode, logger_handle, dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        # self.image_dir = os.path.join(rootdir, 'all')
        # self.ann_dir = os.path.join(rootdir, 'all_anno')
        self.image_dir = os.path.join(rootdir, dataset_cfg['set'])
        self.ann_dir = os.path.join(rootdir, dataset_cfg['set']+"annot")
        #self.segclass_dir = os.path.join(rootdir, 'SegmentationClass')
        #self.set_dir = os.path.join(rootdir, 'ImageSets', 'Segmentation')
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, dataset_cfg['set']+'.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]
    '''pull item'''
    def __getitem__(self, index):
        imageid = self.imageids[index]
        imagepath = os.path.join(self.image_dir, imageid+'.png')
        annpath = os.path.join(self.ann_dir, imageid+'.png')
        sample = self.read(imagepath, annpath, self.dataset_cfg.get('with_ann', True))
        segmen_copy = sample['segmentation'].copy()
        segmen_copy[sample['segmentation']==11]=255
        sample['segmentation'] = segmen_copy
        sample.update({'id': imageid})
        if self.mode == 'TRAIN':
            sample = self.synctransform(sample, 'without_totensor_normalize_pad')
            #sample['edge'] = self.generateedge(sample['segmentation'].copy())
            sample = self.synctransform(sample, 'only_totensor_normalize_pad')
        else:
            sample = self.synctransform(sample, 'all')
        return sample
    '''length'''
    def __len__(self):
        return len(self.imageids)