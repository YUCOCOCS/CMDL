'''
Function:
    Build config file
Author:
    Zhenchao Jin
'''
import os
import time
import shutil
import importlib


'''BuildConfig'''
def BuildConfig(cfg_file_path, tmp_cfg_dir='tmpp_cfg'):
    # assert whether file exists
    assert os.path.exists(cfg_file_path), 'cfg_file_path %s not exist' % cfg_file_path

    # get config file info
    cfg_file_path = os.path.abspath(os.path.expanduser(cfg_file_path)) #获得绝对路径

    cfg_info, ext = os.path.splitext(cfg_file_path)  # 将后缀名分离出来
    assert ext in ['.py'], 'only support .py type, but get %s' % ext
    cfg_dir, cfg_name = '/'.join(cfg_info.split('/')[:-1]), cfg_info.split('/')[-1] #获得文件的目录和文件名
    # base_cfg.py must exist
    base_cfg_file_path = os.path.join(cfg_dir, 'base_cfg' + ext) # 判断的是base_cfg文件存不存在
    assert os.path.exists(base_cfg_file_path), 'base_cfg_file_path %s not exist' % base_cfg_file_path
    # make temp dir for loading config
    if not os.path.exists(tmp_cfg_dir):
        try: os.mkdir(tmp_cfg_dir)
        except: pass
    # copy config file and the base config file
    shutil.copyfile(cfg_file_path, os.path.join(tmp_cfg_dir, cfg_name + ext))  # 将配置文件复制到临时文件中
    shutil.copyfile(base_cfg_file_path, os.path.join(tmp_cfg_dir, 'base_cfg' + ext))
    time.sleep(0.5)
    # load module from the temp dir
    try:
        cfg = importlib.import_module(f'{tmp_cfg_dir}.{cfg_name}', __package__) #加载配置文件
    except:
        import sys
        sys.path.insert(0, '.')
        cfg = importlib.import_module(f'{tmp_cfg_dir}.{cfg_name}', __package__)
    # return cfg
    return cfg, cfg_file_path