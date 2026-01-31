import os
import time
import torch
import shutil
import collections
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf
import torch.nn.functional as F
from torch.utils.data import DataLoader
from misc.utils import warning_print
from tensorboardX import SummaryWriter


class baseTrainer(object):
    def save(self, epoch, model, optimizer, checkpoint_dir):
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   '{}/net.pickle'.format(checkpoint_dir))
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   '{}/net_epoch_{}.pickle'.format(checkpoint_dir, epoch))

    def load(self, net, optim, models_dir, resume_model_name=''):
        warning_print('resuming models...')
        models = {}
        for f in os.listdir(models_dir):
            if os.path.splitext(f)[1] == '.pickle':
                f = '{}/{}'.format(models_dir, f)
                id = os.path.getctime(f)
                models[id] = f
        if len(models) < 1:
            warning_print('there is no models to resume')
            return
        if resume_model_name.endswith('.pickle'):
            resume_model_name = resume_model_name.replace('.pickle', '')
        if len(resume_model_name) > 0:
            model_name = '{}/{}.pickle'.format(models_dir, resume_model_name)
        else:
            index = sorted(models)[-1]
            model_name = models[index]
        model_dict = torch.load(model_name)

        # 单gpu模型与多gpu模型兼容方案
        # 如果模型采用了多gpu模型，则原有模型的子模块包含在module下面
        if 'DataParallel' in str(type(net)):
            # net 为多gpu类判断  模型的类型
            if not list(model_dict['state_dict'].keys())[0].startswith('module.'):
                # 保存的是单gpu类型的模型
                new_dict = collections.OrderedDict()
                for key in model_dict['state_dict'].keys():
                    new_dict['module.' + key] = model_dict['state_dict'][key]  # 验证通过
                net.load_state_dict(new_dict, strict=False)
            else:
                net.load_state_dict(model_dict['state_dict'], strict=False)  # 验证通过
        else:
            # net 是 单gpu类型的
            if list(model_dict['state_dict'].keys())[0].startswith('module.'):
                # 保存的模型是多gpu的
                # 处理的过程较为繁琐
                new_dict = collections.OrderedDict()
                for key in model_dict['state_dict'].keys():
                    new_dict[key.replace('module.', '')] = model_dict['state_dict'][key]
                net.load_state_dict(new_dict, strict=False)
            else:
                net.load_state_dict(model_dict['state_dict'], strict=False)  # 验证通过

        # step = model_dict['epoch']
        optim_state = model_dict['optimizer']
        if optim is not None:
            optim.load_state_dict(optim_state)
            pass
        warning_print('finish to resume models {}'.format(model_name))
