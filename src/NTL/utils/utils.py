import torch
import numpy as np
import random 
from termcolor import cprint
import torchvision
import os
from .ntl_utils.utils_cifar_stl import get_cifar_data, get_stl_data
from .ntl_utils.utils_visda import get_visda_data_tgt, get_visda_data_src
from .model import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg19, vgg19_bn


def auto_save_name(config):
    if os.path.exists('saved_models') is False:
        os.makedirs('saved_models')
    
    if config.task_name in ['SL', 'sNTL', 'sCUTI']:
        save_path = f'saved_models/{config.task_name}_{config.domain_src}_{config.NTL_network}.pth'
    else:
        save_path = f'saved_models/{config.task_name}_{config.domain_src}_{config.domain_tgt}_{config.NTL_network}.pth'
    return save_path

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

domain_dict = {'cifar': get_cifar_data,
               'stl': get_stl_data,
               'visda_t': get_visda_data_src,
               'visda_v': get_visda_data_tgt
               }

model_dict = {'vgg11': vgg11,
              'vgg11bn': vgg11_bn,
              'vgg13': vgg13,
              'vgg13bn': vgg13_bn,
              'vgg19': vgg19,
              'vgg19bn': vgg19_bn,
              'resnet18': torchvision.models.resnet18}

model_dict_cmi = {'vgg11bncmi': 'vgg11',
                  'vgg13bncmi': 'vgg13',
                  'resnet50cmi': 'resnet50_imagenet',
                  'resnet34cmi': 'resnet34_imagenet',
                  'resnet34cmi1': 'resnet34',
                  'resnet18cmi': 'resnet18_imagenet',
                  'resnet18cmi1': 'resnet18',
                  'resnet50cmi_wobn': 'resnet50_wobn_imagenet',
                  'resnet34cmi_wobn': 'resnet34_wobn_imagenet',
                  'resnet18cmi_wobn': 'resnet18_wobn_imagenet',
                  'wide_resnet50_2cmi': 'wide_resnet50_2_imagenet',
                  'wrn40_2cmi': 'wrn40_2',
                  'wrn40_1cmi': 'wrn40_1',
                  'wrn16_2cmi': 'wrn16_2',
                  'wrn16_1cmi': 'wrn16_1',
                  }

domain_cifar_stl = ['cifar', 'stl']
domain_visda = ['visda_v', 'visda_t']
