from NTL.utils.ntl_utils.utils_cifar_stl import get_cifar_data, get_stl_data
from NTL.utils.ntl_utils.utils_visda import get_visda_data_src, get_visda_data_tgt
from JailNTL.data import jailntl_data_setup
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from NTL.utils import Cus_Dataset
import random
import os, sys
import wandb


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def split(config):
    # define transform
    data_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((config.image_size, config.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
    
    # get data
    if config.domain_src in ('cifar, stl'):
        domain_dict = {
            'cifar': get_cifar_data,
            'stl': get_stl_data
        }
    elif config.domain_src in ('visda_t', 'visda_v'):
        domain_dict = {
            'visda_t': get_visda_data_src,
            'visda_v': get_visda_data_tgt
        }
    dataset_names = list(domain_dict.keys())
    dataset_funcs = list(domain_dict.values())

    dataset_split_seed = 2021
    setup_seed(dataset_split_seed)
        
    for name, dataset_funcs in zip(dataset_names, dataset_funcs):
        data = dataset_funcs()
        datafile_val = Cus_Dataset(dataset_name=name, mode='val',
                            dataset=data, begin_ind=0, size=1000,
                            spec_dataTransform=data_transforms,
                            config=config)
        datafile_train = Cus_Dataset(dataset_name=name, mode='train',
                            dataset=data, begin_ind=1000, size=8000,
                            spec_dataTransform=data_transforms,
                            config=config)
        torch.save({'train': datafile_train,
                    'val': datafile_val},
                   f'data_presplit/{name}_{config.image_size}.pth')
    
        pass

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Please input the domain pair! e.g. python src/data_split.py -s cifar -t stl")
        sys.exit(0)
    domain_src, domain_tgt = sys.argv[2], sys.argv[4]
    if domain_src == 'visda_t' and domain_tgt == 'visda_v':
        wandb.init(project='JailNTL_datasplit', config='config/pretrain/visda.yml')
    elif domain_src == 'cifar' and domain_tgt == 'stl':
        wandb.init(project='JailNTL_datasplit', config='config/pretrain/cifarstl.yml')
    elif domain_src == 'stl' and domain_tgt == 'cifar':
        wandb.init(project='JailNTL_datasplit', config='config/pretrain/stlcifar.yml')
    config = wandb.config
    
    # split data
    print('Split data', 'magenta')
    if not os.path.exists('./data_presplit'):
        os.makedirs('./data_presplit')
    split(config)
    
    # prepare data for JailNTL
    config.update({'jailntl_shot_num': 80})
    jailntl_data_setup(config)

