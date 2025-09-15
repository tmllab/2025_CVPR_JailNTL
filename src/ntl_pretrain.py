import torch
import wandb
from NTL.utils.utils import *
from NTL.utils.load_utils import *
from data_split import split
import NTL.train as train
from NTL.eval import eval_src
from NTL.utils import Cus_Dataset
import sys


def auto_save_name(config):
    if os.path.exists('saved_models') is False:
        os.makedirs('saved_models')
    
    if config.task_name in ['SL', 'sNTL', 'sCUTI']:
        save_path = f'saved_models/{config.task_name}_{config.domain_src}_{config.NTL_network}.pth'
    else:
        save_path = f'saved_models/{config.task_name}_{config.domain_src}_{config.domain_tgt}_{config.NTL_network}.pth'
    return save_path

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Please input the domain pair! e.g. python src/ntl_train.py -s cifar -t stl")
        sys.exit(0)
    # prepare config
    domain_src, domain_tgt = sys.argv[2], sys.argv[4]
    if domain_src == 'visda_t' and domain_tgt == 'visda_v':
        wandb.init(project='JailNTL_pretrain', config='config/pretrain/visda.yml')
    elif domain_src == 'cifar' and domain_tgt == 'stl':
        wandb.init(project='JailNTL_pretrain', config='config/pretrain/cifarstl.yml')
    elif domain_src == 'stl' and domain_tgt == 'cifar':
        wandb.init(project='JailNTL_pretrain', config='config/pretrain/stlcifar.yml')
    config = wandb.config
       
    # prepare data
    if config.data_pre_split:
        cprint('Split data', 'magenta')
        split(config)
    else:
        cprint('Load pre-split data', 'magenta')
    setup_seed(config.seed)
    dataloader_train, dataloader_val, datasets_name = load_data_tntl(config)
    model_ntl = load_model(config)
    model_ntl.eval()

    # NTL training
    if config.NTL_train: 
        cprint('Train model from scratch', 'magenta')
        cprint(f'method: {config.task_name}', 'yellow')
        
        if config.task_name in ['tNTL', 'sNTL']:
            trainer_func = train.train_tntl
        elif config.task_name in ['tCUTI', 'sCUTI']:
            trainer_func = train.train_tCUTI
        else:
            raise NotImplementedError
        trainer_func(config, dataloader_train, dataloader_val, 
                     model_ntl, datasets_name=datasets_name)
        
        if config.NTL_save:
            save_path = auto_save_name(config)
            print(f'save path: {save_path}')
            torch.save(model_ntl.state_dict(), save_path)
    else: 
        cprint('load saved parameters', 'magenta')
        
        if config.NTL_save_path == 'auto':
            save_path = auto_save_name(config)
        else: 
            save_path = config.NTL_save_path
        print(save_path)
        model_ntl.load_state_dict(torch.load(save_path))
        
        eval_src(config, dataloader_val,
                 model_ntl, datasets_name=datasets_name)
