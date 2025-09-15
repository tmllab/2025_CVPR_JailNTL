from .utils import *
from torch.utils.data import DataLoader
import copy
import cv2, os
from .cmi_registry import get_model as get_cmi_model
from NTL.utils import Cus_Dataset


def load_cmi(config):
    name = config.NTL_network
    pretrain = config.NTL_pretrain
    
    if pretrain: cprint('Pretrain in datafree implemention is useless')
    model = get_cmi_model(model_dict_cmi[name],
                          num_classes=config.num_classes, 
                          pretrained=pretrain)
    return model.to(config.device)

def load_model(config):
    cprint('Init NTL model', 'magenta')
    
    if 'cmi' in config.NTL_network: return load_cmi(config)

    model = model_dict[config.NTL_network](
                pretrained=config.NTL_pretrain, 
                num_classes=config.num_classes, 
                img_size=config.image_size).to(config.device)
    return model

def load_data(config):
    cprint('load data', 'magenta')
    
    # source domain
    loaded_src = torch.load(
        f'data_presplit/{config.domain_src}_{config.image_size}.pth', weights_only=False)
    datafile_src_train = loaded_src['train']
    datafile_src_val = loaded_src['val']
    dataloader_train = DataLoader(datafile_src_train, batch_size=config.batch_size,
                                    shuffle=True, num_workers=config.num_workers,
                                    drop_last=True)
    dataloader_val = DataLoader(datafile_src_val, batch_size=config.batch_size,
                                shuffle=False, num_workers=config.num_workers,
                                drop_last=False)
    
    # target domain
    if config.domain_src in domain_visda:
        dataset_tgts_name = domain_visda
    elif config.domain_src in domain_cifar_stl:
        dataset_tgts_name = domain_cifar_stl
    else:
        raise Exception
    dataset_tgts_name.remove(config.domain_src)
    datasets_name = [config.domain_src] + dataset_tgts_name
    dataloader_tgt_val = []
    for tgt in dataset_tgts_name:
        loaded_tgt = torch.load(f'data_presplit/{tgt}_{config.image_size}.pth')
        # datafile_tgt_train = loaded_tgt['train']
        datafile_tgt_val = loaded_tgt['val']
        dataloader_tgt_val.append(DataLoader(datafile_tgt_val,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers,
                                             drop_last=False))
    dataloader_train = [dataloader_train]
    dataloader_val = [dataloader_val] + dataloader_tgt_val

    return dataloader_train, dataloader_val, datasets_name

def load_data_tntl(config):
    # source domain
    loaded_src = torch.load(f'data_presplit/{config.domain_src}_{config.image_size}.pth',
                            weights_only=False)
    datafile_src_train = loaded_src['train']
    datafile_src_val = loaded_src['val']
    dataloader_train = DataLoader(datafile_src_train, batch_size=config.batch_size,
                                    shuffle=True, num_workers=config.num_workers,
                                    drop_last=True)
    dataloader_val = DataLoader(datafile_src_val, batch_size=config.batch_size,
                                shuffle=False, num_workers=config.num_workers,
                                drop_last=False)
    
    # target domain
    dataset_tgts_name = [config.domain_tgt]
    datasets_name = [config.domain_src] + dataset_tgts_name
    dataloader_tgt_train = []
    dataloader_tgt_val = []
    for tgt in dataset_tgts_name:
        loaded_tgt = torch.load(f'data_presplit/{tgt}_{config.image_size}.pth',
                                weights_only=False)
        datafile_tgt_train = loaded_tgt['train']
        datafile_tgt_val = loaded_tgt['val']
        dataloader_tgt_train.append(DataLoader(datafile_tgt_train,
                                             batch_size=config.batch_size,
                                             shuffle=True,
                                             num_workers=config.num_workers,
                                             drop_last=True))
        dataloader_tgt_val.append(DataLoader(datafile_tgt_val,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers,
                                             drop_last=False))
    dataloader_train = [dataloader_train] + dataloader_tgt_train
    dataloader_val = [dataloader_val] + dataloader_tgt_val

    return dataloader_train, dataloader_val, datasets_name


domain_cifar_stl = ['cifar', 'stl']
domain_visda = ['visda_v', 'visda_t']

