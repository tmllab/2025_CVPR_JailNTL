"""
This module includes functions to evaluate a model on source and target datasets.
"""

import torch
import sys
sys.path.append('..')
from NTL.utils.evaluators import classification_evaluator

def eval_src(config, valloaders, model, print_freq=0, datasets_name=None):
    evaluators = [classification_evaluator(
        v) for v in valloaders]

    acc1s = []
    acc5s = []
    for evaluator in evaluators:
        eval_results = evaluator(model, device=config.device)
        (acc1, acc5), _ = eval_results['Acc'], eval_results['Loss']
        acc1s.append(acc1)
        acc5s.append(acc5)

    print('[Evaluate] | src_val_acc1: %.1f, tgt_val_acc1: ' %
            (acc1s[0]), end='')
    for dname, acc_tgt in zip(datasets_name[1:], acc1s[1:]):
        print(f'{dname}: {acc_tgt:.2f} ', end='')
    tgt_mean = torch.mean(torch.tensor(acc1s[1:])).item()
    print(f'| tgt_mean: {tgt_mean:.2f}')
    
    return round(float(acc1s[0]), 2), round(float(tgt_mean), 2)
