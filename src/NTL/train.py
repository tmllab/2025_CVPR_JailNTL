"""
This module include the training functions for the NTL and CUTI methods.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import wandb

import sys
sys.path.append('..')
from NTL.utils.evaluators import classification_evaluator


def train_tntl(config, dataloaders, valloaders, model, datasets_name):
    
    evaluators = [classification_evaluator(v) for v in valloaders]
    
    lr = config.NTL_lr
    print('use learning rate: ', lr)
    print(f'use {config.NTL_optimizer} optimizer')
    if config.NTL_optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif config.NTL_optimizer == 'sgd':
        print('use SGD optimizer')
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                    momentum=config.NTL_momentum, weight_decay=config.NTL_weight_decay)
    lambda1 = lambda epoch:0.999**epoch
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    #criterion = torch.nn.CrossEntropyLoss()
    criterion_KL = torch.nn.KLDivLoss()

    device = config.device
    cnt = 0
    for epoch in range(config.NTL_epochs):
        model.train()
        for i, zipped in enumerate(zip(dataloaders[0], dataloaders[1])):
            img1 = zipped[0][0].to(device).float()
            label1 = zipped[0][1].to(device).float()
            img2 = zipped[1][0].to(device).float()
            label2 = zipped[1][1].to(device).float()
            
            if 'cmi' in config.NTL_network:
                # out1, fe1 = model(img1, return_features=True)
                # out2, fe2 = model(img2, return_features=True)
                if epoch == 0 and i == 0:
                    print('NTL combine training')
                out, feat = model(torch.cat((img1, img2), 0), return_features=True)
                out1, out2 = out.chunk(2, dim=0)
                fe1, fe2 = feat.chunk(2, dim=0)
            else:
                # out1, out2, fe1, fe2 = model(img1, img2)
                if config.NTL_network in ['vgg11', 'vgg13', 'vgg19']:
                    if epoch == 0 and i == 0:
                        print('NTL seperate training for vgg without bn')
                    out1, out2, fe1, fe2 = model(img1, img2)
                else:
                    if epoch == 0 and i == 0:
                        print('NTL combine training')
                    out, feat = model.forward_f(torch.cat((img1, img2), 0))
                    out1, out2 = out.chunk(2, dim=0)
                    fe1, fe2 = feat.chunk(2, dim=0)

            out1 = F.log_softmax(out1,dim=1)
            loss1 = criterion_KL(out1, label1)
            #loss = criterion(out, label.squeeze())

            out2 = F.log_softmax(out2,dim=1)
            loss2 = criterion_KL(out2, label2)#?change to 0.01 when different dataset, 0.1 on watermark

            # set important parameters
            if 'NTL_alpha' in dict(config).keys() and 'NTL_beta' in dict(config).keys():
                alpha = config.NTL_alpha
                beta = config.NTL_beta
            else:
                alpha = 0.1
                beta = 0.1
                if epoch == 0 and i == 0:
                    print('use default paras: alpha and beta')

            mmd_loss = MMD_loss()(fe1.view(fe1.size(0), -1), fe2.view(fe2.size(0), -1)) * beta
            loss2 = loss2 * alpha
            if loss2 > 1:
                loss2 = torch.clamp(loss2, 0, 1)#0.01
            if mmd_loss > 1:
                mmd_loss_1 = torch.clamp(mmd_loss, 0, 1)
            else:
                mmd_loss_1 = mmd_loss
            
            loss = loss1 - loss2 * mmd_loss_1
            # loss = loss1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            cnt += 1
            # if i % 100 == 0:
            #     print('Epoch:{0},Frame:{1}, train_loss {2}'.format(
            #         epoch, cnt*config.batch_size, loss.item()))
            #     print('mmd loss: ', mmd_loss.item())
        
        model.eval()
        acc1s, acc5s = [], []
        for evaluator in evaluators:
            eval_results = evaluator(model, device=config.device)
            (acc1, acc5), _ = eval_results['Acc'], eval_results['Loss']
            acc1s.append(acc1)
            acc5s.append(acc5)
        # scheduler.step()
        # is_best = acc1 > best_acc1
        # best_acc1 = max(acc1, best_acc1)

        wandb_log_dict = {
            'epoch': epoch,
            'loss': loss.item(),
            'mmd_loss': mmd_loss.item(),
            'Acc_src': acc1s[0],
        }
        ########## insert log of mddloss 

        print('[Train] | epoch %03d | train_loss: %.3f, mmd_loss: %.3f | src_val_acc1: %.1f, tgt_val_acc1: ' %
              (epoch, loss.item(), mmd_loss.item(), acc1s[0]), end='')
        for dname, acc_tgt in zip(datasets_name[1:], acc1s[1:]):
            print(f'{dname}: {acc_tgt:.2f} ', end='')
            wandb_log_dict[f'Acc_tgt_{dname}'] = acc_tgt
        tgt_mean = torch.mean(torch.tensor(acc1s[1:])).item()
        print(f'| tgt_mean: {tgt_mean:.2f}')
        
        wandb.log(wandb_log_dict)

    pass

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

def calc_ins_mean_std(x, eps=1e-5):
        """extract feature map statistics"""
        size = x.size()
        assert (len(size) == 4)
        N, C = size[:2]
        var = x.contiguous().view(N, C, -1).var(dim=2) + eps
        std = var.sqrt().view(N, C, 1, 1)
        mean = x.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return mean, std

class CUTI(nn.Module):
    def __init__(self):
        super(CUTI, self).__init__()
    def forward(self, x):
        if self.training:
            batch_size, C = x.size()[0]//2, x.size()[1]
            style_mean, style_std = calc_ins_mean_std(x[:batch_size])
            conv_mean = nn.Conv2d(C, C, 1, bias=False).cuda()
            conv_std = nn.Conv2d(C, C, 1, bias=False).cuda()
            mean = torch.sigmoid(conv_mean(style_mean))
            std = torch.sigmoid(conv_std(style_std))
            x_a = x[batch_size:]*std+mean
            x = torch.cat((x[:batch_size], x_a), 0)
        return x
    
def train_tCUTI(config, dataloaders, valloaders, model, datasets_name):

    def validate_class(val_loader, model, epoch, num_class=10):
        model.eval()
        correct = 0
        total = 0
        c_class = [0 for i in range(num_class)]
        t_class = [0 for i in range(num_class)]
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            y_pred = model(images)
            _, predicted = torch.max(y_pred.data, 1)
            total += labels.size(0)
            true_label = torch.argmax(labels[:, 0], axis=1)
            correct += (predicted == true_label).sum().item()
            for j in range(predicted.shape[0]):
                t_class[true_label[j]] += 1
                if predicted[j] == true_label[j]:
                    c_class[true_label[j]] += 1
            
        acc = 100.0 * correct / total
        print('   * EPOCH {epoch} | Ave_Accuracy: {acc:.3f}%'.format(epoch=epoch, acc=acc))
        model.train()
        return acc
    
    def forward_CUTI(model, x, y=None, choice=0):
        if y == None:
            x = model.features(x)
            x = x.view(x.size(0), -1)
            x = model.classifier1(x)
            return x
        elif choice % 2 == 0:
            input = torch.cat((x, y), 0)
            input = model.features(input)
            x, y = input.chunk(2, dim=0)

            x = x.view(x.size(0), -1)
            x = model.classifier1(x)

            y = y.view(y.size(0), -1)
            y = model.classifier1(y)

            return x, y

        elif choice % 2 == 1:
            input = torch.cat((x, y), 0)
            input = model.chan_ex(model.features[:3](input))
            input = model.chan_ex(model.features[3:6](input))
            input = model.chan_ex(model.features[6:11](input))
            input = model.chan_ex(model.features[11:16](input))
            input = model.chan_ex(model.features[16:](input))

            x, y = input.chunk(2, dim=0)

            x = x.view(x.size(0), -1)
            x = model.classifier1(x)

            y = y.view(y.size(0), -1)
            y = model.classifier1(y)

            return x, y

    def forward_CUTI_resnetcmi(model, x, y=None, choice=0):
        bs = x.shape[0]
        if y == None:
            return model(x)
        elif choice % 2 == 0:
            input = torch.cat((x, y), 0)
            output = model(input)
            x = output[:bs, :]
            y = output[bs:, :]
            return x, y
        elif choice % 2 == 1:
            input = torch.cat((x, y), 0)

            out = model.conv1(input)
            out = model.bn1(out)
            out = F.relu(out)
            # CUTI
            out = model.chan_ex(out)

            out = model.layer1(out)
            # CUTI
            out = model.chan_ex(out)

            out = model.layer2(out)
            # CUTI
            out = model.chan_ex(out)

            out = model.layer3(out)
            # CUTI
            out = model.chan_ex(out)

            out = model.layer4(out)
            # CUTI
            out = model.chan_ex(out)

            out = F.adaptive_avg_pool2d(out, (1,1))
            feature = out.view(out.size(0), -1)
            # output = model.linear(feature)
            output = model.fc(feature)
            
            x = output[:bs, :]
            y = output[bs:, :]
            return x, y
        
    evaluators = [classification_evaluator(v) for v in valloaders]
    
    model.chan_ex = CUTI()
    lr = config.NTL_lr
    print('use learning rate: ', lr)
    print(f'use {config.NTL_optimizer} optimizer')
    if config.NTL_optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif config.NTL_optimizer == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                    momentum=config.NTL_momentum, weight_decay=config.NTL_weight_decay)
    lambda1 = lambda epoch:0.999**epoch
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    criterion_KL = torch.nn.KLDivLoss()

    if config.NTL_network in ['vgg11', 'vgg13', 'vgg19', 'vgg11bn', 'vgg13bn', 'vgg19bn']:
        forward_func = forward_CUTI
    elif config.NTL_network in ['resnet18cmi', 'resnet18cmi_wobn', 'resnet34cmi_wobn', 'resnet34cmi', 'resnet50cmi', 'wide_resnet50_2cmi']:
        forward_func = forward_CUTI_resnetcmi
    # elif config.NTL_network in ['vgg11bncmi', 'vgg13bncmi']:
    #     forward_func = forward_CUTI_vggcmi
    else:
        raise NotImplementedError

    device = config.device
    # cnt = 0
    for epoch in range(config.NTL_epochs):
        model.train()
        for i, zipped in enumerate(zip(dataloaders[0], dataloaders[1])):
            img1 = zipped[0][0].to(device).float()
            label1 = zipped[0][1].to(device).float()
            img2 = zipped[1][0].to(device).float()
            label2 = zipped[1][1].to(device).float()
            
            out1, out2 = forward_func(model, img1, img2, i)

            out1 = F.log_softmax(out1, dim=1)
            loss1 = criterion_KL(out1, label1)

            out2 = F.log_softmax(out2, dim=1)
            loss2 = criterion_KL(out2, label2)

            # set important parameters
            if 'CUTI_alpha' in dict(config).keys():
                alpha = config.CUTI_alpha
            else:
                alpha = 0.1
                # alpha = 0.5
                # print('use default paras: alpha and beta')

            loss2 = loss2 * alpha
            if loss2 > 1:
                loss2 = torch.clamp(loss2, 0, 1)

            loss = loss1 - loss2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        model.eval()
        # acc1 = validate_class(valloaders[0], model, epoch, num_class=config.num_classes)
        # acc2 = validate_class(valloaders[1], model, epoch, num_class=config.num_classes)
        # print("epoch = {:02d}, acc_s = {:.3f}, acc_t = {:.3f}".format(epoch, acc1, acc2) + '\n')

        acc1s, acc5s = [], []
        for evaluator in evaluators:
            eval_results = evaluator(model, device=config.device)
            (acc1, acc5), _ = eval_results['Acc'], eval_results['Loss']
            acc1s.append(acc1)
            acc5s.append(acc5)

        wandb_log_dict = {
            'epoch': epoch,
            'loss': loss.item(),
            'Acc_src': acc1s[0],
            # 'Acc_src': acc1,
        }

        print('[Train CUTI] | epoch %03d | train_loss: %.3f | src_val_acc1: %.1f, tgt_val_acc1: ' %
              (epoch, loss.item(), acc1s[0]), end='')
            # (epoch, loss.item(), acc1), end='')
        for dname, acc_tgt in zip(datasets_name[1:], acc1s[1:]):
        # for dname, acc_tgt in zip(datasets_name[1:], [acc2]):
            print(f'{dname}: {acc_tgt:.2f} ', end='')
            wandb_log_dict[f'Acc_tgt_{dname}'] = acc_tgt
        tgt_mean = torch.mean(torch.tensor(acc1s[1:])).item()
        print(f'| tgt_mean: {tgt_mean:.2f}')
        wandb.log(wandb_log_dict)
    pass
