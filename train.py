import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import thop
import pandas as pd

from albumentations.augmentations import transforms
from albumentations.augmentations import geometric

from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize

from model import BMFSegNet
import losses
from dataset import Dataset

from metrics import iou_score, indicators
from utils import AverageMeter, str2bool

from tensorboardX import SummaryWriter

import shutil
import os
import subprocess

from pdb import set_trace as st
from dataset import get_train_val_test_loader_from_train, get_train_val_test_indices

LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    parser.add_argument('--dataseed', default=2981, type=int,
                        help='')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='BMFSegNet')
    
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=128, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=128, type=int,
                        help='image height')
    parser.add_argument('--input_list', type=list_type, default=[128, 256])

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='busi', help='dataset name')      
    parser.add_argument('--data_dir', default='inputs', help='dataset dir')

    parser.add_argument('--output_dir', default='outputs', help='ouput dir')


    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')

    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
                        
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')


    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument('--num_workers', default=4, type=int)




    config = parser.parse_args()

    return config

class BoundaryLoss(nn.Module):
    def __init__(self, inner_expand=3, outer_expand=3, cosine_weight=1.0, visualize=False):
        super().__init__()
        self.inner_expand = inner_expand
        self.outer_expand = outer_expand
        self.cosine_weight = cosine_weight
    
    def extract_boundary(self, mask):
        sobel_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=7)
        sobel_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=7)
        edge = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        edge = (edge > 0).astype(np.uint8)
        return edge

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        target_np = target.cpu().numpy().astype(np.uint8)

        boundary_loss = 0
        cosine_loss = 0
        batch_size = target.shape[0]

        for i in range(batch_size):
            gt_mask = target_np[i, 0]
            pred_mask = (pred[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)

            gt_edge = self.extract_boundary(gt_mask)
            pred_edge = self.extract_boundary(pred_mask)

            gt_inner = binary_dilation(gt_edge, iterations=self.inner_expand) & (gt_mask == 1)
            gt_outer = binary_dilation(gt_edge, iterations=self.outer_expand) & (gt_mask == 0)

            pred_inner = binary_dilation(pred_edge, iterations=self.inner_expand) & (pred_mask == 1)
            pred_outer = binary_dilation(pred_edge, iterations=self.outer_expand) & (pred_mask == 0)

            gt_inner_tensor = torch.tensor(gt_inner, dtype=torch.float32, device=pred.device)
            gt_outer_tensor = torch.tensor(gt_outer, dtype=torch.float32, device=pred.device)
            pred_inner_tensor = torch.tensor(pred_inner, dtype=torch.float32, device=pred.device)
            pred_outer_tensor = torch.tensor(pred_outer, dtype=torch.float32, device=pred.device)

            inner_loss_gt = F.mse_loss(pred[i, 0] * gt_inner_tensor, torch.ones_like(pred[i, 0]) * gt_inner_tensor)
            outer_loss_gt = F.mse_loss(pred[i, 0] * gt_outer_tensor, torch.zeros_like(pred[i, 0]) * gt_outer_tensor)

            target_tensor = torch.tensor(gt_mask, dtype=torch.float32, device=pred.device)
            inner_loss_pred = F.mse_loss(target_tensor * pred_inner_tensor, torch.ones_like(target_tensor) * pred_inner_tensor)
            outer_loss_pred = F.mse_loss(target_tensor * pred_outer_tensor, torch.zeros_like(target_tensor) * pred_outer_tensor)

            boundary_loss += inner_loss_gt + outer_loss_gt 

            pred_flat = pred[i, 0].view(-1)
            target_flat = target_tensor.view(-1)
            cosine_loss += 1 - F.cosine_similarity(pred_flat, target_flat, dim=0).mean()

        final_loss = (boundary_loss / batch_size) + self.cosine_weight * (cosine_loss / batch_size)
        return final_loss

def train(config, train_loader, model, criterion, boundary_loss, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'boundary_loss': AverageMeter()} 

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            boundary_loss_value = 0
            for output in outputs:
                loss += criterion(output, target)
                boundary_loss_value += boundary_loss(output, target) 

            loss /= len(outputs)
            boundary_loss_value /= len(outputs)

            total_loss = loss + boundary_loss_value 
            iou, dice, _ = iou_score(outputs[-1], target)

        else:
            output = model(input)
            loss = criterion(output, target)
            boundary_loss_value = boundary_loss(output, target) 

            total_loss = loss + boundary_loss_value  
            #total_loss = loss
            iou, dice, _ = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['boundary_loss'].update(boundary_loss_value.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('boundary_loss', avg_meters['boundary_loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('boundary_loss', avg_meters['boundary_loss'].avg),
        ('iou', avg_meters['iou'].avg),
    ])

def validate(config, val_loader, model, criterion, boundary_loss):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'boundary_loss': AverageMeter()} 

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config.get('deep_supervision', False):
                outputs = model(input)
                loss = 0
                boundary_loss_value = 0
                for output in outputs:
                    loss += criterion(output, target)
                    boundary_loss_value += boundary_loss(output, target)  
                loss /= len(outputs)
                boundary_loss_value /= len(outputs)
                iou, dice, _ = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                boundary_loss_value = boundary_loss(output, target)  
                iou, dice, _ = iou_score(output, target)

            total_loss = loss + boundary_loss_value
            #total_loss = loss

            avg_meters['loss'].update(total_loss.item(), input.size(0))
            avg_meters['boundary_loss'].update(boundary_loss_value.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('boundary_loss', avg_meters['boundary_loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('boundary_loss', avg_meters['boundary_loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg)
    ])

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    seed_torch()
    config = vars(parse_args())

    exp_name = config.get('name')
    output_dir = config.get('output_dir')

    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    
    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f)
        
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    boundary_loss = BoundaryLoss().cuda()

    cudnn.benchmark = True

    # create model
    model = BMFSegNet(in_chans=3, out_chans=1, depths=[2,2,2,2], feat_size=[48, 96, 192, 384])
    model = model.cuda()
    param_groups = [{'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']} for param in model.parameters()]

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)
    elif config['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(param_groups)
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError
    dataset_name = config['dataset']
    img_ext = '.png'

    if dataset_name == 'busi-1':
        mask_ext = '_mask.png'
    elif dataset_name == 'Dataset001_COVID-19':
        mask_ext = '.png'
    elif dataset_name == 'BUS':
        mask_ext = '.png'
    elif dataset_name == 'PH2':
        mask_ext = '.png'
   
    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    data = '……/BUS/images'
    train_img_ids, val_img_ids, test_img_ids = get_train_val_test_indices(data)

    train_transform = Compose([
        RandomRotate90(),
        geometric.transforms.Flip(),
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_dir'] ,config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], drop_last=False)

    log = OrderedDict([('epoch', []), ('lr', []), ('loss', []), ('boundary_loss', []), ('iou', []), ('val_loss', []), ('val_iou', [])])

    best_iou = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        train_log = train(config, train_loader, model, criterion, boundary_loss, optimizer)
        val_log = validate(config, val_loader, model, criterion, boundary_loss)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - boundary_loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['boundary_loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model.pth')
            best_iou = val_log['iou']
            print("=> saved best model")

        torch.cuda.empty_cache()
    
if __name__ == '__main__':
    main()
