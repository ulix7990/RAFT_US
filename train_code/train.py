from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raft import RAFT
import evaluate
import datasets

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def unsupervised_loss(flow_preds, image1, image2, gamma=0.8, smoothness_weight=0.1):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    photometric_loss = 0.0
    smoothness_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        
        # Photometric loss
        flow = flow_preds[i]
        height, width = flow.shape[2:]
        grid = coords_grid(flow.shape[0], height, width, device=flow.device)
        warped_grid = grid + flow
        # Normalize grid to [-1, 1]
        warped_grid[:, 0, :, :] = 2.0 * warped_grid[:, 0, :, :].clone() / max(width - 1, 1) - 1.0
        warped_grid[:, 1, :, :] = 2.0 * warped_grid[:, 1, :, :].clone() / max(height - 1, 1) - 1.0
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        warped_image2 = F.grid_sample(image2, warped_grid, padding_mode='border')
        photometric_loss += i_weight * (image1 - warped_image2).abs().mean()

        # Smoothness loss
        flow_grad_x = (flow[:, :, :, :-1] - flow[:, :, :, 1:]).abs()
        flow_grad_y = (flow[:, :, :-1, :] - flow[:, :, 1:, :]).abs()
        smoothness_loss += i_weight * (flow_grad_x.mean() + flow_grad_y.mean())

    total_loss = photometric_loss + smoothness_weight * smoothness_loss

    metrics = {
        'photometric_loss': photometric_loss.item(),
        'smoothness_loss': smoothness_loss.item(),
        'total_loss': total_loss.item(),
    }

    return total_loss, metrics

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    if args.unsupervised:
        train_loader = datasets.UnsupervisedFlowDataset(root=args.dataset_root)
    else:
        train_loader = datasets.fetch_dataloader(args)
        
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 5000
    add_noise = True

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2 = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)

            if args.unsupervised:
                loss, metrics = unsupervised_loss(flow_predictions, image1, image2, args.gamma)
            else:
                flow, valid = [x.cuda() for x in data_blob[2:]]
                loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                if not args.unsupervised:
                    results = {}
                    for val_dataset in args.validation:
                        if val_dataset == 'chairs':
                            results.update(evaluate.validate_chairs(model.module))
                        elif val_dataset == 'sintel':
                            results.update(evaluate.validate_sintel(model.module))
                        elif val_dataset == 'kitti':
                            results.update(evaluate.validate_kitti(model.module))

                    logger.write_dict(results)
                
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--unsupervised', action='store_true', help='use unsupervised training')
    parser.add_argument('--dataset_root', type=str, default='datasets/Sintel', help='root directory of the dataset')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)