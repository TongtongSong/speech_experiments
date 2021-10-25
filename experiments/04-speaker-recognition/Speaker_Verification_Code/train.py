from ecapa import ECAPA_TDNN
import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
import argparse
import yaml

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim as optim

from datasets import SpkTrainDataset
from xvector import Xvector
from loss import *


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU device number.')
parser.add_argument('--n_folder', type=int, default=0, help='Number of folder.')

parser.add_argument('--loss_type', type=str, default='softmax', help='softmax or amsoftmax.')

parser.add_argument('--batchsize', type=int, default=64, help='Batchsize.')
parser.add_argument('--speaker_size', type=int, default=300, help='Speaker size.')

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(str(device))
log_dir = 'saved_model/' + str(args.n_folder).zfill(3)  # where to save checkpoints
f = open('config.yaml', 'r')
opts = yaml.load(f, Loader=yaml.FullLoader)['data']
f.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 定义数据的 dataloader
    trainset = SpkTrainDataset(opts)
    train_loader = DataLoader(trainset, batch_size = args.batchsize, shuffle = True, collate_fn = trainset.collate_fn)

    # ===================> code <=======================
    # 定义模型


    # 定义损失函数


    # 定义学习器


    # ===================> code <=======================
    
    # training
    train(train_loader, model, criterion, optimizer, log_dir)


def train(train_loader, model, criterion, optimizer, log_dir):
    for epoch in range(opts['epoch']):
        model.train()

        losses = AverageMeter()
        accuracy = AverageMeter()

        for t, (data) in enumerate(train_loader):
            feats, labels = data
            feats = feats.to(device)
            labels = labels.to(device)

            # ===================> code <=======================





            # ===================> code <=======================

            losses.update(loss.item(), feats.size(0))
            accuracy.update(acc, feats.size(0))

            print(
                'Train Stage: {epoch}/{t} ===>\t Loss {losses.avg:.3f}\t Acc {accuracy.avg:.3f}'.format(epoch=epoch, t=t, losses=losses, accuracy=accuracy)
            )

        if epoch % 10 == 0:
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict()}, '{}/checkpoint_{}.pth'.format(log_dir, str(epoch).zfill(3)))


main()


