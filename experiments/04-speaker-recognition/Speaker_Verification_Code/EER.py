import os
import time
import argparse
import warnings
import pandas as pd

import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as AUC
import matplotlib.pyplot as plt
import numpy as np

from xvector import Xvector
from ecapa import ECAPA_TDNN
from datasets import SpkTestDataset

import yaml
import torch

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

parser = argparse.ArgumentParser()
# Loading setting
parser.add_argument('--gpu', type=int, default=0, help='GPU device number.')
parser.add_argument('--n_folder', type=int, default=0, help='Number of folder.')
parser.add_argument('--cp_num', type=int, default=100, help='Number of checkpoint.')
parser.add_argument('--data_type', type=str, default='vox2', help='vox1 or vox2.')

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(str(device))
f = open('config.yaml', 'r')
opts = yaml.load(f)['test']
f.close()
log_dir = opts['model_path']

def main():
    testset = SpkTestDataset(opts)
    test_loader =DataLoader(testset, batch_size = 100, shuffle = True, collate_fn = testset.collate_fn)

    print('Load checkpoint')

    # Load model from checkpoint
    # ==================> code <======================
    # =============== 定义并导入模型 ==================




    # ==================> code <======================
    
    model.eval()

    dict_embedding = {}
    # Enroll and test
    for t, (data) in enumerate(test_loader):
        feats, labels = data

        feats = feats.to(device)
        # ==================> code <======================
        # ============ 为测试数据生成深度嵌入 ==============

        # ==================> code <======================

        for i in range(embeddings.shape[0]):
            dict_embedding[labels[i]] = embeddings[i].cpu()

    verification(opts, dict_embedding)


def verification(opts, dict_embedding):
    f = open(opts['trials'], 'r')
    score_list = []
    label_list = []
    num = 0

    while True:
        line = f.readline()
        if not line: break
        print(line)

        enroll_filename, test_filename, label = line.split(' ')[0], line.split(' ')[1], int(line.split(' ')[2])
        with torch.no_grad():
            enroll_embedding = dict_embedding[enroll_filename].unsqueeze(0)
            test_embedding = dict_embedding[test_filename].unsqueeze(0)

            # ==================> code <======================
            # ================= 计算相似度 ====================

            # ==================> code <======================
            score = score.data.cpu().numpy()[0]
            del enroll_embedding
            del test_embedding
        
        score_list.append(score)
        label_list.append(label)
        num += 1

    f.close()

    roc(score_list, label_list)
    eer, eer_threshold = get_eer(score_list, label_list)


def get_eer(score_list, label_list):
    fpr, tpr, threshold = roc_curve(label_list, score_list, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    intersection = abs(1 - tpr - fpr)
    print("Epoch=%d  EER= %.2f  Thres= %0.5f" % (
    args.cp_num, 100 * fpr[np.argmin(intersection)], eer_threshold))

    return eer, eer_threshold

def roc(score_list, label_list):
    # ==================> code <======================
    # ================ 绘制roc曲线 ===================


    
    pass
    # ==================> code <======================



if __name__ == '__main__':
    main()