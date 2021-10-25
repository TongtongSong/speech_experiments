#author :JunjieLi
#createtime:2021/01

from AVModel import AVModel
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import os 
import time 
import torch.nn as nn
import logging 
import argparse
from Solver import Solver

#set the seed for generating random numbers. 
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)




def main(args,use_gpu):

    logFileName = './log/train_lr'+str(args.lr)+'.log'
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(logFileName,mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    logger.info("***batch_size=%d***"%args.batch_size)


    if args.continue_from:
        files = os.listdir('./log/model')
        files.sort()
        model_name =files[-1] 
        model_saved_path="./log/model/"+model_name
    else:
        model_saved_path='' #means not to load model 

    model = AVModel()
    model = nn.DataParallel(model)

    if use_gpu:
        model.cuda()

    optimizer = optim.Adam([{'params':model.parameters()}],lr=args.lr,weight_decay=1e-5)

    solver = Solver(args,model=model,use_gpu=use_gpu,optimizer=optimizer,logger=logger)
    solver.train()

if __name__=='__main__':

    parser = argparse.ArgumentParser('AVConv-TasNet')
    
    #training
    parser.add_argument('--batch_size',type=int,default=1,help='Batch size')
    parser.add_argument('--num_workers',type=int,default=16,help='number of workers to generate minibatch')
    parser.add_argument('--num_epochs',type=int,default=500,help='Number of maximum epochs')
    parser.add_argument('--lr',type=float,default=1e-3,help='Init learning rate')
    parser.add_argument('--continue_from',default=False,action='store_true')


    args = parser.parse_args()

    use_gpu= torch.cuda.is_available()
    os.makedirs('./log/model/',exist_ok=True)


    main(args,use_gpu)
