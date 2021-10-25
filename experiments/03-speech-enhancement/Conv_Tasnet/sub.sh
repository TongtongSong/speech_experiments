#!/bin/bash
#$ -S /bin/bash

#here you'd best to change testjob as username
#$ -N separate

#cwd define the work environment,files(username.o) will generate here
#$ -cwd

# merge stdo and stde to one file
#$ -j y

export PATH=/Work18/2020/lijunjie/anaconda3/envs/torch1.7/bin:$PATH

echo "job start time: `date`"
# start whatever your job below, e.g., python, matlab, etc.
#ADD YOUR COMMAND HERE,LIKE python3 main.py
#chmod a+x run.sh

CUDA_VISIBLE_DEVICES=3 python separate.py

echo "job end time:`date`"
