import numpy as np
from python_speech_features import mfcc,delta
import scipy.io.wavfile as wav
import logging
def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

def extract_feat(wav_path):
    (rate, sig) = wav.read(wav_path)
    feat = mfcc(sig,rate,numcep=13) # numcep < 26
    return feat

def get_all_feats(wav_scp):
    feats = []
    with open(wav_scp, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            feat = extract_feat(line[1])
            feats.append(feat)
    return np.concatenate(feats, axis = 0)

def Dataloader(wav_scp, text):
    utt2feat = {}
    class2utt = {}
    with open(wav_scp,'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            utt2feat[line[0]] = line[1]
    with open(text, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            if line[1] not in class2utt.keys():
                class2utt[line[1]] = [line[0]]
            else:
                class2utt[line[1]].append(line[0])
    return class2utt, utt2feat

def get_feats(class_, class2utt, utt2wav):
    feats = []
    for utt in class2utt[class_]:
        feat = extract_feat(utt2wav[utt])
        feats.append(feat)
    return np.concatenate(feats, axis = 0)


