from random import sample
import soundfile as sf
from python_speech_features import mfcc, fbank, logfbank, delta
from librosa import stft, magphase
from torch.utils.data import Dataset
import torch
import random
import numpy as np
import os
# from torch.utils.data import DataLoader


class SpkTrainDataset(Dataset):
    def __init__(self, opts):
        self.opts = opts
        self.TRAIN_MANIFEAT = opts["train_manifest"]
        self.feat_type = opts['feat_type']
        self.root = opts['root']
        self.rate = opts['rate']
        self.data_feat, self.data_label, self.speaker_dict = self._load_dataset()
        self.count = len(self.data_feat)

    def _load_dataset(self):
        with open(self.TRAIN_MANIFEAT, 'r') as f:
            all_data = f.read()
        all_data = all_data.split('\n')

        data_feat = []
        data_label = []
        speaker_dict = {}
        speaker_id = 0
        for data in all_data:
            if data == '':
                continue
            speaker, path = data.split(' ')[0], data.split(' ')[1]

            if speaker not in speaker_dict.keys():
                speaker_dict[speaker] = speaker_id
                speaker_id += 1

            data_label.append(speaker_dict[speaker])
            
            source_feat, _ = self._load_audio(os.path.join(self.root, path))
            if source_feat[0] == None:
                print('a')
            feat = self._extract_feature(source_feat)
            feat = feat.astype(np.float32)
            feat = np.array(feat)
            feat = self._fix_length(feat)
            data_feat.append(feat)
        
        data_feat = np.array(data_feat)
        data_label = np.array(data_label)

        return data_feat, data_label, speaker_dict

    def _fix_length(self, feat):
        out_feat = feat

        while out_feat.shape[0] < self.opts['max_length']:
            out_feat = np.concatenate((out_feat, feat), axis=0)
        
        feat_len = out_feat.shape[0]
        start = random.randint(a=0, b=feat_len-self.opts['max_length'])
        end = start + self.opts['max_length']
        out_feat = out_feat[start:end, :]
        return out_feat

    def _load_audio(self, path):
        y, sr = sf.read(path)
        return y, sr

    def _normalize(self, feat):
        return (feat - feat.mean(axis = 0)) / (feat.std(axis = 0) + 2e-12)

    def _delta(self, feat, order = 2):
        if order == 2:
            feat_d1 = delta(feat, N = 1)
            feat_d2 = delta(feat, N = 2)
            feat = np.hstack([feat, feat_d1, feat_d2])
        elif order == 1:
            feat_d1 = delta(feat, N = 1)
            feat = np.hstack([feat, feat_d1])
        return feat

    def _extract_feature(self, data):
        if self.feat_type == 'mfcc':
            feat = mfcc(data, self.rate, winlen = self.opts['win_len'], winstep = self.opts['win_shift'], numcep = self.opts['num_cep'])
        elif self.feat_type == 'fbank':
            feat, _ = fbank(data, self.rate, winlen = self.opts['win_len'], winstep = self.opts['win_shift'], nfilt = self.opts['num_bin'])
        elif self.feat_type == 'logfbank':
            feat = logfbank(data, self.rate, winlen = self.opts['win_len'], winstep = self.opts['win_shift'], nfilt = self.opts['num_bin'])
        elif self.feat_type == 'stft':
            feat = stft(data, n_fft = self.opts['n_fft'], hop_length = int(self.rate * self.opts['win_shift']), win_length = int(self.rate * self.opts['win_len']))
            feat, _ = magphase(feat)
            feat = np.log1p(feat)
            feat = feat.transpose(1,0)
        else:
            raise NotImplementedError("Other features are not implemented!")
        if self.opts['normalize']:
            feat = self._normalize(feat)
        if self.opts['delta']:
            feat = self._delta(feat, order = 2)
        return feat

    def collate_fn(self, batch):
        feats = []
        labels = []
        for id in batch:
            feat = self.data_feat[id]
            label = self.data_label[id]
            feats.append(feat)
            labels.append(label)
        feats = np.array(feats).astype(np.float32)
        labels = np.array(labels).astype(np.int64)
        return torch.from_numpy(feats).transpose_(1, 2), torch.from_numpy(labels)


    # def collate_fn(self, batch):
    #     frame = random.randint(self.lower_frame_num, self.higher_frame_num) # random select a frame number in uniform distribution
    #     duration = (frame - 1) * self.opts['win_shift'] + self.opts['win_len'] # duration in time of one training speech segment
    #     samples_num = int(duration * self.rate) # duration in sample point of one training speech segment
    #     feats = []
    #     for sid in batch:
    #         speaker = self.dataset[sid]
    #         y = []
    #         n_samples = 0
    #         while n_samples < samples_num:
    #             aid = random.randrange(0, len(speaker))
    #             audio = speaker[aid]
    #             t, sr = audio[1], audio[2]
    #             samples_len = int(t * sr)
    #             start = int(random.uniform(0, t) * sr) # random select start point of speech
    #             _y, _ = self._load_audio(audio[0], start = start, stop = samples_len) # read speech data from start point to the end
    #             if _y is not None:
    #                 y.append(_y)
    #                 n_samples += len(_y)
    #         y = np.hstack(y)[:samples_num]
    #         feat = self._extract_feature(y)
    #         feats.append(feat)
    #     feats = np.array(feats).astype(np.float32)
    #     labels = np.array(batch).astype(np.int64)
    #     return torch.from_numpy(feats).transpose_(1, 2), torch.from_numpy(labels)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        idx = idx % self.count
        return idx




class SpkTestDataset(Dataset):
    def __init__(self, opts):
        self.opts = opts
        self.TEST_MANIFEAT = opts["test_manifest"]
        self.feat_type = opts['feat_type']
        self.root = opts['root']
        self.rate = opts['rate']
        self.data_feat, self.data_label, self.speaker_dict = self._load_dataset()
        self.count = len(self.data_feat)

    def _load_dataset(self):
        with open(self.TEST_MANIFEAT, 'r') as f:
            all_data = f.read()
        all_data = all_data.split('\n')

        data_feat = []
        data_label = []
        speaker_dict = {}
        speaker_id = 0
        for data in all_data:
            if data == '':
                continue
            speaker, path = data.split(' ')[0], data.split(' ')[1]

            if speaker not in speaker_dict.keys():
                speaker_dict[speaker] = speaker_id
                speaker_id += 1

            data_label.append(path)
            
            source_feat, _ = self._load_audio(os.path.join(self.root, path))
            if source_feat[0] == None:
                print('a')
            feat = self._extract_feature(source_feat)
            feat = feat.astype(np.float32)
            feat = np.array(feat)
            feat = self._fix_length(feat)
            data_feat.append(feat)
        
        data_feat = np.array(data_feat)
        data_label = np.array(data_label)

        return data_feat, data_label, speaker_dict

    def _fix_length(self, feat):
        out_feat = feat

        while out_feat.shape[0] < self.opts['max_length']:
            out_feat = np.concatenate((out_feat, feat), axis=0)
        
        feat_len = out_feat.shape[0]
        start = random.randint(a=0, b=feat_len-self.opts['max_length'])
        end = start + self.opts['max_length']
        out_feat = out_feat[start:end, :]
        return out_feat

    def _load_audio(self, path):
        y, sr = sf.read(path)
        return y, sr

    def _normalize(self, feat):
        return (feat - feat.mean(axis = 0)) / (feat.std(axis = 0) + 2e-12)

    def _delta(self, feat, order = 2):
        if order == 2:
            feat_d1 = delta(feat, N = 1)
            feat_d2 = delta(feat, N = 2)
            feat = np.hstack([feat, feat_d1, feat_d2])
        elif order == 1:
            feat_d1 = delta(feat, N = 1)
            feat = np.hstack([feat, feat_d1])
        return feat

    def _extract_feature(self, data):
        if self.feat_type == 'mfcc':
            feat = mfcc(data, self.rate, winlen = self.opts['win_len'], winstep = self.opts['win_shift'], numcep = self.opts['num_cep'])
        elif self.feat_type == 'fbank':
            feat, _ = fbank(data, self.rate, winlen = self.opts['win_len'], winstep = self.opts['win_shift'], nfilt = self.opts['num_bin'])
        elif self.feat_type == 'logfbank':
            feat = logfbank(data, self.rate, winlen = self.opts['win_len'], winstep = self.opts['win_shift'], nfilt = self.opts['num_bin'])
        elif self.feat_type == 'stft':
            feat = stft(data, n_fft = self.opts['n_fft'], hop_length = int(self.rate * self.opts['win_shift']), win_length = int(self.rate * self.opts['win_len']))
            feat, _ = magphase(feat)
            feat = np.log1p(feat)
            feat = feat.transpose(1,0)
        else:
            raise NotImplementedError("Other features are not implemented!")
        if self.opts['normalize']:
            feat = self._normalize(feat)
        if self.opts['delta']:
            feat = self._delta(feat, order = 2)
        return feat

    def collate_fn(self, batch):
        feats = []
        labels = []
        for id in batch:
            feat = self.data_feat[id]
            label = self.data_label[id]
            feats.append(feat)
            labels.append(label)
        feats = np.array(feats).astype(np.float32)
        labels = np.array(labels)
        return torch.from_numpy(feats).transpose_(1, 2), labels


    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        idx = idx % self.count
        return idx





# import yaml
# f = open('config.yaml', 'r')
# opts = yaml.load(f)['data']
# f.close()
# trainset = SpkTrainDataset(opts)
# train_loader = DataLoader(trainset, batch_size = 128, shuffle = True, collate_fn = trainset.collate_fn)
# for feat, label in enumerate(train_loader):
#     pass