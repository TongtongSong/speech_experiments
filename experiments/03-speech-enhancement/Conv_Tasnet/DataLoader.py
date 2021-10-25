import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import ast
from Loss import cal_snr
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torchaudio as ta 

def normalization(feature):
    std, mean = torch.std_mean(feature)
    return (feature - mean) / std


def handle_scp(scp_path):
    '''
    Read scp file script
    input: 
          scp_path: .scp file's file path
    output: 
          scp_dict: {'key':'wave file path'}
    '''
    scp_dict = dict()
    line = 0
    lines = open(scp_path, 'r').readlines()
    for l in lines:
        scp_parts = l.strip().split(" ", 1)
        line += 1
        if len(scp_parts) != 2:
            raise RuntimeError("For {}, format error in line[{:d}]: {}".format(
                scp_path, line, scp_parts))
        if len(scp_parts) == 2:
            key, value = scp_parts
        if key in scp_dict:
            raise ValueError("Duplicated key \'{0}\' exists in {1}".format(
                key, scp_path))

        scp_dict[key] = value

    return scp_dict



def load_audio(index_dict, index, timeLen=3, sr=16000):
    '''
    load audio data
    '''
    keys = list(index_dict.keys())
    key = ''
    if type(index) not in [int, str]:
        raise IndexError('Unsupported index type: {}'.format(type(index)))
    elif type(index) == int:
        num_uttrs = len(index_dict)
        if(num_uttrs <= index or index < 0):
            raise KeyError('Interger index out of range,suppose to get 0 to {:d} \
but get {:d}'.format(num_uttrs-1, index))
        key = keys[index]

    audio_data, _=librosa.load(index_dict[key], sr=sr)
    audio_data = audio_data[0:_*3]
    audio_data = np.expand_dims(audio_data, axis=0)

    # if ISMIX == False:

    #     name = key.split('+')[1]
    #     name = name.replace("_norm_", "_")
    #     name = name.replace(".wav", "")
    #     if name in ASR_dict.keys():
    #         audio_data_truth, _ = librosa.load(ASR_dict[name], sr=sr)
    #         audio_data_truth = audio_data_truth
    #         audio_data_truth = np.expand_dims(audio_data_truth, axis=0)
    #     else:
    #         audio_data_truth = audio_data
    #     audio_data_truth = torch.from_numpy(audio_data_truth)
    #     # ft = ta.compliance.kaldi.fbank(audio_data_truth, num_mel_bins=40, sample_frequency=16000, dither=0.0)
        
    #     return audio_data
    # else:
        # audio_data = torch.tensor(audio_data,dtype=torch.float).unsqueeze(0)
    return audio_data




class AudioDataset(Dataset):
    '''
    Load audio data
    batch_size=:
    shuffle=:
    num_works=:
    '''

    def __init__(self, mix_scp=None, s1_scp=None ,sr=16000):
        super(AudioDataset, self).__init__()
        self.mix_audio = handle_scp(mix_scp)
        self.s1_audio = handle_scp(s1_scp)


        self.sr = sr

    def __len__(self):
        return len(self.mix_audio)

    def __getitem__(self, index):
        return{
            'mix': load_audio(self.mix_audio, index, sr=self.sr),
            's1': load_audio(self.s1_audio, index, sr=self.sr),
        }



def collate_fn(dataBatch):

    mix = [torch.tensor(data['mix']) for data in dataBatch]
    mix = torch.cat(mix, dim=0).float().unsqueeze(1)

    s1 = [data['s1'] for data in dataBatch]
    

    audio_s1 = [torch.tensor(data) for data in s1]
    audio_s1 = torch.cat(audio_s1, dim=0).float().unsqueeze(1)

    


    return {
        'mix': mix,
        's1':  audio_s1
    }


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)


if __name__ == '__main__':
    # vdataset = VisualDataset('./data/visual/test_s1.scp','./data/visual/test_s2.scp')

    # vdataloader = VisualDataloder(vdataset,num_workers=0,batch_size=1,shuffle=True)

    # for eg in dataloader:
    #     print(eg['s1'])
    #     break

    dataset = AudioDataset('./data/audio/test/noisy.scp',
                           './data/audio/test/clean.scp')
    dataLoader = AudioDataLoader(dataset, batch_size=2,collate_fn=collate_fn)
    import soundfile

    for eg in dataLoader:
        mix = eg['mix']
        s1 = eg['s1']
        print(s1)
        # s1 = s1 * torch.max(torch.abs(mix)) / torch.max(torch.abs(s1))
        # s1 = s2 * torch.max(torch.abs(mix)) / torch.max(torch.abs(s2))

        # s1 = s1[0].permute(1, 0).numpy()

        # soundfile.write('xx1.wav', s1, samplerate=16000)
        # soundfile.write('xx2.wav',s2,samplerate=16000)
        break

    # print(len(dataset))

    # for audio in dataLoader:
    #     s1 = audio['s1']
    #     s2 = audio['s2']
    #     mix = audio['mix']

    #     print(cal_snr(s1,s2))
    #     print(cal_snr(s2,mix))
    #     print(cal_snr(s1,mix))
    #     print('---------------------')
