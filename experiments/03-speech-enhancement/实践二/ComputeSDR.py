# 计算多通道音频之间的SDR
# 多通道音频的SDR计算  一般提取第0个通道数据作为参考 然后计算单通道音频之间的SDR
from SDR import cal_sdr
import librosa as lb
import soundfile as sf
import numpy as np


mixture = "./clean_20_unsteadyNoise_10.wav"
clean = "./clean_mul_20.wav"
enh = "./estimate_clean_20_unsteadyNoise_10.wav"

mixture_data, sr = sf.read(mixture)
clean_data, sr = sf.read(clean)
enh_data, sr = sf.read(enh)


flag = True  # 是否保存参考通道音频


mixture_data = mixture_data[:, 0]  # 只取参考通道 即第一个通道
if flag:
    sf.write('./clean_170_unsteadyNoise_10_0.wav',
             mixture_data, samplerate=16000)

mixture_data = mixture_data.reshape(1, 1, mixture_data.shape[0])


clean_data = clean_data[:, 0]
if flag:
    sf.write('./clean_mul_170_0.wav', clean_data, samplerate=16000)
clean_data = clean_data.reshape(1, 1, clean_data.shape[0])


enh_data = enh_data.reshape(1, 1, enh_data.shape[0])
print(cal_sdr(clean_data, mixture_data))

# 因为多通道增强后数据长度可能会发生变化 以短的音频数据为基准就可以
print(cal_sdr(clean_data[:, :, 0:241664], enh_data))
