'''
谱减法 基础版
'''

import numpy as np
import librosa
import math
from scipy.io import wavfile


mixture_file = "./mixture-steadyNoise51874.wav"
output_file = "estimate"+mixture_file.split("mixture")[-1]
# audio_data 音频数据  sr 音频采样率
audio_data, sr = librosa.load(mixture_file, sr=16000)


stft_audio = librosa.stft(audio_data, n_fft=512)  # 对音频信号进行短时傅里叶变换
# n_fft 可以自行设置 stft_audio的shape为(1+n_fft/2,n_frames) shape[0]代表frequency bin长度 shape[1]代表时间帧长度
# 可以自行参考
# http://librosa.org/doc/main/generated/librosa.stft.html
mag_audio = np.abs(stft_audio)  # 幅度谱
pha_audio = np.angle(stft_audio)  # 相位谱

# 噪声幅度计算 假设前5帧为silence(noise)  也可以采用其他方式估计噪声
noise_mean = np.zeros((mag_audio.shape[0],))
for i in range(0, 5):
    noise_mean += mag_audio[:,i]
noise_mean /= 5  # 取平均


for i in range(mag_audio.shape[1]):

    mag_audio[:,i] = mag_audio[:,i] - noise_mean

mag_audio_ = np.where(mag_audio > 0, mag_audio, 0)  # 大于0的部分保持不变 负数取0

stft_audio_ = mag_audio_ * np.exp(1.0j*pha_audio)  # 利用原始相位信息进行逆傅里叶变换变换

wav_data = librosa.istft(stft_audio_)

wavfile.write(output_file, sr, (wav_data * 32768).astype(np.int16))
