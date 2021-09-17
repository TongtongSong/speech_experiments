# Feature extractor
# Author: Tongtong Song (TJU)
# Date: 2021/9/1 15:00
# Last modified: 2021/9/5 15:49

import os
import librosa
import numpy as np
from scipy.fftpack import dct
import matplotlib
import matplotlib.pyplot as plt

np.seterr(divide='ignore',invalid='ignore')
def plt_wav(wav,label):
    plt.figure(figsize=(20, 5))
    x = np.arange(0, len(wav), 1)
    plt.plot(x, wav)
    plt.xlabel('T(s)')
    plt.title(label)
    plt.tight_layout()
    plt.savefig("result/" + label + ".png")

def plt_envelope(log_mag_spectrum,envelope,label):

    plt.figure(figsize=(20, 5))
    x = np.arange(0, len(log_mag_spectrum), 1)
    plt.plot(x, log_mag_spectrum)
    plt.plot(x,envelope,c='r')
    plt.xlabel('Freq')
    plt.title(label)
    plt.tight_layout()
    plt.savefig("result/" + label + ".png")

def plt_spectrogram(spec, label):
    """Draw spectrogram
    """
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('T')
    plt.title(label)
    plt.tight_layout()
    plt.savefig("result/"+label+".png")

def preemphasis(signals, alpha=0.95):
    """preemphasis on the input signal.
        x'[n] = x[n] - alpha*x[n-1]
    :param signals: original signals
    :param alpha: The coefficient. 0 is no filter, default is 0.95.
    :return:
        signals: the filtered signal.
    """

    """
        TODO
    """
    return signals

def framesig(signals,frame_len,frame_shift):
    """split signals to frames
        n_frames = (n_samples - frame_length) // frame_shift + 1
    :param signals: signals had pre-emphasised
    :param frame_len: sample number of one frame
    :param frame_shift: sample number to shift
    :return:
        frames:
    """

    """
        TODO
    """
    return frames

def add_windows(frames):
    """
    :param frames: frames to add window
    :return:
        frames: frames that have been processed
        win: window to add on each frame
    """

    """
        TODO
    """
    return frames, win

def get_spectrum(frames):
    """get power spectrum
        you can use np.fft.fft()
        mag_spectrum = |FFT(frame)|
        spectrum_power = (mag_spectrum)**2/n_fft
    :param frames: frames to calculate power spectrum
    :return:
        power_spectrum:
    """
    """
        TODO
    """
    return power_spectrum, log_power_spectrum

def get_fbank(spectrum, sr, n_fliter):
    """
        m = 2595 * log(1 + f/700) # freq to mel
        f = 700 * (10^(m/2595) - 1) # mel to freq
        Hm(k):
            k < f(m-1) or k > f(m+1): 0
            f(m-1) < k < f(m): (k-f(m-1))/(f(m)-f(m-1))
            f(m) < k < f(m+1): (f(m+1)-k)/(f(m+1)-f(m))
    """
    n_fft=int((spectrum.shape[1]-1)*2)
    low_freq = 0
    high_freq = sr//2

    min_mel = 2595 * np.log10(1 + low_freq / 700)
    max_mel = 2595 * np.log10(1 + high_freq / 700)
    mel_points = np.linspace(min_mel, max_mel, n_fliter + 2) # create mel points

    freq_points = 700 * (10 ** (mel_points / 2595) - 1) # mel to freq

    bin = np.floor(freq_points * ((n_fft + 1) / sr))  # freq to fft scale
    fbanks = np.zeros((n_fliter, n_fft // 2 + 1))
    """
        TODO
    """
    feats = np.dot(spectrum, fbanks.T)
    feats = np.where(feats==0,np.finfo(float).eps,feats)
    feats = np.log10(feats)
    return feats


def get_mfcc(fbank,n_mfcc):
    """Get MFCC
      for every frames you can use the following formula:
      f = sqrt(1/(4*N)) if k = 0,
      f = sqrt(1/(2*N)) otherwise.
                   N-1
      y[k] = 2*f * sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
                   n=0
    """
    n_frame,n_fliter = fbank.shape
    assert n_mfcc < n_fliter
    """
        TODO
    """
    # Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    # magnitude of the high frequency DCT coeffs.
    L=22
    feats = feats[:, :n_mfcc]
    nframes, ncoeff = np.shape(feats)
    n = np.arange(ncoeff)
    lift = 1 + (L / 2.) * np.sin(np.pi * n / L)
    feats = lift * feats
    return feats

def get_envelope(frame):
    log_mag_spectrum = np.log(np.abs(np.fft.rfft(frame)))
    """
        TODO
    """
    return log_mag_spectrum,envelope

def main():
    # pre-emphasis config
    alpha = 0.97

    # framesig config
    frame_len = 400  # 25ms, sr=16kHz
    frame_shift = 160  # 10ms, sr=16kHz

    # fbank config
    n_fliter = 40

    # mfcc config
    n_mfcc = 13

    signals, sr = librosa.load('./test.wav',sr=None) #sr=None means using the original audio sampling rate
    plt_wav(signals, '00-original_wave')  # show 10th frame
    plt_wav(signals[1600:2000],'01-10th_frame_wave') # show 10th frame

    signals = preemphasis(signals, alpha)
    plt_wav(signals[1600:2000], '02-10th_frame_preemphasis_wave') # show 10th frame

    frames = framesig(signals, frame_len, frame_shift)

    frames,win = add_windows(frames)
    plt_wav(win, '03-win')
    plt_wav(frames[10], '04-frame_add_win')  # show 10th frame

    power_spectrum,log_power_spectrum = get_spectrum(frames)
    plt_spectrogram(log_power_spectrum.T, '05-log_power_spectrum')

    fbank = get_fbank(power_spectrum, sr, n_fliter)
    plt_spectrogram(fbank.T, '06-fbank')

    mfcc = get_mfcc(fbank,n_mfcc)
    plt_spectrogram(mfcc.T, '07-mfcc')

    ### Choice
    log_mag_spectrum, envelope = get_envelope(frames[10])
    plt_envelope(log_mag_spectrum, envelope, '08-envelope')

if __name__ == '__main__':
    result_path='./result'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    main()