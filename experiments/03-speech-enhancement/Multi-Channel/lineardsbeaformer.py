import librosa
import numpy as np

def str2tuple(string, sep=","):
    """
    Map "1.0,2,0" => (1.0, 2.0)
    """
    tokens = string.split(sep)
    # if len(tokens) == 1:
    #     raise ValueError("Get only one token by " +
    #                      f"sep={sep}, string={string}")
    floats = map(float, tokens)
    return tuple(floats)

linear_topo = "0,100,200,300,400"
topo = np.array(str2tuple(linear_topo))
num_mics = 5

def plane_steer_vector(distance, num_bins, c=340, sr=16000):
    """
    Compute steer vector given projected distance on DoA:
    Arguments:
        distance: numpy array, N
        num_bins: number of frequency bins
    Return:
        steer_vector: F x N
    """
    omega = np.pi * np.arange(num_bins) * sr / (num_bins - 1)
    # temp_1 = distance/c 
    steer_vector = np.exp(-1j * np.outer(omega, distance / c))
    # temp = np.abs(steer_vector)
    return steer_vector


def linear_steer_vector(topo, doa, num_bins, c=340, sr=16000):
    """
    Compute steer vector for linear array:
        [..., e^{-j omega tau_i}, ...], where omega = 2*pi * f
    0   1   ...     N - 1
    *   *   ...     *
    0   d1  ...     d(N-1)
    Arguments:
        topo: linear topo, N
        doa: direction of arrival, in degree
        num_bins: number of frequency bins
    Return:
        steer_vector: F x N
    """
    dist = np.cos(doa * np.pi / 180) * topo
    # 180 degree <---------> 0 degree
    return plane_steer_vector(dist, num_bins, c = c, sr = sr)
    


def weights(doa, num_bins, c=340, sr=16000):
    sv = linear_steer_vector(topo, doa, num_bins, c=c, sr=sr)
    return sv / num_mics  #sv对应的是delay and sum beaformer 中的weight 


def beamform(weight, obs):
    """
    Arguments: (for N: num_mics, F: num_bins, T: num_frames)
        weight: shape as F x N
        obs: shape as N x F x T
    Return:
        stft_enhan: shape as F x T
    """
    # N x F x T => F x N x T
    if weight.shape[0] != obs.shape[1] or weight.shape[1] != obs.shape[0]:
        raise ValueError("Input obs do not match with weight, " +
                            f"{weight.shape} vs {obs.shape}")
    obs = np.transpose(obs, (1, 0, 2))


    temp = np.abs(weight)
    obs = np.einsum("...n,...nt->...t", weight.conj(), obs)

    return obs


def run(doa, obs, c=340, sr=16000):
    """
    Arguments: (for N: num_mics, F: num_bins, T: num_frames)
        doa: direction of arrival, in degree
        obs: shape as N x F x T
    Return:
        stft_enhan: shape as F x T
    """
    if obs.shape[0] != num_mics:
        raise ValueError(
            "Shape of obs do not match with number" +
            f"of microphones, {num_mics} vs {obs.shape[0]}")
    num_bins = obs.shape[1]
    weight = weights(doa, num_bins, c=c, sr=sr)



    return beamform(weight, obs)

if __name__ == "__main__":
    doa = 20  #声源传来的角度   clean的语音的角度  #根据不同的音频 需要自己修改

    import soundfile as sf
    import numpy as np

    wave_data, sr = sf.read('./clean_20_unsteadyNoise_10.wav')
    if sr!=16000:
        wave_data = librosa.resample(load, sr, 16000)
    
    wav_stft = []
    for i in range(wave_data.shape[1]):
        wav_stft.append(librosa.stft(wave_data[:,i]))
    wav_stft = np.array(wav_stft)

    enh_stft = run(doa, wav_stft)
    enh_wav = librosa.istft(enh_stft)

    enh_wav = enh_wav / np.max(np.abs(enh_wav))
    sf.write('./estimate_clean_20_unsteadyNoise_10.wav',enh_wav,samplerate=16000)
