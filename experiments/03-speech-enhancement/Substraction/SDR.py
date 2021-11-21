from posix import ST_SYNCHRONOUS
from mir_eval.separation import bss_eval_sources

# SDR 公式


def cal_sdr(source, estimate_source):
    '''
    source: [B,C,T]  source should be numpy 
    B:batch_size
    C: channel 
    T: length
    '''
    # assert source.shape == estimate_source.shape  # 保证两个音频维度相同
    if source.shape!=estimate_source.shape:
        min_len = min(source.shape[2],estimate_source.shape[2])
        source = source[:,:,0:min_len]
        estimate_source = estimate_source[:,:,0:min_len]
    SDR = 0.
    batch_size = source.shape[0]
    for batch in range(batch_size):
        source_ = source[batch]
        estimate_source_ = estimate_source[batch]
        sdr, _, _, _ = bss_eval_sources(source_, estimate_source_)
        SDR += sdr
    return float(SDR / batch_size)


# example
if __name__ == '__main__':
    import librosa as lb
    clean = './clean-51874.wav'
    mixture = './mixture-steadyNoise51874.wav'
    estimate = './estimate-steadyNoise51874.wav'

    clean_data, _ = lb.load(clean)
    clean_data = clean_data.reshape(1, 1, clean_data.shape[0])
    mixture_data, _ = lb.load(mixture)
    mixture_data = mixture_data.reshape(1, 1, mixture_data.shape[0])
    estimate_data, _ = lb.load(estimate)
    estimate_data = estimate_data.reshape(1, 1, estimate_data.shape[0])



    SDR1 = cal_sdr(clean_data, mixture_data)  # 第一个参数应该是干净的语音  第二个参数是mix或者预测的语音
    print('增强前：', SDR1)
    SDR2 = cal_sdr(clean_data, estimate_data)
    print('增强后：', SDR2)
