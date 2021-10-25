#根据单通道音频去模拟生成多通道音频  

import pyroomacoustics as pra
import wave
import numpy as np
import soundfile as sf
import os


def generate(load_path, save_path):
    room_length = 6
    room_width = 6
    room_height = 3
    rt_60 = 0.5
    fs = 16000

    mic_locations = np.c_[  #麦克风的位置 
        [room_width / 2 - 2, 0, 2],
        [room_width / 2 - 1, 0, 2],
        [room_width / 2, 0, 2],
        [room_width / 2 + 1, 0, 2],
        [room_width / 2 + 2, 0, 2],
    ]

    print(mic_locations)
    print(mic_locations[:, 0])

    source_angle_lst = [20]  # 以中间麦克风为参考点   指定声源的角度
    source_distance_lst = [3]  #声源距离中间麦克风的距离  
    source_rad_lst = []
    for angle in source_angle_lst:
        source_rad = np.pi / 180 * angle
        source_rad_lst.append(source_rad)

    print(source_rad_lst)

    room_dim = [room_length, room_width, room_height]

    absorption, max_order = pra.inverse_sabine(rt_60, room_dim)
    audio_data, sr = sf.read(load_path, dtype=np.int16)

    for distance in source_distance_lst:
        index = 89  #只是一个index 为了命令方便 无特别作用 
        for rad in source_rad_lst:
            print("No.:", index)
            index = index + 1
            if rad < np.pi / 2:
                source_x = room_width / 2 - distance * np.cos(rad)
                source_y = distance * np.sin(rad)
                source_z = 2

            elif rad > np.pi / 2:
                source_x = room_width / 2 + distance * np.cos(np.pi - rad)
                source_y = distance * np.sin(np.pi - rad)
                source_z = 2
            else:
                source_x = room_width / 2
                source_y = distance
                source_z = 2
            source_location = np.array([source_x, source_y, source_z])
            wav_name = "distance_{}m_source_{}.wav".format(distance, index)
            output_path = os.path.join(save_path, wav_name)

            multichannel_audio_data = generate_room(
                room_dim, fs, max_order, source_location, mic_locations, absorption, audio_data)
            print(save_path)

            trans_save_mul_ch_audio(multichannel_audio_data, output_path, fs)


def generate_room(room_dim, fs, max_order, source_location, mic_locations, absorption, audio_data):
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(
        absorption), max_order=max_order)

    c = 345

    dist = np.linalg.norm(source_location - mic_locations[:, 0])
    delay = dist / c
    room.add_source(source_location, signal=audio_data, delay=delay)

    room.add_microphone_array(mic_locations)

    room.simulate()

    orig_max_value = np.max(np.abs(audio_data))

    multichannel_audio_data = room.mic_array.signals[:, 0:len(audio_data)]
    multichannel_audio_data = multichannel_audio_data / \
        np.max(np.abs(multichannel_audio_data)) * orig_max_value
    return multichannel_audio_data


def trans_save_mul_ch_audio(mul_ch_audio_data, output_file_path, fs=16000):
    channels, frames = mul_ch_audio_data.shape

    mul_ch_audio_data = mul_ch_audio_data.transpose(1, 0)

    out_data = np.reshape(mul_ch_audio_data, [frames * channels, 1])
    out_data = out_data.astype(np.int16)

    with wave.open(output_file_path, 'wb') as f:
        f.setframerate(fs)
        f.setsampwidth(2)
        f.setnchannels(channels)
        f.writeframes(out_data.tostring())
        f.close()


if __name__ == "__main__":
    load_path = "./clean.wav"
    save_path = './'
    generate(load_path,save_path)


    #以下是当多通道音频可用时 可以混合clean和noise
    # import librosa
    # import soundfile as sf
    # import numpy as np

    # noise, _ = sf.read('./unsteadyNoise_mul_10.wav')
    # clean, _ = sf.read('./clean_mul_20.wav')

    # mixture = np.zeros(clean.shape)
    # for i in range(clean.shape[0]):
    #     mixture[i, :] = clean[i, :]+noise[i % noise.shape[0], :]

    # # mixture = noise+clean
    # sf.write('./clean_20_unsteadyNoise_10.wav', mixture, samplerate=16000)
