#用来保存预测的语音 
import torch
from AVModel import AVModel 
from DataLoader import AudioDataset,AudioDataLoader,collate_fn
import soundfile
import numpy as np 
import os 
from Loss import cal_si_snr,permute_SI_SNR,permute_SDR
from Loss import cal_sdr
import torch.nn as nn
from Loss import cal_snr
import torch.nn
from collections import OrderedDict
import time 

start = time.time()

output_path = "./log/separate/"
model_path = "./log/model/Checkpoint_0015.pt"
samplerate=16000
num_workers=8

os.makedirs(output_path,exist_ok=True)



model = AVModel()
model = nn.DataParallel(model)
checkpoint = torch.load(model_path)
model_dict = checkpoint['model']


model.load_state_dict(model_dict)
model.cuda()
model.eval()


audio_data_test = AudioDataset('./data/audio/test/noisy.scp','./data/audio/test/clean.scp')

audio_data_loader = AudioDataLoader(audio_data_test,batch_size=1,num_workers=num_workers,collate_fn=collate_fn)

Average_SISNR=0
Average_SDR =0
Average_SNR = 0

mix_SISNR=0
mix_SDR=0

length = len(audio_data_test)

print('start separate >>>>')
with torch.no_grad():
    total_loss = 0 
    total_data_num=0
    for idx,audio in enumerate(audio_data_loader):

        audio_mix = audio['mix']
        # print(audio_mix.shape)
        audio_s1 = audio['s1']
        
        # compute mixture metric loss ----------------------


        mix_SISNR += cal_si_snr(audio_s1,audio_mix).item()


        mix_SDR += cal_sdr(audio_s1.numpy(),audio_mix.numpy())

        
        # predict estimate singal 
        audio_mix = audio_mix.cuda()
        audio_s1 = audio_s1.cuda()

        # video_s1 = video['s1'].cuda()
        # video_s2 = video['s2'].cuda()
        
        output = model(audio_mix)

        # output_s2 = model([audio_mix,audio_s2_truth,audio_s2_truth_mask])
        output = output.cpu()


        # output_s1 = output_s1.cpu()
        # output_s2 = output_s2.cpu()

        # video_s1 = video_s1.cpu()
        # video_s2 = video_s2.cpu()
        audio_mix = audio_mix.cpu()
        audio_s1 = audio_s1.cpu()
        # audio_s2 = audio_s2.cpu()

        output = output*torch.max(torch.abs(audio_mix))/torch.max(torch.abs(output))



        SI_SNR = cal_si_snr(output,audio_s1)
        # compute output metric loss -----------------------
        # SI_SNR1 = cal_si_snr(audio_s1,output_s1).item()
        # SI_SNR2 = cal_si_snr(audio_s2,output_s2).item()
        # Average_SISNR +=SI_SNR
        # Average_SISNR +=SI_SNR[1]



        SDR = cal_sdr(audio_s1.numpy(),output.numpy())
        # SDR2 = cal_sdr(audio_s2.numpy(),output_s2.numpy())
        # temp = [output_list[0].numpy(),output_list[1].numpy()]
        # SDR = cal_SDR(temp,[audio_s1.numpy(),audio_s2.numpy()])
        # Average_SDR+=SDR
        # Average_SDR+=SDR2

        print('%04d:'%idx,'SI_SNR=%.04f'%SI_SNR,'SDR=%.04f'%SDR)


        output = output[0].permute(1, 0).numpy()
        audio_mix = audio_mix[0].permute(1,0).numpy()
        audio_s1 = audio_s1[0].permute(1,0).numpy()
        # audio_s2 = audio_s2[0].permute(1,0).numpy()
        # output_s2 = output_s2[0].permute(1,0).numpy()


        soundfile.write(output_path+'%04d_ests.wav'%idx,output,samplerate=samplerate)
        # soundfile.write(output_path+'%04d_ests2.wav'%idx,output_s2,samplerate=samplerate)
        soundfile.write(output_path+"%04d_noisy.wav"%idx,audio_mix,samplerate=samplerate)
        soundfile.write(output_path+"%04d_clean.wav"%idx,audio_s1,samplerate=samplerate)
        # soundfile.write(output_path+"%04d_s2.wav"%idx,audio_s2,samplerate=samplerate)


    # print('est_SI_SNR=%.04f' % (mix_SISNR / (length)), 'est_SDR:%.04f' % (mix_SDR / ( length)))
    
    # print('MIX_SI_SNR=%.04f'%(mix_SISNR/(2*length)),'MIX_SDR=%.04f'%(mix_SDR/(2*length)),'MIX_SNR=%.04f'%(mix_SNR/(2*length)))

    end = time.time()
    print('time= %d min'%((end-start)/60))

