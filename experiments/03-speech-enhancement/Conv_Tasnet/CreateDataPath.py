'''
@author:JunjieLi
@time:2020/12
map audio and visual file to their pahts 
'''

import os 
import ast 
import shutil
import librosa as lb 


#which is saving data set
# visual_root = '/CDShare2/LRS3_process/lips'  #there are three folders(pretrain,trainval,test) in this folder
mixaudio_root = '//Work18/2018/shihao/SLT2021-speech-separaion/DS_10283_1942/datasets/'#there are three folders(pretrain ,train,test) in this folder

#which is used to save pathfile
# visual_datafolder = './data/visual/'
audio_datafolder = './data/audio/' 

# os.makedirs(visual_datafolder,exist_ok=True)
os.makedirs(audio_datafolder,exist_ok=True)



def CreateAudioPath(inpath,outpath,timeLen=3):
    '''
    create file contains mixaudio s1audio s2audio path
    inpath: root path folder that contain mixaduios 
    outpath: The folder is used to put created files
    '''
    for root,dirs ,files in os.walk(inpath,topdown=True):#由于根目录中含有mix字符串 所以设为True能够将错误文件覆盖
        if ('train' in root):
            os.makedirs(audio_datafolder+'train/',exist_ok=True)
            files.sort()
            if('clean' in root and len(files)!=0):
                s2 = open(os.path.join(audio_datafolder+'train/','clean.scp'),'w')
                for file in files:
                    data, sr = lb.load(root + '/' + file, sr=16000)
                    if(len(data)/sr >=3):
                        s2.write(file+" "+root+'/'+file+'\n')
                s2.close()
        
            elif('noisy' in root and len(files)!=0):
                mix = open(os.path.join(audio_datafolder+'train/','noisy.scp'),'w')
                for file in files:
                    data, sr = lb.load(root + '/' + file, sr=16000)
                    if(len(data)/sr >=3):
                        mix.write(file+" "+root+'/'+file+'\n')
                mix.close()
            
        
        elif ('test' in root):
            os.makedirs(audio_datafolder+'test',exist_ok=True)
            files.sort()
            if('clean' in root and len(files)!=0):
                s2 = open(os.path.join(audio_datafolder+'test/','clean.scp'),'w')
                for file in files:
                    data, sr = lb.load(root + '/' + file, sr=16000)
                    if(len(data)/sr >=3):
                        s2.write(file+" "+root+'/'+file+'\n')
                s2.close()
            elif('noisy' in root and len(files)!=0):
                mix = open(os.path.join(audio_datafolder+'test/','noisy.scp'),'w')
                for file in files:
                    data, sr = lb.load(root + '/' + file, sr=16000)
                    if(len(data)/sr >=3):
                        mix.write(file+" "+root+'/'+file+'\n')
                mix.close()
 


if __name__ =='__main__':
    CreateAudioPath(mixaudio_root,audio_datafolder)
    # CreateVisualPath(audio_datafolder,visual_root,visual_datafolder)
    # CheckLipNum(visual_datafolder)