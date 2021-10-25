import time 
import torch 
from Loss import cal_si_snr,cal_snr
from DataLoader import AudioDataset,AudioDataLoader,collate_fn
import os 


class Solver(object):
    def __init__(self,args,model,use_gpu,optimizer,logger):
        # video_data_trainval = VisualDataset('./data/visual/trainval_s1.scp','./data/visual/trainval_s2.scp')
        audio_data_trainval = AudioDataset('./data/audio/test/noisy.scp','./data/audio/test/clean.scp')

        # video_data_pretrain = VisualDataset('./data/visual/pretrain_s1.scp','./data/visual/pretrain_s2.scp')
        audio_data_pretrain = AudioDataset('./data/audio/train/noisy.scp','./data/audio/train/clean.scp')
        
        #pretrain data
        # self.video_data_loader = VisualDataloder(video_data_pretrain,batch_size=args.batch_size,num_workers=args.num_workers,pin_memory=True)
        self.audio_data_loader = AudioDataLoader(audio_data_pretrain,batch_size=args.batch_size,num_workers=args.num_workers,pin_memory=True,collate_fn=collate_fn)
        self.pretrain_len = len(audio_data_pretrain)

        # print('train',self.pretrain_len)
        #trainval data 
        # self.video_trainval = VisualDataloder(video_data_trainval,batch_size=args.batch_size,num_workers=args.num_workers,pin_memory=True )
        self.audio_trainval = AudioDataLoader(audio_data_trainval,batch_size=args.batch_size,num_workers=args.num_workers,pin_memory=True,collate_fn=collate_fn)
        self.trainval_len = len(audio_data_trainval)
        # print('val',self.trainval_len)

        self.args = args
        self.model = model
        self.use_gpu =use_gpu
        self.optimizer = optimizer
        self.logger = logger

        self._rest()


    def _rest(self):
        self.halving = False
        if self.args.continue_from:
            files = os.listdir('./log/model')
            files.sort()
            checkpoint_name = files[-1]
            checkpoint = torch.load('./log/model/'+checkpoint_name)

            #load model 
            model_dict = self.model.state_dict()
            pretrained_model_dict = checkpoint['model']
            pretrained_model_dict = {k:v for k,v in pretrained_model_dict.items() if k in model_dict}
            model_dict.update(pretrained_model_dict)
            self.model.load_state_dict(model_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            self.logger.info("*** model "+checkpoint_name+" has been successfully loaded! ***")
            #load other params 
            self.start_epoch=checkpoint['epoch']
            self.best_val_sisnr = checkpoint['best_val_sisnr']
            self.val_no_impv = checkpoint['val_no_impv']
            self.pre_val_sisnr = checkpoint['pre_val_sirsnr']

            # print("modify lr")
            # optim_state = self.optimizer.state_dict()
            # optim_state['param_groups'][0]['lr'] = 0.5
            # self.optimizer.load_state_dict(optim_state)

        else:
            self.start_epoch=0
            self.best_val_sisnr = float('inf')
            self.val_no_impv = 0
            self.pre_val_sisnr = float("inf")
            self.logger.info("*** train from scratch ***")




    def train(self):
        self.logger.info("use SI_SNR as loss function")
        for epoch in range(self.start_epoch,self.args.num_epochs):
            self.logger.info("------------")
            self.logger.info("Epoch:%d/%d"%(epoch,self.args.num_epochs))
            #train
            #--------------------------------------
            start = time.time()
            self.model.train()            
            temp = self._run_one_epoch(self.audio_data_loader,state='train')
            tr_loss_sisnr = temp['si_snr']
            tr_loss_snr = temp['snr']
            tr_loss_sisnr = tr_loss_sisnr/self.pretrain_len
            tr_loss_snr = tr_loss_snr/self.pretrain_len
            end = time.time()
            self.logger.info("Train: SI_SNR=%.04f,SNR=%.04f,Time:%d minutes"%(-tr_loss_sisnr,tr_loss_snr,(end-start)//60))

            #validation 
            #--------------------------------------
            start = time.time()
            self.model.eval()
            with torch.no_grad():
                temp = self._run_one_epoch(self.audio_trainval,state='val')
                val_loss_sisnr = temp['si_snr']
                val_loss_snr = temp['snr']
                val_loss_sisnr = val_loss_sisnr/self.trainval_len
                val_loss_snr/=self.trainval_len
            end = time.time()
            self.logger.info("Val: SI_SNR=%.04f,SNR=%.04f,Time:%d minutes"%(-val_loss_sisnr,val_loss_snr,(end-start)//60))
            self.logger.info(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

            #check whether to adjust learning rate and early stop 
            #-------------------------------------
            if val_loss_sisnr >= self.best_val_sisnr:
                self.val_no_impv +=1 
                if self.val_no_impv >=3:
                    self.halving =True 
                if self.val_no_impv >=6:
                    self.logger.info("No improvement for 6 epoches in val dataset, early stop")
                    break
            else:
                self.val_no_impv = 0


            # half the learning rate 
            #-----------------------------------
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr']/2
                self.optimizer.load_state_dict(optim_state)
                self.logger.info("**learning rate is adjusted from [%f] to [%f]"
                                %(optim_state['param_groups'][0]['lr']*2,optim_state['param_groups'][0]['lr']))
                self.halving = False
            
            self.pre_val_sisnr = val_loss_sisnr

            # save the model 
            #----------------------------------
            if val_loss_sisnr < self.best_val_sisnr :
                self.best_val_sisnr = val_loss_sisnr
                checkpoint = {'model':self.model.state_dict(),
                                'optimizer':self.optimizer.state_dict(),
                                'epoch':epoch+1,
                                'best_val_sisnr':self.best_val_sisnr,
                                'pre_val_sirsnr':self.pre_val_sisnr,
                                'val_no_impv':self.val_no_impv}
                torch.save(checkpoint, "./log/model/Checkpoint_%04d.pt" % epoch)

                self.logger.info("***save checkpoint as Checkpoint_%04d.pt***" % epoch)
                # if epoch % 10 == 0:
                #     self.logger.info("***epoch%10==0")



    def _run_one_epoch(self,audio_data_loader,state='train'):
        epoch_loss={'si_snr':0,'snr':0}
        for audio in audio_data_loader:
            audio_mix = audio['mix']
            # print(audio_mix.shape)
            audio_s1 = audio['s1']




            if self.use_gpu:
                audio_mix = audio_mix.cuda()
                audio_s1 = audio_s1.cuda()


            # print('*****',audio_s1_truth_mask.shape)
            audio_est = self.model(audio_mix)
            # return
            # print(s2_name)
            # audio_est_s2 = self.model([audio_mix])

            loss = cal_si_snr(audio_est,audio_s1)
            loss = -loss
            # loss = cal_si_snr(audio_est_s1,audio_s1)
            # loss+=cal_si_snr(audio_est_s2,audio_s2)
            # loss= -loss/2.0   #return negative number 

            epoch_loss['si_snr']+=loss.item()
            
            if state =='train': #
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # with torch.no_grad():
            #     loss = cal_snr(audio_est_s1,audio_s1)
            #     loss+= cal_snr(audio_est_s2,audio_s2)
            #     loss = loss/2 
            # epoch_loss['snr']+=loss.item()
        return epoch_loss













