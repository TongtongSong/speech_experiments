from Modelutils import *
from Normolization import *

import torch
import torch.nn as nn
import cv2
from torchsummary import summary
import math
import torch.optim as optim
import librosa
import numpy as np 



class Block(nn.Module):

    def __init__(self, filter_size, dilation, num_filters, input_filters, padding):
        super(Block, self).__init__()

        self.conv = nn.Conv1d(
            in_channels=input_filters,
            out_channels=num_filters,
            kernel_size=filter_size,
            dilation=dilation,
            padding=padding,
        )

        self.bn = nn.BatchNorm1d(num_filters)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class concatNet(nn.Module):

    def __init__(self, num_repeats, num_blocks, in_channels=256,
                 out_channels=256, kernel_size=3, norm='gln', causal=False):
        super(concatNet, self).__init__()
        # self.liner1 = Conv1D(768, 512, kernel_size=3, stride=1, padding=1)
        # self.liner = Conv1D(512, in_channels, kernel_size=3, stride=1, padding=1)
        self.TCN = self._Sequential_repeat(num_repeats, num_blocks,
                                           in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=kernel_size, norm=norm, causal=causal)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        # out = self.liner(x)
        # out = self.liner2(out)
        out = self.TCN(x)
        out = self.relu(out)

        return out

    def _Sequential_repeat(self, num_repeats, num_blocks, **kwargs):
        repeat_lists = [self._Sequential_block(
            num_blocks, **kwargs) for i in range(num_repeats)]
        return nn.Sequential(*repeat_lists)

    def _Sequential_block(self, num_blocks, **kwargs):
        '''
        Sequential 1-D Conv Block
        input:
            num_blocks:times the block appears
            **block_kwargs
        '''
        Conv1D_Block_lists = [Conv1D_Block(
            **kwargs, dilation=(2**i)) for i in range(num_blocks)]
        return nn.Sequential(*Conv1D_Block_lists)


class Encoder(nn.Module):
    '''
    Encoder of the TasNet
    '''

    def __init__(self, kernel_size, stride, outputDim=256):
        super(Encoder, self).__init__()
        self.encoder = nn.Conv1d(1, outputDim, kernel_size, stride=stride)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.encoder(x)
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    '''
    Decoder of the TasNet
    '''

    def __init__(self, kernel_size, stride, inputDim=256):
        super(Decoder, self).__init__()
        self.decoder = nn.ConvTranspose1d(inputDim, 1, kernel_size, stride)

    def forward(self, x):
        out = self.decoder(x)
        return out


class TCN(nn.Module):
    '''
    in_channels:the encoder out_channels

    '''

    def __init__(self, out_channels, num_repeats, num_blocks,
                 kernel_size, norm='gln', causal=False):
        super(TCN, self).__init__()

        self.TCN = self._Sequential_repeat(num_repeats, num_blocks, in_channels=256, out_channels=out_channels,
                                           kernel_size=kernel_size, norm=norm, causal=causal)

    def forward(self, x):
        c = self.TCN(x)
        return c  # shape [-1,1,256]

    def _Sequential_repeat(self, num_repeats, num_blocks, **kwargs):
        repeat_lists = [self._Sequential_block(
            num_blocks, **kwargs) for i in range(num_repeats)]
        return nn.Sequential(*repeat_lists)

    def _Sequential_block(self, num_blocks, **kwargs):
        '''
        Sequential 1-D Conv Block
        input:
            num_blocks:times the block appears
            **block_kwargs
        '''
        Conv1D_Block_lists = [Conv1D_Block(
            **kwargs, dilation=(2**i)) for i in range(num_blocks)]
        return nn.Sequential(*Conv1D_Block_lists)


class AVModel(nn.Module):

    def __init__(self,):
        super(AVModel, self).__init__()

        self.audio_model_encoder = Encoder(kernel_size=40, stride=20)
        self.audio_model_TCN = TCN(
            out_channels=256, num_repeats=1, num_blocks=8, kernel_size=3)
        # self.video_model = VisualNet()
        self.concat_model = concatNet(
            in_channels=256, out_channels=256, num_repeats=3, num_blocks=8)
        self.decoder = Decoder(kernel_size=40, stride=20, inputDim=256)
        
        self.geneate_mask = nn.Conv1d(in_channels=256,out_channels=256,kernel_size=3,padding=1,stride=1 )
        self.batch2 = nn.BatchNorm1d(256)
        # self.batch3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, data):

        audio_mix = data
        
        encoder_output = self.audio_model_encoder(audio_mix)
        
        TCN_output = self.audio_model_TCN(encoder_output)
        # visual_output = self.video_model(video_s1)
        # visual_output = self.batch1(visual_output)

        # memory = self.batch3(memory)

        # concat_input = torch.cat(( TCN_output,), dim=1)
        # print(concat_input.shape)


        concat_output = self.concat_model(TCN_output)
        
        mask = self.geneate_mask(concat_output)
        # print(mask.shape)
        # mask = torch.chunk(mask, chunks=2, dim=1)
        # print(mask[0].shape)
        # print(torch.stack(mask,dim=0).shape)
        mask = self.relu(mask)
        # print(mask[0].shape)
        # print(encoder_output.shape)

        decoder_input = encoder_output* mask
        output = self.decoder(decoder_input)  # output [B,C,lenght]

        return output


if __name__ == '__main__':

    print('start')
    # net = VisualNet()
    # Videopath = '/Work19/2020/lijunjie/lips/test/0Fi83BHQsMA/00002-0001.jpg'
    # image = cv2.imread(Videopath,cv2.IMREAD_GRAYSCALE)
    # image = torch.from_numpy(image.astype(float))
    # image = image.unsqueeze(0)
    # image = image.unsqueeze(0)
    # image = image.unsqueeze(0)
    # print(image.shape)

    # image = image.float()
    # x = net(image)

    # net_input = torch.ones(32, 3, 10, 224, 224)
    # net_input = net_input.int()
    # # With square kernels and equal stride | 所有维度同一个参数配置
    # conv = nn.Conv3d(3, 64, kernel_size=3, stride=2, padding=1)
    # net_output = conv(net_input)
    # print(net_output.shape)  # shape=[32, 64, 5, 112, 112] | 相当于每一个维度上的卷积核大小都是3，步长都是2，pad都是1

    # # non-square kernels and unequal stride and with padding | 每一维度不同参数配置
    # conv = nn.Conv3d(3, 64, (2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
    # net_output = conv(net_input)
    # print(net_output.shape) # shape=[32, 64, 9, 112, 112]

    # net = VisualNet().to('cuda')
    # summary(net,(1,75,120,120))
    # net = TCN(256,2,2,3).to('cuda')
    # summary(net,(1,48000))
    # net = Encoder(40,20).to('cuda')
    # summary(net,(1,48000))

    # net = concatNet(2).to('cuda')
    # summary(net,(2,256))
    # for name,param in net.named_parameters():
    #     print(name,'        ',param.size())

    model = AVModel()
    net = contxt_encoder().to('cuda')
    summary(net,(256,73))
    # for n,m in model.named_parameters():
    #     print(n,type(m))
    # optimizer = optim.Adam([{'params': model.parameters()}], lr=0.001)
    # print(optimizer.state_dict())
