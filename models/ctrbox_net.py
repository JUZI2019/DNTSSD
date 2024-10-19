import torch.nn as nn
import numpy as np
import torch
from .model_parts import CombinationModule
from . import resnet
import matplotlib.pyplot as plt
import os
class CTRBOX(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super(CTRBOX, self).__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))  # =2
        self.base_network = resnet.resnet101(pretrained=pretrained)
        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':              #  channels[self.l1]=256 ，head_conv=256
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True), # 3*3卷积
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=3, padding=1, bias=True)) # 3*3卷积
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True), # 3*3卷积
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True)) # 1*1卷积
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)


    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


   # 这里的  x.shape  为:[batch_size, channels, height, width]  即：[batch_size, 3, 608, 608]
    def forward(self,x):
        x = self.base_network(x)

        # 查看特征图
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}-1.png'.format(idx)), temp)
        # for idx in range(x[2].shape[1]):
        #     temp = x[2][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}-2.png'.format(idx)), temp)    
        # for idx in range(x[3].shape[1]):
        #     temp = x[3][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}-3.png'.format(idx)), temp)
        # for idx in range(x[4].shape[1]):
        #     temp = x[4][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}-4.png'.format(idx)), temp)            
        # for idx in range(x[5].shape[1]):
        #     temp = x[5][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}-5.png'.format(idx)), temp)

        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])


        # for idx in range(c2_combine.shape[1]):
        #     temp = c2_combine[0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}-c2.png'.format(idx)), temp)

        # for idx in range(c3_combine.shape[1]):
        #     temp = c3_combine[0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}-c3.png'.format(idx)), temp)     

        # for idx in range(c4_combine.shape[1]):
        #     temp = c4_combine[0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}-c4.png'.format(idx)), temp)



        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])  #  'hm'或'cls'两个头的输出应用sigmoid函数
        return dec_dict
