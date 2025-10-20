import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from DWT_IDWT_layer import *
import SimpleITK as sitk

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DWT_Downsample(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(DWT_Downsample, self).__init__()
        self.dwt = DWT_3D(wavename='haar')

    def forward(self, input):
        LLL , LLH , LHL , LHH , HLL , HLH , HHL , HHH = self.dwt(input)
        return LLL , LLH , LHL , LHH , HLL , HLH , HHL , HHH

class Frequency_Attention(nn.Module):#(self, LLL , LLH , LHL , LHH , HLL , HLH , HHL , HHH):
    #watch out dim!!! may be wrong
    #8 DWT image
    def __init__(self,in_feature,out_feature):
        super(Frequency_Attention, self).__init__()
        self.in_feature=in_feature
        self.out_feature=out_feature

    def forward(self,input):
        LLL , LLH , LHL , LHH , HLL , HLH , HHL , HHH=input
        # print("fuck",LLL.shape , LLH.shape , LHL.shape , LHH.shape , HLL.shape , HLH.shape , HHL.shape , HHH.shape)

    
        Dense0=nn.Linear(in_features=self.in_feature, out_features=256, bias=False)
        nn.init.kaiming_normal_(Dense0.weight, nonlinearity='relu')     #initalize he_normal 
        Dense1=nn.Linear(in_features=256, out_features=32, bias=False)
        nn.init.kaiming_normal_(Dense1.weight, nonlinearity='relu')     #initalize he_normal 
        Dense2=nn.Linear(in_features=32, out_features=self.out_feature, bias=False)
        nn.init.kaiming_normal_(Dense2.weight, nonlinearity='relu')     #initalize he_normal 
        
        Dense0=Dense0.to(device)
        Dense1=Dense1.to(device)
        Dense2=Dense2.to(device)
        
        # #Frequency Attention
        # LLL_gap=F.adaptive_avg_pool3d(LLL,(1, 1,self.out_feature))
        # # LLL_gap=LLL_gap.reshape(1,1,256)
        # LLL_w=Dense2(Dense1(Dense0(LLL_gap)))
        LLL=LLL.permute(0,2,3,4,1)
        LLL_gap=F.adaptive_avg_pool3d(LLL,(LLL.shape[1],LLL.shape[2] ,self.out_feature))
        # print("LLL_gap",LLL_gap.shape)
        LLL_w=Dense2(Dense1(Dense0(LLL_gap)))
        # print("LLL_W",LLL_w.shape)

        LLH=LLH.permute(0,2,3,4,1)
        LLH_gap=F.adaptive_avg_pool3d(LLH,(LLH.shape[1],LLH.shape[2] ,self.out_feature))
        # print("LLH_gap",LLH_gap.shape)
        LLH_w=Dense2(Dense1(Dense0(LLH_gap)))
        # print("LLH_W",LLH_w.shape)

        LHL=LHL.permute(0,2,3,4,1)
        LHL_gap=F.adaptive_avg_pool3d(LHL,(LHL.shape[1],LHL.shape[2] ,self.out_feature))
        # print("LHL_gap",LHL_gap.shape)
        LHL_w=Dense2(Dense1(Dense0(LHL_gap)))
        # print("LHL_W",LHL_w.shape)

        LHH=LHH.permute(0,2,3,4,1)
        LHH_gap=F.adaptive_avg_pool3d(LHH,(LHH.shape[1],LHH.shape[2] ,self.out_feature))
        # print("LHH_gap",LHH_gap.shape)
        LHH_w=Dense2(Dense1(Dense0(LHH_gap)))
        # print("LHH_W",LHH_w.shape)

        HLL=HLL.permute(0,2,3,4,1)
        HLL_gap=F.adaptive_avg_pool3d(HLL,(HLL.shape[1],HLL.shape[2] ,self.out_feature))
        # print("HLL_gap",HLL_gap.shape)
        HLL_w=Dense2(Dense1(Dense0(HLL_gap)))
        # print("HLL_W",HLL_w.shape)

        HLH=HLH.permute(0,2,3,4,1)
        HLH_gap=F.adaptive_avg_pool3d(HLH,(HLH.shape[1],HLH.shape[2] ,self.out_feature))
        # print("HLH_gap",HLH_gap.shape)
        HLH_w=Dense2(Dense1(Dense0(HLH_gap)))
        # print("HLH_W",HLH_w.shape)

        HHL=HHL.permute(0,2,3,4,1)
        HHL_gap=F.adaptive_avg_pool3d(HHL,(HHL.shape[1],HHL.shape[2] ,self.out_feature))
        # print("HHL_gap",HHL_gap.shape)
        HHL_w=Dense2(Dense1(Dense0(HHL_gap)))
        # print("HHL_W",HHL_w.shape)

        HHH=HHH.permute(0,2,3,4,1)
        HHH_gap=F.adaptive_avg_pool3d(HHH,(HHH.shape[1],HHH.shape[2] ,self.out_feature))
        # print("HHH_gap",HHH_gap.shape)
        HHH_w=Dense2(Dense1(Dense0(HHH_gap)))
        # print("HHH_W",HHH_w.shape)

        # LLH_gap=F.adaptive_avg_pool3d(LLH,(1, 1,self.out_feature))
        # # LLH_gap=LLH_gap.reshape(1,1,256)
        # LLH_w=Dense2(Dense1(Dense0(LLH_gap)))

        # LHL_gap=F.adaptive_avg_pool3d(LHL,(1, 1,self.out_feature))
        # # LHL_gap=LHL_gap.reshape(1,1,256)
        # LHL_w=Dense2(Dense1(Dense0(LHL_gap)))

        # LHH_gap=F.adaptive_avg_pool3d(LHH,(1, 1,self.out_feature))
        # # LHH_gap=LHH_gap.reshape(1,1,256)
        # LHH_w=Dense2(Dense1(Dense0(LHH_gap)))

        # HLL_gap=F.adaptive_avg_pool3d(HLL,(1, 1,self.out_feature))
        # # HLL_gap=HLL_gap.reshape(1,1,256)
        # HLL_w=Dense2(Dense1(Dense0(HLL_gap)))

        # HLH_gap=F.adaptive_avg_pool3d(HLH,(1, 1,self.out_feature))
        # # HLH_gap=HLH_gap.reshape(1,1,256)
        # HLH_w=Dense2(Dense1(Dense0(HLH_gap)))

        # HHL_gap=F.adaptive_avg_pool3d(HHL,(1, 1,self.out_feature))
        # # HHL_gap=HHL_gap.reshape(1,1,256)
        # HHL_w=Dense2(Dense1(Dense0(HHL_gap)))
        
        # HHH_gap=F.adaptive_avg_pool3d(HHH,(1, 1,self.out_feature))
        # # HHH_gap=HHH_gap.reshape(1,1,256)
        # HHH_w=Dense2(Dense1(Dense0(HHH_gap)))
        # print(LLL_w.shape)
        #Final feature maps in each channel
        # LLL=LLL.reshape(18,18,1,self.out_feature)
        # LLH=LLH.reshape(18,18,1,self.out_feature)
        # LHL=LHL.reshape(18,18,1,self.out_feature)
        # LHH=LHH.reshape(18,18,1,self.out_feature)
        # HLL=HLL.reshape(18,18,1,self.out_feature)
        # HLH=HLH.reshape(18,18,1,self.out_feature)
        # HHL=HHL.reshape(18,18,1,self.out_feature)
        # HHH=HHH.reshape(18,18,1,self.out_feature)
        # print("LLL mal",LLL.shape,LLL_w.shape)
        LLL_fm=LLL*LLL_w    #tensor.matmul?
        LLH_fm=LLH*LLH_w
        LHL_fm=LHL*LHL_w
        LHH_fm=LHH*LHH_w
        HLL_fm=HLL*HLL_w
        HLH_fm=HLH*HLH_w
        HHL_fm=HHL*HHL_w
        HHH_fm=HHH*HHH_w

        LLL_fm=LLL_fm.permute(0,4,1,2,3)
        LLH_fm=LLH_fm.permute(0,4,1,2,3)
        LHL_fm=LHL_fm.permute(0,4,1,2,3)
        LHH_fm=LHH_fm.permute(0,4,1,2,3)
        HLL_fm=HLL_fm.permute(0,4,1,2,3)
        HLH_fm=HLH_fm.permute(0,4,1,2,3)
        HHL_fm=HHL_fm.permute(0,4,1,2,3)
        HHH_fm=HHH_fm.permute(0,4,1,2,3)
        out=torch.cat([LLL_fm , LLH_fm , LHL_fm , LHH_fm , HLL_fm , HLH_fm , HHL_fm , HHH_fm],1)    #
        # out=out.permute(3,2,0,1) #256 8 18 18
        # print("out",out.shape)
        out_conv3d= nn.Conv3d(in_channels=self.out_feature*8, out_channels=self.out_feature*2, kernel_size=1)
        nn.init.kaiming_normal_(out_conv3d.weight, nonlinearity='relu')
        out_conv3d=out_conv3d.to(device)
        out=out_conv3d(out)
        # print("out_conv3d",out.shape)  #768 6 6 6
        # out=out.permute(1,2,3,0)
        # print("out",out.shape) 
        
        # out_conv2d=nn.Conv2d(12, 12, kernel_size=3, padding=1)
        # nn.init.kaiming_normal_(out_conv2d.weight, nonlinearity='relu')

        # out_conv2d_2=nn.Conv2d(12, 12, kernel_size=3, padding=1)
        # nn.init.kaiming_normal_(out_conv2d_2.weight, nonlinearity='relu')

        # out_conv2d=out_conv2d.to(device)
        # out_conv2d_2=out_conv2d_2.to(device)
        # out=out_conv2d(out)
        # # print("out1",out.shape)
        # out=out_conv2d_2(out)
        # # print("out2",out.shape)

       
        # out_bn2d=nn.BatchNorm2d(12, eps=1e-5)
        # out_bn2d.to(device)
        # out=out_bn2d(out)
        # out=F.relu(out)
        # out=nn.Dropout(0.1)(out)
        # out=out.unsqueeze(0)
        return out

class DWT_Attention(nn.Module):
    def __init__(self,in_feature,out_feature):#(self,img_shape=(128, 128, 128), input_dim=4, output_dim=3):
        super(DWT_Attention,self).__init__()
        # self.input_dim = input_dim
        # self.output_dim = output_dim
        # self.img_shape = img_shape
        self.dwt=DWT_Downsample(wavename = 'haar')
        self.FreqAtt=Frequency_Attention(in_feature=in_feature,out_feature=out_feature)
    # def DWT(self,x):#x is downsample from last encoder layer
    #     self.dwt=DWT_3D(wavename='haar')
    #     LLL , LLH , LHL , LHH , HLL , HLH , HHL , HHH=self.dwt(x)
    #     return LLL , LLH , LHL , LHH , HLL , HLH , HHL , HHH

    def forward(self,input):
        # print("input",input.shape)
        out=self.dwt(input)
        out=self.FreqAtt(out)

        return out

# if __name__ == "__main__":
#     img=sitk.ReadImage('/home/user/Documents/nnUNetv2/nnUNet_raw/Dataset503_Abdomen3d/imagesTr/img0001_0000.nii.gz')
#     img=sitk.GetImageFromArray(img)
#     attn_img=DWT_Attention(img)
    

