import SimpleITK as sitk
import glob
import numpy as np
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt  # plt 用于显示图片

def pngs2nii(dir,spacing):
    # image_path="./dataset/Abdomen/WORD_Abdomen/labelsVal2d0to7/"+dir
    image_path="/media/user/2tb/dataset/miccai2015pred/2d_rgb/ground_truth/"+dir
    # image_path="./dataset/Abdomen/majei_abdomen1127_png/MMH/"+dir
    file_list=os.listdir(image_path)
    file_list.sort();
    flag=1
    # print(list(reversed(file_list)))
    for f in file_list:
        # slicing=cv2.imread(image_path+'/'+f,cv2.IMREAD_GRAYSCALE)
        slicing=cv2.imread(image_path+'/'+f)
        slicing=cv2.cvtColor(slicing, cv2.COLOR_BGR2RGB)
        # slicing=Image.open(image_path+'/'+f) 
        slicing=np.array(slicing)
        # slicing=np.flipud(slicing)  #上下翻轉
        if flag:
            out_3d=slicing[np.newaxis,:,:]

            flag=0
        else:
            out_3d=np.concatenate((out_3d,slicing[np.newaxis,:,:]),axis=0)
        # print(out_3d.shape)
    # out_3d=out_3d.transpose((0,2,1))
    # print(out_3d.shape)
    out=sitk.GetImageFromArray(out_3d)
    out.SetSpacing(spacing)
    # sitk.WriteImage(out,'./dataset/Abdomen/WORD_Abdomen/labelsVal3d/'+dir+'.nii.gz')
    sitk.WriteImage(out,"/media/user/2tb/dataset/btcvinit/3d/labelsRGB/"+dir+'.nii.gz')
    # sitk.WriteImage(out,"./dataset/Abdomen/majei_1127_nii/"+dir+'.nii.gz')

# dir_path="./dataset/Abdomen/BTCV_Abdomen/findContours/2d_anidiff/niter20_kappa20_gamma0.05_option1/2d/imagesTr/"
# dir_path="./dataset/Abdomen/BTCV_Abdomen/window_bound/imagesTr/"

# dir_path="./dataset/Abdomen/majei_abdomen1127_png/MMH/"
dir_path='/media/user/2tb/dataset/miccai2015pred/2d_rgb/ground_truth/'
dirs=os.listdir(dir_path)
dirs.sort()
for dir in dirs:
    # image = sitk.ReadImage('./dataset/Abdomen/WORD_Abdomen/WORD-V0.1.0/labelsVal/'+dir+'.nii.gz')
    image = sitk.ReadImage('/home/user/Documents/dataset/Abdomen/Abdomen/RawData/Training/label/'+dir+'.nii.gz')
    # image = sitk.ReadImage('./dataset/Abdomen/Abdomen/RawData/Training/img/img0001.nii.gz')
    spacing=image.GetSpacing()
    pngs2nii(dir,spacing)
    # pngs2nii(dir,(0.67,0.67,3))
print("end process")
#--several .png overlapping together and  change to .nii--#
# image_path = "./dataset/Abdomen/BTCV_Abdomen/BTCV_mask/labelsTr0to8/label0001"

# file_list=os.listdir(image_path)
# file_list.sort();
# #file_names=[os.path.join(image_path,f) for f in file_list]
# for f in file_list:
#     newspacing=[1,1,1]
#     neworigin=[-1,-1,1]
#     reader=sitk.ImageSeriesReader()
#     p=[os.path.join(image_path,f)]
#     reader.SetFileNames(p)
#     vol=reader.Execute()
#     #vol.SetOrigin(vol.GetOrigin())
#     #vol.SetDirection(vol.GetDirection())
#     vol.SetSpacing(newspacing)
#     sitk.WriteImage(vol,"./dataset/Abdomen/BTCV_Abdomen/BTCV_mask/test.nii.gz")
# import SimpleITK as sitk
# import glob
# import os

# dir_path='./dataset/Abdomen/BTCV_Abdomen/BTCV_mask/labelsTr0to8'
# dir_name=os.listdir(dir_path)
# dir_name.sort()
# for dir in dir_name:
#     file_names = glob.glob(dir_path+'/'+dir+'/'+'*.png')
#     reader = sitk.ImageSeriesReader()
#     reader.SetFileNames(file_names)
#     vol = reader.Execute()
#     sitk.WriteImage(vol, dir_path+'/'+dir+'.nii.gz')



