import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import copy
from PIL import Image
import os
import cv2

path='/media/user/2tb/dataset/mmhpred/2d/MMH_frequencyunetr'
out_path='/media/user/2tb/dataset/mmhpred/2d_rgb/MMH_frequencyunetr'
dirs=os.listdir(path)
dirs.sort()

for dir in dirs:
    # print(f)
    files=os.listdir(path+'/'+dir)
    for file in files:
        img = cv2.imread(path+'/'+dir+'/'+file,cv2.IMREAD_GRAYSCALE)
        img = np.asarray(img)
        # img=np.flipud(img)
        a1 = copy.deepcopy(img)
        a2 = copy.deepcopy(img)
        a3 = copy.deepcopy(img)
        #1:Red,2:Green,3:Blue,4:Yellow,5:Orange,6:Pink,7:Purple
        #1:Pancreas,2:gallbladder,3:right kidney,4:liver,5:spleen,6:left kidney,7:stomach
        #terminal=R
        #1:紅色:胰臟，2:綠色:膽囊，3:藍色:右腎，4:黃色:肝臟，5:橘色:脾臟，6:粉紅色:左腎，7:紫色:胃，
        # print(a1[a1>0])
        # a1[a1 == 1] = 255
        a1[a1 == 1] = 255
        a1[a1 == 2] = 0
        a1[a1 == 3] = 0
        a1[a1 == 4] = 255
        a1[a1 == 5] = 255
        a1[a1 == 6] = 255
        a1[a1 == 7] = 160

        #terminal=G
        # a2[a2 == 1] = 255
        a2[a2 == 1] = 0
        a2[a2 == 2] = 255
        a2[a2 == 3] = 0
        a2[a2 == 4] = 255
        a2[a2 == 5] = 97
        a2[a2 == 6] = 0
        a2[a2 == 7] = 32


        #terminal=B
        # a3[a3 == 1] = 255
        a3[a3 == 1] = 0
        a3[a3 == 2] = 0
        a3[a3 == 3] = 255
        a3[a3 == 4] = 0
        a3[a3 == 5] = 0
        a3[a3 == 6] = 255
        a3[a3 == 7] = 240

        a1 = Image.fromarray(np.uint8(a1)).convert('L')
        a2 = Image.fromarray(np.uint8(a2)).convert('L')
        a3 = Image.fromarray(np.uint8(a3)).convert('L')
        out_img = Image.merge('RGB', [a1, a2, a3])
        if not os.path.isdir(out_path+'/'+dir):
            os.mkdir(out_path+'/'+dir)
        out_img.save(out_path+'/'+dir+'/'+file)
        # img = cv2.cvtColor(np.asarray(out_img),cv2.COLOR_RGB2BGR) 
        # cv2.imshow(f,img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
