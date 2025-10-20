import os
import cv2
import glob
import numpy as np
import shutil
from PIL import Image
# # ours from up to down,from left to right
# # "0": "background",
# # "1": "liver",
# # "2": "spleen",
# # "3": "left_kidney",
# # "4": "right_kidney",
# # "5": "stomach",
# # "6": "gallbladder",
# # "7": "pancreas",

#-------------print labelme output pixel----------------
#38(red):liver,75(green):spleen,14(blue):RK,113(brown):LK,52(pink):stomach,89(light blue):gallbladder,128(gray):pancreas
# cnt=0
# pp='/media/user/2tb/dataset/miccai2015pred/2d/frequencyunetr/img0001'
# patient=os.listdir(pp)
# patient.sort()
# for p in patient:
#     cnt+=1
#     path=pp+'/'+p
#     # dirs=os.listdir(path)
#     # dirs.sort()
#     color_all=[0,1,2,3,4,5,6,7]
#     img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
#     for i in range(img.shape[0]):
#         # if color==color_all:
#         #     break
#         for j in range(img.shape[1]):
#             # if img[i][j]not in color and img[i][j]>13:
#             if img[i][j]not in color_all:
#                 print("coloer no in:",pp,img[i][j])   
#                 # color.append(img[i][j])
#                 # print(color,'filename:',pp,'index:',i,j,'value:',img[i][j])
#     if cnt % 500==0:
#         print(cnt)

    # color.sort()       
    # # print("end process")      
    # print(p)
    # print(p,color)
    # color.sort()
    # if not(color == color_all):
    #     print("img:",p)
    #     print(color)

# # ours from up to down,from left to right
# # "0": "background",
# # "1": "liver",
# # "2": "spleen",
# # "3": "left_kidney",
# # "4": "right_kidney",
# # "5": "stomach",
# # "6": "gallbladder",
# # "7": "pancreas",

# # majei_init
# # "0": "background",
# # "4": "liver", 
# # "5": "spleen",
# # "6": "left_kidney",
# # "3": "right_kidney",
# # "7": "stomach",
# # "2": "gallbladder", 
# # "1": "pancreas",
# color=[0]
# img=cv2.imread('./dataset/Abdomen/majei_abdomen/patient/patient_labels/111685_20211204_CT_9_38/111685_20211204_CT_9_38_08.png',cv2.IMREAD_GRAYSCALE)
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         if img[i][j]not in color:    
#             color.append(img[i][j])
#             print("i:",i,"j:",j,"pixel_value=",img[i][j],"color_set:",color)

# # ours from up to down,from left to right
# # "0": "background",
# # "1": "liver",
# # "2": "spleen",
# # "3": "left_kidney",
# # "4": "right_kidney",
# # "5": "stomach",
# # "6": "gallbladder",
# # "7": "pancreas",

#38(red):liver,75(green):spleen,14(blue):RK,113(brown):LK,52(pink):stomach,89(light blue):gallbladder,128(gray):pancreas     

#-----------move file according to patient nums---------
path='/media/user/2tb/dataset/wordpred/frequency_unetr'
path2='/media/user/2tb/dataset/wordpred/3d'
dirs=os.listdir(path)
dirs.sort()
for dir in dirs:
    # if file == '..png':
    #     continue
    # a,b,c,d,e,f=file.split('_')
    # patient_num=a+'_'+b+'_'+c+'_'+d+'_'+e
    # if not os.path.isdir(path2+'/'+patient_num):
    #     os.mkdir(path2+'/'+patient_num)
    shutil.move(path+'/'+dir+'/'+dir+'.nii.gz',path2+'/'+dir+'.nii.gz')

   
#------------change label value to our setting------------
# path='/media/user/2tb/mmh_label/labels'
# path2='/media/user/2tb/mmh_label/labels2d/'
# dirs=os.listdir(path)
# dirs.sort()

# for dir in dirs:
#     # if not os.path.isdir(path2+dir):
#     #     os.mkdir(path2+dir)
#     # dirs2=os.listdir(path+'/'+dir)
#     # dirs2.sort()
#     # for ff in dirs2:
#     img=cv2.imread(path+'/'+dir,cv2.IMREAD_GRAYSCALE)
#     #set initial label under 8
#     img[img==38]=1
#     img[img==75]=2
#     img[img==113]=3
#     img[img==14]=4
#     img[img==52]=5
#     img[img==89]=6
#     img[img==128]=7
    
#     cv2.imwrite(path2+dir,img)

#--------------val json---------------------------
# path='/media/user/2tb/0503mmh_label/3d/imagesTr'
# files=os.listdir(path)
# files.sort()
# for file in files:
#     print("{{\n\"image\":{}\",\n\"label\":{}\",\n}}"
#           .format("\"imagesTr"+'/'+file,"\"labelsTr"+'/'+file))

#-----------rewrite MMH up down flip image----------
# image_path='/media/user/2tb/0503mmh_label/2d/imagesTr2d'
# write_path='/media/user/2tb/0503mmh_label/2d/up_down_flip/imagesTr'
# image_list=os.listdir(image_path)
# image_list.sort()
# for image in image_list:
#     img=cv2.imread(image_path+'/'+image,cv2.IMREAD_GRAYSCALE)
#     img=cv2.flip(img, 0)#verticle flip
#     cv2.imwrite(write_path+'/'+image,img)
