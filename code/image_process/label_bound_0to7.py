import os
import cv2
import numpy as np
#BTCV 13 organs
# V(1) spleen
# V(2) right kidney
# V(3) left kidney
# V(4) gallbladder
# (5) esophagus
# V(6) liver
# V(7) stomach
# (8) aorta
# (9) inferior vena cava
# (10) portal vein and splenic vein
# V(11) pancreas
# (12) right adrenal gland
# (13) left adrenal gland

#word 
# "0": "background",
# "1": "liver",
# "2": "spleen",
# "3": "left_kidney",
# "4": "right_kidney",
# "5": "stomach",
# "6": "gallbladder",
# "7": "esophagus",
# "8": "pancreas",
# "9": "duodenum",
# "10": "colon",
# "11": "intestine",
# "12": "adrenal",
# "13": "rectum",
# "14": "bladder",
# "15": "Head_of_femur_L",
# "16": "Head_of_femur_R"

#majei_init
#"0": "background",
#"4": "liver", 
#"5": "spleen",
#"6": "left_kidney",
#"3": "right_kidney",
#"7": "stomach",
#"2": "gallbladder", 
#"1": "pancreas",

#ours from up to down,from left to right
# "0": "background",
# "1": "liver",
# "2": "spleen",
# "3": "left_kidney",
# "4": "right_kidney",
# "5": "stomach",
# "6": "gallbladder",
# "7": "pancreas",
# path='./dataset/Abdomen/WORD_Abdomen/labelsTr'
# path2='./dataset/Abdomen/WORD_Abdomen/labels0to8/'
path='./dataset/Abdomen/WORD_Abdomen/labelsVal2d'
path2='./dataset/Abdomen/WORD_Abdomen/labelsVal2d0to7/'
dirs=os.listdir(path)
dirs.sort()

for dir in dirs:
    if not os.path.isdir(path2+dir):
        os.mkdir(path2+dir)
    dirs2=os.listdir(path+'/'+dir)
    dirs2.sort()
    for ff in dirs2:
        img=cv2.imread(path+'/'+dir+'/'+ff,cv2.IMREAD_GRAYSCALE)
        # cv2.imshow('tt',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        #organ we don't care set to 0
        img[img==0]=0
        img[img==7]=0
        img[img==9]=0
        img[img==10]=0
        img[img==11]=0
        img[img==12]=0
        img[img==13]=0
        img[img==14]=0
        img[img==15]=0
        img[img==16]=0
        # img[img==0]=0
        # img[img==5]=0
        # img[img==8]=0
        # img[img==9]=0
        # img[img==10]=0
        # img[img==12]=0
        # img[img==13]=0
        #set initial lable out to 8
        # img[img==6]=21
        # img[img==1]=22
        # img[img==3]=23
        # img[img==2]=24
        # img[img==7]=25
        # img[img==4]=26
        # img[img==11]=27
        #reset label
        img[img==8]=7
        # img[img==21]=1
        # img[img==22]=2
        # img[img==23]=3
        # img[img==24]=4
        # img[img==25]=5
        # img[img==26]=6
        # img[img==27]=7
        # cv2.imshow('tt2',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(path2+dir+'/'+ff,img)
