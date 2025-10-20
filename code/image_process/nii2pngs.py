# import numpy as np
# import os #遍历文件夹
# import nibabel as nib
# import imageio #转换成图像

# def nii_to_image(niifile):
#     filenames = os.listdir(filepath)  #读取nii文件
#     slice_trans = []

#     for f in filenames:
#         #开始读取nii文件
#         img_path = os.path.join(filepath, f)
#         img = nib.load(img_path)  #读取nii
#         img_fdata = img.get_fdata()
#         fname = f.replace('.nii', '') #去掉nii的后缀名
#         img_f_path = os.path.join(imgfile, fname)
#         # 创建nii对应图像的文件夹
#         if not os.path.exists(img_f_path):
#             os.mkdir(img_f_path)  #新建文件夹

#         #开始转换图像
#         (x,y,z) = img.shape
#         for i in range(z):   #是z的图象序列
#             slice = img_fdata[i, :, :]  #选择哪个方向的切片自己决定
#             imageio.imwrite(os.path.join(img_f_path, '{}.png'.format(i)), slice)

# if __name__ == '__main__':
#     filepath = '/home/user/Documents/dataset/Abdomen/BTCV_Abdomen/Abdomen/RegData/Training-Training/label/0001'
#     imgfile = '/home/user/Documents/dataset/Abdomen/BTCV_Abdomen/BTCVtest/labelsTr'
#     nii_to_image(filepath)
import scipy, shutil, os
import sys, getopt
import imageio
from tqdm import tqdm
import nibabel as nib
import numpy as np
import SimpleITK as sitk


def matrix2uint8(matrix):
    ''' 
matrix must be a numpy array NXN
Returns uint8 version
    '''
    m_min= np.min(matrix)
    m_max= np.max(matrix)
    matrix = matrix-m_min
    return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 255.0),dtype=np.uint8))

def niito2D(filepath, outputpath):
    center = 50 #窗宽窗位
    width = 400
    inputfiles = os.listdir(filepath)  # 遍历文件夹数据
    outputfile = outputpath
    print('Input file is   : ', inputfiles)
    print('Output folder is: ', outputfile)
    file_count = 0  # 文件计数器
    for inputfile in inputfiles:
        if not os.path.isdir(outputpath+'/'+inputfile[:-7]):
            os.mkdir(outputpath+'/'+inputfile[:-7])
        image = nib.load(filepath + inputfile)
        image_array = image.get_fdata()  # 数据读取

         # 转换成window level,window width
        min = (2 * center - width) / 2.0 + 0.5
        max = (2 * center + width) / 2.0 + 0.5
        dFactor = 255.0 / (max - min)

        # print(len(image_array.shape))
        file_count = file_count + 1
        (x, y , z) = image_array.shape  # 获得数据shape信息：（长，宽，维度-切片数量）
        # print(x,y,z)
        # 不同3D体数据有用的切片数量不同，自行查看，自行设定起止数量 512*512*147
        total_slices = z # 总切片数
        slice_counter = 0 # 从第几个切片开始

        loop = tqdm(range(slice_counter, slice_counter + z)) 
        # loop = tqdm(range(slice_counter, slice_counter + z)) 
        for current_slice in loop:
            if (slice_counter % 1) == 0:
                data = image_array[:, :, current_slice].astype(np.uint8) #labels
                # data = image_array[:, :, current_slice]
                # data=data-min
                # data=np.trunc(data* dFactor)
                # data[data < 0.0] = 0
                # data[data > 255.0] = 255                # 转换为窗位窗位之后的数据
                data=np.rot90(data,-1)
                data=np.flip(data,axis=1)
                # data=np.flipud(data)
                if (slice_counter % 1) == 0:
                    # 切片命名
                    image_name = outputpath+'/'+inputfile[:-7] +'/'+ "{:0>3}".format(str(current_slice+1)) + ".png"
                    # 保存
                    # imageio.imwrite(image_name, data)
                    imageio.imwrite(image_name, data[int((x - 512) / 2):int((x - 512) / 2) + 512])

                    # 移动到输出文件夹
                    src = image_name
                    # shutil.move(src, outputfile)
                    slice_counter += 1
                    
                    loop.set_description(f'文件数：[{file_count}/{len(inputfiles)}]')
    print('Finished converting images')


if __name__ == '__main__':
    # input_path = './dataset/Abdomen/BTCV_Abdomen/Abdomen/RawData/Testing/img/'
    # output_path = './dataset/Abdomen/BTCV_Abdomen/window_bound/imagesTs'
    # input_path ='./unetr/research-contributions/UNETR/BTCV/dataset/datasetmajei/swinunetr_mmh_final/'
    # output_path ='./unetr/research-contributions/UNETR/BTCV/dataset/datasetmajei/swinunetr_mmh_final_pngs'
    
    input_path ='/home/user/Documents/unetr/research-contributions/UNETR/BTCV/dataset/dataset0/imagesTr/'
    output_path ='/media/user/2tb/dataset/btcvinit/2d'
    niito2D(input_path, output_path)









