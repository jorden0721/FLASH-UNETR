import cv2
import numpy as np

def apply_mask_with_transparency(image_path, mask_path, alpha=0.5, color=(0, 255, 0)):
    """
    将分割掩码应用到原图上，并设置透明度
    :param image: 原始图像
    :param mask: 分割掩码，二值图像
    :param alpha: 透明度，值在0到1之间
    :param color: 掩码颜色 (B, G, R)
    :return: 叠加后的图像
    """
    # 创建一个彩色的掩码图像
    image = cv2.imread(image_path)
    image=np.flipud(image)
    mask = cv2.imread(mask_path)

    # 叠加彩色掩码到原图上
    overlay = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)
    
    return overlay

def main():
    # 读取原始图像和分割掩码
    image_path = '/media/user/2tb/dataset/btcvinit/2d/img0022/068.png'
    mask_path = '/media/user/2tb/dataset/miccai2015pred/2d_rgb/nnunet/img0022/068.png'
    
    
    # 设置透明度和颜色
    alpha = 0.5  # 透明度
    color = (0, 255, 0)  # 掩码颜色，绿色
    
    # 应用透明掩码
    result = apply_mask_with_transparency(image_path, mask_path, alpha, color)
    
    # 显示结果
    cv2.imshow('Overlay', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果
    # result_path = 'path_to_save_overlay_image.jpg'
    # cv2.imwrite(result_path, result)

if __name__ == "__main__":
    main()
