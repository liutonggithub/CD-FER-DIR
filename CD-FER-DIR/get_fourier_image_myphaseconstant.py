# -*- coding:utf-8 -*-


from data.DGDataReader import *

import cv2
import numpy as np




def colorful_spectrum_mix_my(img1, img2, alpha, ratio=1.0):
    """Input image size: ndarray of [H, W, C]"""
    lam = np.random.uniform(0, alpha)

    assert img1.shape == img2.shape
    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    # 傅里叶变换得到频率分布
    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))

    # fft 结果是复数，其绝对值结果是振幅img1_abs,img2_abs，相位是img1_pha,img2_pha
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    # 默认结果中心点位置是在左上角 调用fftshift()函数转移到中间位置
    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))

    #  傅里叶逆变换
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12

def get_spectrum(img):
    img_fft = np.fft.fft2(img)
    img_abs = np.abs(img_fft)
    img_pha = np.angle(img_fft)
    return img_abs, img_pha

def get_centralized_spectrum(img):
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    img_abs = np.abs(img_fft)
    img_pha = np.angle(img_fft)
    return img_abs, img_pha

def spectrum_constant_my(img1, img2, alpha, ratio=1.0):
    """Input image size: ndarray of [H, W, C]"""
    lam = np.random.uniform(0, alpha)

    assert img1.shape == img2.shape
    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    # 傅里叶变换得到频率分布
    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))

    # fft 结果是复数，其绝对值结果是振幅img1_abs,img2_abs，相位是img1_pha,img2_pha
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    # # 默认结果中心点位置是在左上角 调用fftshift()函数转移到中间位置
    # img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    # img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    # img1_abs_ = np.copy(img1_abs)
    # img2_abs_ = np.copy(img2_abs)
    # img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
    #     lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
    #                                                                                       h_start:h_start + h_crop,
    #                                                                                       w_start:w_start + w_crop]
    # img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
    #     lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
    #                                                                                       h_start:h_start + h_crop,
    #                                                                                       w_start:w_start + w_crop]
    #
    # img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    # img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    # 把振幅设为常数
    # img1_abs_ = img1_abs.mean()
    # img2_abs_ = img2_abs.mean()
    img1_abs_ = img1_abs.std()
    img2_abs_ = img2_abs.std()
    # img1_abs_ = np.ones((h, w, c))
    # img2_abs_ = np.ones((h, w, c))

    # 把幅度谱和相位谱再合并为复数形式的频域图数据
    img1 = img1_abs_ * (np.e ** (1j * img1_pha))
    img2 = img2_abs_ * (np.e ** (1j * img2_pha))

    #  傅里叶逆变换#还原为空间域图像
    img1 = np.real(np.fft.ifft2(img1, axes=(0, 1)))
    img2 = np.real(np.fft.ifft2(img2, axes=(0, 1)))

    img1_my = np.uint8(np.clip(img1, 0, 255))
    img2_my = np.uint8(np.clip(img2, 0, 255))

    return img1_my, img2_my

def spectrum_constant_phase(img1):
    """Input image size: ndarray of [H, W, C]"""

    h, w, c = img1.shape
    # h_crop = int(h * sqrt(ratio))
    # w_crop = int(w * sqrt(ratio))
    # h_start = h // 2 - h_crop // 2
    # w_start = w // 2 - w_crop // 2

    # 傅里叶变换得到频率分布
    img1_fft = np.fft.fft2(img1, axes=(0, 1))


    # fft 结果是复数，其绝对值结果是振幅img1_abs,img2_abs，相位是img1_pha,img2_pha
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)



    # 把phase相位设为常数
    img1_pha_ = img1_pha.mean()
    # img1_pha_ = img1_pha.std()
    # # img2_abs_ = img2_abs.mean()
    # img1_abs_ = img1_abs.std()
    print(img1_pha_)

    # img1_abs_ = np.ones((h, w, c))
    # img2_abs_ = np.ones((h, w, c))

    # 把幅度谱和相位谱再合并为复数形式的频域图数据
    img1 = img1_abs * (np.e ** (1j * img1_pha_))


    #  傅里叶逆变换#还原为空间域图像
    img1 = np.real(np.fft.ifft2(img1, axes=(0, 1)))


    img1_my = np.uint8(np.clip(img1, 0, 255))


    return img1_my


img1=cv2.imread('F:/paper_4/my_yong/Oulu-CASIA.jpg')
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2=cv2.imread('F:/paper_4/happiness_1.jpg')

# image1,image2= colorful_spectrum_mix_my(img1,img2, alpha=1.0, ratio=1.0)
# image1,image2= spectrum_constant_my(img1,img2, alpha=1.0, ratio=1.0)
image1= spectrum_constant_phase(img1)

# show
cv2.imshow('window_title', image1)
# cv2.imshow('window_title', image2)
# save
# cv2.imwrite('F:/paper_4/0_3.jpg', image1)
# cv2.imwrite('F:/paper_4/0_28.jpg', image2)

cv2.imwrite('F:/paper_4/my_yong/Oulu-CASIA_amplitude.jpg', image1)
# cv2.imwrite('F:/paper_4/happiness_1_constant.jpg', image2)