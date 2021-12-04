# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 00:57:47 2021

@author: Rearo
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import glob


def DFFTnp(img):
    f = np.fft.fft2(img) # двумерное дискретное преобразование Фурье
    fshift = np.fft.fftshift(f) # сдвигает компонент нулевой частоты в центр спектра
    return fshift

def reverseDFFTnp(dfft):
    f_ishift = np.fft.ifftshift(dfft) # обратное к fftshift
    reverse_image = np.fft.ifft2(f_ishift) # двумерное обратное дискретное преобразование Фурье
    return reverse_image

# фильтр Гаусса
def Gaussian(img, fshift):
    ksize = 21
    kernel = np.zeros(img.shape) # ядро
    blur = cv.getGaussianKernel(ksize, -1)
    blur = np.matmul(blur, np.transpose(blur))
    kernel[0:ksize, 0:ksize] = blur
    fkshift = DFFTnp(kernel)
    mult = np.multiply(fshift, fkshift)
    reverse_image = reverseDFFTnp(mult)
    return reverse_image

images = glob.glob('__129.png')
for image in images:
    img = np.float32(cv.imread(image, 0))
    fshift = DFFTnp(img)

    plt.subplot(221), plt.title('Input spectrum'), plt.xticks([]), plt.yticks([])
    plt.imshow(np.abs(fshift), cmap = "gray", norm = LogNorm(vmin = 5))

    w, h = fshift.shape # возьмем ширину и высоту изображения
    maxpix = fshift[w//2][h//2]
    for i in range(w):
        for j in range(h):
            if i != w//2 and j != h//2:
                if abs(np.abs(fshift[i][j])-np.abs(maxpix)) < np.abs(maxpix) - 250000:
                    fshift[i][j] = 0

    plt.subplot(222), plt.title('Notch filter'), plt.xticks([]), plt.yticks([])
    plt.imshow(np.abs(fshift), cmap = "gray", norm = LogNorm(vmin = 5))

    reverse_image = Gaussian(reverseDFFTnp(fshift), fshift)
    
    plt.subplot(223), plt.title('Input image'), plt.xticks([]), plt.yticks([])
    plt.imshow(abs(img), cmap = 'gray')
    plt.subplot(224), plt.title('Gaussian result'), plt.xticks([]), plt.yticks([])
    plt.imshow(abs(reverse_image), cmap = 'gray')

    plt.show()
    
