# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 07:18:01 2022

@author: user
"""


from scipy import ndimage
from skimage import io
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Actividad

# 1) aplicar a una imagen el filtro gausiano

img = io.imread("Esposa.jpg")
plt.imshow(img, cmap='gray')
blurred_img = ndimage.gaussian_filter(img, sigma=3)
plt.imshow(blurred_img, cmap='gray')

# 2) aplicar el filtro de media

filtromedia = ndimage.uniform_filter(img, size=11)
plt.imshow(filtromedia, cmap='gray')

# 3) aplicar el realce o el enfoque

filtro_blurred_img = ndimage.gaussian_filter(blurred_img, 1)

alpha= 15
enfoque = blurred_img+alpha *(blurred_img-filtro_blurred_img)
plt.imshow(enfoque, cmap='gray')

# 4) aplicar filtro canny

canny = cv2.Canny(img, 50, 150)
plt.imshow(canny, cmap='gray')

# 5) aplicar filtro sobel


sobelX = cv2.Sobel(img, cv2.CV_64F, 1,0)
sobelY = cv2.Sobel(img, cv2.CV_64F, 0,1)
    
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
    
sobel_final = cv2.bitwise_or(sobelX, sobelY)
plt.imshow(sobel_final, cmap='gray')


# 6) visualizar todo en un subplot


plt.figure(figsize=(9,3))
plt.subplot(231)
plt.title("Imagen original")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(232)
plt.title("Imagen borrosa")
plt.imshow(blurred_img, cmap='gray')
plt.axis("off")

plt.subplot(233)
plt.title("Imagen filtro media")
plt.imshow(filtromedia, cmap='gray')
plt.axis("off")

plt.subplot(234)
plt.title("Imagen Enfoque")
plt.imshow(enfoque, cmap='gray')
plt.axis("off")

plt.subplot(235)
plt.title("Imagen filtro Canny")
plt.imshow(canny, cmap='gray')
plt.axis("off")

plt.subplot(236)
plt.title("Imagen filtro Sobel")
plt.imshow(sobel_final, cmap='gray')
plt.axis("off")


plt.subplots_adjust(wspace=0, hspace=0, top=2,
                    bottom=0.01, left=0.01,
                    right=0.99)