# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 17:41:08 2022

@author: user
"""

from skimage import io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

# 1 Cargar una imagen cualquiera
imagen = io.imread("ejemplo.jpg")

# 2 Cambiar la imagen a escala de grises
imagen_grises = rgb2gray(imagen)

# 3 Visualizar la imagen a color
plt.imshow(imagen)

# 4 Visualizar la imagen a escala de grises
plt.imshow(imagen_grises, cmap='gray')

# 5 Recortar area de la imagen a color
imgColorRecortada = imagen[5:200, 5:220]

# 6 Recortar la imagen a escala de grises
imgGrisesRecortada = imagen_grises[5:180, 5:150]

# 7 Dimensiones de imagen recortada color
imgColorRecortada.shape

# 8 Dimensiones de imagen recortada grises
imgGrisesRecortada.shape

# Visualizar imagenes recortadas
fig, axs = plt.subplots(2)
fig.suptitle('Imagenes recortadas')
axs[0].imshow(imgColorRecortada)
axs[1].imshow(imgGrisesRecortada)
