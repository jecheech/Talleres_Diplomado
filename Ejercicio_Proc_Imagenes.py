# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:45:34 2022

@author: user
"""
# Librerias necesarias
from skimage import io
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt



#1) Cargar una imagen rgb o escala de grises
img = io.imread("Esposa.jpg")
plt.imshow(img, cmap='gray')


#2)  Recortar la imagen
lx, ly = img.shape
imgCortada = img[lx//4:-lx//4, ly//4:-ly//4]
plt.imshow(imgCortada, cmap='gray')

#3) Agregar elementos a la imagen
imgElementos = img
X,Y = np.ogrid[0:lx, 0:ly]
# Se crean los elementos a agregar
mask = (X-lx)**2+(Y-ly)**2 > lx*ly/2
mask2 = (X-lx)**3+(Y-ly)**3 > lx*ly*250
# Se reemplazan los elementos por colores negros y blancos
imgElementos[mask]=0
imgElementos[mask2]=255
plt.imshow(imgElementos, cmap='gray')
# Se gira la imagen hacia arriba
imgElementosflipup = np.flipud(imgElementos)
plt.imshow(imgElementosflipup, cmap='gray')
# Se gira la imagen hacia el lado
imgElementosfliplr = np.fliplr(imgElementos)
plt.imshow(imgElementosfliplr, cmap='gray')
# Se gira la imagen hacia arriba
imgElementosflip = np.flip(imgElementos, 0)
plt.imshow(imgElementosflip, cmap='gray')
# Se rota la imagen 30 grados
imgRotada = ndimage.rotate(img, 30)
plt.imshow(imgRotada, cmap='gray')
# Se rota la imagen 45 grados
imgRotada2 = ndimage.rotate(img, 45, reshape=False)
plt.imshow(imgRotada2, cmap='gray')


#4) Visualizar en un solo subplot todas las imagenes

plt.subplot(241)
plt.imshow(img, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(242)
plt.imshow(imgCortada, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(243)
plt.imshow(imgElementos, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(244)
plt.imshow(imgElementosflipup, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(245)
plt.imshow(imgElementosfliplr, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(246)
plt.imshow(imgElementosflip, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(247)
plt.imshow(imgRotada, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(248)
plt.imshow(imgRotada2, cmap=plt.cm.gray)
plt.axis("off")

#5) usar los ajustes del subplot y el figsize para mejorar la calidad 

fig = plt.figure(figsize=(12.5, 2.5), facecolor="c")

fig.suptitle("Hola", x= 0.5, y=0.5, fontsize=100,
             fontstyle="normal", fontfamily="fantasy", color="gray",
             backgroundcolor= "blue", rotation=10)

plt.subplot(241)
plt.imshow(img, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(242)
plt.imshow(imgCortada, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(243)
plt.imshow(imgElementos, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(244)
plt.imshow(imgElementosflipup, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(245)
plt.imshow(imgElementosfliplr, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(246)
plt.imshow(imgElementosflip, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(247)
plt.imshow(imgRotada, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(248)
plt.imshow(imgRotada2, cmap=plt.cm.gray)
plt.axis("off")


plt.subplots_adjust(wspace=0.05, hspace=0.3, top =1, bottom =0.1, left = 0, right = 1)