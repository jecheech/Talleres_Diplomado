# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 16:45:04 2022

@author: user
"""
import cv2
# Actividad
# 1) cargar una imagen con Open CV

img = cv2.imread("paisaje.jpg")
img = cv2.resize(img, (743, 640))
cv2.imshow("paisaje", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 2) Cargar un Video Con Open CV

video = cv2.VideoCapture("Balanceo_Tiempos.mp4")

while True:
    estado, imagen = video.read()
    cv2.imshow("video", imagen) 
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    
video.release()
cv2.destroyAllWindows()


# 3) Grabar un video corto con la camara del PC

video_streaming = cv2.VideoCapture(0)
salida = cv2.VideoWriter("Video_Actividad.avi", cv2.VideoWriter_fourcc(*'XVID'),
                         20.0, (640,480))

while (video_streaming.isOpened()):
    estado, img_stream = video_streaming.read()
    if estado == True:
        cv2.imshow("Video Streaming", img_stream)
        salida.write(img_stream)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    else:
        break
    
video_streaming.release()
salida.release()
cv2.destroyAllWindows()


# 4) Buscar contornos de una imagen
# 5) contar los contornos en la imagen del chat

img = cv2.imread("contornos.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgGauss = cv2.GaussianBlur(imgGray, (15,15),0)
imgCanny = cv2.Canny(imgGauss, 150, 250)
contornos, _ = cv2.findContours(imgCanny,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

print("He encontrado {} objetos".format(len(contornos)))

cv2.drawContours(img, contornos, -1, (0,0,255), 2)

cv2.imshow("contornos", img)
cv2.imshow("contornos Escala de Grises", imgGray)
cv2.imshow("contornos desenfoque Gausiano", imgGauss)
cv2.imshow("contornos Filtro Canny", imgCanny)

cv2.waitKey(0)
cv2.destroyAllWindows()


# 5) aplicar la tecnica de dibujo a una foto personal

img = cv2.imread("Foto2.jpg")
img = cv2.resize(img, (480, 640))

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgGrayINV = 255-imgGray

imgGauss = cv2.GaussianBlur(imgGrayINV, (21,21),0)

imgGaussINV = 255-imgGauss

imgLapiz = cv2.divide(imgGray, imgGaussINV, scale=256.0)

cv2.imshow("Foto", img)
cv2.imshow("Foto Escala de Grises invertido", imgGrayINV)
cv2.imshow("Foto desenfoque Gausiano invertido", imgGaussINV)
cv2.imshow("Foto Lapiz", imgLapiz)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 6) aplicar conteo de objetos en tiempo real


video = cv2.VideoCapture(0)
salida = cv2.VideoWriter("Video_Contornos.avi", cv2.VideoWriter_fourcc(*'XVID'),
                         20.0, (640,480))

while True:
    estado, img = video.read()
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGauss = cv2.GaussianBlur(imgGray, (9,9),0)
    imgCanny = cv2.Canny(imgGauss, 50, 150)
    contornos, _ = cv2.findContours(imgCanny,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    cv2.putText(img,  
                "He encontrado {} contornos".format(len(contornos)),  
                (50, 50),  
                font, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4)
    
    salida.write(img)
    cv2.imshow("Video", img)
    #cv2.imshow("Video diff", imgBlur)
    
    key = cv2.waitKey(10)
    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()