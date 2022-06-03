# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 09:32:09 2022

@author: user
"""

import pandas as pd
from datetime import datetime
import numpy as np

datos = pd.DataFrame({'NOMBRE': ['ELSY YULIANA SILGADO RIVERA','DEIMER DAVID MORELO OSPINO','JESUS DAVID PORTILLO VILLA','JONNIER ANDRES TERAN MORALES','JONATHAN EMILIO BRITO AVILEZ','GUSTAVO JAVIER BLANCO JARAMILLO','DAYAN ALZATE HERNANDEZ','WENDY PAOLA MENDOZA BARRERA','ALVARO JOSE ZAMORA CURY','GABRIELA TORRES RAMOS','MIGUEL ALFONSO FABRIS AVILA','JHONY ALBERTO MELENDEZ CAVADIA','ADRIANA LUCIA SUAREZ BALLESTEROS','JAIDER STIVEN ECHEVERRI BEDOYA','NOIVER DARIO BARROSO RAMOS','Miriam Rosa Lopez Diaz','IVAN DARIO PALENCIA ALEAN','ANGELA JOHANNA POSSO DIAZ'],
                     'EDAD': [20,21,22,23,24,25,26,27,28,20,21,22,23,24,25,26,27,28],
                     'SEXO': ['F','M','M','M','M','M','F','F','M','F','M','M','F','M','M','F','M','F'],
                     'PESO': [60,70,72,72,74,74,62,61,76,63,78,80,59,74,70,63,73,64],
                     'ALTURA': [163,178,175,171,177,165,161,172,179,154,199,161,170,171,173,184,173,175],
                     'DINERO_INVERTIR': [10000000,17500000,2000000,3500000,7000000,12250000,21437500,4000000,7000000,8000000,14000000,24500000,42875000,10000000,17500000,30625000,13000000,22750000],
                     'INTERES_ANUAL': [0.05,0.06,0.055,0.063,0.0625,0.065,0.0675,0.07,0.0725,0.075,0.0775,0.05,0.06,0.055,0.063,0.0625,0.065,0.0725],
                     'ANOS_INVERSION': [1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3],
                     'TELEFONO': ['300257943709','300630400602','300307353606','300666525826','300748195684','300676935047','300560691850','300144616992','300183280926','300297927344','300867152922','300136881442','300751105846','300874719927','300311227835','300549598161','300817376161','300567927711'],
                     'HORA_COMPRA_PAN':['7:00:00','7:10:00','11:15:00','13:20:00','18:30:00','21:00:00','16:00:00','14:10:00','18:15:00','19:30:00','22:30:00','7:20:00','17:00:00','18:10:00','16:15:00','8:00:00','18:30:00','10:00:00']})


# Ejercicio 1: 
# A partir de la tabla, usando el peso (en kg) y estatura (en metros),
# calcule el índice de masa corporal de cada individuo y lo almacene en
# una variable, Muestre por pantalla la frase “Tu índice de masa corporal
# es <imc>” donde <imc> es el índice de masa corporal calculado
# redondeado con dos decimales.

for i in datos.index:
    imc = round(((datos["PESO"][i])/((datos["ALTURA"][i])/100)),2)
    print("""hola {} Tu índice de masa corporal es {}""".format(str(datos["NOMBRE"][i]),str(imc)))

# Otro método es agregarlo al DataFrame como una columna
datos["IMC"] = datos["PESO"]/(datos["ALTURA"]/100)


# Ejercicio 2
# A partir de los datos recolectados: Escribir un programa que teniendo en
# cuenta una cantidad a invertir, el interés anual y el número de años,
# muestre por pantalla el capital obtenido en la inversión.

for i in datos.index:
    capital = round(datos["DINERO_INVERTIR"][i]*(((datos["INTERES_ANUAL"][i])/(100+1))**datos["ANOS_INVERSION"][i]),2)
    print("""hola {} el capital obtenido en la inversión {}""".format(str(datos["NOMBRE"][i]),str(capital)))

# Otro método es agregarlo al DataFrame como una columna
datos["CAPITAL_INVERSION"] = round(datos["DINERO_INVERTIR"]*(((datos["INTERES_ANUAL"])/(100+1))**datos["ANOS_INVERSION"]),2)


#  Ejercicio 3
# Una panadería vende barras de pan a $15,000 COP cada una. El pan
# tiene un descuento del 10%, 20%, 30%, 40%, cuando no se vende en las
# primeras 6h, 12h, 18h, 24h, después de horneado. Crear una columna
# en el DataFrame para determinar el porcentaje de descuento obtenido
# de acuerdo a la hora en que fue realizada la compra. Y otra columna
# para el precio final obtenido.

# Asumimos la hora de hornear el pan a las 4:15 am y el precio del pan es 7.000

datos["HORAS_POST_HORNEAR"] = ((pd.to_timedelta(datos["HORA_COMPRA_PAN"])-pd.to_timedelta("4:15:00")).dt.total_seconds())//3600

condiciones = [
    (datos["HORAS_POST_HORNEAR"] >= 0) & (datos["HORAS_POST_HORNEAR"]<6),
    (datos["HORAS_POST_HORNEAR"] >= 6) & (datos["HORAS_POST_HORNEAR"]<12),
    (datos["HORAS_POST_HORNEAR"] <= 12) & (datos["HORAS_POST_HORNEAR"]<18),
    (datos["HORAS_POST_HORNEAR"] <= 18) & (datos["HORAS_POST_HORNEAR"]<24)]
selecciones = [0.1, 0.2, 0.3, 0.4]

datos["PORCENTAJE_DESCUENTO"] =  np.select(condiciones, selecciones, default='Not Specified')
datos["PRECIO_FINAL"] = 7000-(pd.to_numeric(datos["PORCENTAJE_DESCUENTO"], errors='coerce') *7000)

#  Ejercicio 4
# Los teléfonos de sus compañeros tienen el siguiente formato prefijonúmero-extensión donde el prefijo es el código del país +57, y la
# extensión tiene dos dígitos (por ejemplo +57-913724710-##). Debe
# organizar en el DataFrame (nueva columna) las extensiones de forma
# que si el sexo de la persona es M, debe poner como extensión 11 y si el
# sexo es F, debe poner como extensión 10.

condiciones_ext = [
    (datos["SEXO"] == "F"),
    (datos["SEXO"] == "M")]
selecciones_ext = [10, 11]


datos["EXTENCION_CEL"] = np.select(condiciones_ext, selecciones_ext, default='Not Specified')