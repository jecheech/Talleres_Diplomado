# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 08:27:14 2022

@author: Jaider Stiven Echeverri Bedoya
"""
#Librerias incluidas
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

print("#---------------------------Actividad con BD Cars Regresion Polinomial-------------------------------------")
# Importe de la base de datos
cars_df = pd.read_csv('cars2.csv')

# Variables independiente
x_cars = cars_df["Weight"]

#Definimos la variable dependiente
y_cars = cars_df["CO2"]

# Estandarizacion de las variables
scale_cars = StandardScaler()
scale_x_cars =(cars_df["Weight"]-cars_df["Weight"].mean())/cars_df["Weight"].std()

#Train / Test
x_train_cars = scale_x_cars[:28]
y_train_cars = y_cars[:28]

x_test_cars = scale_x_cars[28:]
y_test_cars = y_cars[28:]


# Grafica de variables
plt.scatter(x_train_cars,y_train_cars)
plt.show()

plt.scatter(x_test_cars,y_test_cars)
plt.show()

#Modelo de Regresion Multiple
poli_cars =  np.poly1d(np.polyfit(x_train_cars,y_train_cars,4))
myline = np.linspace(0,2,100)
poli_new_y = poli_cars(myline)

#GRafica del modelo
plt.scatter(x_train_cars,y_train_cars)
plt.plot(myline,poli_new_y )
plt.show()


#Prediccion del modelo
print(poli_cars(0.1))

#R de relacion train y test
r2_train = r2_score(y_train_cars, poli_cars(x_train_cars))
print(r2_train)

r2_test = r2_score(y_test_cars, poli_cars(x_test_cars))
print(r2_test)



print("#---------------------------Actividad con BD Cars Regresion Multiple-------------------------------------")
# Importe de la base de datos
cars_df = pd.read_csv('cars2.csv')

# Variables independiente
x_cars = cars_df[["Weight","Volume"]]

#Definimos la variable dependiente
y_cars = cars_df["CO2"]

# Estandarizacion de las variables
scale_cars = StandardScaler()
scale_x_cars =scale_cars.fit_transform(x_cars)

#Train / Test
x_train_cars = scale_x_cars[:28]
y_train_cars = y_cars[:28]

x_test_cars = scale_x_cars[28:]
y_test_cars = y_cars[28:]


#Modelo de Regresion Multiple
modelo_cars = linear_model.LinearRegression()
modelo_cars.fit(x_train_cars,y_train_cars)

#Prediccion del modelo
pred_scale_x_cars = modelo_cars.predict([x_test_cars[0]])
print(pred_scale_x_cars)

#R de relacion train y test
r2_train = r2_score(y_train_cars, modelo_cars.predict(x_train_cars))
print(r2_train)

r2_test = r2_score(y_test_cars, modelo_cars.predict(x_test_cars))
print(r2_test)

print("#---------------------------Actividad con BD Students-------------------------------------")
# Importe de la base de datos
student_df = pd.read_csv('student_por.csv')

# Variables independiente
x_student = student_df[["age","freetime"]]                    
            
#Definimos la variable dependiente
y_student = student_df["health"]

# Estandarizacion de las variables
scale_student = StandardScaler()
scale_x_student =scale_student.fit_transform(x_student)

#Train / Test
x_train_student = scale_x_student[:519]
y_train_student = y_student[:519]

x_test_student = scale_x_student[519:]
y_test_student = y_student[519:]

#Modelo de Regresion Multiple
modelo_student = linear_model.LinearRegression()
modelo_student.fit(x_train_student,y_train_student)

#Prediccion del modelo
pred_scale_x_student = modelo_student.predict([x_test_student[0]])
print(pred_scale_x_student)

#R de relacion train y test
r2_train = r2_score(y_train_student, modelo_student.predict(x_train_student))
print(r2_train)

r2_test = r2_score(y_test_student, modelo_student.predict(x_test_student))
print(r2_test)

print("#---------------------------Actividad con BD Netflix-------------------------------------")
# Importe de la base de datos
netflix_df = pd.read_excel('Netflix_list.xlsx')
netflix_df["duracion"] = pd.to_numeric(netflix_df['duration'].replace('([^0-9]*)','', regex=True), errors='coerce')


condiciones = [
    (netflix_df["type"] == "Movie"),
    (netflix_df["type"] == "TV Show")
    ]
selecciones = [1.0, 2.0]
netflix_df["Cod_type"] =  np.select(condiciones, selecciones, default='Not Specified')

condiciones2 = [
    (netflix_df["duration"].str.contains("Season").astype(np.bool_)),
    (netflix_df["duration"].str.contains("min").astype(np.bool_))
    ]
selecciones2 = [1.0, 2.0]
netflix_df["duration_type"] =  np.select(condiciones2, selecciones2, default='Not Specified')

# Variables independiente
x_netflix = netflix_df[["Cod_type","duration_type"]][:2000]                    
            
#Definimos la variable dependiente
y_netflix = netflix_df["duracion"][:2000]   

# Estandarizacion de las variables
scale_netflix = StandardScaler()
scale_x_netflix =scale_netflix.fit_transform(x_netflix)

#Train / Test
x_train_netflix = scale_x_netflix[:1600]
y_train_netflix = y_netflix[:1600]

x_test_netflix = scale_x_netflix[1600:]
y_test_netflix = y_netflix[1600:]

#Modelo de Regresion Multiple
modelo_netflix = linear_model.LinearRegression()
modelo_netflix.fit(x_train_netflix,y_train_netflix)

#Prediccion del modelo
pred_scale_x_netflix = modelo_netflix.predict([x_test_netflix[0]])
print(pred_scale_x_netflix)

#R de relacion train y test
r2_train = r2_score(y_train_netflix, modelo_netflix.predict(x_train_netflix))
print(r2_train)

r2_test = r2_score(y_test_netflix, modelo_netflix.predict(x_test_netflix))
print(r2_test)
