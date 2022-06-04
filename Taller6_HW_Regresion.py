# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 15:34:55 2022

@author: user
"""

import pandas as pd
import numpy as np
from sklearn import linear_model


# Se importa desde CSV
cars_df = pd.read_csv('cars.csv')

condiciones = [
    (cars_df["Car"] == "Audi"),
    (cars_df["Car"] == "BMW"),
    (cars_df["Car"] == "Fiat"),
    (cars_df["Car"] == "Ford"),
    (cars_df["Car"] == "Honda"),
    (cars_df["Car"] == "Hundai"),
    (cars_df["Car"] == "Mazda"),
    (cars_df["Car"] == "Mercedes"),
    (cars_df["Car"] == "Mini"),
    (cars_df["Car"] == "Mitsubishi"),
    (cars_df["Car"] == "Opel"),
    (cars_df["Car"] == "Skoda"),
    (cars_df["Car"] == "Suzuki"),
    (cars_df["Car"] == "Toyoty"),
    (cars_df["Car"] == "VW"),
    (cars_df["Car"] == "Volvo"),
    (cars_df["Car"] == "Hyundai")
    ]
selecciones = [1.0, 2.0, 3.0, 4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0]

cars_df["Cod_Car"] =  np.select(condiciones, selecciones, default='Not Specified')
pd.to_numeric(cars_df["Cod_Car"])

# Variables independiente
x = cars_df[["Volume","Weight","CO2"]]

# Variable Dependiente
y = cars_df["Cod_Car"]

# Regresion
reg_mod = linear_model.LinearRegression()
reg_mod.fit(x,y)

#Prediccion
predict_co2 = reg_mod.predict([[1000,790,99]])
print(predict_co2)
print(reg_mod.coef_)
