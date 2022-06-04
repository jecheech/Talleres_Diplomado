# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 16:07:55 2022

@author: user
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#------------------------Regresion Lineal-----------------------
# Se importa desde CSV
df = pd.read_excel('AirQualityUCI.xlsx')

x = df["CO(GT)"]
y = df["NO2(GT)"]

slope,intercept,r,p,std_err = stats.linregress(x,y)

def regresion(x):
    return slope*x + intercept

modelo = list(map(regresion, x))

plt.scatter(x,y)
plt.plot(x,modelo)
plt.show()

#Prediccion
predict_reg_lineal = regresion(5)
print("""La prediccion de la regresion lineal es {} y el r de relacion es {}""".format(str(predict_reg_lineal), str(r)))

#------------------------Regresion Polinomial-----------------------
poli_model = np.poly1d(np.polyfit(x,y, 3))

poli_line = np.linspace(1,22,100)

poli_new_y = poli_model(poli_line)

plt.scatter(x,y)
plt.plot(poli_line, poli_new_y)
plt.show()

#Prediccion
predict_reg_poli = poli_model(5)
r2_poli = r2_score(y,poli_model(x))
print("""La prediccion de la regresion Polinomial es {} y el r de relacion es {}""".format(str(predict_reg_poli), str(r2_poli)))

#------------------------Regresion Multiple-----------------------
xm = df[["CO(GT)"]]

reg_mod = linear_model.LinearRegression()
reg_mod.fit(xm,y)

#Prediccion
predict_multi = reg_mod.predict([[5]])
r_multi = reg_mod.coef_

print("""La prediccion de la regresion Multiple es {} y el r de relacion es {}""".format(str(predict_multi), str(r_multi)))
