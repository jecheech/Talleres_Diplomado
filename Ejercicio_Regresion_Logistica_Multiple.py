# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:02:21 2022

@author: user
"""

# Las librerías utilizadas en este ejemplo para Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Las librerías utilizadas en este ejemplo para Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Las librerías utilizadas en este ejemplo para Preprocesado y modelado
# ==============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Las librerías utilizadas en este ejemplo para Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')


# Datos
# la Data corresponde a una serie de parametros que asociados a e-mails
# pueden indicar una relacion con que sean SPAM o no
# ==============================================================================
url = 'https://raw.githubusercontent.com/JoaquinAmatRodrigo/' \
       + 'Estadistica-machine-learning-python/master/data/spam.csv'
datos = pd.read_csv(url)
datos.head(3)

# Se codifica la variable respuesta como 1 si es spam y 0 si no lo es, 
# y se identifica cuantas observaciones hay de cada clase.
datos['type'] = np.where(datos['type'] == 'spam', 1, 0)

print("Número de observaciones por clase")
print(datos['type'].value_counts())
print("")

print("Porcentaje de observaciones por clase")
print(100 * datos['type'].value_counts(normalize=True))

# El 60.6% de los correos no son spam y el 39.4% sí lo son. 
# Un modelo de clasificación que sea útil debe de ser capaz de predecir 
# correctamente un porcentaje de observaciones por encima del porcentaje 
# de la clase mayoritaria. En este caso, el umbral de referencia que se 
# tiene que superar es del 66.6%.

# Se ajusta un modelo de regresión logística múltiple con el 
# objetivo de predecir si un correo es spam en función de todas las 
# variables disponibles.

# División de los datos en train y test
# de el DF x se elimina la columna type
# Se crea la serie y con la columna type



# ==============================================================================
X = datos.drop(columns = 'type')
y = datos['type']

# Divida matrices o matrices en subconjuntos aleatorios de entrenamiento y prueba.

# Utilidad rápida que envuelve la validación de entrada y la aplicación a 
# los datos de entrada en una sola llamada para dividir (y opcionalmente submuestrear)

X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

# Creación del modelo utilizando matrices como en scikitlearn
# ==============================================================================
# A la matriz de predictores se le tiene que añadir una columna de 1s 
# para el intercept del modelo
# add_constant Agregue una columna de unos a una matriz.
# prependbool Si es verdadero, la constante está en la primera columna. 
# De lo contrario, la constante se anexa (última columna).
# Logit Esta función estima la curva de regresión utilizando el enfoque de 
# probabilidad local para un vector de observaciones binomiales y un vector 
# asociado de valores covariables.
X_train = sm.add_constant(X_train, prepend=True)
modelo = sm.Logit(endog=y_train, exog=X_train,)
modelo = modelo.fit()
print(modelo.summary())

# Predicciones


# Una vez entrenado el modelo, se pueden obtener predicciones para nuevos 
# datos. Los modelos de statsmodels permiten calcular los intervalos de 
# confianza asociados a cada predicción.

# Predicciones con intervalo de confianza 
# ==============================================================================
predicciones = modelo.predict(exog = X_train)

# Clasificación predicha
# ==============================================================================
clasificacion = np.where(predicciones<0.5, 0, 1)
clasificacion


# Accuracy de test

# Se calcula el porcentaje de aciertos que tiene el modelo al 
# predecir las observaciones de test (accuracy).
# Accuracy de test del modelo 
# modelo.predict En la clasificación multietiqueta, esta función calcula la precisión 
# del subconjunto: el conjunto de etiquetas predichas para una muestra debe 
# coincidir exactamente con el conjunto correspondiente de etiquetas en y_true.
# ==============================================================================
X_test = sm.add_constant(X_test, prepend=True)
predicciones = modelo.predict(exog = X_test)
clasificacion = np.where(predicciones<0.5, 0, 1)
accuracy = accuracy_score(
            y_true    = y_test,
            y_pred    = clasificacion,
            normalize = True
           )
print("")
print(f"El accuracy de test es: {100*accuracy}%")


# Matriz de confusión de las predicciones de test
# Se puede apreciar como predijo que no son spam 535 acertadamente y solo 28 no
# Mientras que 320 como que son spam acertadamente y solo se equivoco en 38
# ==============================================================================
confusion_matrix = pd.crosstab(
    y_test.ravel(),
    clasificacion,
    rownames=['Real'],
    colnames=['Predicción']
)
confusion_matrix

# Conclusión


# El modelo logístico creado para predecir la probabilidad de que un 
# correo sea spam es en conjunto significativo (Likelihood ratio p-value = 0). 
# El porcentaje de clasificación correcta en el conjunto del test es del 92.8%, 
# un valor muy por encima del umbral de 60.6% esperado por azar.

# Acorde al p-value individual de cada predictor, solo algunos de 
# ellos aportan información al modelo. Es conveniente identificar cuáles 
# son y excluirlos para simplificar el modelo y evitar la introducción de ruido. 
# Una forma de conseguir reducir la influencia de predictores que no aportan 
# a un modelo de regresión logística es incorporar regularización en su ajuste.
