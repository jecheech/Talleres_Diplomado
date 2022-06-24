# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 07:02:28 2022

@author: user
"""

# Ejemplo regresión logística simple


# Un estudio quiere establecer un modelo que permita 
# calcular la probabilidad de obtener matrícula de honor 
# al final del bachillerato en función de la nota que se 
# ha obtenido en matemáticas. La variable matrícula está 
# codificada como 0 si no se tiene matrícula y 1 si se tiene.


# Las librerías utilizadas en este ejemplo para Tratamiento de datos 
# ==============================================================================
import pandas as pd
import numpy as np

# Las librerías utilizadas en este ejemplo para  Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Las librerías utilizadas en este ejemplo para Preprocesado y modelado
# ==============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind

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

# Datos (La variable matrícula está codificada 
# como 0 si no se tiene matrícula y 1 si se tiene.)
# ==============================================================================
matricula = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1,
                     0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,
                     0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                     0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                     1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0,
                     1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1,
                     1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1,
                     0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                     0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0,
                     0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0,
                     0, 0, 0, 0, 1, 0, 0, 0, 1, 1])

matematicas = np.array([
                  41, 53, 54, 47, 57, 51, 42, 45, 54, 52, 51, 51, 71, 57, 50, 43,
                  51, 60, 62, 57, 35, 75, 45, 57, 45, 46, 66, 57, 49, 49, 57, 64,
                  63, 57, 50, 58, 75, 68, 44, 40, 41, 62, 57, 43, 48, 63, 39, 70,
                  63, 59, 61, 38, 61, 49, 73, 44, 42, 39, 55, 52, 45, 61, 39, 41,
                  50, 40, 60, 47, 59, 49, 46, 58, 71, 58, 46, 43, 54, 56, 46, 54,
                  57, 54, 71, 48, 40, 64, 51, 39, 40, 61, 66, 49, 65, 52, 46, 61,
                  72, 71, 40, 69, 64, 56, 49, 54, 53, 66, 67, 40, 46, 69, 40, 41,
                  57, 58, 57, 37, 55, 62, 64, 40, 50, 46, 53, 52, 45, 56, 45, 54,
                  56, 41, 54, 72, 56, 47, 49, 60, 54, 55, 33, 49, 43, 50, 52, 48,
                  58, 43, 41, 43, 46, 44, 43, 61, 40, 49, 56, 61, 50, 51, 42, 67,
                  53, 50, 51, 72, 48, 40, 53, 39, 63, 51, 45, 39, 42, 62, 44, 65,
                  63, 54, 45, 60, 49, 48, 57, 55, 66, 64, 55, 42, 56, 53, 41, 42,
                  53, 42, 60, 52, 38, 57, 58, 65])

datos = pd.DataFrame({'matricula': matricula, 'matematicas': matematicas})
datos.head(3)


# El primer paso antes de generar un modelo de regresión logística simple 
# es representar los datos para poder intuir si existe una relación 
# entre la variable independiente y la variable respuesta.

# Número de obsercaciones por clase
# ==============================================================================
datos.matricula.value_counts().sort_index()

# Se puede identificar a partir del Gráfico que si existe una relacion
# ya que a partir de 60 puntos de nota en matematica aprox. hay una relacion
# entre tener o no matricula de honor. Esta información es útil para 
# considerar la nota de matemáticas como un buen predictor para el modelo.
# ==============================================================================
fig, ax = plt.subplots(figsize=(6, 3.84))

sns.violinplot(
        x     = 'matricula',
        y     = 'matematicas',
        data  = datos,
        #color = "white",
        ax    = ax
    )

ax.set_title('Distribución notas de matemáticas por clase');

# T-test entre clases
# T-test: Comparación de medias poblacionales independientes 
# Para estudiar si la diferencia observada entre las medias de dos grupos 
# es significativa, se puede recurrir a métodos paramétricos como el basado 
# en Z-scores o en la distribución T-student. En ambos casos, 
# se pueden calcular tanto intervalos de confianza para saber entre que 
# valores se encuentra la diferencia real de las medias poblacionales o 
# test de hipótesis para determinar si la diferencia es significativa.
# ==============================================================================
res_ttest = ttest_ind(
                x1 = matematicas[matricula == 0],
                x2 = matematicas[matricula == 1],
                alternative='two-sided'
            )
print(f"t={res_ttest[0]}, p-value={res_ttest[1]}")

# División de los datos en train y test

# Se ajusta un modelo empleando como variable respuesta matricula y 
# como predictor matematicas. Como en todo estudio predictivo, no solo es 
# importante ajustar el modelo, sino también cuantificar su capacidad para 
# predecir nuevas observaciones. Para poder hacer esta evaluación, se dividen 
# los datos en dos grupos, uno de entrenamiento y otro de test.
# ==============================================================================
X = datos[['matematicas']]
y = datos['matricula']

X_train, X_test, y_train, y_test = train_test_split(
                                        X.values.reshape(-1,1),
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

# Creación del modelo
# ==============================================================================
# Para no incluir ningún tipo de regularización en el modelo se indica
# penalty='none'
modelo = LogisticRegression(penalty='none')
modelo.fit(X = X_train.reshape(-1, 1), y = y_train)

# Información del modelo
# ==============================================================================
print("Intercept:", modelo.intercept_)
print("Coeficiente:", list(zip(X.columns, modelo.coef_.flatten(), )))
print("Accuracy de entrenamiento:", modelo.score(X, y))

# Predicciones probabilísticas
# ==============================================================================
# Con .predict_proba() se obtiene, para cada observación, la probabilidad predicha
# de pertenecer a cada una de las dos clases.
# Una vez entrenado el modelo, se pueden predecir nuevas observaciones.
# en este caso utilizando la muestra almacenada en test
predicciones = modelo.predict_proba(X = X_test)
predicciones = pd.DataFrame(predicciones, columns = modelo.classes_)
predicciones.head(3)

# Predicciones con clasificación final
# ==============================================================================
# Con .predict() se obtiene, para cada observación, la clasificación predicha por
# el modelo. Esta clasificación se corresponde con la clase con mayor probabilidad.
predicciones = modelo.predict(X = X_test)
predicciones

# ==============================================================================

# Statsmodels
# La implementación de regresión logística de Statsmodels, es más completa 
# que la de Scikitlearn ya que, además de ajustar el modelo, 
# permite calcular los test estadísticos y análisis necesarios para 
# verificar que se cumplen las condiciones sobre las que se basa este 
# tipo de modelos. Statsmodels tiene dos formas de entrenar el modelo:

# Indicando la fórmula del modelo y pasando los datos de entrenamiento 
# como un dataframe que incluye la variable respuesta y los predictores. 
# Esta forma es similar a la utilizada en R.

# Pasar dos matrices, una con los predictores y otra con la variable respuesta.
# Esta es igual a la empleada por Scikitlearn con la diferencia de que a la 
# matriz de predictores hay que añadirle una primera columna de 1s.


# División de los datos en train y test
# ==============================================================================
X = datos[['matematicas']]
y = datos['matricula']

X_train, X_test, y_train, y_test = train_test_split(
                                        X.values.reshape(-1,1),
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

# Creación del modelo utilizando el modo fórmula (similar a R)
# ==============================================================================
# datos_train = pd.DataFrame(np.hstack((X_train, y_train)),
#                            columns=['matematicas', 'matricula'])
# modelo = smf.logit(formula = 'matricula ~matematicas', data = datos_train)
# modelo = modelo.fit()
# print(modelo.summary())

# Creación del modelo utilizando matrices como en scikitlearn
# ==============================================================================
# A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo
X_train = sm.add_constant(X_train, prepend=True)
modelo = sm.Logit(endog=y_train, exog=X_train,)
modelo = modelo.fit()
print(modelo.summary())

# Intervalos de confianza para los coeficientes del modelo
# Además del valor de las estimaciones de los coeficientes parciales
# de correlación del modelo, es conveniente calcular sus correspondientes 
# intervalos de confianza.
# ==============================================================================
intervalos_ci = modelo.conf_int(alpha=0.05)
intervalos_ci = pd.DataFrame(intervalos_ci)
intervalos_ci.columns = ['2.5%', '97.5%']
intervalos_ci


# Predicciones


# Una vez entrenado el modelo, se pueden obtener predicciones para nuevos datos. 
# Los modelos de regresión logística de statsmodels devuelven la probabilidad de 
# pertenecer a la clase de referencia.

# Predicción de probabilidades
# ==============================================================================
predicciones = modelo.predict(exog = X_train)
predicciones[:4]

# Para obtener la clasificación final, se convierten los valores de 
# probabilidad mayores de 0.5 a 1 y los mejores a 0.

# Clasificación predicha
# ==============================================================================
clasificacion = np.where(predicciones<0.5, 0, 1)
clasificacion


# Además de la línea de mínimos cuadrados, es recomendable incluir los límites 
# superior e inferior del intervalo de confianza. Esto permite identificar 
# la región en la que, según el modelo generado y para un determinado nivel 
# de confianza, se encuentra el valor promedio de la variable respuesta.

# Predicciones en todo el rango de X
# ==============================================================================
# Se crea un vector con nuevos valores interpolados en el rango de observaciones.
grid_X = np.linspace(
            start = min(datos.matematicas),
            stop  = max(datos.matematicas),
            num   = 200
         ).reshape(-1,1)

grid_X = sm.add_constant(grid_X, prepend=True)
predicciones = modelo.predict(exog = grid_X)


# Gráfico del modelo
# ==============================================================================
fig, ax = plt.subplots(figsize=(6, 3.84))

ax.scatter(
    X_train[(y_train == 1).flatten(), 1],
    y_train[(y_train == 1).flatten()].flatten()
)
ax.scatter(
    X_train[(y_train == 0).flatten(), 1],
    y_train[(y_train == 0).flatten()].flatten()
)
ax.plot(grid_X[:, 1], predicciones, color = "gray")
ax.set_title("Modelo regresión logística")
ax.set_ylabel("P(matrícula = 1 | matemáticas)")
ax.set_xlabel("Nota matemáticas");

# Se calcula el porcentaje de aciertos que tiene el modelo al predecir 
# las observaciones de test (accuracy).

# Accuracy de test del modelo 
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


# Conclusión


# El modelo logístico creado para predecir la probabilidad de que un 
# alumno obtenga matrícula de honor a partir de la nota de matemáticas es e
# n conjunto significativo (Likelihood ratio p-value = 9.831e-11). 
# El p-value del predictor matemáticas es significativo (p-value = 7.17e-08).

# P(matricula)=e−8.9848+0.1439∗nota matematicas1+e−8.9848+0.1439∗nota matematicas
 
# Los resultados obtenidos con el conjunto de test indican que el 
# modelo es capaz de clasificar correctamente el 87.5% de las observaciones.

