# -*- coding: utf-8 -*-
"""
Created on Sat May 28 13:53:36 2022

@author: Jaider Stiven Echeverri Bedoya
"""
import pandas as pd
import random
import numpy as np

my_data = pd.read_csv('netflix_titles.csv')

# Imprima por consola las primeras y últimas 5 filas del arreglo.
print(my_data.head(8))

print(my_data.tail(8))

# Imprima cada uno de los tipos de dato asociado a las etiquetas.
print(my_data.dtypes)

# Guarde un archivo .xlsx, en el cual el nombre del archivo sea “Netflix_list” y el nombre de la hoja sea “títulos”
my_data.to_excel("Netflix_list.xlsx", sheet_name="títulos", index=False)

#Cree una nueva data frame en el cual segmente únicamente: el tipo, la duración, la descripción y el país
new_my_data = my_data.loc[:,['type','duration','description','country']]

# Campo con duracion en numerico
my_data["duracion"] = pd.to_numeric(my_data['duration'].replace('([^0-9]*)','', regex=True), errors='coerce')
#pd.to_numeric(my_data["duracion"], errors='coerce')

# Haga un filtro para las películas que tienen una duración superior a 100 min
movies_100 = my_data[my_data['type'].str.contains('Movie', na=False)]
movies_100_min = movies_100[movies_100['duracion']>100]

#Haga un filtro para los “TV Shows” que tienen más de 3 temporadas.
tv_show = my_data[my_data['type'].str.contains('TV Show', na=False)]
tv_show_3_seasons= tv_show[tv_show['duracion']>3]

#Haga un filtro en el cual solo tenga en cuenta 2 categorías/etiquetas (libre elección)
categorias = my_data.loc[my_data['listed_in'].isin(['Documentaries','International TV Shows, TV Dramas, TV Mysteries'])]

# Recuerde usar casos con indexación numérica y con texto (loc / iloc).

# Modifique los valores del ID de las 5 primeras y las 5 últimas “shows” y de cualquier otra etiqueta de su elección (solo un valor).
my_data.iloc[:5,0]='s1'
tv_show.loc[2671:,'listed_in']='Comedies, Horror Movies'

#Añada una nueva columna “Visto”, en la cual debe colocar 1/0 (aleatorio) si vio o no el show (simulación).

my_data["Visto"] = np.random.randint(0, 2, my_data.shape[0])