# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 16:34:48 2022

@author: user
"""

import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import pydotplus


carseats = sm.datasets.get_rdataset("Carseats", "ISLR")
datos = carseats.data
print(carseats.__doc__)
datos['ventas_altas'] = np.where(datos.Sales > 8, 0, 1)
datos = datos.drop(columns = 'Sales')

d= {'Yes':1, 'No':0}
d1= {'Bad':0, 'Medium':1, 'Good':2}
datos["Urban"] = datos["Urban"].map(d)
datos["US"] = datos["US"].map(d)
datos["ShelveLoc"] = datos["ShelveLoc"].map(d1)

features = ["CompPrice","Income","Advertising", "Population","Price","ShelveLoc","Age","Education","Urban","US"]

X = datos[features]
y = datos["ventas_altas"]

x_train = X[:320]
y_train = y[:320]

x_test = X[320:]
y_test = y[320:]

dtree = DecisionTreeClassifier()
dtree  = dtree.fit(x_train, y_train)

prediction = dtree.predict([[136,70,12,171,152,1,44,18,1,1]])

data = tree.export_graphviz(dtree, out_file = None, feature_names= features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')
img = pltimg.imread("mydecisiontree.png")
imgplot = plt.imshow(img)
plt.show()