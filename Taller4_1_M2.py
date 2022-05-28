# -*- coding: utf-8 -*-
"""
Created on Sat May 28 13:30:14 2022

@author: user
"""
a=1
b=2
c=3
d=4
e=5
f=6


ecu1 = (a+(b/c))/(d+(e/f))
Ecu2 = a-(b/(c-d))

Ecu3 = ecu1
ecu1 = Ecu2
Ecu2 = Ecu3

print(ecu1)
print(Ecu2)

