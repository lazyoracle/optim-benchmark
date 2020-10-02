# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 20:08:48 2019

@author: anura
"""
from benchmark_functions import *
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

  
plt.rcParams['figure.figsize'] = (12, 8) 
plt.rc('font', family='serif')
plt.rc('font', size=12)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 600 


x = np.linspace(-1.5, 1.5, 500)
y = np.linspace(-1.5, 1.5, 500)

    
X, Y = np.meshgrid(x, y)

final_mesh = []
for i in y:
    temp_series = []
    for j in x:
        temp_series.append((j, i))
    final_mesh.append(temp_series)

Z = []
for i in final_mesh:
    z = []
    for j in i:
        temp = -deceptivepath(np.asarray([j[0],j[1]]))
        z.append(temp)
    Z.append(z)
        
Z = np.asarray(Z)


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface');
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');


plt.show()