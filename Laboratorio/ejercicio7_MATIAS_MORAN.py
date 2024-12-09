import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import math
import pandas as pd
import os

# A)

def matriz_vander(x):
    A = np.zeros((len(x),len(x)))
    for q in range(0,len(x)):
        A[:,q] = np.power(x,q)
    return(A)

# B)

def coef_indeter(x,y):
    A = matriz_vander(x)
    coef = npl.solve(A, y)
    return(coef)

# C)

def funcion_interpol(x,y):
    coef = coef_indeter(x,y)
    def function(m):
        A = matriz_vander(m)[:, 0:len(x)]
        y_interpolado = np.dot(A,coef)
        return(y_interpolado)
    return(function)

# D)



x = np.linspace(0,math.pi,10)
y = []

for i in range(len(x)):
    y.append(math.sin(x[i]))

sin_interpolado = funcion_interpol(x,y)
y_interpolado = sin_interpolado(x)


x_real = np.linspace(0,math.pi,100)
y_real = []
y_interpolado  = sin_interpolado(x_real)

for i in range(len(x_real)):
    y_real.append(math.sin(x_real[i]))

plt.scatter(x,y, c = 'g', label = 'puntos interpoladores')
plt.plot(x_real,y_interpolado, 'r', label = 'y = sin_interpolado(x)')
plt.plot(x_real, y_real, 'b', label='y=sin(x)')

plt.legend(loc='upper left')
plt.show()


# D)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
datos = pd.read_csv(os.getcwd() + '\\data_prob7.csv')

x = np.log2(np.array(datos['area_acres']))
y = np.array(datos['pop_log2'])

datos_interpolado = funcion_interpol(x,y)

x_real = np.linspace(np.min(x),np.max(x),100)
y_real = datos_interpolado(x_real)

plt.scatter(x,y, c = 'g', label = 'puntos interpoladores')
plt.plot(x_real, y_real, 'b', label='datos_interpolado(x)')

plt.legend(loc='upper left')
plt.show()

# vemos que los datos parece que solo tienen sentido en el 'medio' del polinomio interpolador
# en los extremos, es decir los valores mas bajos y los valores mas altos la funcion tiene picos
# que por la naturaleza de los datos son erroneos