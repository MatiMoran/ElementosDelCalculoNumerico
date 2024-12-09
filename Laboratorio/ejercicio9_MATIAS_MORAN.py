## EJERCICIO 9
## MATIAS MORAN
## LU 806/19


import numpy as np
import numpy.linalg as npl
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# A)

def matriz_A(x,m,tipo):
    nx = len(x)
    A = np.zeros((nx,m))
    if tipo=='polinomial':
        for i in range(nx):
            for j in range(m):
                A[i][j] = x[i]**j
    elif tipo=='senoidal':
        for i in range(nx):
            for j in range(m):
                A[i][j] = math.sin((j+1)*x[i])
    return(A)

# B)

def cuadrados_minimos(x,y,m,tipo):
    A = matriz_A(x,m,tipo)
    B = np.dot(np.transpose(A), A)
    c = npl.solve(B, np.dot(np.transpose(A), y))
    return(c)
    
# C)

def genera_ajustador(c,tipo):
    def function(s):
        A = matriz_A(s, len(c), tipo)
        resultado = np.dot(A, c)
        return (resultado)
    return (function)

# D)

datos_x = np.linspace(-np.pi/2, np.pi/2, num=10)
datos_y = [1.19,0.84,0.825,0.56,0.376,-0.186,-0.663,-0.682,-0.801,-0.996]
poli_coef = cuadrados_minimos(datos_x, datos_y, 2, 'polinomial')
sen_coef = cuadrados_minimos(datos_x, datos_y, 2, 'senoidal')

poli_fun = genera_ajustador(poli_coef, 'polinomial')
sen_fun = genera_ajustador(sen_coef, 'senoidal')

X = np.linspace(-np.pi/2, np.pi/2, 100)
plt.scatter(datos_x, datos_y)
plt.plot(X, poli_fun(X), label='Ajuste polinomial')
plt.plot(X, sen_fun(X), label='Ajuste senoidal')
plt.legend()
plt.savefig('ajuste_tabla.png')

# vemos que el ajuste polinomial se aproxima mas a la funcion

# E)

datos = pd.read_csv('datos_google.csv')
X = np.array(datos['dia'])
Y = np.array(datos['precio de cierre'])

ms = [2, 4, 6, 8]
x_aju = np.linspace(np.min(X), np.max(X), 500)
plt.clf()
plt.plot(X, Y, 'o', ms=1)   # Ploteamos los datos
for m in ms:   # Para cada m hallamos el polinomio y lo graficamos
    coefs = cuadrados_minimos(X, Y, m, 'polinomial')
    poli_fun = genera_ajustador(coefs, 'polinomial')
    plt.plot(x_aju, poli_fun(x_aju), label='m=' + str(m))
plt.legend()
plt.savefig('ajustes_datos.png')

#por lo que observamos en la imagen el ajuste con m=6 es el mas fiel a los datos ya que al 2 y 4 le faltan complejidad y las 8 tiene un pico demasiado
#grande al final de los datos
