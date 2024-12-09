import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import math


def matriz_A(k, h, N, a):
    # Esta lista almacenará las filas de la matriz
    filas_matriz = []
    for n in range(N):
        fila = []
        for i in range(N):
            if i == n:  # En este caso estoy en la diagonal
                fila.append(1-a*k/h)
            elif i == n-1:  # En este caso estoy en la subdiagonal
                fila.append(a*k/h)
            else:
                fila.append(0)
        filas_matriz.append(fila)
    return np.array(filas_matriz)


def vector_g(j, k, h, N, a, g):
    G = np.zeros(N)
    G[0] = a*k/h * g(j*k)
    return G


def calcular_u0(h, N, f):
    u0 = []
    for i in range(N):
        u0.append(f(h*i))
    return np.array(u0)


def graficar(u_j, j, k, h, N, u_exacta, nombre_gif):
    # Esta función se encarga de generar el grafico para el paso temporal actual

    # Se borra lo que haya quedado graficado desde antes
    plt.clf()
    # Discretizacion de x
    X = np.arange(0, N*h, step=h)

    # Valores de la solucion exacta
    exacta = []
    for x in X:
        exacta.append(u_exacta(x, j*k))

    # Generacion del grafico
    plt.plot(X, u_j, label='solucion numerica')
    plt.plot(X, exacta, label='solucion exacta')
    plt.ylim([0, 1.2])
    plt.legend()
    plt.savefig(f'{nombre_gif}_{j}.png')


def upwind(tf, xf, a, g, f, N, M, nombre_gif):

    # Definimos la solución exacta
    def u_exacta(x, t):
        return f(x - a*t)

    # Calculamos h y k a partir de N y M, respectivamente
    h = xf/N
    k = tf/M

    # Creamos una lista donde guardaremos las fotos para el .gif
    fotos = []

    # Calculamos u_0, que viene dado por uno de los valores de contorno
    u_j = calcular_u0(h,N,f)
    # Graficamos u_0 y se guarda la imagen
    graficar(u_j, 0, k, h, N, u_exacta, nombre_gif)
    # Agregamos la imagen a nuestra lista de fotos
    fotos.append(imageio.imread(f'{nombre_gif}_0.png'))
    # Una vez almacenada la imagen en la lista de fotos, borramos el archivo .png porque ya no lo necesitamos
    os.remove(f'{nombre_gif}_{0}.png')

    # Para cada j, calculamos u_j
    A = matriz_A(k, h, N, a)
    for j in range(1, M+1):
        G = vector_g(j, k, h, N, a, g)
        u_j = A.dot(u_j) + G
        if j % 10 == 0:    # Esto es para guardar una de cada 10 fotos y que el .gif no sea tan pesado
            # Graficamos u_j y se guarda la imagen
            graficar(u_j, j, k, h, N, u_exacta, nombre_gif)
            # Agregamos la imagen a nuestra lista de fotos
            fotos.append(imageio.imread(f'{nombre_gif}_{j}.png'))
            # Una vez almacenada la imagen en la lista de fotos, borramos el archivo .png porque ya no lo necesitamos
            os.remove(f'{nombre_gif}_{j}.png')

    imageio.mimsave(f'{nombre_gif}.gif', fotos)  # con esta función se crea el .gif


def ITEM_E():
    # Definimos las funciones por las que vienen dados los valores de contorno
    def f(x):
        return math.exp(-10 * ((4*x-1)**2))

    def g(t):
        return 0

    a = 1
    xf = 2
    tf = 1.5

    N = 200
    M = 150

    upwind(tf, xf, a, g, f, N, M, "ITEM_E")


def ITEM_F():
    # Definimos las funciones por las que vienen dados los valores de contorno
    def f(x):
        return math.exp(-10 * ((4*x-1)**2))

    def g(t):
        return 0

    a = 1
    xf = 2
    tf = 1.5

    N = 200
    M = 140

    upwind(tf, xf, a, g, f, N, M, "ITEM_F")


def ITEM_G():
    # Definimos las funciones por las que vienen dados los valores de contorno
    def f(x):
        return math.exp(-10 * ((4*x-1)**2))

    def g(t):
        return 0

    a = 1
    xf = 2
    tf = 1.5

    N = 180
    M = 150

    upwind(tf, xf, a, g, f, N, M, "ITEM_G")


def ITEM_H():
    # Definimos las funciones por las que vienen dados los valores de contorno
    def f(x):
        return math.exp(-10 * ((4*x-6)**2))

    def g(t):
        return 0

    a = -1
    xf = 2
    tf = 1.5

    N = 200
    M = 150

    upwind(tf, xf, a, g, f, N, M, "ITEM_H")


ITEM_E()
ITEM_F()
ITEM_G()
ITEM_H()


# Vemos que en el ITEM E la aproximacion de la ecuacion del transporte aproxime muy bien a la funcion ya que se cumple la igualdad en la convergencia
# en el item F vemos que la aproximacion no cumple la condicion entre a,k y h por lo tanto al cabo de un tiempo va diverger y no es convergente y estable
# en el item G vemos que la aproximacion es peor que la del caso E aunque esta cumple la condicion < en vez de =
# en el caso H vemos que como a es negativo el metodo es inestable y aproxima muy mal la solucion exacta
