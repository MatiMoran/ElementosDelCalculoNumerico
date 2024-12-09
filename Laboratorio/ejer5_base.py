import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


def matriz_A(k, h, N, a):
    # Esta lista almacenará las filas de la matriz
    filas_matriz = []
    for n in range(N):
        fila = []
        for i in range(N):
            if i == n:  # En este caso estoy en la diagonal
                # COMPLETAR
            elif i == n-1:  # En este caso estoy en la subdiagonal
                # COMPLETAR
            else:
                # COMPLETAR
        filas_matriz.append(# COMPLETAR)
    return np.array(# COMPLETAR)


def vector_g(j, k, h, N, a, g):
    # COMPLETAR //  Sugerencia: se puede inicializar el vector usando np.zeros y luego cambiar la primera coordenada
    return G


def calcular_u0(h, N, f):
    u0 = []
    for i in range(N):
        u0.append(# COMPLETAR)
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
    # COMPLETAR

    # Creamos una lista donde guardaremos las fotos para el .gif
    fotos = []

    # Calculamos u_0, que viene dado por uno de los valores de contorno
    u_j = # COMPLETAR
    # Graficamos u_0 y se guarda la imagen
    graficar(u_j, 0, k, h, N, u_exacta, nombre_gif)
    # Agregamos la imagen a nuestra lista de fotos
    fotos.append(imageio.imread(f'{nombre_gif}_0.png'))
    # Una vez almacenada la imagen en la lista de fotos, borramos el archivo .png porque ya no lo necesitamos
    os.remove(f'{nombre_gif}_{0}.png')

    # Para cada j, calculamos u_j
    A = # COMPLETAR
    for j in range(1, M+1):
        G = # COMPLETAR
        u_j = # COMPLETAR
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
        return # COMPLETAR

    def g(t):
        return 0

    a = # COMPLETAR
    xf = # COMPLETAR
    tf = # COMPLETAR

    N = # COMPLETAR
    M = # COMPLETAR

    upwind(tf, xf, a, g, f, N, M, # COMPLETAR)


def ITEM_F():
    # Definimos las funciones por las que vienen dados los valores de contorno
    def f(x):
        return # COMPLETAR

    def g(t):
        return 0

    a = # COMPLETAR
    xf = # COMPLETAR
    tf = # COMPLETAR

    N = # COMPLETAR
    M = # COMPLETAR

    upwind(tf, xf, a, g, f, N, M, # COMPLETAR)


def ITEM_G():
    # Definimos las funciones por las que vienen dados los valores de contorno
    def f(x):
        return # COMPLETAR

    def g(t):
        return 0

    a = # COMPLETAR
    xf = # COMPLETAR
    tf = # COMPLETAR

    N = # COMPLETAR
    M = # COMPLETAR

    upwind(tf, xf, a, g, f, N, M, # COMPLETAR)


def ITEM_H():
    # Definimos las funciones por las que vienen dados los valores de contorno
    def f(x):
        return # COMPLETAR

    def g(t):
        return 0

    a = # COMPLETAR
    xf = # COMPLETAR
    tf = # COMPLETAR

    N = # COMPLETAR
    M = # COMPLETAR

    upwind(tf, xf, a, g, f, N, M, # COMPLETAR)


ITEM_E()
ITEM_F()
ITEM_G()
ITEM_H()
