## EJERCICIO 4
## MATIAS MORAN
## LU 806/19

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0
estado_0 = [1.0, 1.0, 1.0]
t_paso = 0.01
t = np.arange(0.0, 40.0, t_paso)


def F(estado, t):
    x, y, z = estado
    rhs = sigma * (y - x), x * (rho - z) - y, x * y - beta * z
    return np.array(rhs)


def integra_euler(f, estado_0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    estados = [estado_0]
    for tn in t:
        vn = np.array(estados[-1])
        paso = vn + dt * F(vn, tn)
        estados.append( paso )
    return np.array(estados)


def paso_runge_kutta(estado, tn, dt):
    k1 = F(estado, tn)
    k2 = F(estado + (dt/2)*k1 , tn + (dt/2))
    k3 = F(estado + (dt/2)*k2 , tn + (dt/2))
    k4 = F(estado + (dt)*k3   , tn + (dt))
    promedio_pesado = 1/6 * dt * (k1 + 2*k2 + 2*k3 + 3*k4)
    return(estado + promedio_pesado)

# funcion que integra utilizando el metodo de runge kuta de orden 4, lo unico que cambiamos el la funcion de paso
def integra_RK4(f, estado_0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    estados = [estado_0]
    for tn in t:
        vn = np.array(estados[-1])
        paso = paso_runge_kutta(vn, tn, dt)
        estados.append( paso )
    return np.array(estados)




fig = plt.figure()
ax = fig.gca(projection='3d')

estados = integra_RK4(0, estado_0, 0, 40, t_paso)
ax.plot(estados[:, 0], estados[:, 1], estados[:, 2])

estados = integra_RK4(0, estado_0, 0, 40, t_paso/10)
ax.plot(estados[:, 0], estados[:, 1], estados[:, 2])

plt.show()

# en esta figura podemos observar como las imagenes se parecen bastante pero tienen diferencias grandes en los valores mas alejados
# a simple vista son similares pero si le hacemos zoom podemos ver claras diferencias



fig = plt.figure()
ax = fig.gca(projection='3d')

estados = integra_RK4(0, estado_0, 0, 40, t_paso/10)
ax.plot(estados[:, 0], estados[:, 1], estados[:, 2])

estados = integra_RK4(0, estado_0, 0, 40, t_paso/50)
ax.plot(estados[:, 0], estados[:, 1], estados[:, 2])

plt.show()


# repetimos el proceso pero esta vez con pasos mas pequeños y podemos ver que las funciones cada vez se aproximan mas una a otra como si fueran a
# converger a la solucion exacta, pero seguimos viendo valores lejanos entre las aproximaciones en valores lejanos, es decir t >> 0.


fig = plt.figure()
ax = fig.gca(projection='3d')

estados = integra_RK4(0, estado_0, 0, 10, t_paso/10)
ax.plot(estados[:, 0], estados[:, 1], estados[:, 2])

estados = integra_RK4(0, estado_0, 0, 10, t_paso/50)
ax.plot(estados[:, 0], estados[:, 1], estados[:, 2])

plt.show()

# en esta ultima imagen repetimos el experimento con los valores pequeño de paso pero dejamos un t_final = 10, para observar si la solucion converge
# a la solucion exacta con T pequeño, lo cual en la imagen vemos que para valores bajos de T esto pasa, ya que los graficos son casi identicos y no
# se empiezan a "separar" hasta que llega un T grande, sospechamos que esto se debe al caracter caotico de la funcion de lorenz