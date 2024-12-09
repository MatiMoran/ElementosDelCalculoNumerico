## EJERCICIO 3
## MATIAS MORAN
## LU 806/19

import math
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt


# A) 

# funcion que devuelve 1000 pares [x,y(x)] para la funcion original a aproximar
def funcion_real():
    x = np.linspace(0,10,1000)
    y = []
    for k in range(0,len(x),1):
        y.append(6*math.exp(-x[k]) - 5*math.exp(-2*x[k]))
    
    return ([x,y])


def resuelve_y(y0,t0,T,N):

    end_val = y0
    y=[]
    t=[]

    y.append(end_val)
    t.append(t0)

    for k in range(1, N+1):
        t.append(t0 + k*(T-t0)/N)
        end_val = end_val + (T-t0)/N * (6*math.exp(-t[k]) - 2*end_val)
        y.append(end_val)

    return([t,y])



# armamos los ejes del grafico
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# ploteamos la funcion original a aproximar
x_real, y_real = funcion_real()
plt.plot(x_real,y_real)
legend = ["y(t) = 6exp(-t) - 5exp(-2t)"]

N_values = [10,50,100,500,1000]

#ploteamos las diferentes aproximaciones
for k in range(len(N_values)):
    a,b = resuelve_y(1,0,10,N_values[k])
    plt.plot(a,b)
    legend.append(format(N_values[k]) + " steps")
    print("Para {} pasos, la diferencia en T=10 es {}".format(N_values[k], 6*math.exp(-10) - 5*math.exp(-20) - b[-1]))


plt.legend(legend)
plt.show()


# Podemos observar que a medida que la cantidad N de pasos aumenta, la diferencia entre y_10 y y(10) se va haciendo cada vez mas y mas pequeña
# Ademas observamos que al pasar de pasos 50 a 500 y de 100 a 1000, es decir aumentar un orden de magnitud los pasos, el orden de magnitud del error
# tambien se reduce en uno, esto no pasa de 10 a 100 porque el paso 10 es demasiado pequeño y no aproxima nada bien a la funcion y(t)
# esto se debe a que la f original se podria decir que es un problema rigido ya que se aproxima a 0 muy rapidamente



# B)

# funcion que devuelve 1000 pares [x,y(x)] para la funcion original a aproximar
def funcion_real2():
    x = np.linspace(0,10,1000)
    y = []
    for k in range(0,len(x),1):
        y.append(math.sqrt(5)*math.sin(math.sqrt(5)*x[k]))
    
    return ([x,y])

# f(t0) = y0, f'(t0) = y1, N = pasos, T = t final
# la funcion devuelve pares t,y aproximando a la funcion original
def resuelve_y2_A(y0,t0,y1,T,N):

    f_end_val = y0
    fdot_end_val = y1
    y=[]
    t=[]

    y.append(f_end_val)
    t.append(t0)

    for k in range(1, N+1):
        t.append(t0 + k*(T-t0)/N)
        f_end_val = f_end_val + (T-t0)/N * fdot_end_val
        fdot_end_val = fdot_end_val + (T-t0)/N * (-5 * y[k-1])
        y.append(f_end_val)

    return([t,y])


# f(t0) = y0, f'(t0) = y1, N = pasos, T = t final
# la funcion devuelve pares t,y aproximando a la funcion original
def resuelve_y2_B(y0,t0,y1,T,N):

    f_end_val = y0
    fdot_end_val = y1
    y=[]
    t=[]

    y.append(f_end_val)
    t.append(t0)

    for k in range(1, N+1):
        t.append(t0 + k*(T-t0)/N)
        f_end_val = f_end_val + (T-t0)/N * fdot_end_val
        y.append(f_end_val)
        fdot_end_val = fdot_end_val + (T-t0)/N * (-5 * y[k])

    return([t,y])
    

# f(t0) = y0, f'(t0) = y1, N = pasos, T = t final
# la funcion devuelve pares t,y aproximando a la funcion original
def resuelve_y2_C(y0,t0,y1,T,N):

    f_end_val = y0
    fdot_end_val = y1
    y=[]
    t=[]

    y.append(f_end_val)
    t.append(t0)

    for k in range(1, N+1):
        t.append(t0 + k*(T-t0)/N)
        fdot_end_val = fdot_end_val + (T-t0)/N * (-5 * y[k-1])
        f_end_val = f_end_val + (T-t0)/N * fdot_end_val
        y.append(f_end_val)

    return([t,y])



# armamos los ejes del grafico
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# ploteamos la funcion original a aproximar
x_real, y_real = funcion_real2()
plt.plot(x_real,y_real)
legend = ["y(t) = sqrt(5)sin(sqrt(5) t)"]

N_values = [10,50,100,500,1000]

#ploteamos las diferentes aproximaciones
for k in range(len(N_values)):
    a,b = resuelve_y2_C(0,0,5,10,N_values[k])
    plt.plot(a,b)
    legend.append(format(N_values[k]) + " steps")
    print("Para {} pasos, la diferencia en T=10 es {}".format(N_values[k], math.sqrt(5)*math.sin(math.sqrt(5)*10) - b[-1]))


plt.legend(legend)
plt.show()


# Para el caso A, Podemos observar que a medida que la cantidad de ciclos aumenta, aunque la funcion es periodica el error se va acumulando y los
# arcos van siendo cada vez mas y mas grande, si agrandamos la cantidad de pasos los ciclos tardan mas en agrandarse
# el error en y(10) con N=1000 es de 0.22 y pareceria ser un metodo de orden menor a 1

# Para el caso B y C, Podemos observar que a partir de una cantidad de pasos, los ciclos no se van agrandando cada vez o se agrandan a un periodo
# muy grande y la funcion aproximada es periodica tambien
# para el caso B y C el error en y(10) con N=1000 es de 0.001021 y pareceria ser un metodo de orden 2
