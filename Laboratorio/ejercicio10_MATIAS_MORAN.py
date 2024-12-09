## EJERCICIO 10
## MATIAS MORAN
## LU 806/19

import math
import numpy as np
import scipy.optimize as scop
import matplotlib.pyplot as plt

#A)

p = lambda x: 1.3/12 * sum((1-x)**np.arange(1,13)) - 1
x0 = 0.05

print('punto A')
print(scop.fsolve(p, x0)[0])
print(p(0.04110665))
print('')
print('')
#vemos que la funcion aplicada en el punto es casi 0.


#B)

f = lambda x: x**2 - x**3
df = lambda x: 2*x - 3*x**2



#C)

j = lambda x: x
dj = lambda x: 1

def NewtonRaphson(f, df, x0):
    ret = x0
    iter = 0
    while math.fabs(f(ret)) > 10**(-8) and iter < 1000:
        ret = ret - f(ret)/df(ret)
        iter += 1

    return ret

print('punto C')
print(scop.fsolve(f, 5)[0])
print(NewtonRaphson(f, df, 5))

print(scop.fsolve(j, 1)[0])
print(NewtonRaphson(j, dj, 1))

print('')
print('')

#vemos que en los 2 casos los resultados son sumamente similares.



#D)


def eulerImplicito(f, df, h, t0, tf, y0):
    values = []
    yn = y0
    iter = t0


    while iter < tf:
        values.append(yn)
        F = lambda x : x - yn - h*f(x)
        DF = lambda x : 1 - h*df(x)
#       yn = scop.fsolve(F, yn)[0]
        yn = NewtonRaphson(F, DF, yn)

        iter += h
    return values



h = 2
r = 0.01
tf = 2/r
ti = 0


y = lambda y: y**2 - y**3
dy = lambda y: 2*y - 3*y**2


plt.plot(np.linspace(ti, tf + h, num = 100), eulerImplicito(y, dy , h, ti, tf, r))
plt.show()