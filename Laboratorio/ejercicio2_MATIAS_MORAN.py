## EJERCICIO 2
## MATIAS MORAN
## LU 806/19

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt


# A) 

def numero_random():
    x = rd.rand()-.5
    return(x)

X = []
N = 100
for q in range(N):
    X.append(numero_random())
    
print(np.mean(X))
plt.plot(X)
plt.savefig("grafico_todos_independientes.png")
plt.clf()

#El promedio es acorde al valor esperado el cual es 0, ya que es una distribucion uniforme entre valores -0.5 y 0.5



# B)

X=  [0]
N = 99
for q in range(N):
    X_nuevo = X[-1] + numero_random()
    X.append(X_nuevo)

print(np.mean(X))
plt.plot(X)
plt.savefig("grafico_random_walk.png")
plt.clf()

#si hay una clara diferencia, el valor puede exceder los limites de -0,5 y 0,5 y el promedio tambien, ya no es cercano a 0



# C)

def genera_X(N):
    X=  [0]
    for q in range(N):
        X_nuevo = X[-1] + numero_random()
        X.append(X_nuevo)
    return(X)

for q in range(50):
    X = genera_X(99)
    plt.plot(X)

plt.savefig("muchastrayectorias.png")
plt.clf()



# D)

def X_con_W(N,W):
    X = [0]
    iterador = 0
    while np.abs(X[-1])<W and iterador < N:
        X_nuevo = X[-1] + numero_random()
        X.append(X_nuevo)
        iterador = iterador +1
    return (X)

W=3
N=100
Ntrayect = 500
iteraciones = []
for q in range(Ntrayect):
    X = X_con_W(N,W)
    iteraciones.append(len(X))

print(np.mean(iteraciones))

# en promedio la cantidad de iteraciones es aproximadamente 75.75


# E)

def minimo_y_maximo(X):
    minimo = np.inf 
    maximo = -np.inf 
    for x in X:
        if minimo>x:
            minimo = x
        if x>maximo:
            maximo = x
    
    return([minimo,maximo])
    
W=3
N=100
Ntrayect = 500
minTotal = []
maxTotal = []
for q in range(Ntrayect):
    X = X_con_W(N,W)
    min_y_max = minimo_y_maximo(X)
    minTotal.append(min_y_max[0])
    maxTotal.append(min_y_max[1])

print(np.mean(minTotal))
print(np.mean(maxTotal))

# el minimo en promedio es -1.78 y el maximo es 1.78
