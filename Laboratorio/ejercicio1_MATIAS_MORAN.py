## EJERCICIO 1
## MATIAS MORAN
## LU 806/19

import math

# A) 
# Calculamos el promedio de x1, x2 y x3:
x1 = 1
x2 = 3
x3 = 7

def promedio(a,b,c):
    return (a+b+c)/3

p = promedio(x1,x2,x3)
print("El promedio es ",p)


# B) 
def calcula_p(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    p = promedio(x1,x2,x3)
    return(p)

nums = [1, 3, 7]

print(calcula_p(nums))



# C) 
def calcula_raices(a,b,c):
    grad = math.sqrt(b**2 - 4*a*c)
    x1 = (-b - grad)/(2*a)
    x2 = (-b + grad)/(2*a)
    return([x1,x2])

a= 1
b = 0
c = -1
print(calcula_raices(a,b,c))





# D)
def calcula_coeficientes(v,x):
    v1 = v[0]
    v2 = v[1]
    x1 = x[0]
    x2 = x[1]
    a = (x2 - v2)/((x1+v1)**2)
    b = -2*v1*a
    c = a*(v1**2) + v2
    return([a,b,c])

v = [0,-1]
x = [1,0]
print(calcula_coeficientes(v,x))



# E)
def calcula_raices2(v,x):
    coeficientes = calcula_coeficientes(v,x)
    a = coeficientes[0]
    b = coeficientes[1]
    c = coeficientes[2]
    raices = calcula_raices(a,b,c)
    return raices
    
v = [0,-1]
x = [1,0]
print(calcula_raices2(v,x))