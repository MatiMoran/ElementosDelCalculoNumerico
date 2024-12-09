import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

# A)

A_1 = np.array([[1,0],[0,2]])
A_2 = np.array([[1,1],[0,2]])
b = np.array([1,1])

x_1 = npl.solve(A_1, b)
x_2 = npl.solve(A_2, b)


# B)
def jacobi(A,b,x0,max_iter,tol):
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    M = D
    N = U + L
    B = -(np.dot(npl.inv(M),N))
    C = np.dot(npl.inv(M) ,b)
    xN = x0
    resto = npl.norm(npl.solve(A, b) - xN)
    iter = 0
    while (iter < max_iter and resto > tol):
        xN = np.dot(B,xN) + C
        iter = iter + 1
        resto = npl.norm(npl.solve(A, b) - xN)
    
    if iter==max_iter:
        print('max_iter alcanzado')
    return([xN,iter])


# C)
print()
print("El resultado de A_1 es: ")
print(x_1)
print("El resultado calculado de A_1 es: ", jacobi(A_1, b, [.0,.0], 100, 0.01)[0])
print("El resultado de A_2 es: ")
print(x_2)
print("El resultado calculado de A_2 es: ", jacobi(A_2, b, [.0,.0], 100, 0.01)[0])
print()


# D)
def gauss_seidel(A,b,x0,max_iter,tol):
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    M = D + L
    N = U
    B = -(np.dot(npl.inv(M),N))
    C = np.dot(npl.inv(M) ,b)
    xN = x0
    resto = npl.norm(npl.solve(A, b) - xN)
    iter = 0
    while (iter < max_iter and resto > tol):
        xN = np.dot(B,xN) + C
        iter = iter + 1
        resto = npl.norm(npl.solve(A, b) - xN)
    
    if iter==max_iter:
        print('max_iter alcanzado')
    return([xN,iter])


# F)
print("El resultado de A_1 es: ")
print(x_1)
print("El resultado calculado de A_1 es: ", gauss_seidel(A_1, b, [.0,.0], 100, 0.01)[0])
print("El resultado de A_2 es: ")
print(x_2)
print("El resultado calculado de A_2 es: ", gauss_seidel(A_2, b, [.0,.0], 100, 0.01)[0])
print()

# G)

def matriz_y_vector(n):
    A = np.random.rand(n, n)
    while npl.det(A) == 0:
        A = np.random.rand(n, n)
    b = np.random.rand(n, 1)
    return([A,b])


# H)

trials_J = []
trials_GS = []
Ntrials = 10**4
trials = 0
x0 = np.transpose(np.array([[0,0,0]]))
max_iter = 10**6
tol = 10**-5
while trials<Ntrials:
    Ab = matriz_y_vector(3)
    A = Ab[0]
    b = Ab[1]
    rGS= gauss_seidel(A,b,x0,max_iter,tol)
    rJ = jacobi(A,b,x0,max_iter,tol)
    trials = trials + 1
    print(trials)
    trials_GS.append(rGS[1])
    trials_J.append(rJ[1])

plt.scatter(trials_J,trials_GS)
plt.ylabel("Gauss-Seidel")
plt.xlabel("Jacobi")
plt.yscale("log")
plt.xscale("log")
plt.show()


# Aunque haya algunos casos donde el metodo de jacobi sea mejor que el de GS, en la mayoria 
# de casos el metodo de GS es mucho mejor que el de jacobi y converge mucho mas rapido