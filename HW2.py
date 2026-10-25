import numpy as np
import math
import matplotlib.pyplot as plt
import time
from scipy.linalg import solve_triangular
from scipy.optimize import root_scalar

#Quadratic products

# Generating random matrix and vector.
A = np.random.randn(100,100)
S = A.T @ A 
S_inv = np.linalg.inv(S)

v = np.random.randn(100)

#Method 1
def QP1(S,v):
    S_inv = np.linalg.inv(S)
    return v.T @ S_inv @ v

#Method 2
def QP2(S,v):
    x = np.linalg.solve(S,v)
    return v.T @ x

#Method3
def QP3(S,v):
    L = np.linalg.cholesky(S)
    w = solve_triangular(L, v, lower=True)
    return w.T @ w

print(QP1(S,v), QP2(S,v), QP3(S,v))


#Part 2 of problem 1

dimensions=range(10,501,10)

QP1_time = []
QP2_time = []
QP3_time = []

for d in dimensions:
    
    A = np.random.rand(d,d)
    S = A.T @ A
    V = np.random.rand(10,d)
    
    start = time.perf_counter()
    for v in V:
        QP1(S,v)
    QP1_time.append(time.perf_counter() - start)
    
    start = time.perf_counter()
    for v in V:
        QP2(S,v)
    QP2_time.append(time.perf_counter() - start)
    
    start = time.perf_counter()
    for v in V:
        QP3(S,v)
    QP3_time.append(time.perf_counter() - start)
    
print(QP1_time,QP2_time,QP3_time)
    
plt.figure(figsize=(9, 5))

plt.plot(dimensions, QP1_time, label="Method 1: Inverse")
plt.plot(dimensions, QP2_time, label="Method 2: Linear systems")
plt.plot(dimensions, QP3_time, label="Method 3: Cholesky Decomposition")

plt.xlabel("Dimension d")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime Comparison of Quadratic Products")
plt.legend()
plt.grid(False)
plt.show()

    
    
#Q3

def f(x):
    return 2/x - np.exp(x)/(np.exp(x)-1)


x = np.linspace(0.1, 10, 1000)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, f(x), color='blue', label='f(x)')

plt.axhline(0, color='red', linewidth=0.8)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of f(x)')
plt.legend()
plt.grid(False)
plt.show()


# Bisection method
bisec_iterates = []

def bisec(x):
    bisec_iterates.append(x)
    return f(x)

root_bisec = root_scalar(bisec,
                         bracket=[1,3],
                         method='bisect',
                         maxiter=8,
                         xtol=1e-8)

for k,x in enumerate(bisec_iterates,start=1):
    print(f"k={k}, x = {x:.12f}")
          

print("Final Root using Bisection Method:", root_bisec.root)
print("Number of iterations:", root_bisec.iterations)

#Newton-Raphson method
newton_iterates = []
def fp(x):
    return -2/x**2 + np.exp(x)/(np.exp(x)-1)**2

def newton(x):
    newton_iterates.append(x)
    return f(x)
    
root_newton = root_scalar(newton,
                          x0=1,
                          fprime=fp,
                          method='newton',
                          maxiter=8,
                          xtol=1e-8)

for k,x in enumerate(newton_iterates):
    print(f"k={k}, x={x:.12f}")

print("Final Root using Newton-Raphson Method:", root_newton.root)
print("Number of iterations:", root_newton.iterations)

# Secant method
secant_iterates = []

def secant(x):
    secant_iterates.append(x)
    return f(x)

root_secant = root_scalar(secant,
                          x0=1,
                          x1=3,
                          method='secant',
                          maxiter=8,
                          xtol=1e-8)

for k,x in enumerate(secant_iterates):
    print(f"k={k}, x={x:.12f}")

print("Final Root using Secant Method:", root_secant.root)
print("Number of iterations:", root_secant.iterations)



#3(b) plotting log-error

x_star = 1.59362426004004

def log_errors(iterates, x_star):
    x = np.array(iterates, dtype=float)

    errors = np.abs(x - x_star)

    # Remove consecutive pairs containing zero errors
    valid = (errors[:-1] > 0) & (errors[1:] > 0)

    log_ek = np.log(errors[:-1][valid])
    log_ek1 = np.log(errors[1:][valid])

    return log_ek, log_ek1

# Plot
plt.figure(figsize=(8, 6))

for iterates, label in [
    (bisec_iterates[2:], "Bisection"),
    (newton_iterates, "Newton-Raphson"),
    (secant_iterates, "Secant")
]:
    x, y = log_errors(iterates, x_star)
    plt.plot(x, y, 'o-', label=label)

plt.xlabel(r'$\log|x^{(k)}-x^*|$')
plt.ylabel(r'$\log|x^{(k+1)}-x^*|$')
plt.title('Convergence of Root-Finding Methods')
plt.legend()
plt.grid(False)
plt.show()


                          





    
    
    
    






