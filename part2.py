# Program to simulate transformed Black-Scholes PDE
# V_tau = alpha V_X + gamma V_XX - rV
# using FCTS

# First import some stuff
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.linalg import inv

# Parameters for simulation
N = 500 #Number of steps on X-grid
M = 500 #Number of steps on tau-grid
X_MAX = 8
X_MIN = -5
N_grid = [i for i in range(1,N+1)]
X_grid = np.linspace(X_MIN, X_MAX, N)

# Black-Scholes parameters
r = 0.04
vol = 0.3
S0 = 100
K = 110
T = 1

# Useful shortcuts


# European call value at maturity
def phi(u):
    return max(u-K, 0)

def BS(S0):
    d1 = (math.log(S0/K) + r + 0.5*(vol**2))/(vol)
    d2 = d1 - vol*math.sqrt(T)
    BS_value = S0*norm.cdf(d1) - K*math.e**(-r*T)*norm.cdf(d2)
    return BS_value

# Simulate the PDE using FCTS scheme
def simulatePDE_FCTS(N, M):
    # Functional grid
    tau_grid = np.linspace(0, T, M)
    X_grid = np.linspace(X_MIN, X_MAX, N)
    # Define helpful parameters
    alpha = r-0.5*(vol**2)
    gamma = 0.5*(vol**2)
    dt = T/M
    dX = (X_MAX - X_MIN)/(N-1)
    eta = (alpha*dt)/(2*dX)
    zeta = (gamma*dt)/((dX)**2)
    
    # Matrices A and B
    A = np.zeros((N, N))
    B = np.zeros((N, N))
    for i in range(N):
        B[i,i] = 1
        A[i,i] = 1- r*dt - 2*zeta
    for i in range(N-1):
        A[i, i+1] = eta + zeta
        A[i+1, i] = -eta + zeta
    for i in range(N):
        A[0,i] = 0
        A[N-1, i] = 0
    A[0,0] = 1
    A[N-1, N-1]=1
    V = np.zeros((N, M)) #V[i,j] = V_i^j
    # Boundary conditions on X
    for j in range(M):
        V[0, j] = 0
        #V[N-1, j] = math.e**(X_MAX)
        V[N-1, j] = math.e**(X_MAX)
    # Boundary condition on maturity
    for i in range(N):
        V[i, 0] = phi(math.e**(X_grid[i]))
    # As B is the identity, we have V^n+1 = AV^n and hence V^n = A^n V^0
    for j in range(M-1):
        V[:, j+1] = np.matmul(A, V[:, j])
        for j in range(M):
            V[0, j] = 0
            V[N-1, j] = math.e**(X_MAX)
        
    return V

# Simulate PDE using Crank-Nicholson scheme
def simulatePDE_Crank(N, M):
    # Functional
    tau_grid = np.linspace(0, T, M)
    X_grid = np.linspace(X_MIN, X_MAX, N)
    # Helpful parameters
    alpha = r-0.5*(vol**2)
    gamma = 0.5*(vol**2)
    dt = T/M
    dX = (X_MAX - X_MIN)/(N)
    beta = (alpha*dt)/(4*dX)
    delta = 0.5*gamma*(dt)/((dX)**2)
    
    # Matrices A and B
    A = np.zeros((N, N))
    B = np.zeros((N, N))
    # Build up the matrices
    for i in range(N):
        A[i,i] = 1 - 2*delta - 0.5*r*dt
        B[i,i] = 1 - 2*delta + 0.5*r*dt
    for i in range(N-1):
        A[i, i+1] = delta  + beta
        A[i+1, i] = -beta + delta
        B[i, i+1] = -delta - beta
        B[i+1, i] = beta - delta
    for i in range(N):
        A[0, i] = 0
        A[N-1, i] = 0
        B[0,i] = 0
        B[N-1, i] = 0
    B[0,0] = 1
    B[N-1, N-1] = 1
    A[0,0] = 1
    A[N-1, N-1] = 1
    C = inv(B)
    D = np.matmul(C, A)
    # Build up value grid
    V = np.zeros((N, M))
    # Boundary conditions on X
    for j in range(M):
        V[0, j] = 0
        V[N-1, j] = math.e**(X_MAX)
        #V[N-1, j] = phi(math.e**(X_MAX))
    # Boundary condition on maturity
    for i in range(N):
        V[i, 0] = phi(math.e**(X_grid[i]))
    for j in range(M-1):
        V[:, j+1] = np.matmul(D, V[:, j])
        #for j in range(M):
         #   V[0, j] = 0
          #  V[N-1, j] = phi(math.e**(X_MAX))
    return V
    
#difference_grid = np.zeros(N)
#for i in range(len(N_grid)):
 #   V = simulatePDE_Crank(N_grid[i], M)
  #  temp = V[int(N_grid[i]/2), M-1]
   # X_grid = np.linspace(X_MIN, X_MAX, N_grid[i])
    #temp2 = BS(math.e**(X_grid[int(N_grid[i]/2)]))
    #difference_grid[i] = abs(temp2 - temp)*((N_grid[i])**2)

    
    

V = simulatePDE_FCTS(N, M)
#V = simulatePDE_Crank(N, M)

#V_grid = np.zeros(N)
#for i in range(N):
 #   V_grid[i] = V[i, M-1]

#delta_grid = np.zeros(N-1)
#dX = (X_MAX - X_MIN)/(N-1)
#for i in range(N-1):
 #   delta_grid[i] = (V_grid[i+1] - V_grid[i])/dX
  #  delta_grid[i] = delta_grid[i]*(math.e**(-X_grid[i]))
    
#S_grid = np.zeros(N-1)
#for i in range(N-1):
 #   S_grid[i] = math.e**(X_grid[i])

#BS_grid = np.zeros(N)
#for i in range(N):
 #   BS_grid[i] = BS(math.e**(X_grid[i])) 

#plt.plot(N_grid, difference_grid, label='Difference between BS and X_mid fair value', color='b')

#one_grid = np.ones(N-1)
#zero_grid = np.zeros(N-1)

#plt.plot(S_grid, delta_grid, label='Delta approximation', color='b')
#plt.plot(S_grid, one_grid, color = 'r', linestyle='dashed')
#plt.plot(S_grid, zero_grid, color = 'r', linestyle='dashed')
plt.plot(X_grid, V[:, M-1], label='PDE Numerical solution', color='b', linestyle='dashed')
#plt.plot(X_grid, BS_grid, label='Black-Scholes value', color='r', linestyle='dashdot')
#plt.plot(X_grid, V[:, M-1] - BS_grid, label='PDE (FTCS) minus Black-Scholes', color='g')
plt.xlabel('Value of X')
plt.ylabel('Indicated value')
#plt.xlim(X_MIN-0.01, X_MAX+0.01)
#plt.ylim(-5, 1)
#plt.xlim(50, N)
#plt.ylim(0, 0.05)
plt.legend()
plt.show()