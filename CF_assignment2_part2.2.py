import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm


T = 1
N = 50
M = 3000


S = 100
K = 99
r = 0.06
vol = 0.2
H = 100
h_star = 0.01

epsilon = 0.005

# Generate standard normal variables in NM array
def generateNormal(N, M):
    normals = np.zeros((M, N))
    for i in range(M):
        temp = np.random.normal(size = N)
        normals[i, :] = temp
    #print(temp[-1])
    return normals
        
# Function to generate S_T using Euler forward
def eulerForward(N, S, vol, T, r, normals):
    dt = T/N
    S_grid = np.zeros(N+1)
    S_grid[0] = S
    for i in range(1,N+1):
        S_grid[i] = S_grid[i-1] + S_grid[i-1]*(r*dt + vol*normals[i-1]*math.sqrt(dt))
    return S_grid

# European call
#def f(S_T, K):
 #   return max(S_T - K, 0)

# Digital option  
#def f(S_T, K):
#    k=0
 #   if S_T > K:
  #      k=1
   # return k
   
# Smoothening of digital option
def f2(S_T, K):
    return 1/(1+math.e**(-10*(S_T - K)))

# Pathway method function
def f(S_T, K):
    return (S_T)*math.e**(-10*(S_T - K))*((1+math.e**(-10*(S_T-K)))**(-2))

def valueOption(N, T, f, K, r, vol, S, normals):
    S_grid = eulerForward(N, S, vol, T, r, normals)
    return math.e**(-r*T)*f(S_grid[N], K)

# Function that does monte Carlo 
def monteCarloOption(N, M, T, f, K, r, vol, normals):
    sum = 0
    for i in range(M):
        sum += valueOption(N, T, f, K, r, vol, S, normals)
    sum = sum*math.e**(-r*T)/M
    return sum

# Approximate the sensitivity delta = dV/dS
def deltaSensitivity(N, M, T, f, K, r, vol, epsilon):
    normals = generateNormal(N,M)
    normals2 = generateNormal(N, M)
    #for i in range(M):
     #   for j in range(N):
      #      normals2[i,j] = -normals2[i,j]
    V1 = np.zeros(M)
    V2 = np.zeros(M)
    sum = 0
    for i in range(M):
        v_1 = valueOption(N, T, f, K, r, vol, S, normals[i, :])
        # TODO
        v_2 = valueOption(N, T, f, K, r, vol, S + epsilon, normals2[i, :])
        temp = (v_2 - v_1)/epsilon
        sum += temp
    return sum/M

# Yields the theoretical BS value for delta
def realDelta(S, K, r, T, vol):
    d1 = (math.log(S/K) + (r+0.5*vol**2)*T)/(vol*math.sqrt(T))
    d2 = d1 - vol*math.sqrt(T)
    return norm.cdf(d1)

# Compute the MSE for h if we use bump and revalue
def computeMSE(N, M, T, f, K, r, vol, h):
    normals = generateNormal(N,M)
    normals2 = normals
    for i in range(M):
        for j in range(N):
            normals2[i,j] = -normals2[i,j]
    V1 = np.zeros(M)
    sum = 0
    sum2 = 0
    for i in range(M):
        v1 = valueOption(N, T, f, K, r, vol, S, normals[i, :])
        v2 = valueOption(N, T, f, K, r, vol, S + h, normals[i, :])
        V1[i] = (v2 - v1)/h
        sum += V1[i]
    sum = sum/M
    bias = sum - realDelta(S, K, r, T, vol)
    for j in range(M):
        sum2 += (V1[j] - sum)**2
    var = (1/(M-1))*sum2
    return var + bias**2


def monteCarlo(S, N, M, T, K, f, r, vol, h_star, same_seed):
    normals = generateNormal(N,M)
    normals2 = normals
    if same_seed:
        for i in range(M):
            for j in range(N):
                normals2[i,j] = -normals[i,j]
    V1 = np.zeros(M)
    sum = 0
    for i in range(M):
        v1 = valueOption(N, T, f, K, r, vol, S, normals[i, :])
        v2 = valueOption(N, T, f, K, r, vol, S + h_star, normals[i, :])
        V1[i] = (v2 - v1)/h_star
        sum += V1[i]
    return sum/M
    

def pathwayMethod(S, N, M, T, K, f, r, vol):
    normals = generateNormal(N, M)
    sum = 0
    for i in range(M):
        sum += valueOption(N, T, f, K, r, vol, S, normals[i, :])
    sum = 10*(sum/(S*M))
    return sum

def likelihoodMethod(S, N, M, T, K, f, r, vol):
    normals = generateNormal(N, M)
    sum = 0
    for i in range(M):
        S_grid = eulerForward(N, S, vol, T, r, normals[i, :])
        ST = S_grid[N]
        sum += f(ST, K)*(math.log(ST/S) - (r - 0.5*(vol**2))*T)
    sum = (sum*math.e**(-r*T))/(S*M*T*(vol**2))
    return sum

M_grid = [100*i for i in range(1, 50)]
approx_grid = np.zeros(len(M_grid))
approx_grid2 = np.zeros(len(M_grid))
approx_grid3 = np.zeros(len(M_grid))
for i in range(len(M_grid)):
    approx_grid[i] = pathwayMethod(S, N, M_grid[i], T, K, f, r, vol)
    #approx_grid[i] = monteCarlo(S, N, M_grid[i], T, K, f2, r, vol, h_star, True)
    approx_grid3[i] = likelihoodMethod(S, N, M_grid[i], T, K, f2, r, vol)


    
#alpha_grid = [1, 3, 5, 10, 15]
#x_grid = np.linspace(95, 105, 200)
#graph_grid = np.zeros((len(alpha_grid), len(x_grid)))
#for i in range(len(alpha_grid)):
#    for j in range(len(x_grid)):
#        graph_grid[i,j] = f(x_grid[j], K, alpha_grid[i])
        
#delta_0 = realDelta(S, K, r, T, vol)
#delta0_grid = np.zeros(len(M_grid) )
#for i in range(len(delta0_grid)):
 #   delta0_grid[i] = delta_0
#h_grid = np.linspace(0.01, 300, H)
#MSE_grid = np.zeros(H)
#for i in range(H):
 #   MSE_grid[i] = computeMSE(N, M, T, f, K, r, vol, h_grid[i])

#for i in range(len(alpha_grid)):
#    plt.plot(x_grid, graph_grid[i, :], label = 'Logistic curve for alpha = ' + str(alpha_grid[i]))

#plt.plot(h_grid, MSE_grid, label = 'Mean-squared error of estimation of Delta for values of h')
plt.plot(M_grid, approx_grid, label = 'Pathway method', color = 'b')
plt.plot(M_grid, approx_grid3, label = 'Likelihood ratio method', color = 'r')
plt.xlabel('Value of M')
plt.ylabel('Approximation of delta parameter')
plt.legend()
plt.show()