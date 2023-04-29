import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def bottom_boundary_condition(K,T,S_min, r, t):
    return np.zeros(t.shape)

def top_boundary_condition(K,T,S_max, r, t):
    return S_max-np.exp(-r*(T-t))*K

def final_boundary_condition(K,T,S):
    return np.maximum(S-K,0)

def compute_abc( K, T, sigma, r, S, dt, dS ):
    # a = .5 *
    a = -sigma**2 * S**2/(2* dS**2 ) + r*S/(2*dS)
    b = r + sigma**2 * S**2/(dS**2)
    c = -sigma**2 * S**2/(2* dS**2 ) - r*S/(2*dS)
    return a,b,c

def compute_lambda( a,b,c ):
    return scipy.sparse.diags( [a[1:],b,c[:-1]],offsets=[-1,0,1])

def compute_W(a,b,c, V0, VM):
    M = len(b)+1
    W = np.zeros(M-1)
    W[0] = a[0]*V0
    W[-1] = c[-1]*VM
    return W

from numpy import log as ln
def price_call_explicit(K, T, r, sigma, N, M):
    """"
    N: Time Steps
    M: Price Steps
    """
    # Choose the shape of the grid
    dt = T/N
    S_min = 0
    S_max = K*np.exp(8*sigma*np.sqrt(T))
    dS = (S_max-S_min)/M
    S = np.linspace(S_min,S_max,M+1)  # convert to log space
    # print(S)

    t = np.linspace(0,T,N+1)
    V = np.zeros((N+1,M+1)) #...

    # Set the boundary conditions
    V[:,-1] = top_boundary_condition(K,T,S_max,r,t)
    V[:,0]  = bottom_boundary_condition(K,T,S_max,r,t)
    V[-1,:] = final_boundary_condition(K,T,S)

    # Apply the recurrence relation
    a,b,c  = compute_abc(K,T,sigma,r,S[1:-1],dt,dS)
    # print(a, b, c)
    Lambda = compute_lambda( a,b,c)
    print(Lambda.toarray())
    identity = scipy.sparse.identity(M-1)

    for i in range(N,0,-1):
        W = compute_W(a, b, c, V[i,0], V[i,M])
        # print(W, "\n")  #  ;   break

        # Use `dot` to multiply a vector by a sparse matrix
        V[i-1,1:M] = (identity-Lambda*dt).dot( V[i,1:M] ) - W*dt

    return V, t, S


V, t, S = price_call_explicit( K=110, T=1., r=.05, sigma=.3, N=10, M=10)

S_, T_, V_ = [], [], []
for _row in range(V.shape[0]):
    for _column in range(V.shape[1]):
        S_.append(S[_column])   
        T_.append(t[_row])
        V_.append(V[_row, _column])

S_, T_, V_ = np.array(S_), np.array(T_), np.array(V_)
S_.shape, T_.shape, V_.shape


# gc.collect()
""" 3D Graph """
plt.rcParams['font.size'] = '15'

fig = plt.figure(figsize=(15,12))
ax = plt.axes(projection='3d')
ax.plot_trisurf(T_, S_, V_, cmap='viridis', edgecolor='none')  # 'viridis' , cm.jet

# plt.xticks(plt.xticks()[0][1:-1], [f'{10**x*100:.2f}' for x in plt.xticks()[0][1:-1]])

ax.set_ylabel('\n\nStock Value (USD)')   ;   ax.set_xlabel('\n\n Time (t) Years')   ;   ax.set_zlabel('\n\n\nOption Value (USD)')
# ax.dist = 10
ax.set_title("Option Value for Different Parameters")
plt.show()