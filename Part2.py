import numpy as np
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
import matplotlib.pyplot as plt

def buildTree(S, vol , T, N):
    r = 0.06
    dt = T / N
    matrix = np.zeros((N + 1, N + 1))

    # Appendix A2.3  (9)
    u = np.exp(vol * np.sqrt(dt)) 
    d = np.exp(-vol * np.sqrt(dt)) 
    # Appendix A2.2 (3)
    p = (np.exp(r*dt) - d)/(u - d) 

    for i in np.arange(N + 1):
        for j in np.arange(i + 1): 
            matrix[i, j] = S * d ** (i-j) * u ** (j)

    return matrix


def valueOptionMatrix(tree,T,r,K,vol,N,S):
    dt=T/N
    u= np.exp(vol * np.sqrt(dt)) 
    d = np.exp(-vol * np.sqrt(dt)) 
    p = (np.exp(r*dt) - d)/(u - d)
    
    columns=tree.shape[1]
    rows=tree.shape[0]
    for c in np.arange(columns):
      
        tree[rows-1,c]= max(0, tree[rows-1,c] - K)

    for i in np.arange(rows-1)[::-1]:
        for j in np.arange(i+1):
            down=tree[i+1,j]
            up=tree[i+1,j+1]
            # Assignment
            tree[i,j]= (np.exp(-r*dt)) * (p * up + (1 - p) * down)
    
    Fair_price = tree[0,0]
    # A2.1 (2)
    Delta = (tree[1, 1] - tree[1, 0])/(S * (u - d))
    #print(tree)
    return Delta, Fair_price

def American_option(tree,T,r,K,vol,N,S,Optiontype):
    dt=T/N
    u= np.exp(vol * np.sqrt(dt)) 
    d = np.exp(-vol * np.sqrt(dt)) 
    p = (np.exp(r*dt) - d)/(u - d)
    
    columns=tree.shape[1]
    rows=tree.shape[0]
    for c in np.arange(columns):
        S=tree[rows-1,c]
        if Optiontype == "Call": tree[rows-1, c] = max(0, S - K)
        elif Optiontype == "Put": tree[rows-1,c]= max(0, K - S)
        else: print("wrong type of options")

    for i in np.arange(rows-1)[::-1]:
        for j in np.arange(i+1):
            down=tree[i+1,j]
            up=tree[i+1,j+1]
            if Optiontype == "Call":
                tree[i,j] = max(tree[i,j]-K, (np.exp(-r*dt)) * (p * up + (1 - p) * down))
            elif Optiontype == "Put":
                tree[i,j] = max(K-tree[i,j], (np.exp(-r*dt)) * (p* up + (1 - p) * down))
            else: print("wrong type of options")

    Fair_price = tree[0,0]
    #print(tree)
    # A2.1 (2)
    return Fair_price



Sigma = 0.2
S = 100
T = 1
N = 5
r = 0.06
K = 99

tree = buildTree(S,Sigma,T,N)
delta, price = valueOptionMatrix(tree,T,r,K,Sigma,N,S)
print(price)

# black scholes theoretical value
def d1(Stockprice,vol):
    return (log(Stockprice/K)+(r+vol**2/2)*T)/(vol*sqrt(T))
def d2(Stockprice,vol):
    return d1(Stockprice,vol)-vol*sqrt(T)
def bs_call(Stockprice,vol):
    BS_delta = norm.cdf(d1(Stockprice,vol))
    Claim = Stockprice*BS_delta-K*exp(-r*T)*norm.cdf(d2(Stockprice,vol))
    return BS_delta, Claim

BS_delta, Claim_BS = bs_call(S,Sigma)
print('The Option Price determined from Black Scholes is: ', Claim_BS)

American_tree = buildTree(S,Sigma,T,N)
American = American_option(American_tree,T,r,K,Sigma,N,S,Optiontype = "Call")
print(American)



def Euler(S_0, r, sigma, T, M):
    dt = T / M
    S_m = np.zeros(M+1)
    S_m[0] = S_0

    for m in range(1, M+1):
        Normal = np.random.normal(0,1)
        S_m[m] = S_m[m-1] + r*S_m[m-1]*dt + sigma*S_m[m-1]*sqrt(dt)*Normal
    return S_m

S0 = 100
sigma = 0.2
r = 0.06
T = 1
M = 254
Time_Interval = np.linspace(0,1,M+1)
Stockprice = Euler(S0, r, sigma, T, M)

print(Time_Interval)
plt.plot(Time_Interval, Stockprice)
plt.show()