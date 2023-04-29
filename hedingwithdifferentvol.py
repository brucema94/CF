import numpy as np
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
import matplotlib.pyplot as plt

def d1(Stockprice,vol,t):
    return (log(Stockprice/K)+(r+vol**2/2)*(T-t))/(vol*sqrt(T-t))
def d2(Stockprice,vol,t):
    return d1(Stockprice,vol,t)-vol*sqrt(T-t)
def bs_call(Stockprice,vol,t):
    BS_delta = norm.cdf(d1(Stockprice,vol,t))
    Claim = Stockprice*BS_delta-K*exp(-r*(T-t))*norm.cdf(d2(Stockprice,vol,t))
    return BS_delta, Claim

def Euler(S_0, r, sigma, T, M):
    S_m = np.zeros((M+1,6))
    dt = T / M
    S_m[0,1] = S_0
    for m in range(1, M+1):
        Normal = np.random.normal(0,1)
        #time interval
        S_m[m,0] = m * dt 
        #stockprice
        S_m[m,1] = S_m[m-1,1] + r*S_m[m-1,1]*dt + sigma*S_m[m-1,1]*sqrt(dt)*Normal
    return S_m

def delta_Simulation(S_m, S_0, vol, M):
    dt = T / M
    Initial_delta, Initial_value = bs_call(S_0,vol,0)

    S_m[0,2] = Initial_delta
    S_m[0,3] = Initial_value - Initial_delta * S_0
    S_m[0,4] = Initial_value
    for m in range(1, M+1):
        delta1, Optionvalue = bs_call(S_m[m,1],vol,S_m[m,0])
        # delta
        S_m[m,2] = delta1
        # balance
        S_m[m,3] = S_m[m-1,3] * np.exp(r*dt) + ( S_m[m-1,2] - delta1 )* S_m[m,1]
        # replicating portfolio
        S_m[m,4] = S_m[m,3] + delta1 * S_m[m,1]
        # BS Option value
        S_m[m,5] = Optionvalue
    return S_m

M = 365
S_0 = 100
r = 0.06
T = 1
K = 99
sigma = 0.2
volitilities = [0.1,0.5,1]

Stock_simulation = Euler(S_0, r, sigma, T, M)


for a in volitilities:

    Simulation = delta_Simulation(Stock_simulation, S_0, a, M)
    plt.plot(Simulation[1:-1,0], Simulation[1:-1,4],label = f'$vol = {a}$')

baseline = delta_Simulation(Stock_simulation, S_0, 0.2, M)
plt.plot(baseline[1:-1,0], baseline[1:-1,5],label ='baseline_BS')
plt.plot(baseline[1:-1,0], baseline[1:-1,4],label ='baseline_portfolio')

plt.rcParams['font.size'] = '7'
plt.ylabel('replicating portfolio value')
plt.xlabel('time')
plt.legend(loc="upper right")
plt.show()
