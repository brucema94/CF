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
    dt = T / M
    S_m = np.zeros((M+1,5))
    S_m[0,1] = S_0
    S_m[0,2] = bs_call(S_0,sigma,0)[0]
    S_m[0,3] = bs_call(S_0,sigma,0)[0]

    for m in range(1, M+1):
        Normal = np.random.normal(0,1)
        S_m[m,0] = m * dt 
        S_m[m,1] = S_m[m-1,1] + r*S_m[m-1,1]*dt + sigma*S_m[m-1,1]*sqrt(dt)*Normal
        if m < M:
            delta1, Optionvalue = bs_call(S_m[m,1],sigma,S_m[m,0])
            S_m[m,2] = delta1
            S_m[m,4] = delta1 * S_m[m,1] - Optionvalue 
        #if m % 7 == 0 & m < M: 
        #    delta2 = bs_call(S_m[m,1],sigma,S_m[m,0])[0]
        #    S_m[m,3] = delta2
        
    return S_m

S_0 = 100
sigma = 0.2
r = 0.06
T = 1
M = 365
K = 99

Simulation = Euler(S_0, r, sigma, T, M)

fig, (plotP, plotQ) = plt.subplots(2, 1,sharex=True)
fig.suptitle('Hedging simulation Daily vs Weekly')

plotQ.plot(Simulation[1:-1,0], Simulation[1:-1,1])
plotQ.set_ylabel('Stockprice')
plotQ.set_xlabel('time')
plotP.plot(Simulation[1:-1,0], Simulation[1:-1,2], label = 'Daily')
plotP.plot(Simulation[7:-1:7,0], Simulation[7:-1:7,2], label = 'Weekly')
plotP.set_ylabel('Delta')

plotP.legend(loc="upper right")
plt.show()