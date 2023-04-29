import numpy as np
from math import log, sqrt, pi, exp, cos, sin, e
from scipy.stats import norm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from decimal import *
import time


r = 0.04
sigma = 0.3
S0 = 100
K = 110
T = 1

def d1(Stockprice,vol):
    return (log(Stockprice/K)+(r+vol**2/2)*T)/(vol*sqrt(T))
def d2(Stockprice,vol):
    return d1(Stockprice,vol)-vol*sqrt(T)
def bs_call(Stockprice,vol):
    BS_delta = norm.cdf(d1(Stockprice,vol))
    Claim = Stockprice*BS_delta-K*exp(-r*T)*norm.cdf(d2(Stockprice,vol))
    return Claim


def Fourier_COS(S0, K, T, sigma, r, N):
    start = time.perf_counter()
    a = np.log(S0/K) + r * T - 12 * np.sqrt(sigma**2 * T)
    b = np.log(S0/K) + r * T + 12 * np.sqrt(sigma**2 * T)
    d = b
    c = 0

    coefficient_d = pi * ((d - a)/(b - a))
    coefficient_c = pi * ((c - a)/(b - a))
    i = complex(0,1)
    V = 0

    for k in range(N + 1):
        coefficient_k = (k * pi)/(b - a)
        
        Fk = 2/(b - a) * np.real(e ** (i * coefficient_k * (r - 1/2 * sigma **2) * T 
                - (1/2 * sigma**2 * T * (coefficient_k)**2)) * e **(i * k *pi *(log(S0/K)- a)/(b - a)))
        chik = (1/(1 + (coefficient_k)**2)) * (cos(k * coefficient_d) * (e ** d) - cos(k * coefficient_c)
            + coefficient_k * (sin(k * coefficient_d) * (e ** d)) - coefficient_k * sin(k * coefficient_c))
        
        if k == 0:
            psik = d - c
        else:
            psik = (sin(k * coefficient_d) - sin(k * coefficient_c))/coefficient_k

        if k == 0:
            Fk *= 1/2

        Gk = 2/(b - a) * K * (chik - psik)
        V += Fk * Gk

    V *= e**(-r * T) * ((b - a) / 2)
    end = time.perf_counter()
    Computing = end - start
    return V, Computing


N_grid = np.linspace(1,192,191,dtype=int)
DIFF = np.zeros(192)
Computingtimegrid = np.zeros(192)
V_BS = bs_call(S0,sigma)


for n in range(1,192): 
    V_COS, computingtime = Fourier_COS(S0, K, T, sigma, r, n)
    Computingtimegrid[n] = 1000 * computingtime
    DIFF[n] = (V_COS - V_BS)

print(DIFF)

plt.plot(N_grid,DIFF[1:], color='r')
plt.title('Computing time with different coefficients')
plt.xlabel('Fourier cosine coefficients')
plt.ylabel('Computing time in msec')
plt.show()
