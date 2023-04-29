import numpy as np
import numpy as np
from math import log, sqrt
from scipy.stats import gmean
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import sem

np.random.seed(1000)

def Euler(S_0, r, sigma, T, N):
    S_n = np.zeros(N+1)
    dt = T / N
    S_n[0] = S_0
    for n in range(1, N+1):
        Normal = np.random.normal(0,1)
        S_n[n] = S_n[n-1] + r*S_n[n-1]*dt + sigma*S_n[n-1]*sqrt(dt)*Normal
    return S_n

def Asian_analytical(S0, r, sigma, T, N, K):

    sigma_avg = sigma * np.sqrt((2 * N + 1)/(6 * (N + 1)))
    r_avg = 0.5 * (r - 0.5 * sigma **2 + sigma_avg **2)

    d1 = (log(S0/K) + (r_avg + 1/2 * sigma_avg ** 2) * T)/(sigma_avg * np.sqrt(T))
    d2 = (log(S0/K) + (r_avg - 1/2 * sigma_avg ** 2) * T)/(sigma_avg * np.sqrt(T))

    Analytical_value = np.exp(-r * T) * (S0 * np.exp(r_avg * T) * norm.cdf(d1) - K * norm.cdf(d2))

    return Analytical_value


def Asian_MC(S0, r, sigma, T, N, M, K):
    Option_price_arithmetic = np.zeros(M)
    Option_price_geometric = np.zeros(M)

    for m in range(M):
        Stockprice = Euler(S0, r, sigma, T, N)
        geometric_average = gmean(Stockprice)
        Option_price_geometric[m] = np.exp(-r*T) * max(0,  geometric_average - K)
        arithmetic_average = np.mean(Stockprice)
        Option_price_arithmetic [m] = np.exp(-r*T) * max(0, arithmetic_average - K)

    return Option_price_arithmetic, Option_price_geometric


S0 = 100
T = 1
N = 50
K = 99
r = 0.06
sigma = 0.2
M = 10000

# this bit is separated to show that only till 100000 the standard MC method give a good enough variance 

Analytical_value = Asian_analytical(S0, r, sigma, T, N, K)
#Aritforerror, Geoforerror = Asian_MC(S0, r, sigma, T, N, M, K)
#Ariterror = sem(Aritforerror)
#Geoerror = sem(Geoforerror)

#print(Analytical_value)
#print(Ariterror)
#print(Geoerror)

IterationIndex = 100
Analytical_plot = np.zeros(IterationIndex)
Iteration = np.linspace(1000,5000,IterationIndex,dtype=int)
GeoMC = np.zeros((IterationIndex,2))
AritMC = np.zeros((IterationIndex,2))

for i in range(IterationIndex): 
    Arit, Geo = Asian_MC(S0, r, sigma, T, N, Iteration[i], K)
    AritMC[i,0] = np.mean(Arit)
    AritMC[i,1] = sem(Arit)
    GeoMC[i,0] = np.mean(Geo)
    GeoMC[i,1] = sem(Geo)
    Analytical_plot[i] = Analytical_value


fig, (plotP, plotQ) = plt.subplots(2, 1,sharex=True)
fig.suptitle('Asian Option Pricing with MC simulation vs Analytical solution')

plotP.plot(Iteration, GeoMC[:,0],label = "Geo_mean")
plotP.plot(Iteration, AritMC[:,0],label = "Arit_mean")
plotP.plot(Iteration,Analytical_plot,label = "Analytical")
plotP.set_xlabel("Number of path")
plotP.set_ylabel("Option price")
plotP.legend(loc="lower right")
plotQ.plot(Iteration, GeoMC[:,1],label = "Geo_std_error")
plotQ.plot(Iteration, AritMC[:,1],label = "Arit_std_error")
plotQ.set_xlabel("Number of path")
plotQ.set_ylabel("Standard error")
plotQ.legend()

plt.show()
