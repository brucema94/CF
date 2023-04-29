import numpy as np
import numpy as np
from math import log, sqrt
from scipy.stats import gmean
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import sem

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


def Control_variate(S0, r, sigma, T, N, M, K):
    Analytical_value = Asian_analytical(S0, r, sigma, T, N, K)
    Arit, Geo = Asian_MC(S0, r, sigma, T, N, M, K)

    beta = np.cov(Arit,Geo)[0,1]/np.var(Geo)
    CV_estimate = Arit - beta * (Geo - Analytical_value)

    return CV_estimate

S0 = 100
T = 1
N = 50
K = 99
r = 0.06
sigma = 0.2
M = 5000

IterationIndex = 100
Analytical_plot = np.zeros(IterationIndex)
pointgrid = np.linspace(5,100,IterationIndex,dtype=int)
GeoMC = np.zeros((IterationIndex,2))
AritMC = np.zeros((IterationIndex,2))
CV_estimate = np.zeros((IterationIndex,2))

for i in range(IterationIndex): 
    Analytical_value = Asian_analytical(S0, r, sigma, T, pointgrid[i], K)
    Arit, Geo = Asian_MC(S0, r, sigma, T, pointgrid[i], M, K)
    CV = Control_variate(S0, r, sigma, T, pointgrid[i], M, K)
    CV_estimate[i,0] = np.mean(CV)
    CV_estimate[i,1] = sem(CV)
    AritMC[i,0] = np.mean(Arit)
    AritMC[i,1] = sem(Arit)
    GeoMC[i,0] = np.mean(Geo)
    GeoMC[i,1] = sem(Geo)
    Analytical_plot[i] = Analytical_value

fig, (plotP, plotQ) = plt.subplots(2, 1,sharex=True)
fig.suptitle('Asian Option Pricing with MC simulation vs Analytical solution')

plotP.plot(pointgrid, AritMC[:,0],label = "Arit_mean")
plotP.plot(pointgrid, CV_estimate[:,0],label = "Control Variate")
plotP.set_xlabel("Number of path")
plotP.set_ylabel("Option price")
plotP.legend(loc="upper right")
plotQ.plot(pointgrid, CV_estimate[:,1],label = "Control_Variate_error")
plotQ.set_xlabel("Number of path")
plotQ.set_ylabel("Standard error")
plotQ.legend()

plt.show()