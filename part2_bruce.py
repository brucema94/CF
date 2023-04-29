import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt

class CrankNicolsonBS:
    def __init__(self, AmEuflag, cpflag, S0, K, T, vol, r, d):
        self.AmEuflag = AmEuflag
        self.cpflag = cpflag
        self.S0 = S0
        self.K = K
        self.T = T
        self.vol = vol
        self.r = r
        self.d = d
        
    def CN_option_info(self, AmEuflag, cpflag, S0, K, T, vol, r, d):
        mu = (r-d-0.5*vol*vol)
        
        # range x in [-x_max, x_max]
        x_max = vol*np.sqrt(T)*5
        
        #number of steps along x
        N = 500
        dx = 2*x_max/N

        # grid along x dimension:
        X = np.linspace(-x_max,x_max,N+1)
        n = np.arange(0,N+1)

        #number of time steps
        J = 500
        dt = T/J
        
        #grid along time dimension:
        Tau = np.arange(J)*dt
        
        # set up the matrix
        a = 0.25*dt*vol*vol/(dx*dx)
        b = 0.25*dt*mu/dx
        c = 0.5*dt*r
        
        A = (1+c+2*a)*np.eye(N+1) + (-a-b)*np.eye(N+1,k=1) + (b-a)*np.eye(N+1,k=-1)
        B = (1-c-2*a)*np.eye(N+1) + (a+b)*np.eye(N+1,k=1) + (a-b)*np.eye(N+1,k=-1)
        Ainv = np.linalg.inv(A)
        
        if cpflag == 'c':
            # Option payoff at maturity
            V = np.clip(S0*np.exp(X)-K,0,1e10)
        elif cpflag == 'p':
            V = np.clip(K-S0*np.exp(X),0,1e10)
        
        cut = int(N/4)

        if AmEuflag == 'Am':
            V0 = V.copy()
            for j in range(J):
                if j == J-1:
                    V1 = V
                V = B.dot(V)
                V = Ainv.dot(V)
                # apply early exercise boundary conditions:
                V = np.where(V>V0, V, V0)
                if j%50==0: plt.plot(S0*np.exp(X[cut:-cut]), V[cut:-cut]) 
                    
        elif AmEuflag == 'Eu':
            for j in range(J):
                if j == J-1:
                    V1 = V
                V = B.dot(V)
                V = Ainv.dot(V)
                V[0] = 0
                V[N] = S0*np.exp(x_max) - K*np.exp(-r*j*dt)
                if j%50==0: plt.plot(S0*np.exp(X[cut:-cut]), V[cut:-cut])
                    
        #mid grid point:
        n_mid = int(N/2)
        price = V[n_mid]
        delta = (V[n_mid+1]-V[n_mid-1])/(S0*np.exp(dx)-S0*np.exp(-dx))
        gamma = ((V[n_mid+1]-V[n_mid])/(S0*(np.exp(dx)-1))-(V[n_mid]-V[n_mid-1])/(S0*(1-np.exp(-dx))))/(S0*np.exp(dx)-S0*np.exp(-dx))*2
        theta = -(V[n_mid]-V1[n_mid])/(dt)
        
        return price, delta, gamma, theta
    

Eu_call = CrankNicolsonBS('Eu', 'c', 100, 110, 1, 0.3, 0.04, 0)
Eu_call_info = Eu_call.CN_option_info('Eu', 'c', 100, 110, 1, 0.3, 0.04, 0)
print('European call\nCrank-Nicolson\nPrice:',Eu_call_info[0],'\nDelta',Eu_call_info[1],'\nGamma:',Eu_call_info[2],'\nTheta:',Eu_call_info[3],'\n')