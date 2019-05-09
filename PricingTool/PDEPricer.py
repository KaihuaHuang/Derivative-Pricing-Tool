# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:17:57 2019

@author: William
"""

import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt

class PDEPricer():
    r = 0.025
    def __init__(self, S0, K, vol, d, T, pcFlag = 1, r = None):
        '''
        r is class attribute and set to be 2.5% by default
        '''
        self.S0 = S0
        self.K = K
        self.vol = vol
        self.d = d
        self.T = T
        self.pcFlag = pcFlag
        if r is not None:
            PDEPricer.r = r
    def vanillaPayoff(self, pcFlag = None, K = None):
        if K is None:
            K = self.K
        if pcFlag is None:
            pcFlag = self.pcFlag
        return lambda x: max(pcFlag*(x-self.K),0)
    
    def delta(self):
        Vm = self.Vmatrix
        if self.logSpace:
            return (Vm[int(self.stockGridNum/2)+1,0] - Vm[int(self.stockGridNum/2)-1,0])/(self.S0 * (np.exp(self.dx)-np.exp(-self.dx)))
        else:
            ds = self.ds
        return (Vm[int(self.stockGridNum/2)+1,0] - Vm[int(self.stockGridNum/2),0])/ds
    
    def gamma(self):
        du = self.delta()
        Vm = self.Vmatrix
        if self.logSpace:
            dsdn = self.S0*(1-np.exp(-self.dx))
            dsup = (np.exp(self.dx)-1) * self.S0
            deltaUp = (Vm[int(self.stockGridNum/2)+1,0] - Vm[int(self.stockGridNum/2),0])/dsup
            deltaDn = (Vm[int(self.stockGridNum/2),0] - Vm[int(self.stockGridNum/2)-1,0])/dsdn
            return (deltaUp-deltaDn)/((dsup + dsdn)/2)
        else:
            ds = self.ds
            dn = (Vm[int(self.stockGridNum/2),0] - Vm[int(self.stockGridNum/2)-1,0])/ds
            return (du-dn)/ds
        
        
        
    
    def vega(self,deltaVol = 0.001,Bermuda = False):
        price = self.Vmatrix[int(self.stockGridNum/2),0]
        initialVmatrix = self.Vmatrix
        self.vol = self.vol + deltaVol
        if(Bermuda):
            priceUp = self.BermudaVal(self.vanillaPayoff(), self.stockGridNum, self.timeGridNum, self.method, self.logSpace, self.frequency)
        else:
            priceUp = self.optionVal(self.vanillaPayoff(),self.stockGridNum,self.timeGridNum,self.method,self.logSpace,self.American)
        self.vol = self.vol - deltaVol
        self.Vmatrix = initialVmatrix
        return (priceUp-price)/deltaVol
    
    def theta(self):
        price = self.Vmatrix[int(self.stockGridNum/2),0]
        priceUp = self.Vmatrix[int(self.stockGridNum/2),1]
        return (priceUp - price)/self.dt
    
    def rho(self, deltaRf = 0.001,Bermuda = False):
        price = self.Vmatrix[int(self.stockGridNum/2),0]
        initialVmatrix = self.Vmatrix
        PDEPricer.r = PDEPricer.r + deltaRf
        if(Bermuda):
            priceUp = self.BermudaVal(self.vanillaPayoff(), self.stockGridNum, self.timeGridNum, self.method, self.logSpace, self.frequency)
        else:
            priceUp = self.optionVal(self.vanillaPayoff(),self.stockGridNum,self.timeGridNum,self.method,self.logSpace,self.American)
        PDEPricer.r = PDEPricer.r - deltaRf
        self.Vmatrix = initialVmatrix
        return (priceUp-price)/deltaRf
    
    def greeks(self,deltaVol =  0.001, deltaRf = 0.001,Bermuda = False):
        result = {}
        result['delta'] = self.delta()
        result['gamma'] = self.gamma()
        result['vega'] = self.vega(deltaVol,Bermuda)
        result['rho'] = self.rho(deltaRf,Bermuda)
        result['theta'] = self.theta()
        return result
    
    def optionVal(self, payoff, stockGridNum = 100, timeGridNum = 500, method = 'Explicit', logSpace = False, American = False):
        '''
        2*S0 should larger than K for call option
        '''
        self.stockGridNum = stockGridNum
        self.timeGridNum = timeGridNum
        self.method = method
        self.dt = self.T/timeGridNum
        self.Tau = np.arange(timeGridNum)*self.dt
        n = np.arange(stockGridNum+1)
        self.logSpace = logSpace
        self.payoff = payoff
        self.American = American
        if logSpace:
            mu = PDEPricer.r - self.d - 0.5*self.vol**2
            x_max = self.vol*np.sqrt(self.T)*5 
            self.dx = 2*x_max/stockGridNum
            self.X = np.linspace(-x_max,x_max,stockGridNum+1)
            V = np.array(list(map(payoff,np.exp(self.X)*self.S0)))
            VInitial = V 
            Vmatrix = np.zeros((stockGridNum+1,timeGridNum+1))
            Vmatrix[:,-1] = V
            if method == 'Crank-Nicolson':
                a = 0.25*(self.vol**2/self.dx**2 + mu/self.dx)*np.ones(stockGridNum+1)             
                b = (PDEPricer.r/2 - 1/self.dt + 0.25*2*self.vol**2/self.dx**2)*np.ones(stockGridNum+1)
                c = (0.25*mu/self.dx - 0.25*self.vol**2/self.dx**2)*np.ones(stockGridNum+1)
                d = (0.25*-2*self.vol**2/self.dx**2 - 1/self.dt-PDEPricer.r/2)*np.ones(stockGridNum+1)
                A = diags([c[1:],b,-a[:-1]],[-1,0,1]).toarray()
                B = diags([-c[1:],d,a[:-1]],[-1,0,1]).toarray()
                Binv = np.linalg.inv(B)
                for j in range(timeGridNum):
                    V = Binv.dot(A).dot(V)
                    if American:
                        index = V < VInitial
                        V[index] = VInitial[index]
                    V[0] = payoff(self.S0*np.exp(self.X[0]))
                    V[stockGridNum] = payoff(self.S0*self.X[-1])
                    Vmatrix[:,-2-j] = V
                self.Vmatrix = Vmatrix
                return V[int(stockGridNum/2)]
            
            elif method == 'Explicit':
                h = self.dt/self.dx
                k = h/self.dx
                C = (1-k*self.vol*self.vol-PDEPricer.r*self.dt)*np.eye(stockGridNum+1) + \
                    0.5*(k*self.vol*self.vol+h*mu)*np.eye(stockGridNum+1,k=1) + \
                    0.5*(k*self.vol*self.vol-h*mu)*np.eye(stockGridNum+1,k=-1)
                for j in range(timeGridNum):
                    V = C.dot(V)
                    if American:
                        index = V < VInitial
                        V[index] = VInitial[index]
                    V[0] = payoff(self.S0*np.exp(self.X[0]))
                    V[stockGridNum] = payoff(self.S0*self.X[-1])
                    Vmatrix[:,-2-j] = V
                self.Vmatrix = Vmatrix
                return V[int(stockGridNum/2)]
            
            elif method == 'Implicit':
                h = self.dt/self.dx
                k = h/self.dx
                D = (1+k*self.vol*self.vol+PDEPricer.r*self.dt)*np.eye(stockGridNum+1) - \
                    0.5*(k*self.vol*self.vol+h*mu)*np.eye(stockGridNum+1,k=1) - \
                    0.5*(k*self.vol*self.vol-h*mu)*np.eye(stockGridNum+1,k=-1)
                Dinv = np.linalg.inv(D)
                for j in range(timeGridNum):
                    V = Dinv.dot(V)
                    if American:
                        index = V < VInitial
                        V[index] = VInitial[index]
                    V[0] = payoff(self.S0*np.exp(self.X[0]))
                    V[stockGridNum] = payoff(self.S0*self.X[-1])
                    Vmatrix[:,-2-j] = V
                self.Vmatrix = Vmatrix
                return V[int(stockGridNum/2)]
            else:
                raise Exception('Avalible Method under Log Space: Explicit, Implicit, Crank-Nicolson')
                
                    
                
        else:
            s_max = 2*self.S0
            self.ds = s_max/stockGridNum
            self.S = np.linspace(0,s_max,stockGridNum+1)
            V = np.array(list(map(payoff, self.S)))
            VInitial = V 
            Vmatrix = np.zeros((stockGridNum+1,timeGridNum+1))
            Vmatrix[:,-1] = V
            if method == 'Explicit':
                pd = 0.5*self.dt*(self.vol*self.vol*n*n-(PDEPricer.r-self.d)*n)
                pu = 0.5*self.dt*(self.vol*self.vol*n*n+(PDEPricer.r-self.d)*n)
                pc = 1- self.dt*(self.vol*self.vol*n*n+PDEPricer.r)
    
                A = diags([pc,pu[:-1],pd[1:]],[0,1,-1]).toarray()
                print(A[:5,:5])
                for j in range(timeGridNum):
                    V = A.dot(V)
                    if American:
                        index = V < VInitial
                        V[index] = VInitial[index]
                    V[0]=payoff(self.S[0])
                    V[stockGridNum] = payoff(s_max*np.exp(self.T*(PDEPricer.r-self.d)))*np.exp(PDEPricer.r*self.dt*(timeGridNum-j-1))
                    Vmatrix[:,-2-j] = V
                self.Vmatrix = Vmatrix
                return V[int(stockGridNum/2)]
            
            elif method == 'Implicit':
                dp = -0.5*self.dt*(self.vol*self.vol*n*n-(PDEPricer.r-self.d)*n)
                up = -0.5*self.dt*(self.vol*self.vol*n*n+(PDEPricer.r-self.d)*n)
                cp = 1 + self.dt*(self.vol*self.vol*n*n+PDEPricer.r)
    
                B = diags([cp,up[:-1],dp[1:]],[0,1,-1]).toarray()
                Binv = np.linalg.inv(B)
                for j in range(timeGridNum):
                    V = Binv.dot(V)
                    if American:
                        index = V < VInitial
                        V[index] = VInitial[index]
                    V[0]=payoff(self.S[0])
                    V[stockGridNum] = payoff(s_max*np.exp(self.T*(PDEPricer.r-self.d)))*np.exp(PDEPricer.r*self.dt*(timeGridNum-j-1))
                    Vmatrix[:,-2-j] = V
                self.Vmatrix = Vmatrix
                return V[int(stockGridNum/2)]
            else:
                raise Exception('Avalible Method: Explicit, Implicit')
            
            
    def plot(self, interval = 50, stockCut = 4, logSpace = True):
        iterations = int(self.timeGridNum/interval) + 1
        cut = int(self.stockGridNum/stockCut)
        if logSpace:
            for i in range(iterations):
                if i == iterations-1:
                    plt.plot(self.S0*np.exp(self.X[cut:-cut]),self.Vmatrix[cut:-cut,0])
                else:
                    plt.plot(self.S0*np.exp(self.X[cut:-cut]),self.Vmatrix[cut:-cut,-i*interval-1])
                plt.title(f'Option Value when TTM = {round(min(i*interval*self.dt,self.T),1)}')
                plt.show()
        else:
            for i in range(iterations):
                if i == iterations-1:
                    plt.plot(self.S[cut:-cut],self.Vmatrix[cut:-cut,0])
                else:
                    plt.plot(self.S[cut:-cut],self.Vmatrix[cut:-cut,-i*interval-1])
                plt.title(f'Option Value when TTM = {round(min(i*interval*self.dt,self.T),1)}')
                plt.show()
    
    def BermudaVal(self, payoff, stockGridNum = 100, timeGridNum = 500, method = 'Crank-Nicolson', logSpace = True, frequency = 12):
        self.stockGridNum = stockGridNum
        self.timeGridNum = timeGridNum
        self.method = method
        self.dt = self.T/timeGridNum
        self.Tau = np.arange(timeGridNum)*self.dt
        self.frequency = frequency
        n = np.arange(stockGridNum+1)
        self.logSpace = logSpace
        self.payoff = payoff
        checkInterval = (np.arange(frequency) + 1) * self.T / frequency
        if logSpace:
            mu = PDEPricer.r - self.d - 0.5*self.vol**2
            x_max = self.vol*np.sqrt(self.T)*5 
            self.dx = 2*x_max/stockGridNum
            self.X = np.linspace(-x_max,x_max,stockGridNum+1)
            V = np.array(list(map(payoff,np.exp(self.X)*self.S0)))
            VInitial = V 
            Vmatrix = np.zeros((stockGridNum+1,timeGridNum+1))
            Vmatrix[:,-1] = V
            if method == 'Crank-Nicolson':
                a = 0.25*(self.vol**2/self.dx**2 + mu/self.dx)*np.ones(stockGridNum+1)             
                b = (PDEPricer.r/2 - 1/self.dt + 0.25*2*self.vol**2/self.dx**2)*np.ones(stockGridNum+1)
                c = (0.25*mu/self.dx - 0.25*self.vol**2/self.dx**2)*np.ones(stockGridNum+1)
                d = (0.25*-2*self.vol**2/self.dx**2 - 1/self.dt-PDEPricer.r/2)*np.ones(stockGridNum+1)
                A = diags([c[1:],b,-a[:-1]],[-1,0,1]).toarray()
                B = diags([-c[1:],d,a[:-1]],[-1,0,1]).toarray()
                Binv = np.linalg.inv(B)
                for j in range(timeGridNum):
                    V = Binv.dot(A).dot(V)
                    if (self.timeGridNum - j -1)*self.dt in checkInterval:
                        index = V < VInitial
                        V[index] = VInitial[index]
                    V[0] = payoff(self.S0*np.exp(self.X[0]))
                    V[stockGridNum] = payoff(self.S0*self.X[-1])
                    Vmatrix[:,-2-j] = V
                self.Vmatrix = Vmatrix
                return V[int(stockGridNum/2)]
            
'''
S0 = 100
K = 100
vol = 0.4
T = 1
r = 0.025
d = 0.0175

# (S0, K, T, vol, r, d) = (10, 10, 1, 0.3, 0., 0.2)
PDE = PDEPricer(S0, K, vol, d, T, 1, r)
print('Price:',PDE.BermudaVal(PDE.vanillaPayoff(),500,480,method = 'Crank-Nicolson',logSpace = True, frequency = 12))
print('Delta:',PDE.delta())
print('Gamma:',PDE.gamma())
print('Theta:',PDE.theta())
print(PDE.greeks(Bermuda = True))
#print(PDE.plot())
'''
S0 = 100
K = 110
B = 200
vol = 0.5
T = 1
r = 0.025
d = 0.0175
#(S0, K, T, vol, r, d) = (100, 100, .5, 0.3, 0., 0.)
PDE = PDEPricer(S0, K, vol, d, T, 1, r)
print('Price:',PDE.optionVal(PDE.vanillaPayoff(),200,500,method = 'Implicit',logSpace = True))