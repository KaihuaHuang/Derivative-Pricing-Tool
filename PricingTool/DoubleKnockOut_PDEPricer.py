# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 20:05:41 2019

@author: kaihu
"""

import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt

class DoubleKnockOut_PDEPricer():
    r = 0.025
    def __init__(self, S0, K, vol, d, T, upperBarrier, downBarrier, pcFlag = 1, r = None):
        '''
        r is class attribute and set to be 2.5% by default
        '''
        self.S0 = S0
        self.K = K
        self.vol = vol
        self.d = d
        self.T = T
        self.ub = upperBarrier
        self.db = downBarrier
        self.pcFlag = pcFlag
        if r is not None:
            DoubleKnockOut_PDEPricer.r = r
    
    def vanillaPayoff(self, pcFlag = None, K = None):
        if K is None:
            K = self.K
        if pcFlag is None:
            pcFlag = self.pcFlag
        return lambda x: max(pcFlag*(x-self.K),0)
    
    def optionVal(self, payoff, stockGridNum = 100, timeGridNum = 500, method = 'Explicit', logSpace = False, American = False):
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
            mu = DoubleKnockOut_PDEPricer.r - self.d - 0.5*self.vol**2
            x_max = self.vol*np.sqrt(self.T)*5 
            self.dx = 2*x_max/stockGridNum
            self.X = np.linspace(-x_max,x_max,stockGridNum+1)
            V = np.array(list(map(payoff,np.exp(self.X)*self.S0)))
            V[np.log(self.ub/self.S0) < self.X] = 0
            V[np.log(self.db/self.S0) > self.X] = 0
            VInitial = V 
            Vmatrix = np.zeros((stockGridNum+1,timeGridNum+1))
            Vmatrix[:,-1] = V
            if method == 'Crank-Nicolson':
                a = 0.25*(self.vol**2/self.dx**2 + mu/self.dx)*np.ones(stockGridNum+1)             
                b = (DoubleKnockOut_PDEPricer.r/2 - 1/self.dt + 0.25*2*self.vol**2/self.dx**2)*np.ones(stockGridNum+1)
                c = (0.25*mu/self.dx - 0.25*self.vol**2/self.dx**2)*np.ones(stockGridNum+1)
                d = (0.25*-2*self.vol**2/self.dx**2 - 1/self.dt-DoubleKnockOut_PDEPricer.r/2)*np.ones(stockGridNum+1)
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
                    V[np.log(self.ub/self.S0) < self.X] = 0
                    V[np.log(self.db/self.S0) > self.X] = 0
                    Vmatrix[:,-2-j] = V
                self.Vmatrix = Vmatrix
                return V[int(stockGridNum/2)]
            
            elif method == 'Explicit':
                h = self.dt/self.dx
                k = h/self.dx
                C = (1-k*self.vol*self.vol-DoubleKnockOut_PDEPricer.r*self.dt)*np.eye(stockGridNum+1) + \
                    0.5*(k*self.vol*self.vol+h*mu)*np.eye(stockGridNum+1,k=1) + \
                    0.5*(k*self.vol*self.vol-h*mu)*np.eye(stockGridNum+1,k=-1)
                for j in range(timeGridNum):
                    V = C.dot(V)
                    if American:
                        index = V < VInitial
                        V[index] = VInitial[index]
                    V[0] = payoff(self.S0*np.exp(self.X[0]))
                    V[stockGridNum] = payoff(self.S0*self.X[-1])
                    V[np.log(self.ub/self.S0) < self.X] = 0
                    V[np.log(self.db/self.S0) > self.X] = 0
                    Vmatrix[:,-2-j] = V
                self.Vmatrix = Vmatrix
                return V[int(stockGridNum/2)]
            
            elif method == 'Implicit':
                h = self.dt/self.dx
                k = h/self.dx
                D = (1+k*self.vol*self.vol+DoubleKnockOut_PDEPricer.r*self.dt)*np.eye(stockGridNum+1) - \
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
                    V[np.log(self.ub/self.S0) < self.X] = 0
                    V[np.log(self.db/self.S0) > self.X] = 0
                    Vmatrix[:,-2-j] = V
                self.Vmatrix = Vmatrix
                return V[int(stockGridNum/2)]
            else:
                raise Exception('Avalible Method under Log Space: Explicit, Implicit, Crank-Nicolson')
        else:
            s_max = 2*self.S0
            self.ds = s_max/stockGridNum
            self.S = np.linspace(0,s_max,stockGridNum+1)
            self.upBarrierIndex = int(self.ub/self.ds)
            self.dnBarrierIndex = int(self.db/self.ds)
            V = np.array(list(map(payoff, self.S)))
            VInitial = V 
            Vmatrix = np.zeros((stockGridNum+1,timeGridNum+1))
            Vmatrix[:,-1] = V
            if method == 'Explicit':
                pd = 0.5*self.dt*(self.vol*self.vol*n*n-(DoubleKnockOut_PDEPricer.r-self.d)*n)
                pu = 0.5*self.dt*(self.vol*self.vol*n*n+(DoubleKnockOut_PDEPricer.r-self.d)*n)
                pc = 1- self.dt*(self.vol*self.vol*n*n+DoubleKnockOut_PDEPricer.r)
            
                A = diags([pc,pu[:-1],pd[1:]],[0,1,-1]).toarray()
                for j in range(timeGridNum):
                    V = A.dot(V)
                    if American:
                        index = V < VInitial
                        V[index] = VInitial[index]
                    V[self.upBarrierIndex:] = 0
                    V[:self.dnBarrierIndex] = 0
                    V[0]=payoff(self.S[0])
                    V[stockGridNum] = payoff(s_max*np.exp(self.T*(DoubleKnockOut_PDEPricer.r-self.d)))*np.exp(DoubleKnockOut_PDEPricer.r*self.dt*(timeGridNum-j-1))
                    Vmatrix[:,-2-j] = V
                self.Vmatrix = Vmatrix
                return V[int(stockGridNum/2)]
            
            elif method == 'Implicit':
                dp = -0.5*self.dt*(self.vol*self.vol*n*n-(DoubleKnockOut_PDEPricer.r-self.d)*n)
                up = -0.5*self.dt*(self.vol*self.vol*n*n+(DoubleKnockOut_PDEPricer.r-self.d)*n)
                cp = 1 + self.dt*(self.vol*self.vol*n*n+DoubleKnockOut_PDEPricer.r)
        
                B = diags([cp,up[:-1],dp[1:]],[0,1,-1]).toarray()
                Binv = np.linalg.inv(B)
                for j in range(timeGridNum):
                    V = Binv.dot(V)
                    if American:
                        index = V < VInitial
                        V[index] = VInitial[index]
                    V[self.upBarrierIndex:] = 0
                    V[:self.dnBarrierIndex] = 0
                    V[0]=payoff(self.S[0])
                    V[stockGridNum] = payoff(s_max*np.exp(self.T*(DoubleKnockOut_PDEPricer.r-self.d)))*np.exp(DoubleKnockOut_PDEPricer.r*self.dt*(timeGridNum-j-1))
                    Vmatrix[:,-2-j] = V
                self.Vmatrix = Vmatrix
                return V[int(stockGridNum/2)]
            else:
                raise Exception('Avalible Method: Explicit, Implicit')
    
    def delta(self, index = None):
        if index is None:
            index = int(self.stockGridNum/2)
        Vm = self.Vmatrix
        if self.logSpace:
            return (Vm[index+1,0] - Vm[index-1,0])/(self.S0 * (np.exp(self.dx)-np.exp(-self.dx)))
        else:
            ds = self.ds
        return (Vm[index+1,0] - Vm[index,0])/ds
    
    def gamma(self, index = None):
        if index is None:
            index = int(self.stockGridNum/2)
        du = self.delta()
        Vm = self.Vmatrix
        if self.logSpace:
            dsdn = self.S0*(1-np.exp(-self.dx))
            dsup = (np.exp(self.dx)-1) * self.S0
            deltaUp = (Vm[index+1,0] - Vm[index,0])/dsup
            deltaDn = (Vm[index,0] - Vm[index-1,0])/dsdn
            return (deltaUp-deltaDn)/((dsup + dsdn)/2)
        else:
            ds = self.ds
            dn = (Vm[index,0] - Vm[index-1,0])/ds
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
        DoubleKnockOut_PDEPricer.r = DoubleKnockOut_PDEPricer.r + deltaRf
        if(Bermuda):
            priceUp = self.BermudaVal(self.vanillaPayoff(), self.stockGridNum, self.timeGridNum, self.method, self.logSpace, self.frequency)
        else: 
            priceUp = self.optionVal(self.vanillaPayoff(),self.stockGridNum,self.timeGridNum,self.method,self.logSpace,self.American)
        DoubleKnockOut_PDEPricer.r = DoubleKnockOut_PDEPricer.r - deltaRf
        self.Vmatrix = initialVmatrix
        return (priceUp-price)/deltaRf
    
    def gammaPlot(self,stockCut = 4, logSpace = True):
        cut = int(self.stockGridNum/stockCut)
        self.gammaArray = []
        for i in range(cut,self.stockGridNum-cut + 1):
                self.gammaArray.append(self.gamma(i))
        if logSpace:
            plt.plot(self.S0*np.exp(self.X[cut:-cut]),self.gammaArray)
            plt.title(f'Gamma vs Stock Price')
            plt.show()
    
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
    
    
    def deltaPlot(self,stockCut = 4, logSpace = True):
        cut = int(self.stockGridNum/stockCut)
        self.deltaArray = []
        for i in range(cut,self.stockGridNum-cut + 1):
                self.deltaArray.append(self.delta(i))
        if logSpace:
            plt.plot(self.S0*np.exp(self.X[cut:-cut]),self.deltaArray)
            plt.title(f'Delta vs Stock Price')
            plt.show()
        
    
        
        
        
        
        


S0 = 100
K = 90
uB = 120
dB = 80
vol = 0.1
T = 1
r = 0.025
d = 0.0175

PDE = DoubleKnockOut_PDEPricer(S0, K, vol, d, T, uB, dB, -1, r)
print('price:',PDE.optionVal(PDE.vanillaPayoff(),200,500,method = 'Crank-Nicolson',logSpace = True))
print('Delta:',PDE.delta())
print('Gamma:',PDE.gamma())
print('Theta:',PDE.theta())
PDE.plot(interval = PDE.timeGridNum)
PDE.deltaPlot()
PDE.gammaPlot()

