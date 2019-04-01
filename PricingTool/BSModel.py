# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:49:54 2019

@author: William
"""
from math import log, sqrt, exp, pi
from scipy.stats import norm

class BlackScholes:
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
            BlackScholes.r = r
    
    def bsValue(self, vol = None):
        if vol is None:
            self.d1 = (log(self.S0 / self.K) + (BlackScholes.r - self.d + 0.5 * self.vol**2)*self.T)/(self.vol * sqrt(self.T))
            self.d2 = self.d1 - self.vol * sqrt(self.T)
        else:
            self.d1 = (log(self.S0 / self.K) + (BlackScholes.r - self.d + 0.5 * vol**2)*self.T)/(vol * sqrt(self.T))
            self.d2 = self.d1 - vol * sqrt(self.T)
        
        
        return self.pcFlag * self.S0 * exp(-self.T * self.d) * norm.cdf(self.d1 * self.pcFlag) - self.pcFlag * self.K * exp(-BlackScholes.r*self.T) * norm.cdf(self.d2 * self.pcFlag)
   
    def delta(self, vol = None):
        if vol is not None:
            d1 = (log(self.S0 / self.K) + (BlackScholes.r - self.d + 0.5 * vol**2)*self.T)/(vol * sqrt(self.T))
            return norm.cdf(d1)
        else:
            if not hasattr(self,'d1'):
                self.d1 = (log(self.S0 / self.K) + (BlackScholes.r - self.d + 0.5 * self.vol**2)*self.T)/(self.vol * sqrt(self.T))
            return self.pcFlag * norm.cdf(self.pcFlag * self.d1) * exp(-self.T * self.d)
        
    def gamma(self, vol = None):
        if vol is not None:
            d1 = (log(self.S0 / self.K) + (BlackScholes.r - self.d + 0.5 * vol**2)*self.T)/(vol * sqrt(self.T))
            return exp(-self.T * self.d)/(self.S0*vol*sqrt(self.T)) * 1 / sqrt(2*pi) * exp(-d1**2/2)
        else:
            if not hasattr(self,'d1'):
                self.d1 = (log(self.S0 / self.K) + (BlackScholes.r - self.d + 0.5 * self.vol**2)*self.T)/(self.vol * sqrt(self.T))
            return exp(-self.T * self.d)/(self.S0*self.vol*sqrt(self.T)) * 1 / sqrt(2*pi) * exp(-self.d1**2/2)
    
    def vega(self, vol = None):
        if vol is not None:
            d1 = (log(self.S0 / self.K) + (BlackScholes.r - self.d + 0.5 * vol**2)*self.T)/(vol * sqrt(self.T))
            return self.S0*sqrt(self.T)*norm.pdf(d1)*exp(-self.T * self.d)
        else:
            if not hasattr(self,'d1'):
                self.d1 = (log(self.S0 / self.K) + (BlackScholes.r - self.d + 0.5 * self.vol**2)*self.T)/(self.vol * sqrt(self.T))
            #return self.K * exp(-BlackSchole.r*self.T)*norm.pdf(self.d2)*sqrt(self.T)
            return self.S0*sqrt(self.T)*norm.pdf(self.d1)*exp(-self.T * self.d)

    def theta(self,vol = None):
        '''
        Return theta per year
        '''
        if vol is not None:
            d1 = (log(self.S0 / self.K) + (BlackScholes.r - self.d + 0.5 * vol**2)*self.T)/(vol * sqrt(self.T))
            d2 = d1 - vol * sqrt(self.T)
            return -exp(-self.d*self.T)*self.S0*norm.pdf(d1)*vol/(2*sqrt(self.T))-\
                    self.pcFlag*BlackScholes.r*self.K*exp(-BlackScholes.r*self.T)*norm.cdf(self.pcFlag*d2) + \
                    self.pcFlag*self.d*self.S0*exp(-self.d*self.T)*norm.cdf(self.pcFlag*d1)
        else:
            if not hasattr(self,'d1'):
                self.d1 = (log(self.S0 / self.K) + (BlackScholes.r - self.d + 0.5 * self.vol**2)*self.T)/(self.vol * sqrt(self.T))
            if not hasattr(self,'d2'):
                self.d2 = (log(self.S0 / self.K) + (BlackScholes.r - self.d + 0.5 * self.vol**2)*self.T)/(self.vol * sqrt(self.T)) - self.vol * sqrt(self.T)
            return -exp(-self.d*self.T)*self.S0*norm.pdf(self.d1)*self.vol/(2*sqrt(self.T))-\
                    self.pcFlag*BlackScholes.r*self.K*exp(-BlackScholes.r*self.T)*norm.cdf(self.pcFlag*self.d2) + \
                    self.pcFlag*self.d*self.S0*exp(-self.d*self.T)*norm.cdf(self.pcFlag*self.d1)
    def rho(self, vol = None):
        if vol is not None:
            d2 = (log(self.S0 / self.K) + (BlackScholes.r - self.d + 0.5 * vol**2)*self.T)/(vol * sqrt(self.T)) - vol * sqrt(self.T)
            return self.K*self.T*exp(-BlackScholes.r*self.T)*norm.cdf(d2)
        else:
            if not hasattr(self,'d2'):
                self.d2 = (log(self.S0 / self.K) + (BlackScholes.r - self.d + 0.5 * self.vol**2)*self.T)/(self.vol * sqrt(self.T)) - self.vol * sqrt(self.T)
            return self.pcFlag*self.K*self.T*exp(-BlackScholes.r*self.T)*norm.cdf(self.pcFlag*self.d2)
    
    def greeks(self,vol = None):
        result = {}
        result['delta'] = self.delta(vol)
        result['gamma'] = self.gamma(vol)
        result['vega'] = self.vega(vol)
        result['rho'] = self.rho(vol)
        result['theta'] = self.theta(vol)
        return result
     
'''
S0 = 110
K = 100
vol = 0.3
T = 1
r = 0.05
d = 0.0
(S0, K, T, vol, r, d) = (10, 10, 1, 0.3, 0., 0.2)
option = BlackSchole(S0, K, vol, d, T, 1, r)
print('BSprice:',option.bsValue())

print('delta:',option.delta())
print('gamma:',option.gamma())
print('vega:',option.vega())
print('theta:',option.theta())
print('rho:',option.rho())
print(option.greeks())
'''