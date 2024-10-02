# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:20:14 2024

@author: rafae
"""
import numpy as np
from scipy.integrate import quad, dblquad, tplquad
from scipy.optimize import minimize
from scipy.stats import binom
import time
import pandas as pd

# Funções Weibull
def fx(x):
    return (b1 / a1**b1) * (x**(b1 - 1)) * np.exp(-(x / a1)**b1)

def fy(y):
    return (b2 / a2**b2) * (y**(b2 - 1)) * np.exp(-(y / a2)**b2)

def fh(h):
    return (b3 / a3**b3) * (h**(b3 - 1)) * np.exp(-(h / a3)**b3)

def Rx(x):
    return 1 - (1 - np.exp(-(x / a1)**b1))

def Ry(y):
    return 1 - (1 - np.exp(-(y / a2)**b2))


def CompExpCost(i):
    Sum=0
    for j in range(1,i,1):
        Sum+=binom.pmf(j, i-1, (1-p))*j*Ci
    return Sum

# Cenários
def P1(K, T):
    prob, length, cost = 0, 0, 0
    for i in range(1, K+1):
        Pi = ((1-q)**(i-1)) * tplquad(lambda h, y, x: fx(x) * fy(y) * fh(h), (i-1)*T, i*T, 0, lambda x: (i*T) - x, 0, lambda x, y: (i*T) - x - y)[0]
        prob += Pi
        length += ((1-q)**(i-1)) * tplquad(lambda h, y, x: (x + y + h) * fx(x) * fy(y) * fh(h), (i-1)*T, i*T, 0, lambda x: (i*T) - x, 0, lambda x, y: (i*T) - x - y)[0]
        cost += Pi * (Cf + CompExpCost(i))+((1-q)**(i-1)) * tplquad(lambda h, y, x: (Cmd*h)*fx(x) * fy(y) * fh(h), (i-1)*T, i*T, 0, lambda x: (i*T) - x, 0, lambda x, y: (i*T) - x - y)[0]
    return prob, length, cost

# Outros cenários...
def P2(K, T):
    prob, length, cost = 0, 0, 0
    for i in range(1, K):
        pi = ((1-q)**(i-1)) * q * Rx(i*T) * dblquad(lambda h, y: fy(y) * fh(h), 0, T, 0, lambda y: T - y)[0]
        prob += pi
        length += ((1-q)**(i-1)) * q * Rx(i*T) * dblquad(lambda h, y: (i*T + y + h) * fy(y) * fh(h), 0, T, 0, lambda y: T - y)[0]
        cost += pi * (Cf + Ci + CompExpCost(i))+((1-q)**(i-1)) * q * Rx(i*T) * dblquad(lambda h, y: (Cmd*h)*fy(y) * fh(h), 0, T, 0, lambda y: T - y)[0]
    return prob, length, cost

def P3(K, T):
    prob, length, cost = 0, 0, 0
    for i in range(1, K):
        for j in range(i+1, K+1):
            pi = ((1-q)**(i-1)) * ((p + (1-p) * beta)**(j-i)) * tplquad(lambda h, y, x: fx(x) * fy(y) * fh(h), (i-1)*T, i*T, lambda x: ((j-1)*T) - x, lambda x: (j*T) - x, 0, lambda x, y: (j*T) - x - y)[0]
            prob += pi
            length += ((1-q)**(i-1)) * ((p + (1-p) * beta)**(j-i)) * tplquad(lambda h, y, x: (x + y + h) * fx(x) * fy(y) * fh(h), (i-1)*T, i*T, lambda x: ((j-1)*T) - x, lambda x: (j*T) - x, 0, lambda x, y: (j*T) - x - y)[0]
            cost += pi * (Cf + CompExpCost(j)) + ((1-q)**(i-1)) * ((p + (1-p) * beta)**(j-i)) * tplquad(lambda h, y, x: (Cmd*h)*fx(x) * fy(y) * fh(h), (i-1)*T, i*T, lambda x: ((j-1)*T) - x, lambda x: (j*T) - x, 0, lambda x, y: (j*T) - x - y)[0]
    return prob, length, cost

def P4(K,T):
    prob, length, cost = 0, 0, 0
    for i in range (1,K-1):
        for j in range (i+2,K+1):
            pi = ((q*(1-q)**(i-1))*((p+(1-p)*beta)**(j-i-1))*Rx(i*T)*dblquad(lambda h, y: fy(y)*fh(h), ((j-1)-i)*T, (j-i)*T, 0, lambda y: j*T-i*T-y)[0]) 
            prob += pi
            length += ((q*(1-q)**(i-1))*((p+(1-p)*beta)**(j-i-1))*Rx(i*T)*dblquad(lambda h, y: (i*T+y+h)*fy(y)*fh(h), ((j-1)-i)*T, (j-i)*T, 0, lambda y: j*T-i*T-y)[0]) 
            cost += pi *(Cf + Ci + CompExpCost(j-1)) + ((q*(1-q)**(i-1))*((p+(1-p)*beta)**(j-i-1))*Rx(i*T)*dblquad(lambda h, y: (Cmd*h)*fy(y)*fh(h), ((j-1)-i)*T, (j-i)*T, 0, lambda y: j*T-i*T-y)[0]) 
    return prob, length, cost


#CENÁRIO 5
def P5(K,T):
    prob, length, cost = 0, 0, 0
    for i in range (1,K):
        for j in range (i,K):
            pi = ((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(1-beta)*(1-p)*(dblquad(lambda y, x: fx(x)*fy(y), (i-1)*T, i*T, lambda x: (j*T)-x, np.inf))[0]
            prob += pi
            length += ((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(1-beta)*(1-p)*(dblquad(lambda y, x: j*T*fx(x)*fy(y), (i-1)*T, i*T, lambda x: (j*T)-x, np.inf))[0]
            cost += pi*(Cr + Ci + CompExpCost(j))
            #resultado_P5+= ((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(1-beta)*Ry((j*T)-x)*quad(lambda x: fx(x), (i-1)*T, i*T)[0]
    return prob, length, cost

#CENÁRIO 6
def P6(K,T):
    prob, length, cost = 0, 0, 0
    for i in range (1,K-1):
        for j in range (i+1,K):
            pi = (q*((1-q)**(i-1))*(((p+(1-p)*beta)**(j-i-1))*(1-beta)*(1-p)*Rx(i*T))*quad(lambda y: fy(y), (j-i)*T, np.inf)[0]) 
            prob += pi
            length += (q*((1-q)**(i-1))*(((p+(1-p)*beta)**(j-i-1))*(1-beta)*(1-p)*Rx(i*T))*quad(lambda y: j*T*fy(y), (j-i)*T, np.inf)[0]) 
            cost += pi * (Cr + 2*Ci + CompExpCost(j-1))
    return prob, length, cost

#CENÁRIO 7
def P7(K,T):
    prob,  length, cost = 0, 0, 0
    for i in range (1,K):
        for j in range (i,K):
            pi = (((1-q)**(i-1))*((p)**(j-i))*(1-p)*(tplquad(lambda h, y, x: fx(x)*fy(y)*fh(h), (i-1)*T, i*T, 0, lambda x: i*T-x, lambda x, y: (((j)*T)-x-y), np.inf))[0])
            prob += pi
            length += (((1-q)**(i-1))*((p)**(j-i))*(1-p)*(tplquad(lambda h, y, x: j*T*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, 0, lambda x: i*T-x, lambda x, y: (((j)*T)-x-y), np.inf))[0])
            cost += pi * (Cr + Ci + CompExpCost(j)) + (((1-q)**(i-1))*((p)**(j-i))*(1-p)*(tplquad(lambda h, y, x: (Cmd*(i*T-(x+y)))*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, 0, lambda x: i*T-x, lambda x, y: (((j)*T)-x-y), np.inf))[0])
    return prob, length, cost
 

#CENÁRIO 8
def P8(K,T):
   prob, length, cost = 0, 0, 0
   for i in range (1,K-1):
       for j in range (i+1,K):
           pi = ((((1-q)**(i-1))*q*(p**(j-i-1))*(1-p)*Rx(i*T))*dblquad(lambda h, y: fy(y)*fh(h), 0, T, lambda y: j*T-y-i*T, np.inf)[0]) 
           prob += pi
           length += ((((1-q)**(i-1))*q*(p**(j-i-1))*(1-p)*Rx(i*T))*dblquad(lambda h, y: j*T*fy(y)*fh(h), 0, T, lambda y: j*T-y-i*T, np.inf)[0]) 
           cost += pi * ((Cr + 2*Ci + CompExpCost(j-1))) + ((((1-q)**(i-1))*q*(p**(j-i-1))*(1-p)*Rx(i*T))*dblquad(lambda h, y: (Cmd*(j*T-(i*T+y)))*fy(y)*fh(h), 0, T, lambda y: j*T-y-i*T, np.inf)[0]) 
   return prob , length , cost
   
#CENÁRIO 9
def P9(K,T):
    prob, length, cost = 0, 0, 0
    for i in range (1,K-1):
        for j in range (i+1,K):
            for l in range (j,K):
                pi = ((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(p**(l-j))*(1-p)*(tplquad(lambda h, y, x: fx(x)*fy(y)*fh(h), (i-1)*T, i*T, lambda x: ((j-1)*T)-x, lambda x: (j*T)-x, lambda x, y: ((l*T)-x-y), np.inf))[0]
                prob += pi
                length += ((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(p**(l-j))*(1-p)*(tplquad(lambda h, y, x: l*T*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, lambda x: ((j-1)*T)-x, lambda x: (j*T)-x, lambda x, y: ((l*T)-x-y), np.inf))[0]
                cost += pi * (Cr + Ci + CompExpCost(l)) + ((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(p**(l-j))*(1-p)*(tplquad(lambda h, y, x: (Cmd*(l*T-(x+y)))*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, lambda x: ((j-1)*T)-x, lambda x: (j*T)-x, lambda x, y: ((l*T)-x-y), np.inf))[0]
    return prob, length, cost

#CENÁRIO 10
def P10(K,T):
    prob, length, cost = 0, 0, 0
    for i in range (1,K-2):
        for j in range (i+2,K):
            for l in range (j,K):
                pi = ((q*(1-q)**(i-1))*(((p+(1-p)*beta)**(j-i-1))*(p**(l-j))*(1-p)*Rx((i)*T))*dblquad(lambda h, y: fy(y)*fh(h), (j-1)*T-i*T, j*T - (i)*T, lambda y: (l*T)-y-(i)*T, np.inf) [0]) 
                prob += pi
                length += ((q*(1-q)**(i-1))*(((p+(1-p)*beta)**(j-i-1))*(p**(l-j))*(1-p)*Rx((i)*T))*dblquad(lambda h, y: l*T*fy(y)*fh(h), (j-1)*T-i*T, j*T - (i)*T, lambda y: (l*T)-y-(i)*T, np.inf) [0]) 
                cost += pi * ((Cr + 2*Ci + CompExpCost(l-1))) + ((q*(1-q)**(i-1))*(((p+(1-p)*beta)**(j-i-1))*(p**(l-j))*(1-p)*Rx((i)*T))*dblquad(lambda h, y: (Cmd*(l*T-(i*T+y)))*fy(y)*fh(h), (j-1)*T-i*T, j*T - (i)*T, lambda y: (l*T)-y-(i)*T, np.inf) [0]) 
    return prob, length, cost

def P11(K,T):
    prob, length, cost = 0, 0, 0
    for i in range (1,K+1):
        pi = ((1-q)**(i-1))*((p+(1-p)*beta)**(K-i))*(dblquad(lambda y, x: fx(x)*fy(y), (i-1)*T, i*T, lambda x: (K*T)-x, np.inf))[0]
        prob += pi
        length += ((1-q)**(i-1))*((p+(1-p)*beta)**(K-i))*(dblquad(lambda y, x: K*T*fx(x)*fy(y), (i-1)*T, i*T, lambda x: (K*T)-x, np.inf))[0]
        cost += pi * (Cr + CompExpCost(K))
    return prob, length, cost

def P12(K,T):
    prob, length, cost = 0, 0, 0
    for i in range (1,K):
        pi = (q*(1-q)**(i-1))*((p+(1-p)*beta)**(K-i-1))*Rx((i)*T)*(quad(lambda y: fy(y), (K*T)-(i)*T, np.inf))[0]
        prob += pi
        length += (q*(1-q)**(i-1))*((p+(1-p)*beta)**(K-i-1))*Rx((i)*T)*(quad(lambda y: K*T*fy(y), (K*T)-(i)*T, np.inf))[0]
        cost += pi * (Cr + Ci + CompExpCost(K-1))
    return prob, length, cost

#CENÁRIO 13
def P13(K,T):
    prob, length, cost = 0, 0, 0
    for i in range (1,K+1):
        pi = ((1-q)**(i-1))*(p**(K-i))*(tplquad(lambda h, y, x: fx(x)*fy(y)*fh(h), (i-1)*T, i*T, 0, lambda x: (i*T)-x, lambda x, y: ((K*T)-x-y), np.inf))[0]
        prob += pi
        length += ((1-q)**(i-1))*(p**(K-i))*(tplquad(lambda h, y, x: K*T*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, 0, lambda x: (i*T)-x, lambda x, y: ((K*T)-x-y), np.inf))[0]
        cost += pi * (Cr + CompExpCost(i)) + ((1-q)**(i-1))*(p**(K-i))*(tplquad(lambda h, y, x: (Cmd*(K*T-(x+y)))*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, 0, lambda x: (i*T)-x, lambda x, y: ((K*T)-x-y), np.inf))[0]
    return prob, length, cost

def P14(K,T):
    prob, length, cost = 0, 0, 0
    for i in range (1,K):
        pi = ((1-q)**(i-1))*q*(p**(K-i-1))*Rx(i*T)*(dblquad(lambda h, y: fy(y)*fh(h), 0, T, lambda y: (K*T)-y-i*T, np.inf))[0]
        prob += pi
        length += ((1-q)**(i-1))*q*(p**(K-i-1))*Rx(i*T)*(dblquad(lambda h, y: K*T*fy(y)*fh(h), 0, T, lambda y: (K*T)-y-i*T, np.inf))[0]
        cost += pi * (Cr + Ci + CompExpCost(i)) + ((1-q)**(i-1))*q*(p**(K-i-1))*Rx(i*T)*(dblquad(lambda h, y: (Cmd*(K*T-(i*T+y)))*fy(y)*fh(h), 0, T, lambda y: (K*T)-y-i*T, np.inf))[0]
    return prob, length, cost

def P15(K,T):
   prob, length, cost = 0, 0, 0
   for i in range (1,K):
       for j in range (i+1,K+1):
           pi = (((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(p**(K-j))*(tplquad(lambda h, y, x: fx(x)*fy(y)*fh(h), (i-1)*T, i*T, lambda x: (j-1)*T - x, lambda x: (j*T)-x, lambda x, y: ((K*T)-x-y), np.inf))[0]) 
           prob += pi
           length += (((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(p**(K-j))*(tplquad(lambda h, y, x: K*T*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, lambda x: (j-1)*T - x, lambda x: (j*T)-x, lambda x, y: ((K*T)-x-y), np.inf))[0]) 
           cost += pi * (Cr + CompExpCost(j)) + (((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(p**(K-j))*(tplquad(lambda h, y, x: (Cmd*(K*T-(x+y)))*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, lambda x: (j-1)*T - x, lambda x: (j*T)-x, lambda x, y: ((K*T)-x-y), np.inf))[0]) 
   return prob, length, cost

def P16(K,T):
   prob, length, cost = 0, 0, 0
   for i in range (1,K-1):
       for j in range (i+2,K+1):
           pi = ((1-q)**(i-1))*q*((p+(1-p)*beta)**(j-i-1))*(p**(K-j))*Rx(i*T)*(dblquad(lambda h, y: fy(y)*fh(h), (j-1)*T-i*T, j*T-i*T, lambda y: (K*T)-y-i*T, np.inf))[0]
           prob += pi
           length += ((1-q)**(i-1))*q*((p+(1-p)*beta)**(j-i-1))*(p**(K-j))*Rx(i*T)*(dblquad(lambda h, y: K*T*fy(y)*fh(h), (j-1)*T-i*T, j*T-i*T, lambda y: (K*T)-y-i*T, np.inf))[0]
           cost += pi * (Cr + Ci + CompExpCost(j-1)) + ((1-q)**(i-1))*q*((p+(1-p)*beta)**(j-i-1))*(p**(K-j))*Rx(i*T)*(dblquad(lambda h, y: (Cmd*(K*T-(i*T+y)))*fy(y)*fh(h), (j-1)*T-i*T, j*T-i*T, lambda y: (K*T)-y-i*T, np.inf))[0]
   return prob, length, cost 

def P17(K,T):
   prob, length, cost = 0, 0, 0
   for i in range (1,K):
       for j in range (i+1,K+1):
           pi = ((1-q)**(i-1))*(p**(j-i))*(tplquad(lambda h, y, x: fx(x)*fy(y)*fh(h), (i-1)*T, i*T, 0, lambda x: (i*T)-x, lambda x, y: (((j-1)*T)-x-y), lambda x, y: ((j*T)-x-y)))[0] 
           prob += pi
           length += ((1-q)**(i-1))*(p**(j-i))*(tplquad(lambda h, y, x: (x+y+h)*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, 0, lambda x: (i*T)-x, lambda x, y: (((j-1)*T)-x-y), lambda x, y: ((j*T)-x-y)))[0] 
           cost += pi * (Cf + CompExpCost(i)) + ((1-q)**(i-1))*(p**(j-i))*(tplquad(lambda h, y, x: (Cmd*h)*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, 0, lambda x: (i*T)-x, lambda x, y: (((j-1)*T)-x-y), lambda x, y: ((j*T)-x-y)))[0] 
   return prob, length, cost 
   

def P18(K,T):
   prob, length, cost = 0, 0, 0
   for i in range (1,K-1):
       for j in range (i+2,K+1):
           pi = ((1-q)**(i-1))*q*(p**(j-i-1))*Rx(i*T)*(dblquad(lambda h, y: fy(y)*fh(h), 0, T, lambda y: ((j-1)*T)-y-i*T,lambda y: (j*T)-y-i*T))[0]
           prob += pi
           length += ((1-q)**(i-1))*q*(p**(j-i-1))*Rx(i*T)*(dblquad(lambda h, y: (i*T+y+h)*fy(y)*fh(h), 0, T, lambda y: ((j-1)*T)-y-i*T,lambda y: (j*T)-y-i*T))[0]
           cost += pi * (Cf + Ci + CompExpCost(i)) + ((1-q)**(i-1))*q*(p**(j-i-1))*Rx(i*T)*(dblquad(lambda h, y: (Cmd*h)*fy(y)*fh(h), 0, T, lambda y: ((j-1)*T)-y-i*T,lambda y: (j*T)-y-i*T))[0]
   return prob, length, cost 

def P19(K,T):
   prob, length, cost = 0, 0, 0
   for i in range (1,K-1):
       for j in range (i+1,K):
           for l in range (j+1,K+1):
               pi = ((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(p**(l-j))*(tplquad(lambda h, y, x: fx(x)*fy(y)*fh(h), (i-1)*T, i*T, lambda x: ((j-1)*T)-x, lambda x: (j*T)-x, lambda x, y: (((l-1)*T)-x-y), lambda x, y:((l*T)-x-y)))[0]
               prob += pi
               length += ((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(p**(l-j))*(tplquad(lambda h, y, x: (x+y+h)*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, lambda x: ((j-1)*T)-x, lambda x: (j*T)-x, lambda x, y: (((l-1)*T)-x-y), lambda x, y:((l*T)-x-y)))[0]
               cost += pi * (Cf + CompExpCost(j)) + ((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(p**(l-j))*(tplquad(lambda h, y, x: (Cmd*h)*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, lambda x: ((j-1)*T)-x, lambda x: (j*T)-x, lambda x, y: (((l-1)*T)-x-y), lambda x, y:((l*T)-x-y)))[0]
   return prob, length, cost 
   
def P20(K,T):
    prob, length, cost = 0, 0, 0
    for i in range (1,K-2):
        for j in range (i+2,K):
            for l in range (j+1,K+1):
                pi = (q*((1-q)**(i-1))*(((p+(1-p)*beta)**(j-i-1))*(p**(l-j))*Rx(i*T))*dblquad(lambda h, y: fy(y)*fh(h), (j-1)*T - i*T, j*T - i*T, lambda y: ((l-1)*T)-y-(i*T), lambda y:(l*T)-y-(i*T)) [0])
                prob += pi
                length += (q*((1-q)**(i-1))*(((p+(1-p)*beta)**(j-i-1))*(p**(l-j))*Rx(i*T))*dblquad(lambda h, y: (i*T+y+h)*fy(y)*fh(h), (j-1)*T - i*T, j*T - i*T, lambda y: ((l-1)*T)-y-(i*T), lambda y:(l*T)-y-(i*T)) [0])
                cost += pi * (Cf + Ci + CompExpCost(j-1)) +  (q*((1-q)**(i-1))*(((p+(1-p)*beta)**(j-i-1))*(p**(l-j))*Rx(i*T))*dblquad(lambda h, y: (Cmd*h)*fy(y)*fh(h), (j-1)*T - i*T, j*T - i*T, lambda y: ((l-1)*T)-y-(i*T), lambda y:(l*T)-y-(i*T)) [0])
    return prob, length, cost 

def P21(K, T):
    prob = ((1-q)**(K-1)) * Rx(K*T)
    length = ((1-q)**(K-1)) * Rx(K*T) * K*T
    cost = prob * (Cr + CompExpCost(K))
    return prob, length, cost

def objeto(K, T):
    prob, length, cost = 0, 0, 0
    cenarios = [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21]  # Inclua todas as funções de cenário aqui
    for cenario in cenarios:
        p, l, c = cenario(K, T)
        prob += p
        length += l
        cost += c
    return prob, length, cost

def taxa_de_custo(K, T):
    prob, length, cost = objeto(K, T)
    return cost/length

#####Optimization##############################################################
file_path = "Resultados modelo Three Stage degradation with defaulting V2.xlsx"
df = pd.read_excel(file_path)
resultados = []

for i in range(13,86,1):
    K_max = 15
    melhor_resultado = float('inf')
    melhor_K = 0
    melhor_T = 0.2
    contagem_sem_melhoria = 0
    
    a1 = df.iloc[i, 1]  # Coluna B (posição 1)
    b1 = df.iloc[i, 2]  # Coluna C (posição 2)
    a2 = df.iloc[i, 3]  # Coluna D (posição 3)
    b2 = df.iloc[i, 4]  # Coluna E (posição 4)
    a3 = df.iloc[i, 5]  # Coluna F (posição 5)
    b3 = df.iloc[i, 6]  # Coluna G (posição 6)
    Ci = df.iloc[i, 7]  # Coluna H (posição 7)
    Cr = df.iloc[i, 8]  # Coluna I (posição 8)
    Cf = df.iloc[i, 9]  # Coluna J (posição 9)
    Cmd = df.iloc[i, 10]  # Coluna K (posição 10)
    beta = df.iloc[i, 11]  # Coluna L (posição 11)
    q = df.iloc[i, 12]  # Coluna M (posição 12)
    p = df.iloc[i, 13]  # Coluna N (posição 13)
    
    inicio = time.time()
    print("Iniciando otimização...\n")
    
    for K in range(1, K_max + 1):
        res = minimize(lambda T: taxa_de_custo(K, T[0]), [0.2], method='SLSQP', bounds=[(0.01, 2.0)])
        resultado_atual = res.fun
        resultados.append((K, res.x[0], resultado_atual))
        print(K)
        print(f"K = {K}, T otimizado = {res.x[0]:.4f}, Taxa de custo = {resultado_atual:.4f}")
        
        if resultado_atual < melhor_resultado:
            melhor_resultado = resultado_atual
            melhor_K = K
            melhor_T = res.x[0]
            contagem_sem_melhoria = 0
        else:
            contagem_sem_melhoria += 1
        
        if contagem_sem_melhoria >= 3:
            break
    ###############################################################################
    duracao = time.time() - inicio
    
    df.iloc[i, 14] = melhor_K  # Coluna O (posição 14)
    df.iloc[i, 15] = melhor_T  # Coluna P (posição 15)
    df.iloc[i, 16] = melhor_resultado 
    df.iloc[i, 18] = duracao 
    
    # Salvar o arquivo após cada caso resolvido
    df.to_excel(file_path, index=False)

    print("\n########RESULTADO FINAL########")
    print(f"Melhor K = {melhor_K}")
    print(f"Melhor T = {melhor_T}")
    print(f"Melhor taxa de custo = {melhor_resultado}")
    print(f"Duração: {duracao} segundos")