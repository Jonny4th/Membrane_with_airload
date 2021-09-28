# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:55:13 2021

@author: ASUA

Solve for moadl frequencies of airloaded kettledrum

Caution: use RUN CELL only.

"""
import numpy as np
import scipy.special as sps
from time import time
import workers # editable package for multiprocessing
import gc
import sys

#%% Non-multiprocessing
def MemMatrix(w, T, m, lim, s, ca, a, method = 'numpy1'):
    """
    Create a matrix to be used to find eigen-frequency and vector for a free-membrane in air.
    w = input angular frequency
    T = membrane tension
    m = m mode number (number of line nodes)
    lim = highest n mode number (number of circular nodes), determine the size of the matrix
    method: method for I summation, which will also determine the C summation method
             'numpy1' = 2-part, 7 args
             'numpy2' = 3-part 1st, 7 args + 1 singular point
             'numpy3' = 3-part 2nd, 7 args + 1 singular point
             'mpmath' = 2-part
             if/onlyif method = 'mpmath', C summation will be 'mpmath' (This is automaticly done in workers.MME function).
    """
    Matrix = np.zeros([lim,lim], dtype=complex)
    for i, n in enumerate(np.arange(lim)+1):
        for j, n2 in enumerate(np.arange(lim)+1):
            Matrix[i,j] = workers.MME(n, n2, w, m, lim, T, s, ca, a, method = method)
    return Matrix

def KettleMatrix(w, T, m, lim,s,ca,a,V, method = 'numpy1'):
    """
    Create a matrix to be used to find eigen-frequency and vector for a kettledrum.
    w = input angular frequency
    T = membrane tension
    m = m mode number (number of line nodes)
    lim = highest n mode number (number of circular nodes), determine the size of the matrix
    method: method for I summation, which will also determine the C summation method
             'numpy1' = 2-part, 7 args
             'numpy2' = 3-part 1st, 7 args + 1 singular point
             'numpy3' = 3-part 2nd, 7 args + 1 singular point
             'mpmath' = 2-part
             if/onlyif method = 'mpmath', C summation will be 'mpmath' (This is automaticly done in workers.MME function).
    """
    Matrix = np.zeros([lim,lim], dtype=complex)
    for i, n in enumerate(np.arange(lim)+1):
        for j, n2 in enumerate(np.arange(lim)+1):
            Matrix[i,j] = workers.KME(n, n2, w, m, lim, T,s,ca,a,V, method = method)
    return Matrix

def HoleMatrix(w, T, m, lim,s,ca,a,V,d, method = 'numpy1'):
    """
    Create a matrix to be used to find eigen-frequency and vector for a holed-kettledrum.
    w = input angular frequency
    T = membrane tension
    m = m mode number (number of line nodes)
    lim = highest n mode number (number of circular nodes), determine the size of the matrix
    method: method for I summation, which will also determine the C summation method
             'numpy1' = 2-part, 7 args
             'numpy2' = 3-part 1st, 7 args + 1 singular point
             'numpy3' = 3-part 2nd, 7 args + 1 singular point
             'mpmath' = 2-part
             if/onlyif method = 'mpmath', C and S summation will be 'mpmath' (This is automaticly done in workers.HME function).
    """
    #prepare metrix
    Matrix = np.zeros([lim,lim], dtype=complex)
    for i, n in enumerate(np.arange(lim)+1):
        for j, n2 in enumerate(np.arange(lim)+1):
            Matrix[i,j] = workers.HME(n, n2, w, m, lim, T,s,ca,a,V,d, method = method)
    return Matrix

#%%Christian's Kettledrum Parameters
# ca = 344.0 # speed of sound in air [m/s]
# a = 0.656/2 # membrane radius [m]
# V= 0.14 # kettle volume [m^3]
# d = 0.014 # hole radius [m]
# rho_air_0 = 1.21 # air density at 25 deg C in [kg/m^3]
# s = 0.2653 # membrane density in [kg/m^2]
# T = 3990.0 # membrane tension [N/m]

#%% My Parameters
ca = 345.13  # air sound speed at 23.4 deg C in [m/s]
a = 0.093  # membrane radius in [m]
V = 0.000477
d = 0.01 # hole radius in [m]
rho_air_0 = 1.21  # air density at 25 deg C in [kg/m^3]
s = 0.35306  # membrane density in [kg/m^2]
T = 1558. # [N/m]

#%%Calculating Parameter
limit = 6
Mode_Num_Calculation = np.array([(1,1),(2,1),(0,2),(3,1),(1,2),(4,1),(2,2),(0,3),(5,1)])
Mode_Freq_Calculation = []
initial_freq_modifier = 0.8
iteration_weight = 0.2
torelance = 0.05
max_iteration = 50

sys.exit('\nThis is a stop for you to check parameters and "Mode_Num_Calculation" variable.\nPlease continue by using RUN CELL0.\nThe calculation may take pretty LONG TIME. PROCEED WITH CAUTION.\n(You may interupt the program with Ctrl+C.)') # This stop is to let you check the parameter and use RUN CELL for the calculation.

#%%
"The calculation may take pretty long time. Preceed with caution."

#%% For Ketteldrum w/ or w/o hole
Mode_Freq_Calculation = []
print('Calculation for T = ' + str(T))
for number, modes in enumerate(Mode_Num_Calculation):
    m = modes[0]
    n = modes[1]
    print('Start calculating mode number: (%d%d)' % (m,n))
    initial_w = sps.jn_zeros(m,n)[-1]/a*np.sqrt(T/s)
    f1 = initial_w/2/np.pi*initial_freq_modifier
    print('0 : %.2f Hz' % f1)
    t = time()
    for i in range(max_iteration):
        #Matrix = KettleMatrix(f1*2*np.pi, T, m, limit, s, ca, a, V)
        Matrix = HoleMatrix(f1*2*np.pi, T, m, limit, s, ca, a, V, d)
        eigenvalues, eigenvectors = np.linalg.eig(Matrix)
        mode_num, mode_freq = workers.Draw_Modeshape(eigenvalues, eigenvectors, m, draw =False)
        f2 = mode_freq[np.where(mode_num == n)][0]
        print('%d : %.2f Hz' % (i+1,f2))
        if abs(f2-f1) < torelance:
            print("Time = %.2f sec, (%d%d) mode freq: %.2f Hz" %(time()-t,m,n,f2))
            Mode_Freq_Calculation.append(f2)
            break
        elif i == max_iteration-1:
            Mode_Freq_Calculation.append((f2+f1)/2)
        f1 += (f2 - f1)*iteration_weight
        print('\t-> %f Hz' % (f1))
        gc.collect()
            
print('Calulation took '+str((time()-t)/60)+' minutes.')
print('Calculation for T = ' + str(T))
for i in range(len(Mode_Num_Calculation)):
    print(str(Mode_Num_Calculation[i])+' : '+str(Mode_Freq_Calculation[i])+' Hz')
#%%

#%% Exclusively for Free Membrane Iteration 
Mode_Freq_Calculation = []
for number, modes in enumerate(Mode_Num_Calculation):
    m = modes[0]
    n = modes[1]
    print('Start calculating mode number: (%d%d)' % (m,n))
    initial_w = sps.jn_zeros(m,n)[-1]/a*np.sqrt(T/s)
    f1 = initial_w/2/np.pi-10
    print('0 : %.2f Hz' % f1)
    t = time()
    for i in range(max_iteration):
        Matrix = MemMatrix(f1*2*np.pi, T, m, limit, s, ca, a)
        eigenvalues, eigenvectors = np.linalg.eig(Matrix)
        mode_num, mode_freq = workers.Draw_Modeshape(eigenvalues, eigenvectors, m, draw =False)
        f2 = mode_freq[np.where(mode_num == n)][0]
        print('%d : %.2f Hz' % (i+1,f2))
        if abs(f2-f1) < 0.05:
            print("Time = %.2f sec, (%d%d) mode freq: %.2f Hz" %(time()-t,m,n,f2))
            Mode_Freq_Calculation.append(f2)
            break
        elif i == max_iteration-1:
            Mode_Freq_Calculation.append( (f2+f1)/2)
        f1 += (f2 - f1)*0.1
        print('\t-> %f Hz' % (f1))
        gc.collect()
            
print('Calulation took '+str((time()-t)/60)+' minutes.')
print('Calculation for T = ' + str(T))
for i in range(len(Mode_Num_Calculation)):
    print(str(Mode_Num_Calculation[i])+' : '+str(Mode_Freq_Calculation[i])+' Hz')
    