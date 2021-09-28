# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 14:52:59 2020
Functions for calculating kettledrum resonance according to Christian et al. and Rienstra.
@author: ASUA

update 10/03/2021
- Fix Imn integrate to output the same sign as the one writen by Rienstra. 
  The output is automaticaly conjugated.
- Fix Cmn on case of m = 0, where y_01 = 0. The term (1-(m/y)^2) is terminated.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sps
import scipy.integrate as intg
import mpmath as mp

rho_air_0 = 1.21 # air density at 25 deg C in [kg/m^3]

#customized j0 function, numpy
def BZ(order, nth_zero, d=0):
    """
    order : order of the Bessel function.
    nth_zero : order of the zero of interest.
    d : option for derivative Bessel function (now only support 0 and 1).
    """
    if d == 0:
        return sps.jn_zeros(order, nth_zero)[-1]
    elif d == 1:
        if order == 0:
            if nth_zero == 1:
                return 0
            else:
                return sps.jnp_zeros(order, nth_zero-1)[-1]
        else:
            return sps.jnp_zeros(order, nth_zero)[-1]

#customized j0 function, mpmath7
def BZmp(order, nth_zero, d=0):
    """
    order : order of the Bessel function.
    nth_zero : order of the zero of interest.
    d : option for derivative Bessel function (now only support 0 and 1).
    """
    #if d == 1 and order == 0:
    #    nth_zero +=1 # compensate for the fact that mpmath's besseljzero returns '0' for order = 1, nth_zero = 1, d = 1.
    return mp.besseljzero(order, nth_zero, d)

#2-part Integration mpmath package
def complex_mpquad_2part(func, low_lim, up_lim, RIpoint, w,m,n,n2,ca,a):
    """
    func = input function
    low_lim = lower limit
    up_lim = upper limit
    RIpoint = real-imaginary changing point
    w = input angular frequency
    m = m mode number
    n = n index
    n2 = n'' index
    """
   
    eps = np.finfo(float).eps
    def real_func(*args, **kwargs):
        return np.real(func(*args, **kwargs))

    def imag_func(*args, **kwargs):
        return np.imag(func(*args, **kwargs))
    
    imag_integral = mp.quad(lambda x: func(x, w,m,n,n2,ca,a), [low_lim, RIpoint-eps])
    real_integral = mp.quad(lambda x: func(x, w,m,n,n2,ca,a), [RIpoint+eps, up_lim])
    
    return complex(real_integral) + complex(imag_integral)

#2-part Integration with scipy integrate quad
def complex_quad_2part(func, low_lim, up_lim, RIpoint, singular_point = None, **kwargs):
    """
    singular_point = singularity in the function (other than real-imaginary changing point)
    """
   
    eps = np.finfo(float).eps
    def real_func(*args, **kwargs):
        return np.real(func(*args, **kwargs))

    def imag_func(*args, **kwargs):
        return np.imag(func(*args, **kwargs))
    
    imag_integral = intg.quad(imag_func, low_lim, RIpoint-eps, **kwargs)
    real_integral = intg.quad(real_func, RIpoint+eps, up_lim, **kwargs)
    
    return real_integral[0] - 1j*imag_integral[0]

#3-part Integration, 1 imag and 2 manual-seperated real, with scipy integrate quad
def complex_quad_3part(func, low_lim, up_lim, RIpoint, singular_point = None, **kwargs):

    def real_func(*args, **kwargs):
        return np.real(func(*args, **kwargs))

    def imag_func(*args, **kwargs):
        return np.imag(func(*args, **kwargs))
    
    imag_integral = intg.quad(imag_func, low_lim, RIpoint, **kwargs)
    real_integral = intg.quad(real_func, RIpoint, singular_point, **kwargs) 
    real_integral2 = intg.quad(real_func, singular_point, up_lim, **kwargs) #manually seperated here
    
    return real_integral[0] + 1j*imag_integral[0]+real_integral2[0]

#3-part Integration, 1 imag and 2 function-seperated real, with scipy integrate quad
def complex_quad_3part2(func, low_lim, up_lim, RIpoint, singular_point = None, **kwargs):
   
    eps = np.finfo(float).eps
    def real_func(*args, **kwargs):
        return np.real(func(*args, **kwargs))

    def imag_func(*args, **kwargs):
        return np.imag(func(*args, **kwargs))
    
    imag_integral = intg.quad(imag_func, low_lim, RIpoint-eps, **kwargs)
    real_integral = intg.quad(real_func, RIpoint+eps, up_lim, points = singular_point, **kwargs) # use function-ready argument for seperation
     
    return real_integral[0] + 1j*imag_integral[0]

#2-part fixed Gaussian quadrature Integration
def complex_quad_FG_2part(func, low_lim, up_lim, RIpoint, singular_point = None, n = 5, **kwargs):
    
    def real_func(*args, **kwargs):
        return np.real(func(*args, **kwargs))

    def imag_func(*args, **kwargs):
        return np.imag(func(*args, **kwargs))
    
    imag_integral = intg.fixed_quad(imag_func, low_lim, RIpoint, n = n, **kwargs)
    real_integral = intg.fixed_quad(real_func, RIpoint, up_lim, n = n, **kwargs)
    
    return real_integral[0]+ 1j*imag_integral[0]

#integrant for scipy
def dI(l,w,m,n,n2,ca,a):
    return l / np.lib.scimath.sqrt( l**2 - (w/ca)**2 ) * sps.jv(m, a*l)**2 / ((a*l)**2 - BZ(m,n)**2) / ((a*l)**2 - BZ(m,n2)**2)

#integrant for mpmath
def dImp(l,w,m,n,n2,ca,a):
    return l / mp.sqrt( l**2 - (w/ca)**2 ) * mp.besselj(m, a*l)**2 / ((a*l)**2 - BZmp(m,n)**2) / ((a*l)**2 - BZmp(m,n2)**2)

#Ultimate Complex Integration Function for I term
def I(low_lim, up_lim, RIpoint, w, m, n, n2, ca, a, singular_point = None, method = 'numpy1'):
    """
    low_lim = lower limit
    up_lim  = upper limit
    RIpoint = real-imaginary limit
    w = input angular frequency
    m = m mode number of interest
    n = n index
    n2 = n'' index
    singular_point = singularity points, for methods: 'numpy2' and 'numpy3'
    Methods: 'numpy1' = 2-part, 7 args
             'numpy2' = 3-part 1st, 7 args + 1 singular point
             'numpy3' = 3-part 2nd, 7 args + 1 singular point
             'mpmath' = 2-part
    Return: Complex
    """
    if method == 'numpy1':
        return 4*np.pi*a**2*BZ(m,n)*BZ(m,n2)*complex_quad_2part(dI, low_lim, up_lim, RIpoint, args=(w,m,n,n2,ca,a))
    elif method == 'numpy2':
        if singular_point == None:
            return print("Please specify 'singular_point'")
        return 4*np.pi*a**2*BZ(m,n)*BZ(m,n2)*complex_quad_3part(dI, low_lim, up_lim, RIpoint, singular_point = singular_point, args=(w,m,n,n2,ca,a))
    elif method == 'numpy3':
        if singular_point == None:
            return print("Please specify 'singular_point'")
        return 4*np.pi*a**2*BZ(m,n)*BZ(m,n2)*complex_quad_3part2(dI, low_lim, 1000, RIpoint, singular_point = singular_point, args=(w,m,n,n2,ca,a))
    elif method == 'mpmath':
        return 4*np.pi*a**2*BZ(m,n)*BZ(m,n2)*complex_mpquad_2part(dImp, low_lim, up_lim, RIpoint, w,m,n,n2,ca,a)
    else:
        print("Wrong method input. Choose: 'numpy1', 'numpy2', 'numpy3', or 'mpmath'")
        
#K-integrant for scipy
def dK(l,w,m,n,n2,ca,a):
    return l / np.lib.scimath.sqrt( l**2 - (w/ca)**2 ) * sps.jv(m, a*l)**2 / ((a*l)**2 - BZ(m,n2)**2)

#K-integrant for mpmath
def dKmp(l,w,m,n,n2,ca,a):
    return l / mp.sqrt( l**2 - (w/ca)**2 ) * mp.besselj(m, a*l)**2 / ((a*l)**2 - BZmp(m,n2)**2)

#Ultimate Complex Integration Function for K term
def K(low_lim, up_lim, RIpoint, w, m, n, n2, ca, a, singular_point = None, method = 'numpy1'):
    """
    Methods: 'numpy1' = 2-part, 7 args
             'numpy2' = 3-part 1st, 7 args + 1 singular point
             'numpy3' = 3-part 2nd, 7 args + 1 singular point
             'mpmath' = 2-part
    Return: Complex
    """
    if method == 'numpy1':
        return 4*np.pi/a*BZ(m,n)*BZ(m,n2)*complex_quad_2part(dK, low_lim, up_lim, RIpoint, args=(w,m,n,n2,ca,a))
    elif method == 'numpy2':
        if singular_point == None:
            return print("Please specify 'singular_point'")
        return 4*np.pi/a*BZ(m,n)*BZ(m,n2)*complex_quad_3part(dK, low_lim, up_lim, RIpoint, singular_point = singular_point, args=(w,m,n,n2,ca,a))
    elif method == 'numpy3':
        if singular_point == None:
            return print("Please specify 'singular_point'")
        return 4*np.pi/a*BZ(m,n)*BZ(m,n2)*complex_quad_3part2(dK, low_lim, 1000, RIpoint, singular_point = singular_point, args=(w,m,n,n2,ca,a))
    elif method == 'mpmath':
        return 4*np.pi/a*BZ(m,n)*BZ(m,n2)*complex_mpquad_2part(dKmp, low_lim, up_lim, RIpoint, w,m,n,n2,ca,a)
    else:
        print("Wrong method input. Choose: 'numpy1', 'numpy2', 'numpy3', or 'mpmath'")

# numpy gamma
def gamma(w, m, n, ca, a):
    return np.sqrt( (complex(w)/ca)**2 - (BZ(m,n,1)/a)**2 )

# mpmath gamma
def gamma_mp(w, m, n, ca, a):
    return mp.sqrt( (complex(w)/ca)**2 - (BZmp(m,n,1)/a)**2 )

#C element, for simple loop summation
def Q(n1, w, m, n0, n2, ca, a, L):
    if m == 0:
        return BZ(m,n0) * BZ(m,n2) / (BZ(m,n0)**2 - BZ(m,n1,1)**2) / (BZ(m,n2)**2 - BZ(m,n1,1)**2) / gamma(w,m,n1,ca,a) / np.tan(gamma(w,m,n1,ca,a)*L)
    else:
        return BZ(m,n0) * BZ(m,n2) / (BZ(m,n0)**2 - BZ(m,n1,1)**2) / (BZ(m,n2)**2 - BZ(m,n1,1)**2) / (1 - (m/BZ(m,n1,1))**2) / gamma(w,m,n1,ca,a) / np.tan(gamma(w,m,n1,ca,a)*L)

#for mpmath
def Qmp(n1, w, m, n0, n2, ca, a, L):
    if m == 0:
        return BZmp(m,n0) * BZmp(m,n2) / ( BZmp(m,n0)**2 - BZmp(m,n1,1)**2)/( BZmp(m,n2)**2 - BZmp(m,n1,1)**2) / gamma_mp(w,m,n1,ca,a) * mp.cot(gamma_mp(w,m,n1,ca,a)*L)
    else:
        return BZmp(m,n0) * BZmp(m,n2) / ( BZmp(m,n0)**2 - BZmp(m,n1,1)**2)/( BZmp(m,n2)**2 - BZmp(m,n1,1)**2) / (1 - (m/BZmp(m,n1,1))**2) / gamma_mp(w,m,n1,ca,a) * mp.cot(gamma_mp(w,m,n1,ca,a)*L)

def C(w,m,n0,n2,lim_n1,ca,a, V, method = "loop"):
    """
    w = angular frequency
    m = m mode number of interest
    n0 = n index
    n2 = n'' index
    lim_n1 = n' index limit, if lim_n1 = inf, the code will override the method into mpmath.
    Methods : loop = use for loop for partial sum.
              mpmath = use mpmath package and sum with mpmath.nsum function;
              if lim_n1 = inf, the code will override the method into mpmath.
    """
    L = V/np.pi/a**2
    if method == 'mpmath' or lim_n1 == mp.inf:
        return mp.nsum(lambda x: Qmp(x,w,m,n0,n2,ca,a,L), [1,lim_n1])
    
    elif method == 'loop':
        y = 0j
        for n1 in np.arange(lim_n1)+1:
            y += Q(n1,w,m,n0,n2,ca,a,L)
        return y
    
    else:
        return print("Wrong method input. Choose: 'loop' or 'mpmath'")

def S(w, n, lim,ca,a,V, method = 'loop'):
    """
    w = input angular frequency
    n = n mode number of interest
    lim = n'' index limit
    Methods : loop = use for loop for partial sum.
              mpmath = use mpmath package and sum with mpmath.nsum function;
              if lim_n1 = inf, the code will override the method into mpmath.
    """
    L = V/np.pi/a**2 
    if method == 'mpmath' or lim == mp.inf:
        #declare single S term to be used in mpmath.nsum
        def s(n1):
            return 1/(a * (BZmp(0,n)**2 - BZmp(0,n1,1)**2) * gamma_mp(w, 0, n1,ca,a) * mp.sin(gamma_mp(w,0,n1,ca,a)*L/a) * mp.besselj(0,BZmp(0,n1,1)))
        return mp.nsum(lambda x: s(x), [1,lim])
    
    elif method == 'loop':
        Sum = 0+0j
        for n1 in np.arange(lim)+1:
            Sum += 1/(a * (BZ(0,n)**2 - BZ(0,n1,1)**2) * gamma(w, 0, n1,ca,a) * np.sin(gamma(w,0,n1,ca,a)*L/a) * sps.jv(0,BZ(0,n1,1)))
        return Sum

#Membrane Matrix Element
def MME(n,n2,w,m,lim, T, s, ca, a, method = 'numpy1'):
    cm = np.sqrt(T/s)
    if method == 'mpmath':
        pi = mp.pi
        x1 = BZmp(m,n) # for x_mn
        x2 = BZmp(m,n2) # for x_mn'
    else:
        pi = np.pi 
        x1 = BZ(m,n)
        x2 = BZ(m,n2)
    return T/4/pi/rho_air_0 * ( (w/cm)**2 - (x2/a)**2 ) * ( ( (w/ca)**2 - (x1/a)**2 ) * I(0, np.inf, w/ca, w, m, n, n2, ca, a, method = method) - a * K(0, np.inf, w/ca, w, m, n, n2, ca, a, method = method) )

#Kettledrum Matrix Elements
def KME(n,n2,w,m,lim, T,s,ca,a,V, method = 'numpy1'):
    cm = np.sqrt(T/s)
    if method == 'mpmath':
        Cmethod = method
    else:
        Cmethod = 'loop'
    element = (cm*w)**2 * rho_air_0/T * (4*C(w,m,n,n2, lim,ca,a,V, method = Cmethod) - I(0, np.inf, w/ca, w, m, n, n2,ca,a, method = method)/2/np.pi)
    if n == n2 :
        if method == 'mpmath':
            element += (cm * BZmp(m,n)/a)**2
        else:
            element += (cm * BZ(m,n)/a)**2
    return element

#Holed-Kettledrum Matrix Elements
def HME(n,n2,w,m,lim, T,s,ca,a,V,d, method = 'numpy1'):
    cm = np.sqrt(T/s)
    if method == 'mpmath':
        Cmethod = method
        pi = mp.pi
        x1 = BZmp(m,n) # for x_mn
        x2 = BZmp(m,n2) # for x_mn'
    else:
        Cmethod = 'loop'
        pi = np.pi 
        x1 = BZ(m,n)
        x2 = BZ(m,n2)
        
    element = (cm*w)**2 * rho_air_0/T * (4*C(w,m,n,n2,lim,ca,a,V, method = Cmethod) - I(0, np.inf, w/ca, w, m, n, n2,ca,a, method = method)/2/pi)
    if n == n2 :
        element += (cm * x1/a)**2
    if m == 0:
        element += -8*(cm*w)**2*rho_air_0*d*x1*x2/T/pi * S(w, n, lim,ca,a,V, method = Cmethod) * S(w, n2, lim,ca,a,V, method = Cmethod)
    return element

def node_count(array):
    """
    Input can be arrya or metrix.
    Count the number of zero-passing points by looking at sign(+/-) change.
    """
    return np.count_nonzero(np.diff(np.signbit(array.T[1:-1]).T), axis = 1)+1
    
def Draw_Modeshape(eigenvalues, eigenvectors, m, draw = False):
    r = np.arange(0,1.001,0.001) # normalized radial displacement.
    lim = len(eigenvalues) # number of modes and plots
    eivec = np.real(eigenvectors) # extracts real value from the inputs  
    eta0 = np.ones([lim,len(r)]) # for unperturbed mode shapes
    
    for i in range(lim):
        eta0[i] = eta0[i]*sps.jn(m,r*sps.jn_zeros(m,i+1)[-1]) # unperturbed mode shapes
    eta = np.zeros([lim, len(r)], dtype=float) # for perturbed mode shapes
    eigenfreq = np.real(np.sqrt(eigenvalues))/2/np.pi # calculation resonance frequencies
    
    for i in range(lim):
        for j in range(lim):
            eta[i] += eivec[j][i]*eta0[j] # summation for final perturbed mode shapes
    
    mode = node_count(eta)

    if draw == True:
        fix, ax = plt.subplots(lim, figsize=[5,9.5])
        for i in range(lim):
            ax[i].axhline(color = 'k')
            ax[i].grid()
            ax[i].set_title("Vector#: " + str(i) + ", mode: " + str(m)+str(mode[i]) + ", frequency: "+str(eigenfreq[i])+"Hz", loc='left')
            ax[i].plot(r,eta[i]) 
        plt.tight_layout()
        
    return mode, eigenfreq