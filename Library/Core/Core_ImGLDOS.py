#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

 
#All functions for calculating the LDOS in any (non-absorbing) layer in a stratified stack


#For electric, magnetic, and magneto-electric dipoles

#Overall structure of this file
#(1) All the *integrands* (electric, magnetic, and crossed)
#(2) Integration routines, with and without the guided mode contributions

#Note that the electric dipole precisely follows the formulation     #Amos and Barnes PRB 55, 7249
#The magnetic and magnetoelectric are in same nomenclature

Parameter input throughout (geometry)
        - We assume a point dipole at height z in a slab of index nslab and height dslab
        - The half infinite substrate n2 is separated from the slab itself by 
          a set of layers of index and thickness listed in nd2 list, counting away from the slab. This is on the z=0 side
        - On the z=dslab side is half infinite superstrate n3. Inbetween the slab and the superstrate is the stack nd3list
 

@author: dpal,fkoenderink
"""
#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%%

from Library.Core import Core_Smatrix as RT


import numpy as np
import scipy.integrate as integrate
 


from numba import jit, config

config.DISABLE_JIT = False
cachechoice=True 



@jit(nopython=True,cache=cachechoice)
def EHC_AMOSintegrand(u,k0,z,dslab,nslab,n2,n3,nd2list,nd3list):
    ''' integrand definition for the electric, magnetic and crossed-dipole LDOS, in the nomenclature of Amos and Barnes. This means:
        Parameter input throughout (dslab,nslab,n2,n3,nd2list,nd3list )
        - We assume a point dipole at height z in a slab of index nslab and height dslab
        - The half infinite substrate n2 is separated from the slab itself by 
          a set of layers of index and thickness listed in nd2 list, counting away from the slab. This is on the z=0 side
        - On the z=dslab side is half infinite superstrate n3. Inbetween the slab and the superstrate is the stack nd3list
    
        u,k0 are a dimensionless parallel momentum as defined in Amos & Barnes, and the free-space wavenumber k0=w/c        
    '''        
    # l is shorthand for dimensionsless i k_z
    # u is dimensionless parallel momentum
    l=np.cdouble(-1.0j*np.sqrt(1.0+0.0j-u*u)) 
     
    # propagation factors
    eb1=np.exp(-2.0*nslab*k0*l*z)
    eb2=np.exp(-2.0*nslab*k0*l*(dslab-z))
    eb12=eb1*eb2 
    
    # parallel momentum in actual units.
    kpar=nslab*k0*u
    
 
    # required p and s reflection coefficients taken from S-matrix algorithm
    # Notice how [0] stands for "reflection" instead of "transmission" coefficient
    r12p=RT.rt_p(k0,kpar,nslab,n2,nd2list)[0]
    r13p=RT.rt_p(k0,kpar,nslab,n3,nd3list)[0]
    r12s=RT.rt_s(k0,kpar,nslab,n2,nd2list)[0]
    r13s=RT.rt_s(k0,kpar,nslab,n3,nd3list)[0]
    
    #Amos and Barnes PRB 55, 7249
    #function returns [Eq3, Eq4] with 
    #Equation 3 in A&B,  integrand-only for zperp is first output
    #Equation 4 in A&B,  integrand-only for zpar
    zP_integrandperp=u*u*u/l*((1.0-eb1*r12p)*(1.0-eb2*r13p))/(1.0-eb12*r12p*r13p)
    zP_integrandpar=u/l*( ((1.0+eb1*r12s)*(1.0+eb2*r13s))/(1.0-eb12*r12s*r13s) + 
                 +(1.-u*u)*((1.0+eb1*r12p)*(1.0+eb2*r13p))/(1.0-eb12*r12p*r13p))


    # For the cross term
    zC_integrand=u*(((1.0-eb1*r12s)*(1.0+eb2*r13s))/(1.0-eb12*r12s*r13s) - 
                    ((1.0+eb1*r12p)*(1.0-eb2*r13p))/(1.0-eb12*r12p*r13p))


   
    ## For the magnetic:  exactly as in the Amos and Barnes integrand for E, but
    ## using the substitutions
    ## (1) s -> p
    ## (2) all reflection coefficients gain a minus sign. 
    
    r12p,r12s=-1.0*r12s,-1.0*r12p
    r13p,r13s=-1.0*r13s,-1.0*r13p    

    
    zM_integrandperp=u*u*u/l*((1.0-eb1*r12p)*(1.0-eb2*r13p))/(1.0-eb12*r12p*r13p)
    zM_integrandpar=u/l*( ((1.0+eb1*r12s)*(1.0+eb2*r13s))/(1.0-eb12*r12s*r13s) + 
                 +(1.-u*u)*((1.0+eb1*r12p)*(1.0+eb2*r13p))/(1.0-eb12*r12p*r13p))

    

    # prefactors 3/4 and 3/2 listed in Amos & Barnes included in shipout routine
    return [0.75*zP_integrandpar,1.5*zP_integrandperp,0.75*zM_integrandpar,1.5*zM_integrandperp,0.75*zC_integrand]



'''Paulus trajectory for complex integration Phys. Rev. E 62, 5797
'''


@jit(nopython=True,cache=cachechoice)
def integrationtrajectory(t,kmin,kmax):
    ''' Slightly deformed as an ellipse between 0 and kmax, just bypassing all
        guided mode poles
        Phys. Rev. E 62, 5797 See figure 3.
        Michael Paulus, Phillipe Gay-Balmaz, and Olivier J. F. Martin
    '''
    ct=np.cos(t)
    st=np.sin(t) 
    x=kmax*(0.5+0.5*ct-1.0j*kmin*st)
    v=(-0.5*st - 1.0j*kmin*ct)*kmax
    return x,v
  
    

 
def AMOSMultiplex(u,k0,z,dslab,nslab,n2,n3,nd2list,nd3list):  
    # integrand simply on the real axis    
    return np.imag(np.array(EHC_AMOSintegrand(u,k0,z,dslab,nslab,n2,n3,nd2list,nd3list)))

 
def AMOSPaulusMultiplex(t,kmax,k0,z,dslab,nslab,n2,n3,nd2list,nd3list):    
    # integrand on the contour, where t will run from 0 to pi.
    kmin=0.01 # max deviation from real axis, in units of k0. Empirical choice
    x,v=integrationtrajectory(t,kmin,kmax)
    zz=EHC_AMOSintegrand(x,k0,z,dslab,nslab,n2,n3,nd2list,nd3list)
    return np.imag(-v*np.array(zz))

def LDOSinlayer(k0,z,dslab,nslab,n2,n3,nd2list,nd3list):
    #perform the integral over the contour, and then onwards far enough to fake infinity.
    kmax=np.max(np.abs(np.real( np.append(np.append(nd2list[:,0],nd3list[:,0]),[n2,n3]))))*2.0
#    kmaxmax=kmax+k0*1.E3 # should be infinity. Integrand decay very rapidly except right at a metal
    kmaxmax=np.inf
    zt1=integrate.quad_vec(lambda x: AMOSPaulusMultiplex(x,kmax,k0,z,dslab,nslab,n2,n3,nd2list,nd3list),0.0,np.pi,epsabs=1E-5,epsrel=1e-5)[0]
    zt2=integrate.quad_vec(lambda x: (AMOSMultiplex(x,k0,z,dslab,nslab,n2,n3,nd2list,nd3list)),kmax,kmaxmax,epsabs=1E-5,epsrel=1e-5)[0]
  
    zt=nslab*(zt1+zt2)
    return np.real(zt)

