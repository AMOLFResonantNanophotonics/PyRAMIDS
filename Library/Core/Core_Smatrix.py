#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code calculates the optical response for any stratified multilayer system at any input k||, 
using the S matrix formalism reported by Lifeng Li in JOSA A 13 1024-1335  1996

The code is divided as follows
General routines:   (1) kz calculation from n, k0=omega/c and k||,  with proper sign of imaginary part pinned by the code.
                        (2) Redheffer star product
                        (3) Lifeng Li code, JOSA A 13 1025
                                     - Interface t and S matrices
                                     - Layer S-matrices that include propagation delay also
                                     - Recursion to build composite S-matrix of a stack
                              Almost entirely identical for both polarizations. Therefore polarization is passed as a boolean
                              through al lthe routines (only used in interface t-matrix)
                        (4) Routine to report  the amplitude r and t (meaning, in half-infinites around the layers)
                            Two routines, one for s and one for p                            

                        (5) Routine to report the up- and down field coefficient in a given layer,
                            given a stack illuminated from the front. 
                            Includes polarization resolved wrappers, calling one routine that works for both
 



@author: fkoenderink, dpal
"""
#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%%


import numpy as np
from numba import jit, config

#config.DISABLE_JIT = True
config.DISABLE_JIT = False
cachechoice=True
fastmathchoice=True

##########    GENERAL FUNCTIONS
@jit(nopython=True,cache=cachechoice,fastmath=fastmathchoice)
def kz(n,k0,kpar):  
    """ Input:  refractive index n (complex scalar), k0 (omega/c wavenumber in free space) and parallel momentum
        Returns: complex valued perpendicular momentum. Machine precision offset etc to pin datatype and sign of argumnent
    """
    return 1.0e-23+np.sqrt(0.0+0.0j+n*n*k0*k0-kpar*kpar)


@jit(nopython=True,cache=cachechoice,fastmath=fastmathchoice)
def  dotstar(a11,a12,a21,a22,b11,b12,b21,b22):
    """Lifeng JOSA A 13 1024, Equation 23a
        For 2x2 matrices, this is the Redheffer-star product. 
        Since here each element aij and bij is itself a scalar and not a submatrix, the term-ordering is irrelevant    
        Input of the matrix as four entries for a and for b is inelegant but  expedites numba jit
    """
    denom=1.0/(1.0+0.0j-a12*b21)
    return b11*a11*denom, b12 + b11*a12*b22*denom,a21+a22*a11*b21*denom,a22*b22*denom 
    


## Lifeng Li building blocks
@jit(nopython=True,cache=cachechoice,fastmath=fastmathchoice)
def layer_tsp(n1,n2,k0,kpar,sorp):
        '''Lifeng Li JOSA A 13 1024 eq 7
            Interface t-matrix  
            Describing the boundary condition field jump/continuity
            
            For s-polarization,  the proper modes to choose as u,d modes are for the E-field. 
            The parallel component of E is  nicely continuous.
            the Hpar = k x E brings in a second continuity condition, now invoking kz
            
            For p-polarization,  the proper modes to choose as u,d modes are for the H-field. 
            The parallel component of H is  nicely continuous.        
            # the Dpar = k x H brings in the lower row as kz / eps for the 2nd continuity equation
        
            Input:  n1 and n2 complex refractive indices of layer 1 and 2
                    k0   = w/c free space wavenumber (real)
                    kpar = parallel momentum. Conserved quantity, so equal in both layers 
                    sorp = Boolean True for s.
            Output: 4 elements of the t-matrix (this outputformat: to enable numba jit )
        '''
            
        kz2=kz(n2,k0,kpar)*(1.0*sorp+(not sorp)*1.0/(n2*n2))
        kz1=kz(n1,k0,kpar)*(1.0*sorp+(not sorp)*1.0/(n1*n1))
    
        # #Lifeng Eq. 2a  
        # W2=np.array([[1.0+0.0j, 1.0+0.0j],[kz2, -kz2]])
        # W2=np.linalg.inv(W2)
        # W1=np.array([[1.0+0.0j, 1.0+0.0j],[kz1, -kz1]])
        # t=np.dot(W2,W1)
        # return t[0][0],t[0][1],t[1][0],t[1][1]
        x=0.5*kz1/kz2
        t00=0.5+0.0j+x
        t01=0.5+0.0j-x
        t10=0.5+0.0j-x
        t11=0.5+0.0j+x
        return t00,t01,t10,t11
        
        


@jit(nopython=True,cache=cachechoice,fastmath=fastmathchoice)
def interfaces_sp(n1,n2,k0,kpar,sorp) :
        ''' Lifeng JOSA A 13 1024 Equation 14a
            interface s matrix from interface t matrix
            Input:  n1 and n2 complex refractive indices of layer 1 and 2
                    k0   = w/c free space wavenumber (real)
                    kpar = parallel momentum. Conserved quantity, so equal in both layers 
                    sorp = Boolean True for s, and False for p polarization.
            Output: 4 elements  (this outputformat: to enable numba jit )
        '''
 
        t11,t12,t21,t22=layer_tsp(n1,n2,k0,kpar,sorp)  
        
        return t11-t12*t21/t22, t12/t22,-t21/t22, 1.0/t22
             


# S matrix for a layer, as given by Lifeng Li
@jit(nopython=True,cache=cachechoice,fastmath=fastmathchoice)
def layer_ssp(n1,n2,k0,kpar,d,sorp) :
    ''' Lifeng JOSA A 13 1024 Equation 13
        Layer s matrix from 
        - Interface s matrix
        - propagation phase
        This layer s matrix takes through layer 1, and to the "2" side of the 1-to-2 interface
        
        Input:  n1 and n2 complex refractive indices of layer 1 and 2
                k0   = w/c free space wavenumber (real)
                kpar = parallel momentum. Conserved quantity, so equal in both layers 
                d  =  thickness of layer n1
                kpar = parallel momentum. Conserved quantity, so equal in both layers 
                sorp = Boolean True for s, and False for p polarization.
        Output: 4 elements  (this outputformat: to enable numba jit )
    '''

    kzz=kz(n1,k0,kpar)
    ezd= np.exp(1.0j*kzz*d) # propagation phase for layer of index n1 and thickness d  
    s11,s12,s21,s22=interfaces_sp(n1,n2,k0,kpar,sorp)
    return s11*ezd,s12,s21*ezd*ezd,s22*ezd
    


# Actual S-matrix recursion 
@jit(nopython=True,cache=cachechoice,fastmath=fastmathchoice)    
def RecurSsp(k0,kpar,n2,n3,ndlist,sorp):
    ''' Lifeng JOSA A 13 1024 Equation 25a,  
        Recursive building of S matrix of full stack
        
        Input:   k0   = w/c free space wavenumber (real scalar)
                 kpar = parallel momentum. Conserved quantity, so equal in both layers 
                 n2 and n3 refractive indices of half-infinite sub and superstrate(could be complex)
                 ndlist for N layers, list of 2N numbers: indices (complex) and thicknesses (real)
                 sorp = Boolean True for s, and False for p polarization.
        Output: returns 4 numbers that are te S matrix elements
    '''
    # Initialize with identity 
    S11=1.0+0.0j+0*kpar
    S12=0.0+0.0j+0*kpar
    S21=0.0+0.0j+0*kpar
    S22=1.0+0.0j+0*kpar

    nlist=np.append(n2,np.append(ndlist[:,0],n3)) # to start the recurrence with crossing the first interface, zero thickness of the substrate
    dlist=np.append(0.0,ndlist[:,1])
    nd=(dlist.shape)[0] # counts the number of entries to recurse through

     
    for m in range(nd) : 
        s11,s12,s21,s22=layer_ssp(nlist[m],nlist[m+1],k0,kpar,dlist[m],sorp) # get layer S matrix
        S11,S12,S21,S22=dotstar(S11,S12,S21,S22,s11,s12,s21,s22) #call Redheffer product
    return S11,S12,S21,S22


###  Routines specific for incidence from one side (n2):  
    
@jit(nopython=True,cache=cachechoice,fastmath=fastmathchoice)
def rt_p(k0,kpar,n2,n3,ndlist):
    ''' 
    Reflection and transmission amplitude of a stack for p-polariz on basis of S matrix algorithm
    Input:   k0   = w/c free space wavenumber (real scalar)
             kpar = parallel momentum. Conserved quantity, so equal in both layers 
             n2 and n3 refractive indices of half-infinite sub and superstrate(could be complex)
             ndlist for N layers, list of 2N numbers: indices (complex) and thicknesses (real)
    Output: returns r and t as an np.array.Assumed inputside is n2
    
    Code is identical to rt_s, except for the fact that the natural output is H, which then needs to cast to E
    '''
    sorp=False # meaning: we do p-polarization
    # result that comes out directly is for  H-field
    S11,S12,S21,S22=RecurSsp(k0,kpar,n2,n3,ndlist,sorp)
   

    # .. and so needs to be recast for E-field
    r=-S21+0.0j
    t=S11*n2/n3+0.0j
    
    return [r,t]
    #return np.array([r,t],dtype=np.cdouble)  



@jit(nopython=True,cache=cachechoice,fastmath=fastmathchoice)
def rt_s(k0,kpar,n2,n3,ndlist):
    ''' 
    Reflection and transmission amplitude of a stack for s-polariz on basis of S matrix algorithm
    Input:   k0   = w/c free space wavenumber (real scalar)
             kpar = parallel momentum. Conserved quantity, so equal in both layers 
             n2 and n3 refractive indices of half-infinite sub and superstrate(could be complex)
             ndlist for N layers, list of 2N numbers: indices (complex) and thicknesses (real)
    Output: returns r and t as an np.array.Assumed inputside is n2
    '''
    sorp=True # meaning, we do s polarization
    S11,S12,S21,S22=RecurSsp(k0,kpar,n2,n3,ndlist,sorp)
    
    t=S11
    r=S21 
    return [r,t]
    #return np.array([r,t],dtype=np.cdouble) 


 
@jit(nopython=True,cache=cachechoice,fastmath=fastmathchoice)
def udcoef(k0,kpar,nin,nout,ndlist,nlay,sorp):
    ''' 
    Given a stack, return the coefficients for the up and down propagating E-field in a given target layer, and assuming s-polarization
    Procedure is:
        1. Get S matrix from recursion
        2. Solve r and t field, given incidence from input side
        3. Use partial S matrix to zoom in on intermediate layer and get coefficients
        
    Input:   k0   = w/c free space wavenumber (real scalar)
             kpar = parallel momentum. Conserved quantity, so equal in both layers 
             nin and noutrefractive indices of half-infinite sub and superstrate. 
             nin is the input side, assumed adjacent to first element in ndlist
             ndlist for N layers, list of 2N numbers: indices (complex) and thicknesses (real)
    Output: returns u,d in target layer nlay 
    '''
    #  1. Solve for the overal reflection transmission problem as follows
    S11,S12,S21,S22=RecurSsp(k0,kpar,nin,nout,ndlist,sorp)
    
    # 2. input side, incoming and outgoing. u,d coefficients are for out of plane Efield for s-polariz
    u0=1.0+0.0j+0.0*S11
    d0=S21
    
    
    #  now apply the partial S matrix to calculate to layer nlay
    
    #nlay = 0 codes for the input side
    #nlay = 1,2,3,n  codes for the layers
    #nlay = n+1 codes for the outputside
    
    numlay=ndlist.shape[0]
    if nlay==0 :  #reflection side
        u,d=u0,d0
    
    elif nlay > numlay :  # transmission side
        u,d= S11,0.0*S11
    else :      
        n3= ndlist[nlay-1,0] # intermediate layer
        ndlist=ndlist[0:nlay-1]

        S11,S12,S21,S22=RecurSsp(k0,kpar,nin,n3,ndlist,sorp) # partial S matrix from front to intermediate layer
        d=(d0-S21*u0)/S22   # Expand [u,d0]=S [u0,d] (lifeng li Eq 17a for layer nlay). Invert the 2nd eq. to express d
        u=S11*u0+S12*d      # note how this uses the d from previous step.

    return u,d





 
@jit(nopython=True,cache=cachechoice,fastmath=fastmathchoice)
def Epfromudcoef(k0,kpar,nin,nout,ndlist,nlay):
    ''' 
    Given a stack, return the coefficients for the up and down propagating E-field in a 
    given target layer, and assuming p-polarization
     Input:   k0   = w/c free space wavenumber (real scalar)
             kpar = parallel momentum. Conserved quantity, so equal in both layers 
             nin and noutrefractive indices of half-infinite sub and superstrate. 
             nin is the input side, assumed adjacent to first element in ndlist
             ndlist for N layers, list of 2N numbers: indices (complex) and thicknesses (real)
    Output: returns u,d in target layer nlay . 
     
    The output is translated already to  E-field, although intrinsically p-polarization is H-field formulated
    '''
    sorp=False
    Huy,Hdy=udcoef(k0,kpar,nin,nout,ndlist,nlay,sorp)  # see below. This returns H-field coefficients
    
    nlist=np.append(nin,np.append(ndlist[:,0],nout))
    
    nslab=nlist[nlay]  #nlay=0: input side. Max nlay is output side
                 
    
    # E follows from H through 
    # Curl H = dD/dt
    # ikxH = i omega eps E 
    #  E = k x H /(omega eps) 
    # Additional factor nin so that the field is for normalized input E-field strength
    
    kzs=kz(nslab,k0,kpar)
    Eux=  Huy*kzs/(k0*nslab*nslab)*nin
    Euz= -Huy*kpar/(k0*nslab*nslab)*nin
    Edx= -Hdy*kzs/(k0*nslab*nslab)*nin
    Edz= -Hdy*kpar/(k0*nslab*nslab)*nin    
    
    
    
    return [Eux,Euz,Huy],[Edx,Edz,Hdy] # meaning Ex, Ez and Hy


 
@jit(nopython=True,cache=cachechoice,fastmath=fastmathchoice)
def Esfromudcoef(k0,kpar,nin,nout,ndlist,nlay):
    ''' 
    Given a stack, return the coefficients for the up and down propagating E-field in a 
    given target layer, and assuming s-polarization
    Input:   k0   = w/c free space wavenumber (real scalar)
             kpar = parallel momentum. Conserved quantity, so equal in both layers 
             nin and noutrefractive indices of half-infinite sub and superstrate. 
             nin is the input side, assumed adjacent to first element in ndlist
             ndlist for N layers, list of 2N numbers: indices (complex) and thicknesses (real)
    Output: returns u,d in target layer nlay . 
     
    The output is for E-field
    '''
    sorp=True
    Euy,Edy=udcoef(k0,kpar,nin,nout,ndlist,nlay,sorp)  # see below. This in itself already returns E-field coefficients
   
    ## H follows from E through
    ## Curl E = -d B / dt
    ## ik x E=-i omega mu H
    ## H = - k x E / (omega mu)
    ##     
    nlist=np.append(nin,np.append(ndlist[:,0],nout))
    
    nslab=nlist[nlay]  #nlay=0: input side. Max nlay is output side
    
    
    # suppose that 
    #   k points in the plus z direction,
    #   E points in the +y direction
    #    
    # Additional factor nin so that the field is for normalized input E-field strength
    
    kzs=kz(nslab,k0,kpar)
    
    Hux=-Euy*kzs/k0/nin
    Huz= Euy*kpar/k0/nin
    Hdx= Edy*kzs/k0/nin
    Hdz= Edy*kpar/k0/nin
    
    
    return [Euy,Hux,Huz],[Edy,Hdx,Hdz]


def SP_FieldsAtZ(k0,kpar,nin,nout,ndlist,nlay,zz,fastmath=fastmathchoice):
    # given a location zz, return the instantaneous field,
    # assuming plane wave in at kpar, from medium nin
    
    EHus,EHds=Esfromudcoef(k0,kpar,nin,nout,ndlist,nlay)
    EHup,EHdp=Epfromudcoef(k0,kpar,nin,nout,ndlist,nlay)
    nlist=np.append(nin,np.append(ndlist[:,0],nout))
    nslab=nlist[nlay]
    kzs=kz(nslab,k0,kpar)
    
    
    
    uphase=np.exp(1.0j*np.outer(kzs,zz))
    dphase=1./uphase
 
  
    EHs=[]
    EHp=[]
    for m in range(3): # first + adds elements to array, is not mathematical summation
        EHs=EHs+[(np.outer(EHus[m],1+0.0*zz)*uphase +np.outer(EHds[m],1+0.0*zz)*dphase)]
        EHp=EHp+[(np.outer(EHup[m],1+0.0*zz)*uphase +np.outer(EHdp[m],1+0.0*zz)*dphase)]
               
   
    return np.array(EHs), np.array(EHp)


