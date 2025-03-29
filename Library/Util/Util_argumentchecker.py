#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dpal,fkoenderink

Checks all the user inputs for its validatiy, data structure, and also molds them it relevant correct form

"""

#%%

import numpy as np
 

def checkRealPositive(nrefr):
    #retruns False unless nrefr is positive and real
    success=True
    if not np.isreal(nrefr):
        success=False
    else :
        if nrefr <= 0.0:
            success=False
    return success

def checkThetaAndPhi(the,phi):
    #if either theta or phi are scalar: turn into lists
    #if either theta or phi are entered as lists exceeding 1 in dimension, throw error
    #if only theta or phi is a list, make the other a list of equal size and constant value
    #if both are lists, meshgrid and flatten.
    #if theta or phi are not real, throw error
  
    # if np.isscalar(the) and np.isscalar(phi): #bona fide input in principle, but should be cast to a list
    #     the=[the]
    #     phi=[phi]
    
    the=np.array(the) #just in case the input were non numpy lists
    phi=np.array(phi)
    
    if len(the.shape)==0:
        the=np.tile(np.array(the)[np.newaxis],(1))
    if len(phi.shape)==0:
        phi=np.tile(np.array(phi)[np.newaxis],(1))
        
    
    shape=the.shape 
    if np.size(the)*np.size(phi)==0 :
        raise ValueError('Either theta or phi is specified as empty by user')
  
    if np.isreal(the).prod()*np.isreal(phi).prod()==0:
        raise ValueError('User has specified complex-valued angles theta or phi..')
    

    if np.ndim(the)>1 or np.ndim(phi) >1 :
        raise ValueError('Theta and/or phi can be at most 1D arrays')   
    if np.size(the)==1:
            the=((the.flatten())[0])*np.ones(np.shape(phi))
            shape=np.shape(the)
    if np.size(phi)==1:
            phi=((phi.flatten())[0])*np.ones(np.shape(the))    
            shape=np.shape(the)
    if np.size(the)!=np.size(phi):
        the,phi=np.meshgrid(the,phi)
        shape=np.shape(the)
        #print('Using meshgrid to generate (flattened) theta-phi grid ')
        the=the.flatten()
        phi=phi.flatten()
    return the,phi,shape # export shape, so that radiation patterns can be reshaped

def checkPuamMu(pu,mu):
    pu=checkDip(pu)
    mu=checkDip(mu)
    return pu,mu

def checkDip(pu):
    message='Dipole moment must be a 3-vector'
    if np.isscalar(pu):
        raise ValueError(message)
    if np.ndim(pu)>1:
        raise ValueError(message)
    if np.size(pu)!=3:
        raise ValueError(message)
    return np.array(pu)
        

def checkFullk0znd(k0,z,nstack,dstack):
    k0=checkk0(k0)
    z=checkz(z)
    nstack,dstack=checkStackdefinition(nstack,dstack)
    
    return k0,z,nstack,dstack

def checkStackdefinition(nstack,dstack):

    dstack=checkdstackonly(dstack)
    if np.isscalar(nstack):
        raise ValueError('Found a single entry for nstack. Need at least 2 (sub and superstrate refractive indices).')        
    
    if np.ndim(nstack)>1:
        raise ValueError('Stack specification requires two 1D lists. nstack is higher-D. Do not input higher dimensions')        
    if (len(nstack)-len(dstack))!=2:
        raise ValueError('The stack definition is inconsistent:\nThe refractive index list must have 2 more entries than the layer thickness list to accomodate the sub and superstrate.')
    if not all(np.imag(nstack)>=0):
        print('Fingers crossed. Encountered refr. index with negative im. part (gain, not loss), as input.')

   
    return np.array(nstack),np.array(dstack) #just in case they were not yet np arrays

 
def checkdstackonly(dstack):
    if np.isscalar(dstack):
        dstack=np.array([dstack])
    if np.ndim(dstack)>1:
        raise ValueError('Stack specification dstack can not be higher-than-1D') 
    if not all(np.isreal(i) for i in dstack):
        raise ValueError('Encountered an imaginary part for layer thickness in dstack as user input.')    
    else :
        if not all(i>=0. for i in dstack):
            raise ValueError('Encountered a negative layer thickness in dstack as user input.')    
    return np.array(dstack) #just in case they were not yet np arrays



def checkequaldimension(x,y):
    x=np.array(x)
    y=np.array(y)
    return (x.shape == y.shape)
    

def checkBFPinput(x,y,s,p):
    a=checkequaldimension(x,y)
    b=checkequaldimension(s,p)
    c=checkequaldimension(x,s)
    if (a and b and c) == False:
        raise ValueError('Incompatible sizes in check BFP input')

    return


def checkk0(k0): 
    if not np.isscalar(k0):
        if np.size(k0)>1 :
            raise ValueError('Found list of k0 (free space wavenum). Can not vectorize over k0')
        else :
            print('Warning: Found list of length 1 for k0 as user input -cast to scalar')
            k0=np.array(k0).flatten()[0]            
    if not np.isreal(k0):
        raise ValueError('Encountered complex-valued free-space wavenumber as input. k0 must be real')
    else :
        if not k0>0:    
            raise ValueError('Encountered negative-valued free-space wavenumber as input. k0 must be positive')             
    return k0

def checkz(z):
    if np.isscalar(z):
        z=np.array([z])
    if z.shape==():
        z=np.array([z])
    if np.size(z)==0:
        raise ValueError('Empty z-list of dipole coordinates')
    if np.ndim(z)>1:
        print('Warning: found higher-dimensional array of z-lists. Can only vectorize LDOS/radpat code over 1d lists for z')
        print('Fingers crossed - will flatten your list')
        z=z.flatten()
    if np.isreal(z).prod() !=1:
        raise ValueError('Check user input for z (dipole position list. I found complex-valued  entries')
    
    return np.array(z) 

def checkr(r):
    r=np.array(r)
    if r.shape[0]!=3:
        r=np.transpose(r) # fingers crossed that tranposing helps ... 
    if r.shape[0]!=3:
        raise ValueError('rs must be a 3 vector or an array of 3 vectors')
        
    if (np.isreal(r)==False).any():
        raise ValueError('position coordinates must be real')
    return r
    

def checkGreenrsourcerdetect(rdetect,rsource):
    rdetect=checkr(rdetect)
    rsource=checkr(rsource) 
    
    if rsource.ndim==1 and rdetect.ndim==1 :
    
        dx=np.array([rdetect[0]-rsource[0]])
        dy=np.array([rdetect[1]-rsource[1]])
        zsource=np.array([rsource[2]])
        zdetect=np.array([rdetect[2]])
        
    elif rsource.ndim==2 and rdetect.ndim==1 :
        dx = rdetect[0]-rsource[0,:]
        dy = rdetect[1]-rsource[1,:]
        zsource=rsource[2,:]
        zdetect=0*zsource+rdetect[2]
       
    elif rsource.ndim==1 and rdetect.ndim==2 :
        dx = rdetect[0,:]-rsource[0]
        dy = rdetect[1,:]-rsource[1]
        zdetect=rdetect[2,:]
        zsource=0*zdetect+rsource[2]

    elif rsource.ndim==2 and rdetect.ndim==2 :
        
        if rsource.shape[1]!=rdetect.shape[1] :
            raise ValueError('rdetect and rsource have incompatible sizes')
            
        else :
            dx = rdetect[0,:]-rsource[0,:]
            dy = rdetect[1,:]-rsource[1,:]
            zdetect=rdetect[2,:]        
            zsource=rsource[2,:]


         
    
    if np.isreal(dx).prod() !=1:
        raise ValueError('Check user input for x - complex valued entries')
    if np.isreal(dy).prod() !=1:
        raise ValueError('Check user input for y - complex valued entries')
    if np.isreal(zdetect).prod() !=1:
        raise ValueError('Check user input for z1 - complex valued entries')
    if np.isreal(zsource).prod() !=1:
        raise ValueError('Check user input for z0 - complex valued entries')
    

    R=np.sqrt(dx*dx+dy*dy)
    phi=np.angle(dx+1.0j*dy)

        
    return dx,dy,zdetect,zsource,R,phi        
            

    

            
def checkKparlist(kparlist):
    if np.isscalar(kparlist):
        kparlist=np.array([kparlist])
    if np.size(kparlist)==0:
        raise ValueError('Empty list of kpar - nothing to do')
    if np.ndim(kparlist)>1:
        raise ValueError('kpar-list higher-D than 1D-array..') 
    return np.array(kparlist)

