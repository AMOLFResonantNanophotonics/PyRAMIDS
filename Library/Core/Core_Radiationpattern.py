#!/usr/bin/env python3
 
"""
    This file contains all the far-field related scripts for electromagnetic sources embedded in stratfied system.  This includes:
    
    - Far field in  halfspaces  on basis of asymptotic dyadic Green function. The use is:
        
        (1) We assume a general electric and magnetic dipole. Cartesian coordinates are used
            for pu=[px,py,pz] mu = [mx,my,mz]. z is perpendicular to the stack. Note that
            For arbitrary dipoles, the relative magnitude and phase matter - coherent addition!
            
            
        (2) The fields are requested for lists of theta,phi coordinates.  
        
        (3) The output is in s- and p-polarized basis, and reports electric field.
        
        (4) The problem spec in terms of geometry is:
                k0=w/c          free space wavenumber
                h, dslab,nslab  The dipole is at height h in slab of thickness dslab, and index nslab
                n2, n3          lower and upper half infinite substrate. 
                nd2list,nd3list stack between the slab and halfinfinite space, counting from slab                          
        
    - Fluxes derived from fields
    - Integration of fluxes to obtain total radiated power. Available for canonical dipole orientations only

    First, this script provides all the routines in "slab-centric" coordinates.
    Appended are wrapper routines to cast the parameter-input in user-friendly form
 


@author: dpal,fkoenderink
"""
#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%%

import numpy as np
import scipy.integrate as integrate


from Library.Core import Core_Smatrix as RT
from Library.Util import Util_argumentchecker as check

from numba import jit, config

#config.DISABLE_JIT = True
config.DISABLE_JIT = False
cachechoice = True

'''  Field calculation  '''
@jit(nopython=True,cache=cachechoice)
def PEdown(pu,mu,the,phi,k0,h,dslab,nslab,n2,n3,nd2list,nd3list,rdipxy):
    ''' Far field on basis of asymptotic dyadic Green function. The use is:
        
        (1) We assume a general electric and magnetic dipole. Cartesian coordinates are used
            for pu=[px,py,pz] mu = [mx,my,mz]. z is perpendicular to the stack. Note that
            For arbitrary dipoles, the relative magnitude and phase matter - coherent addition!
            
            
        (2) The fields are requested for lists of theta,phi coordinates. This routine is for the lower  half space, of index n2

        (3) The output is in s- and p-polarized basis, and reports electric field.
        
        (4) The problem spec in terms of geometry is:
                k0=w/c          free space wavenumber
                h, dslab,nslab  The dipole is at height h in slab of thickness dslab, and index nslab
                n2, n3          lower and upper half infinite substrate. 
                nd2list,nd3list stack between the slab and halfinfinite space, counting from slab                          
    '''

    kpar=k0*n2*np.sin(the)          # parallel momentum, common to all layers
    kzslab=RT.kz(nslab,k0,kpar)     # z-component of k in the slab, and in the lower half space of index n2
    kz2=RT.kz(n2,k0,kpar)
    
    
    #propagation phase factors
    ed=np.exp(1.0j*kzslab*dslab) 
    eh=np.exp(1.0j*kzslab*h)
    P3 = ed 
    P4 = ed*ed/eh
    P5 = eh
    
    
    # reflection and transmission coefficients from the source towards the lowerhalf infinite space
    rtp2 = RT.rt_p(k0, kpar, nslab, n2, nd2list); 
    rp2 = rtp2[0];tp2=rtp2[1]
    rts2 = RT.rt_s(k0, kpar, nslab, n2, nd2list); 
    rs2 = rts2[0];ts2=rts2[1]
 
    # reflection coefficients from the source towards the upper  half infinite space
    rp3 = RT.rt_p(k0, kpar, nslab, n3, nd3list)[0]
    rs3 = RT.rt_s(k0, kpar, nslab, n3, nd3list)[0]
   
    #auxilliary definitions,  labelled p and s. Note the geometric-series denominator.
    prep2 = tp2/(1.0 - rp3*rp2*P3*P3)
    pres2 = ts2/(1.0 - rs3*rs2*P3*P3)

    # aux quantities to account for multiple reflections and final transmisions
    # "downxy" labels meaning:
           # this code is for down (towards n2)
           # x=p,s stands for polarization label of Fresnel coefficient that enters
           # y=p,m labels plus and minus signs
    gtdownpp = prep2*(P5 - P4*rp3) 
    gtdownpm = prep2*(P5 + P4*rp3)
    gtdownsp = pres2*(P5 + P4*rs3)
    gtdownsm = pres2*(P5 - P4*rs3)
    
   #Electric field from electric dipole
   #The structure of these equations is exactly as in Novotny & Hecht (1st edition)
   #Equation 10.32 and equation 10.36, but transmission coefficients have the full multiple reflection series in the denominator
    Edownsfromp=np.sqrt(0.0j+n2/nslab)*kz2/kzslab*gtdownsp*(pu[0]*np.sin(phi)+pu[1]*np.cos(phi))
    Edownpfromp=np.sqrt(0.0j+n2/nslab)*n2/nslab*(-np.sin(the)*kz2/kzslab*gtdownpp*pu[2]+np.cos(the)
          *gtdownpm*(pu[0]*np.cos(phi) - pu[1]*np.sin(phi)))



    #Electric field from magnetic dipole
    #The structure from the equations is identical, however, there are the following changes:
    #    (1) Calculating H-field from dipole m is exactly like calculating E from p
    #        However swapping rp and rs
    #    (2) Next, we need E, from H. This swaps s and p, and adds minus sign
    # 
    #Hence read equation first: RHS to the right of the np.sqrt Verify swap of p and s
    #Then read step (2). Note the minus sign,  and the p,s swap in the lefthandside       

    Edownpfromm=-np.sqrt(0.0j+n2/nslab)*kz2/kzslab*gtdownpp*(-mu[0]*np.sin(phi)+mu[1]*np.cos(phi))
    Edownsfromm=-np.sqrt(0.0j+n2/nslab)*n2/nslab*(-np.sin(the)*kz2/kzslab*gtdownsp*mu[2]+np.cos(the)
          *gtdownsm*(mu[0]*np.cos(phi) + mu[1]*np.sin(phi)))       

     
    #coherent addition leads to total electric field in s and p    
    Edowns = Edownsfromp + Edownsfromm
    Edownp = Edownpfromp + Edownpfromm      
    
    phasefac=np.exp(1.0j*kpar*(rdipxy[0]*np.cos(phi)+rdipxy[1]*np.sin(phi)))

#    prefac=np.real(3.0/(8.0*np.pi))/np.sum(np.abs(pu)**2+np.abs(mu)**2)*nslab #normalization factor - power of a free space dipole
    prefac=np.real(3.0/(8.0*np.pi))*nslab #normalization factor - power of a free space dipole
    P=prefac*(np.abs(Edowns)**2+np.abs(Edownp)**2) #metric for flux

    return P,Edowns*phasefac, Edownp*phasefac

@jit(nopython=True,cache=cachechoice)
def PEup(pu,mu,the,phi,k0,h,dslab,nslab,n2,n3,nd2list,nd3list,rdipxy):
    ''' Far field on basis of asymptotic dyadic Green function. The use is:
        
        (1) We assume a general electric and magnetic dipole. Cartesian coordinates are used
            for pu=[px,py,pz] mu = [mx,my,mz]. z is perpendicular to the stack. Note that
            For arbitrary dipoles, the relative magnitude and phase matter - coherent addition!
            
            
        (2) The fields are requested for lists of theta,phi coordinates. This routine is for the upper half space, of index n3

        (3) The output is in s- and p-polarized basis, and reports electric field.
        
        (4) The problem spec in terms of geometry is:
                k0=w/c          free space wavenumber
                h, dslab,nslab  The dipole is at height h in slab of thickness dslab, and index nslab
                n2, n3          lower and upper half infinite substrate. 
                nd2list,nd3list stack between the slab and halfinfinite space, counting from slab                          
    '''
        
    
    kpar=k0*n3*np.sin(the)   # parallel momentum, common to all layers
    kzslab=RT.kz(nslab,k0,kpar) # z-component of k in the slab, and in the upper half infinite medium (n3)
    kz3=RT.kz(n3,k0,kpar)
    
    
    #required propagation phases
    ed=np.exp(1.0j*kzslab*dslab) 
    eh=np.exp(1.0j*kzslab*h)
    
    P1 = ed/eh
    P2 = ed*eh
    P3 = ed
    
#    P1 = np.exp(1.0j*kzslab*(dslab-h))
#    P2 = np.exp(1.0j*kzslab*(dslab+h))
#    P3 = np.exp(1.0j*kzslab*(dslab))

    # reflection and transmission coefficients from the source towards the upper half infinite space
    rtp3 = RT.rt_p(k0, kpar, nslab, n3, nd3list) 
    rp3 = rtp3[0];tp3=rtp3[1]
    rts3 = RT.rt_s(k0, kpar, nslab, n3, nd3list)
    rs3 = rts3[0];ts3=rts3[1]

    # reflection coefficients from the source towards the lower half infinite space
    rp2 = RT.rt_p(k0, kpar, nslab, n2, nd2list)[0]
    rs2 = RT.rt_s(k0, kpar, nslab, n2, nd2list)[0]

    #auxilliary definitions,  labelled p and s. Note the geometric-series denominator.
    prep3 = tp3/(1.0 - rp3*rp2*P3*P3)  
    pres3 = ts3/(1.0 - rs3*rs2*P3*P3)
    
    # aux quantities to account for multiple reflections and final transmisions
    # "upxy" labels meaning:
           # this code is for up (towards n3)
           # x=p,s stands for polarization label of Fresnel coefficient that enters
           # y=p,m labels plus and minus signs
    gtuppp = prep3*(P1 - P2*rp2) 
    gtuppm = prep3*(P1 + P2*rp2)
    gtupsp = pres3*(P1 + P2*rs2) 
    gtupsm = pres3*(P1 - P2*rs2) 
    
    #Electric field from electric dipole
    #The structure of these equations is exactly as in Novotny & Hecht (1st edition)
    #Equation 10.32 and equation 10.36, bbut transmission coefficients have the full multiple reflection series in the denominator
    Eupsfromp=np.sqrt(0.0j+n3/nslab)*kz3/kzslab*gtupsp*(-pu[0]*np.sin(phi)+pu[1]*np.cos(phi))
    Euppfromp=np.sqrt(0.0j+n3/nslab)*n3/nslab*(-np.sin(the)*kz3/kzslab*gtuppp*pu[2]+np.cos(the)
          *gtuppm*(pu[0]*np.cos(phi) + pu[1]*np.sin(phi)))
    
    
    #Electric field from magnetic dipole
    #The structure from the equations is identical, however, there are the following changes:
    #    (1) Calculating H-field from dipole m is exactly like calculating E from p
    #        However swapping rp and rs
    #    (2) Next, we need E, from H. This swaps s and p, and adds minus sign
    # 
    #Hence read equation first: RHS to the right of the np.sqrt Verify swap of p and s
    #Then read step (2). Note the minus sign,  and the p,s swap in the lefthandside       

    Euppfromm=-np.sqrt(0.0j+n3/nslab)*kz3/kzslab*gtuppp*(mu[0]*np.sin(phi)+mu[1]*np.cos(phi))
    Eupsfromm=-np.sqrt(0.0j+n3/nslab)*n3/nslab*(-np.sin(the)*kz3/kzslab*gtupsp*mu[2]+np.cos(the)
          *gtupsm*(mu[0]*np.cos(phi) - mu[1]*np.sin(phi)))    
    
    
    
    #coherent addition leads to total electric field in s and p
    Eups = Eupsfromp + Eupsfromm 
    Eupp = Euppfromp + Euppfromm 
    
    
#    prefac=np.real(3.0/(8.0*np.pi))/np.sum(np.abs(pu)**2+np.abs(mu)**2)*nslab #normalization factor - power of a free space dipole
    
    prefac=np.real(3.0/(8.0*np.pi))*nslab #normalization factor - power of a free space dipole
    P=prefac*(np.abs(Eups)**2+np.abs(Eupp)**2) #metric for flux

    phasefac=np.exp(1.0j*kpar*(rdipxy[0]*np.cos(phi)+rdipxy[1]*np.sin(phi)))
                
    return P,Eups*phasefac, Eupp*phasefac




''' Flux integrals for favorite dipole orientations to get total radiatedpowers in each half-space

The rationale is 

(1) that for LDOS one typically wants to know radiative LDOS (total integrated flux) just for 
parallel and perpendicular electric and magnetic dipoles, as well as for the cross term.

(2) for these specific orientations you can avoid a full integral over phi.  Even if the phi-dependence is generally somewhat nontrivial for arbitrary orientations, 
for perpendicular dipoles you simply have cylindrical symmetry, and for parallel dipoles you simply have a cos^2phi and sin^2phi term, so that the phi-integral is analytical
 
Therefore: special orientations boil down to just theta-integrals

the rationale throughout for the integrals is then:
    
    (A) For z-oriented dipoles simply multiply  Pup with 2*pi (for phi integral)  sintheta (for Jacobian) and nslab (normalization factor)
    (B) For x-oriented instead (i) the factor is pi * sin(theta) for Jacobian and phi integral, and you need to account for the sliced along, and crossed to, the dipole orientaiton
        This is effectuyated by adding [1.,0,0] and [0.,1.,0.] terms
        
    (C) Chiral dipoles have also a correction Pcross that either adds or subtract to total radiated power depending on handedness
        Here the pseudochirality that actually matters is px my* versus py mx*
        You need both elements to be nonzero, and have the right quarter wave phase slip.
        
        This explains the Pcross terms 
 '''
 


#@jit(nopython=True,cache=cachechoice)
def PupintegrandPuMu(pu,mu,the,k0,h,dslab,nslab,n2,n3,nd2list,nd3list):
    #as function of phi, any radiation pattern is a quadratic form on the unit circle
    #this is exactly integrated by any DFT with sufficient points (>=3)
    #for DFT this means: (A) equal weight, (B) equal spacing, (C) don't double count 0 and 2pi
    
    MM=3
    fac=2.0*np.pi/MM
    jacobian=np.abs(np.sin(the))*fac
 
    
    P=PEup(pu,mu,the,0.0,k0,h,dslab,nslab,n2,n3,nd2list,nd3list,np.array([0.0,0.0]))[0] 
    
    for m in range(1,MM) :
        phi=2.0*np.pi*m/MM
        P=P+PEup(pu,mu,the,phi,k0,h,dslab,nslab,n2,n3,nd2list,nd3list,np.array([0,0]))[0]
        
  
    P=jacobian*P
    return np.transpose(P)

#@jit(nopython=True,cache=cachechoice)
def PdownintegrandPuMu(pu,mu,the,k0,h,dslab,nslab,n2,n3,nd2list,nd3list):
    #as function of phi, any radiation pattern is a quadratic form on the unit circle
    #this is exactly integrated by any DFT with sufficient points (>=3)
    #for DFT this means: (A) equal weight, (B) equal spacing, (C) don't double count 0 and 2pi
    MM=3
    fac=2.0*np.pi/MM
  
    P=PEdown(pu,mu,the,0.0,k0,h,dslab,nslab,n2,n3,nd2list,nd3list,np.array([0.0,0.0]))[0]
    for m in range(1,MM) : 
        phi=2.0*np.pi*m/MM 
        P=P+PEdown(pu,mu,the,phi,k0,h,dslab,nslab,n2,n3,nd2list,nd3list,np.array([0.0,0.0]))[0]    
    
    jacobian=np.abs(np.sin(the))*fac
    P=jacobian*P
                 
    return np.transpose(P)



''' run the adaptive integration routine quad_vec from scipy.integrate to do the theta-integrals''' 

def TotalRadiatedfromlayerPuMu(pu,mu,k0,h,dslab,nslab,n2,n3,nd2list,nd3list):
    
    if check.checkRealPositive(n3) : #up is into n3
        up=  integrate.quad_vec(lambda x: PupintegrandPuMu(pu,mu,x,k0,h,dslab,nslab,n2,n3,nd2list,nd3list),0.0,np.pi/2.0,epsabs=1e-03,epsrel=1e-03)[0]
    else:
        up=np.zeros([1,len(h)])
    if check.checkRealPositive(n2) :
        down=integrate.quad_vec(lambda x: PdownintegrandPuMu(pu,mu,x,k0,h,dslab,nslab,n2,n3,nd2list,nd3list),np.pi/2.0,np.pi,epsabs=1e-03,epsrel=1e-03)[0] 
    else:
        down=np.zeros([1,len(h)])
    
     
    return np.vstack((up+down,up, down))

