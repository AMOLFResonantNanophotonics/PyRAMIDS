
"""
#All user-level routines for LDOS evaluation in layered stacks

#This file wraps core ImG LDOS routines in user-centric coordinates
#(first interface at z=0) and returns electric, magnetic, and crossed
#LDOS contributions at requested z positions.
#
#Conventions used throughout
#  - k0 = 2*pi/lambda_vac
#  - nstack = [n0, n1, n2, ..., n_{m-1}, n_m]
#  - dstack = [d1, d2, ..., d_{m-1}]
#  - LDOS core follows Amos and Barnes (PRB 55, 7249) nomenclature.
#
#Main user functions
#  - LDOS: canonical electric/magnetic/crossed LDOS channels
#  - LDOSatanyPandM: LDOS projected for arbitrary (p, m) dipole vectors
#  - LDOSintegrandplottrace: k_parallel-resolved LDOS integrands
#  - LDOSintegrandplotdispersion: dispersion map built from integrand traces

@author: dpal,fkoenderink
"""


#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%%

import numpy as np

from Library.Util import Util_argumentrewrapper as cc
from Library.Util import Util_argumentchecker as check

from Library.Core import Core_ImGLDOS as ImGL



''' Coordinate wrapped interface to core ImG
Routines above (Core folder) are slab centric i.e.,  consider a dipole at height z from slab boundary, where the slab might be layer m out of n=1..N layers
                The base routines then require to enumerate separately the slabs  m+1 ... N on one end of the slab of interest,
                and also the slabs 1 to m-1 in reverse order, counting away from the source
 
User-centric is: consider  a stratified system with first interface at z=0, and then  a set of finite layers of thickness d1,d2,d3, dN
               the source is at a coordinate z in the system
                
The routine below casts the LDOS function in user centric coordinates, using the imported helper function
''' 
def LDOS(k0,zlist,nstack,dstack) :
    """Compute canonical LDOS components at user-specified z positions.

    Intuition
    ---------
    This is the main wrapper for LDOS in user-centric coordinates. Points are
    grouped layer-by-layer so core slab-centric routines can be vectorized.

    Parameters
    ----------
    k0 : float
        Free-space wavenumber (`k0 = 2*pi/lambda_vac`).
    zlist : array-like
        Dipole z positions in user-centric coordinates.
    nstack, dstack : array-like
        Layer refractive indices and thicknesses.

    Returns
    -------
    tuple(ndarray, ndarray, ndarray, ndarray, ndarray)
        `(rhoE_par, rhoE_per, rhoM_par, rhoM_per, rhoC)`.
    """
 
    #type check the arguments
    k0,zlist,nstack,dstack=check.checkFullk0znd(k0,zlist,nstack,dstack)

    # classify the points z in terms of in which layer they reside
    lib=cc.pinpointdomain(zlist,dstack)
    unilib=list(set(lib))  # library of unique domains that are visited by zlist
    
    
    # allocate zero-arrays by value to hold the outcome
    outparfromE=np.zeros(np.shape(zlist))
    outperfromE=np.zeros(np.shape(zlist))
    outparfromH=np.zeros(np.shape(zlist))
    outperfromH=np.zeros(np.shape(zlist))
    outcrossfromC=np.zeros(np.shape(zlist))
   
    skip=False #in case z traverse an absorbing, metallic or gainy layer, avoid popping out an unphysical answer.  
    # loop over all the unique domains in unilib.
    # Reason for this approach: amos.LDOS is vectorized but only as long as zlist is in one slab only, and not out of it.
    for m in unilib :  # loop over all slab-spaces in which there are sample points
         indices=[i for i, x in enumerate(lib) if  x == m]
         zz,dslab,nslab,n2,n3,nd2list,nd3list=cc.providecoordinates(m,zlist[indices],nstack,dstack)
 
         if check.checkRealPositive(nslab) :
             ld=ImGL.LDOSinlayer(k0,zz,dslab,nslab,n2,n3,nd2list,nd3list)
           
             outparfromE[indices]=ld[0]
             outperfromE[indices]=ld[1]
             outparfromH[indices]=ld[2]
             outperfromH[indices]=ld[3]
             outcrossfromC[indices]=ld[4]
         else : 
             skip=True

    if skip:
        print('For layers with refractive index not positive or real, 0.0 entered as value')
    #output is LDOSE_||, LDOSE_perp,  LDOSM_||, LDOSM_perp, (all positive)
    #and LDOSC which can be pos and negative (since LDOS for any dipole has both p and m depends on LDOSE,M and C together)
    
    return outparfromE,outperfromE,outparfromH,outperfromH,outcrossfromC


def LDOSatanyPandM(pu,mu,k0,zlist,nstack,dstack):
    """Project LDOS for arbitrary electric/magnetic dipole vectors.

    Intuition
    ---------
    Combines canonical LDOS channels into the LDOS seen by a specific dipole
    definition `(pu, mu)` including the crossed magnetoelectric term.

    Parameters
    ----------
    pu, mu : array-like, length 3
        Electric and magnetic dipole vectors.
    k0, zlist, nstack, dstack :
        Same definition as in `LDOS`.

    Returns
    -------
    ndarray
        Normalized projected LDOS at each z position.
    """

    pu,mu=check.checkPuamMu(pu, mu)
    
    rhoE_par,rhoE_per, rhoM_par,rhoM_per,rhoC=LDOS(k0,zlist,nstack,dstack)    
    out=    (np.abs(pu[0])**2+np.abs(pu[1])**2)*rhoE_par+(np.abs(pu[2])**2)*rhoE_per
    out=out+(np.abs(mu[0])**2+np.abs(mu[1])**2)*rhoM_par+(np.abs(mu[2])**2)*rhoM_per
    out=out+2.0*rhoC*np.imag(np.conjugate(pu[1])*mu[0]-pu[0]*np.conjugate(mu[1]))
    norm=(np.abs(pu)**2+np.abs(mu)**2)
    
    return out/norm.sum()


def LDOSintegrandplottrace(k0,kparlist,zlist,nstack,dstack,guidevisible=1):
    """Return k_parallel-resolved LDOS integrands for plotting.

    Intuition
    ---------
    Equivalent to looking at LDOS versus `k||` before integration.
    Useful to identify radiative/leaky/guided contributions.

    Parameters
    ----------
    k0 : float
        Free-space wavenumber.
    kparlist : array-like
        Parallel wavevector values in units of `k0`.
    zlist : array-like
        z positions of interest.
    nstack, dstack : array-like
        Layer refractive indices and thicknesses.
    guidevisible : float, optional
        Controls small imaginary offset added to reveal guided-mode features.

    Returns
    -------
    ndarray
        Array of shape `(5, Nz, Nkpar)` with `(Epar, Eperp, Hpar, Hperp, C)`.
    """
    #type check the arguments / format massage
    k0,zlist,nstack,dstack=check.checkFullk0znd(k0, zlist, nstack, dstack)
    kparlist=check.checkKparlist(kparlist)
    
    # classify the points z in terms of in which layer they reside
    lib=cc.pinpointdomain(zlist,dstack)
    unilib=list(set(lib))  # library of unique domains that are visited by zlist
    
    nk=np.size(kparlist)
    # loop over all the unique domains in unilib.
    # Reason for this approach: amos.LDOS is vectorized but only as long as zlist is in one slab only, and not out of it.
    out=np.zeros([5,np.size(zlist),nk])
    
    skip=False #in case z traverse an absorbing, metallic or gainy layer, avoid popping out an unphysical answer.  
    for mm in range(nk):
        for m in unilib :  # loop over all slab-spaces in which there are sample points
             indices=[i for i, x in enumerate(lib) if  x == m]
             zz,dslab,nslab,n2,n3,nd2list,nd3list=cc.providecoordinates(m,zlist[indices],nstack,dstack)     
             
             
             if check.checkRealPositive(nslab) :
                 
                 out[:,indices,mm]=np.real(nslab*ImGL.AMOSMultiplex(kparlist[mm]/nslab-1.E-4j*guidevisible, k0, zz, dslab, nslab, n2, n3, nd2list, nd3list))
                 
             else :
                 skip=True
                 
    if skip:
        print('For layers with refractive index not positive or real, 0.0 entered as value')
    return out
#    return np.abs(out+np.nextafter(0.,1.))#tiny offset upwards avoids error when plotting on a logscale



def LDOSintegrandplotdispersion(k0list,kparlist,z,nstack,dstack,guidevisible=1):
    """Build a dispersion map by stacking `LDOSintegrandplottrace` over k0.

    Intuition
    ---------
    Produces a `(k0, k||)` map at fixed `z`, using the same channel ordering
    as `LDOSintegrandplottrace`.

    Returns
    -------
    ndarray
        Array of shape `(5, Nk0, Nkpar)`.
    """
    out=np.zeros([5,np.size(k0list),np.size(kparlist)])
    for index, k0 in enumerate(k0list):
        out[:,index,:]=np.squeeze(LDOSintegrandplottrace(k0,kparlist,np.array([z]),nstack,dstack,guidevisible))
    return out
