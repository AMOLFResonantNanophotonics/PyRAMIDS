"""
#All user-level routines for dyadic Green tensor evaluation in layered stacks

#This file wraps the core Green-tensor routines in user-centric coordinates.
#It provides full Green, scattered Green, and free-space Green components.
#
#Current implementation assumes source and detector are in the same layer.
#
#Conventions used throughout
#  - k0 = 2*pi/lambda_vac
#  - nstack = [n0, n1, n2, ..., n_{m-1}, n_m]
#  - dstack = [d1, d2, ..., d_{m-1}]
#
#Main user functions
#  - Greensafe: total (scattered + free-space) dyadic Green tensor
#  - GreenS: scattered and free-space Green tensors as separate outputs
#  - Green: scattered Green tensor only
#  - GreenFree: homogeneous-space Green tensor only

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

from Library.Core import Core_Greenslab as Gr


def Greensafe(k0,nstack,dstack,rdetect,rsource):
    """Return total dyadic Green tensor (scattered + free-space).

    Intuition
    ---------
    Main user wrapper when you want the physically complete Green tensor.
    Internally this is `GreenS + GreenFree`.

    Parameters
    ----------
    k0 : float
        Free-space wavenumber.
    nstack, dstack : array-like
        Layer refractive indices and thicknesses.
    rdetect, rsource : array-like
        Detection/source coordinates accepted by the argument checker.

    Returns
    -------
    ndarray
        Complex array of shape (6, 6, Npos).
    """
    GS,GF=GreenS(k0, nstack, dstack, rdetect, rsource)
    return GS + GF

 


def GreenS(k0,nstack,dstack,rdetect,rsource):
    """Return scattered and free-space dyadic Green tensors.

    Intuition
    ---------
    Use this when you want to inspect the layered contribution and the
    homogeneous reference contribution separately.

    Returns
    -------
    tuple(ndarray, ndarray)
        `(GS, GF)` each of shape (6, 6, Npos).
    """
    GS= GreenEvalwrapped(k0,nstack,dstack,rdetect,rsource,diff=True)
    GF= GreenFree(k0,nstack,dstack,rdetect,rsource)
    
    return GS,GF
 
def Green(k0,nstack,dstack,rdetect,rsource):
    """Return scattered dyadic Green tensor.

    Intuition
    ---------
    Wrapper for the layered correction only (no free-space term).

    Returns
    -------
    ndarray
        Complex array of shape (6, 6, Npos).
    """
    return GreenEvalwrapped(k0,nstack,dstack,rdetect,rsource,diff=False)




def GreenFree(k0,nstack,dstack,rdetect,rsource):
    """Return homogeneous-medium dyadic Green tensor.

    Intuition
    ---------
    Baseline free-space/same-medium dyadic Green tensor for the local slab.

    Notes
    -----
    Source and detector positions must be in the same layer.

    Returns
    -------
    ndarray
        Complex array of shape (6, 6, Npos).
    """
    #type check the arguments
    nstack,dstack=check.checkStackdefinition(nstack,dstack)
    k0=check.checkk0(k0)
    
    dx,dy,zdetect,zsource,R,phi = check.checkGreenrsourcerdetect(rdetect,rsource)

    m1=cc.pinpointdomain(zsource,dstack)
    m2=cc.pinpointdomain(zdetect,dstack)
    if any(m1!=m2):
        raise('In this Green function implementation, z1 and z0 must be in same layer')
    
    
    munique=np.unique(m1)
    Npos=np.size(R)
    GG=0.0j*np.zeros((6,6,Npos))
    for m in munique:
        
        indices = [i for i in range(len(m1)) if m1[i] == m]
        Dx=dx[indices]
        Dy=dy[indices]
        Zdet=zdetect[indices]
        Zsourc=zsource[indices]
        
        Zdet,dslab,nslab,n2,n3,nd2list,nd3list=cc.providecoordinates(m,Zdet,nstack,dstack)
        Zsourc,dslab,nslab,n2,n3,nd2list,nd3list=cc.providecoordinates(m,Zsourc,nstack,dstack)
        
        k=k0*nslab
        rvec=np.array([Dx,Dy,Zdet-Zsourc])
        GG[:,:,indices]=Gr.FreeDyadG(k, rvec)

    return GG


def GreenEvalwrapped(k0,nstack,dstack,rdetect,rsource,diff=False):
    """Internal slab-centric wrapper used by public Green routines.

    Handles user-to-slab coordinate conversion and source/detector ordering.
    """
    
    #In internal routines 
    #    z0 is the source
    #    z is the detection point
    #    appear in order "z,z0" in argument lists
    #    supposedly this means that the parallel
    #    displacemet appears as (R cos f, R sin f )=(rdetect[0,1]-rsource[0,1])
    
    #    The derivation is for z > z0

    
    #type check the arguments
    nstack,dstack=check.checkStackdefinition(nstack,dstack)
    k0=check.checkk0(k0)
    dx,dy,zdetect,zsource,R,phi = check.checkGreenrsourcerdetect(rdetect,rsource)

    # make sure all the z1 and z2 match up to be in the same layer.
    m1=cc.pinpointdomain(zsource,dstack)
    m2=cc.pinpointdomain(zdetect,dstack)
    if any(m1!=m2):
        raise('In this Green function implementation, z1 and z0 must be in same layer')
    
    
    # now loop over all the required positions (the check routines already verified this is well defined)
    Npos=np.size(R)
    GG=0.0j*np.zeros((6,6,Npos))

    munique=np.unique(m1)
    
    for m in munique:
        
        
        indices = [i for i in range(len(m1)) if m1[i] == m]
        
        RR=R[indices]
        Pphi=phi[indices]
        Z1=zdetect[indices]
        Z0=zsource[indices]

        G=0.0j*np.zeros((6,6,len(indices)))
        
        Z1,dslab,nslab,n2,n3,nd2list,nd3list=cc.providecoordinates(m,Z1,nstack,dstack)
        Z0,dslab,nslab,n2,n3,nd2list,nd3list=cc.providecoordinates(m,Z0,nstack,dstack)
        
        flipindicator=(Z1 < Z0)*1.0
        indic = [i for i in range(len(flipindicator)) if flipindicator[i] == 0]
        
    
        
        
        if len(indic)>0: 
            G[:,:,indic]=Gr.GreenEval(k0,dslab,nslab,n2,n3,nd2list,nd3list,RR[indic],Pphi[indic],Z1[indic],Z0[indic],diff)
 
        indic = [i for i in range(len(flipindicator)) if flipindicator[i] == 1]

        if len(indic)>0:
            Gdum=Gr.GreenEval(k0,dslab,nslab,n2,n3,nd2list,nd3list,RR[indic],Pphi[indic]+np.pi,Z0[indic],Z1[indic],diff)
            GEH=Gdum[0:3,3:6,:];  GHE=Gdum[3:6,0:3,:]
            Gdum[0:3,3:6,:]=-GEH
            Gdum[3:6,0:3,:]=-GHE
            
            
        
            G[:,:,indic]=np.transpose(Gdum,(1,0,2))
        
        
    
        GG[:,:,indices]=G
 
         
    return GG*(0.5j*(k0*nslab)**3)

 
