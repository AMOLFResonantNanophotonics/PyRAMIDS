#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
    GS,GF=GreenS(k0, nstack, dstack, rdetect, rsource)
    return GS + GF

 


def GreenS(k0,nstack,dstack,rdetect,rsource):
    GS= GreenEvalwrapped(k0,nstack,dstack,rdetect,rsource,diff=True)
    GF= GreenFree(k0,nstack,dstack,rdetect,rsource)
    
    return GS,GF
 
def Green(k0,nstack,dstack,rdetect,rsource):
    return GreenEvalwrapped(k0,nstack,dstack,rdetect,rsource,diff=False)




def GreenFree(k0,nstack,dstack,rdetect,rsource):
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

 