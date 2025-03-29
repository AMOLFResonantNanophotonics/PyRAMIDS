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

from Library.Core import Core_Smatrix as Smatrix
from Library.Util import Util_argumentrewrapper as cc
from Library.Util import Util_argumentchecker as check

import numpy as np


#########################################################################################
#########################################################################################
####################                                                    #################
####################                                                    #################
####################        Plane wave interrogation of stacks          #################
####################                                                    #################
####################                                                    #################
#########################################################################################
#########################################################################################


def checkplanewaveproblem(k0,kpar,nstack,dstack): 
    k0=check.checkk0(k0)
    kpar=check.checkKparlist(kpar)
    nstack,dstack=check.checkStackdefinition(nstack,dstack)
    nin,nout,ndlist,nmid=cc.stackseparator(nstack, dstack)
    if check.checkRealPositive(nin)==False:
        raise ValueError('Refl and transm requires input halfspace to be transparent and nonlossy')

    return k0,kpar,nstack,dstack,nin,nout,ndlist,nmid

# Plane wave layer reflectivity and transmission
def IntensityRT(k0,kpar,nstack,dstack):
    ''' Wrapper routine that queries S-matrix for intensity Reflectance, and transmittance.
        k0 = free space wavenumber
        kpar = parallel momentum of the input wave, can go up to nin. Goes with sine input angle
        nstack = [nin, n1,n2..nM, nout]  refractive indices
        dstack = [d1,d2,d3 .., dN]  
    '''
    
    k0,kpar,nstack,dstack,nin,nout,ndlist,dum=checkplanewaveproblem(k0,kpar,nstack,dstack)
    
        
        
    rts=Smatrix.rt_s(k0,kpar,nin,nout,ndlist)
    rtp=Smatrix.rt_p(k0,kpar,nin,nout,ndlist)
    projectionfactor = (Smatrix.kz(nout,k0,kpar)/Smatrix.kz(nin,k0,kpar))
    
    
    Rs, Rp=((np.abs(rts[0]))**2),  ((np.abs(rtp[0]))**2)
    
 
    if check.checkRealPositive(nin)==True:
        Ts, Tp = ((np.abs(rts[1]))**2)*np.real(projectionfactor) , ((np.abs(rtp[1]))**2)*np.real(projectionfactor)
        # print(Ts)
    else:
        Ts, Tp=0.0*Rs,0.0*Rp
 
    return np.array([Rs,Ts,1.0-Rs-Ts]),np.array([Rp,Tp,1.0-Rp-Tp])



def PerLayerAbsorption(k0,kpar,nstack,dstack):
    ''' Wrapper routine that queries S-matrix to solve the plane wave problem for incidence from medium 1,  
        and to return the absorped power per layer for each of the finite thickness layers in the stack.   
        
        k0 = free space wavenumber
        kpar = parallel momentum of the input wave, can go up to nin. Goes with sine input angle
        nstack = [nin, n1,n2..nM, nout]  refractive indices
        dstack = [d1,d2,d3 .., dN]  
    '''
    # query the sensibility of the problem parameters: 
    k0,kpar,nstack,dstack,nin,nout,ndlist,nmid=checkplanewaveproblem(k0,kpar,nstack,dstack)
    
    def LayerAbsorptionFromLayerE(u,d,kz,dslab):
        ''' Auxiliary routine to calculate absorption in a layer of thickness dslab, 
            Given already determined forward and backwardpropagating E-field amplitudes 
            
            Implements generalized version of Nanophotonics 9, 3985–4007, (2020)   Eq. 16
            
            Compared to that equation the prefactor (1/2 omega Im eps, generalized to non-normal incidence)
            Is left out - it is inserted as prefac in PerLayerAbsorption
            
            Then, one only needs to know kz
        '''
         
        kprime=np.real(kz)
        kpprime=np.imag(kz)+1.0E-23 # offset to pin imaginary part in non-lossy case.
        
        
        if(dslab >0):
            e2kppd=np.exp(2.0*kpprime*dslab)
                 
            delta=np.angle(u)-np.angle(d)
            u=np.abs(u)
            d=np.abs(d)
            a=((u*u)*(1.0-1.0/e2kppd)+(d*d)*(e2kppd-1.0))/(2.0*kpprime)
            b=u*d*(np.sin(2*kprime*dslab+delta)-np.sin(delta))/kprime
            
        else: # this signals:  last halfspace
            u=np.abs(u)
            a=(u*u)/(2.0*kpprime)
            b=0.0
            
        return a+b


    
    # allocate space for the result
    
    abss=[]
    absp=[]
    

    # ndlist enumerates all the finite-thickness slabs one by one
    # I add the half infinite back space with negative thickness
    # in the layer absorption routine this is picked out with an if-statement
    
    ndlist=np.vstack([ndlist,np.stack([nout,-1.0])])
    nlay=ndlist.shape[0]
    
    for m in range(nlay): # loop over all layers in the stack.
        
         # refractive index and thickess of the specific layer
         nslab=ndlist[m,0]
         dslab=ndlist[m,1]
         
         kz= Smatrix.kz(nslab,k0,kpar)
         kzin=Smatrix.kz(nin,k0,kpar)
         prefac=np.imag(nslab*nslab)*np.real(k0*(k0/kzin)) 
         
         #  s-polarization. Call S-matrix to provide up- and down wave amplitudes. 
         #  Then query absorption via Nanophotonics 9, 3985–4007, (2020)   Eq. 16
         EHu,EHd=Smatrix.Esfromudcoef(k0, kpar, nin, nout, ndlist, m+1)  
         
         abss=abss+[prefac*LayerAbsorptionFromLayerE(EHu[0],EHd[0],kz,dslab)]
    
          

         #  p-polarization. Call S-matrix to provide up- and down wave amplitudes. 
         #  Then query absorption via Nanophotonics 9, 3985–4007, (2020)   Eq. 16
         #
         #  Complication is that now the u and d coefficients are for H-fields, and the incident
         #  field was also expressed in E.  This  explains the prefactor, and the two terms. The logic is
         #  E follows from H by derivative.  This leads to two E components - along x, and also along z, which is normal to the stack
         #  Both give rise to absorption. 
         
         EHu,EHd=Smatrix.Epfromudcoef(k0, kpar, nin, nout, ndlist, m+1)
         abspx=LayerAbsorptionFromLayerE(EHu[0],EHd[0],kz,dslab)
         abspz=LayerAbsorptionFromLayerE(EHu[1],EHd[1],kz,dslab)
         
         absp=absp+[(abspx+abspz)*prefac]
         

    return np.array(abss).real,np.array(absp).real




def OnAxisLocalFieldandAbsorption(k0,kpar,nstack,dstack,zlist):
    ''' Wrapper routine that queries S-matrix to solve the plane wave problem for incidence from medium 1,  
        and to return the local E and H fields on the x=y=0 axis, separated out 
        in the s and p solution. Also the routine returns the local absorption density
         
        k0 = free space wavenumber
        kpar = parallel momentum of the input wave, can go up to nin. Goes with sine input angle
        nstack = [nin, n1,n2..nM, nout]  refractive indices
        dstack = [d1,d2,d3 .., dN]  
    '''
    # Argument checking     
    zlist=check.checkz(zlist)
    k0,kpar,nstack,dstack,nin,nout,ndlist,dum=checkplanewaveproblem(k0,kpar,nstack,dstack)
 
     
    kzin=Smatrix.kz(nin,k0,kpar)
    
    #allocatespace to hold the result
    
    S = np.zeros([3,kpar.size,zlist.size],dtype=complex)
    P = np.zeros([3,kpar.size,zlist.size],dtype=complex) 
    As= np.zeros([kpar.size,zlist.size], dtype=float) 
    Ap= np.zeros([kpar.size,zlist.size], dtype=float) 
    
    #uniquely assign each z in zlist to a layer
    zlist=np.array(zlist);    lib=cc.pinpointdomain(zlist,dstack);  unilib=list(set(lib))
      
    
    for m in unilib :  # loop over all layers in the stack  in which there are sample points
          indices=[i for i, x in enumerate(lib) if  x == m]  # retrieve the indices of the pertinent z-points
    
    
          zz,dslab,nslab,n2,n3,nd2list,nd3list=cc.providecoordinates(m,zlist[indices],nstack,dstack) #for those z-points construct the functional arguments



          if m==0:
              zz=zlist[indices]
          s,p=Smatrix.SP_FieldsAtZ(k0,kpar,nin,nout,ndlist,m,zz)      
          S[:,:,indices]=s
          P[:,:,indices]=p 

          prefac=np.imag(nslab*nslab)*np.real(k0*(k0/kzin))
          prefac=np.outer(prefac,np.ones(zz.size))
         
          As[:,indices]=prefac*np.abs(s[0,:,:])**2   #  Absorption density (per length along the z-axis)
          Ap[:,indices]=prefac*(np.abs(p[0,:,:])**2+np.abs(p[1,:,:])**2)   #  Absorption density (per length along the z-axis)

    return S,P,As,Ap 



def CartesianField(theta,phi,scoeff,pcoeff,rlist,k0,nstack,dstack):
    ''' Routine provides Cartesian field components for E and H 
    ## For a range of input wave vectors, and at a list of positions
    ##
    ##
    ## rlist:  positions r that are offered r[:,3] at which you want to know the answer
    ##  
    ## thetas, phis   specify input wavevectors for which you want the answer, 
    ## these are angles in medium nin (negative z)
    ## 
    ##  Each illumination at s and p complex prefactors, meaning amplitudes and phase
    ## 
    ##  k0,nstack,dstack as usual
    '''
    
# Argument checking plane wave problem      
    k0,kpar,nstack,dstack,nin,nout,ndlist,dum=checkplanewaveproblem(k0,0.0*k0,nstack,dstack)
    
# Argument checking positionlist
    rlist=check.checkr(rlist) 

    zlist=rlist[2,:]
    
    
# Argument checking input angles and amplitudes. Note that cos theta must be positive
    theta,phi,shape=check.checkThetaAndPhi(theta, phi) 
    if len(shape)==2:
        raise ValueError('Theta and phi can not be lists of different lengths in CartesianField')
    if check.checkequaldimension(theta, scoeff) == False:
        raise ValueError('number of desired spol amplitudes does not match number of desired angles in CartesianField')
    if check.checkequaldimension(scoeff, pcoeff) == False:
        raise ValueError('number of desired spol an p pol amplitudes inconsistent in CartesianField')

    if (np.cos(theta)<0).any():
        raise ValueError('CartesianField only works for input k-vectors into positive z (0<theta<pi/2')
    
 
    
# Unit vectors for all the distinct k-parallels that are specified by theta
    khat, shat, phat =cc.spherical2cartesian(theta, phi)
    
    yhat=np.array(shat)
    xhat = 0*yhat;  xhat[0,:] = yhat[1,:]; xhat[1,:]=  -yhat[0,:]
    
  
    xhat=np.tile(xhat[:,:,np.newaxis],(1,1,zlist.size))  # replicate, to accomodate all positions without forloop
    yhat=np.tile(yhat[:,:,np.newaxis],(1,1,zlist.size))
  
# Get the field strengths at each z, for S and P.      
    kpar=k0*nin*np.sin(theta)
    S,P,As,Ap=OnAxisLocalFieldandAbsorption(k0,kpar,nstack,dstack,zlist)
     
# Inflate size, to avoid for loops
    S=S*np.tile(scoeff[np.newaxis,:,np.newaxis],(3,1,zlist.size))
    P=P*np.tile(pcoeff[np.newaxis,:,np.newaxis],(3,1,zlist.size))
    

#  Actual S, P to cartesian
    EcartSx=S[0,:,:]*yhat[0,:,:]
    EcartSy=S[0,:,:]*yhat[1,:,:]
    EcartSz=0.0  # by construction of yhat


# Magnetic field H is in units H = Z0 * H_SI; (Z = Z0/n) .... scale by n for medium with refractive index n
    HcartSx=S[1,:,:]*xhat[0,:,:] * nin
    HcartSy=S[1,:,:]*xhat[1,:,:] * nin
    HcartSz=S[2,:,:] * nin
    
    EcartPx=P[0,:,:]*xhat[0,:,:]
    EcartPy=P[0,:,:]*xhat[1,:,:] 
    EcartPz=P[1,:,:] 
  
    HcartPx=P[2,:,:]*yhat[0,:,:] * nin
    HcartPy=P[2,:,:]*yhat[1,:,:] * nin
    HcartPz=0.0 * nin
    
 #  Stuff the result in a huge array. E and H = 6 coeffic. Array (6, Ntheta, Npos)   
    EH=np.array([EcartSx+EcartPx, EcartSy+EcartPy,EcartSz+EcartPz,
                 HcartSx+HcartPx, HcartSy+HcartPy,HcartSz+HcartPz])
    
#   Apply the sideways phasefactor implied by kpar
    phasefactor=np.exp(1.0j*k0*nin*(np.outer(khat[0],rlist[0,:])+np.outer(khat[1],rlist[1,:]))) 

    return EH*np.tile(phasefactor[np.newaxis,:,:],(6,1,1))


 
