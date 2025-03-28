#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dpal,fkoenderink
"""

 

import numpy as np 

from scipy.special import jv


from Library.Core import Core_Smatrix as RT
import scipy.integrate as integrate


from numba import jit, config

config.DISABLE_JIT = True
#config.DISABLE_JIT = False
cachechoice=False
 



def GreenEval(k0,dslab,nslab,n2,n3,nd2list,nd3list,R,phi,z,z0,diff=False):
    
    #perform the integral over the elliptical contour, and then onwards far enough to fake infinity.
    kmax=np.max(np.abs(np.real( np.append(np.append(nd2list[:,0],nd3list[:,0]),[n2,n3]))))*3.0
    
 
    G1=integrate.quad_vec(lambda x: PaulusGreenfunctionIntegrand(x,kmax,k0,dslab,nslab,n2,n3,nd2list,nd3list,R,phi,z,z0,diff),0.0,np.pi,epsabs=1E-7,epsrel=1e-7)[0]
    kmaxmax=np.inf
    G2=integrate.quad_vec(lambda x: (GreenfunctionIntegrand(x,k0,dslab,nslab,n2,n3,nd2list,nd3list,R,phi,z,z0,diff)),kmax,kmaxmax,epsabs=1E-7,epsrel=1e-7)[0]
    
    return G1+G2




@jit(nopython=True,cache=cachechoice)
def Paulusintegrationtrajectory(t,kmin,kmax):
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
 
   
def PaulusGreenfunctionIntegrand(t,kmax,k0,dslab,nslab,n2,n3,nd2list,nd3list,R,phi,z,z0,diff=False):    
    # integrand on the contour, where t will run from 0 to pi.
    kmin=0.01 # max deviation from real axis, in units of kslab. Empirical choice
    x,v=Paulusintegrationtrajectory(t,kmin,kmax)
    zz=GreenfunctionIntegrand(x,k0,dslab,nslab,n2,n3,nd2list,nd3list,R,phi,z,z0,diff)
    return -v*zz


  
def GreenfunctionIntegrand(u,k0,dslab,nslab,n2,n3,nd2list,nd3list,R,phi,z,z0,diff=False):
    # u is k-parallel in units of kslab
        #z0 is the source
        #z is the detection point

    #  parallel momentum in actual units:
    kpar=nslab*k0*u 
 

    # Fresnel coefficients NOTE THE MINUS SIGN. CONVENTION NOVOTNY
    rdownp= - RT.rt_p(k0,kpar,nslab,n2,nd2list)[0]
    rdowns=   RT.rt_s(k0,kpar,nslab,n2,nd2list)[0]  
    rupp  = - RT.rt_p(k0,kpar,nslab,n3,nd3list)[0]
    rups  =   RT.rt_s(k0,kpar,nslab,n3,nd3list)[0]
    
    
         
    m0,m1,m2=M0M1M2(nslab,k0,kpar,dslab,rdowns,rdownp,rups,rupp,phi,z,z0)
 
 
    if diff == True:
        m0F,m1F,m2F=M0M1M2(nslab,k0,kpar,dslab,0,0,0,0,phi,z,z0)
        
        m0=m0-m0F
        m1=m1-m1F
        m2=m2-m2F
    
    kR=kpar*R
    # j0,j1,j2=jv([0,1,2],kR) # you can not vectorize both over the order n, and the coordinate kR, so instead do:
    j0=jv(0,kR)
    j1=jv(1,kR)
    j2=jv(2,kR)
    
  
    return (j0*m0+j1*m1+j2*m2)  

##################################3




@jit(nopython=True,cache=cachechoice)
def M0M1M2(nslab,k0,kpar,dslab,rdowns,rdownp,rups,rupp,phi,z,z0):    
    
    As,Ap,Bs,Bp,Cs,Cp,Ds,Dp,bs,bp,cs,cp = prefactorconstruction(nslab,k0,kpar,dslab,rdowns,rdownp,rups,rupp,z,z0)
    m0,m1,m2=M0M1M2construction(As,Ap,Bs,Bp,Cs,Cp,Ds,Dp,bs,bp,cs,cp,phi)
    
    return m0,m1,m2


@jit(nopython=True,cache=cachechoice)
def M0M1M2construction(As,Ap,Bs,Bp,Cs,Cp,Ds,Dp,bs,bp,cs,cp,f):
#After azimuthal integration you are left with proportionalities to Bessel functions j0, j1, j2 of argument kR
#Proportionality factor is a 6x6 matrix, elements determined by prefactor routine (below)
 

#  For M0 (the j0 term)

     # M0 = np.array([
     #          0        1   2                 3       4        5
     #----------------------------------------------------------------
     #  0   [As+Dp,      0,  0,                0,   -(Bs+Cp) ,  0 ],
     #  1   [0,      As+Dp,  0,              (Bs+Cp),  0,       0 ],
     #  2   [0,          0,  -2.*Dp*bp*cp,     0,      0,       0 ],
     #     #
     #  3   [0,    -(Bp+Cs), 0,            -(Ap+Ds),   0,        0],
     #  4   [Bp+Cs,      0,  0,                0,   -(Ap+Ds),    0],
     #  5   [0,          0,  0,                0,     0,     2.*Ds*bs*cs]
     #         ])


## FOLLOWING IN ORDER NOT TO BREAK NUMBA JIT
  
     M0=0.0j*np.zeros((6,6,f.size)) 
     M0[0,0,:]=As+Dp
     M0[1,1,:]=As+Dp
     M0[2,2,:]=-2.0*Dp*bp*cp

     M0[3,3,:]=-(Ap+Ds)
     M0[4,4,:]=-(Ap+Ds)
     M0[5,5,:]=2.0*Ds*bs*cs
     
     M0[0,4,:]=-(Bs+Cp)
     M0[1,3,:]= (Bs+Cp)
     
     M0[3,1,:]=-(Bp+Cs)
     M0[4,0,:]=  Bp+Cs
     
     
     # M2=np.array([
     #          0                 1       2               3       4        5
     #----------------------------------------------------------------

     #  0  [(As-Dp)*cff,      (As-Dp)*sff,  0,      (Bs-Cp)*sff, -(Bs-Cp)*cff, 0],
     #  1  [(As-Dp)*sff,     -(As-Dp)*cff,  0,     -(Bs-Cp)*cff, -(Bs-Cp)*sff, 0],
     #  2  [0+0.0j,              0,         0,          0,            0,       0],
     #    #
     #  3  [ (Bp-Cs)*sff,    -(Bp-Cs)*cff,  0,     -(Ap-Ds)*cff, -(Ap-Ds)*sff, 0],
     #  4  [-(Bp-Cs)*cff,    -(Bp-Cs)*sff,  0,     -(Ap-Ds)*sff,  (Ap-Ds)*cff, 0],
     #  5  [0+0.0j,0,0, 0,0,0]
     #    ])

     
     
     cff=np.cos(2.0*f)
     sff=np.sin(2.0*f)
     
     
     M2=0.0j*np.zeros((6,6,f.size)) 
     M2[0,0,:]= (As-Dp)*cff
     M2[0,1,:]= (As-Dp)*sff
     M2[1,0,:]= (As-Dp)*sff
     M2[1,1,:]=-(As-Dp)*cff
     
     M2[3,3,:]=-(Ap-Ds)*cff
     M2[3,4,:]=-(Ap-Ds)*sff
     M2[4,3,:]=-(Ap-Ds)*sff
     M2[4,4,:]= (Ap-Ds)*cff
      
     M2[0,3,:]= (Bs-Cp)*sff
     M2[0,4,:]=-(Bs-Cp)*cff
     M2[1,3,:]=-(Bs-Cp)*cff
     M2[1,4,:]=-(Bs-Cp)*sff
     
     M2[3,0,:]= (Bp-Cs)*sff
     M2[3,1,:]=-(Bp-Cs)*cff
     M2[4,0,:]=-(Bp-Cs)*cff
     M2[4,1,:]=-(Bp-Cs)*sff
     
     
     
     
#  For M1 (the j1 term)

     # M1=np.array([
     #          0                 1       2               3       4        5
     #----------------------------------------------------------------
     # 0  [0,                  0,      pbD*cf,     0,       0,       -sbB*sf],
     # 1  [0,                  0,      pbD*sf,     0,       0,        sbB*cf],
     # 2  [-pcD*cf,        -pcD*sf,        0,    -pcC*sf, pcC*cf,           0],
     #   #
     # 3  [0,                  0,     -pbB*sf,     0,      0,       -sbD*cf],
     # 4  [0,                  0,      pbB*cf,     0,      0,       -sbD*sf],
     # 5  [-scC*sf,          scC*cf,      0,     scD*cf, scD*sf,           0]
     #     ])


     sf=np.sin(f)
     cf=np.cos(f)
      
     pbD=2.0j*bp*Dp
     pcD=2.0j*cp*Dp
     sbD=2.0j*bs*Ds
     scD=2.0j*cs*Ds
     
     sbB=2.0j*bs*Bs
     pcC=2.0j*cp*Cp
     
     pbB=2.0j*bp*Bp
     scC=2.0j*cs*Cs
     
     M1=0.0j*np.zeros((6,6,f.size)) 
     M1[0,2,:]= pbD*cf
     M1[1,2,:]= pbD*sf
     M1[2,0,:]=-pcD*cf
     M1[2,1,:]=-pcD*sf
     
     M1[3,5,:]=-sbD*cf
     M1[4,5,:]=-sbD*sf
     M1[5,3,:]= scD*cf
     M1[5,4,:]= scD*sf
        
     M1[0,5,:]=-sbB*sf
     M1[1,5,:]= sbB*cf
     
     M1[2,3,:]=-pcC*sf
     M1[2,4,:]= pcC*cf
     
     M1[3,2,:]=-pbB*sf
     M1[4,2,:]= pbB*cf
     M1[5,0,:]=-scC*sf
     M1[5,1,:]= scC*cf




# Here are the matrices

     
 
     return M0, M1, M2




@jit(nopython=True,cache=cachechoice)
def prefactorconstruction(nslab,k0,kpar,d,rdown_s,rdown_p,rup_s,rup_p,z,z0):
    # Prefactors for Green function derived using method of Y. Chen Op Ex paper
    # See Mathematica derivation
    # 
    #
    #z0 is the source
    #z is the detection point
     
    kslab=nslab*k0 +0.0j
    kz=RT.kz(nslab,k0,kpar) 
    
    kparonkz=kpar/kz
    kparonkslab=kpar/kslab
 
    # Propagation factors
    etad=       np.exp(2.j*kz*d)
    etaz0=      np.exp(2.j*kz*z0)
    eta2dz=     np.exp(2.j*kz*(d-z))
   
    etazmin=    np.exp(1.j*kz*(z-z0))
    etazplus=   np.exp(1.j*kz*(z+z0))
    
    etaz2dplus=np.exp(1.j*kz*(2.*d-(z+z0)))
    etaz2dmin= np.exp(1.j*kz*(2.*d-(z-z0)))
     
    
    # s-polarisation   
    preS=1.0/(1.-rdown_s*rup_s*etad)


    
    rd=rdown_s*etazplus
    ru=  rup_s*etaz2dplus
    rud=rdown_s*rup_s*etaz2dmin




    
    As=preS*kparonkz*               ( etazmin+ rd  +ru  + rud)
    Bs=preS*kparonkslab*            (-etazmin+ rd  -ru  + rud)
    Cs=preS*kparonkslab*            ( etazmin+ rd  -ru  - rud)
    Ds=preS*kpar*kz/(kslab*kslab)*  (-etazmin+ rd  +ru  - rud)    
      
    bs= -  kparonkz*(1.+etaz0*rdown_s)   /  (1.-etaz0*rdown_s)
    cs=    kparonkz*(1.+eta2dz*rup_s)    / ( 1.-eta2dz*rup_s)
 


                              
    preP=-1.0/(1.-rdown_p*rup_p*etad) # note the important minus sign

    rd=rdown_p*etazplus
    ru=  rup_p*etaz2dplus
    rud=rdown_p*rup_p*etaz2dmin

    Ap=preP*kparonkz*               ( etazmin+ rd + ru + rud )
    Bp=preP*kparonkslab*            (-etazmin+ rd - ru + rud )
    Cp=preP*kparonkslab*            ( etazmin+ rd - ru - rud )
    Dp=preP*kpar*kz/(kslab*kslab)*  (-etazmin+ rd + ru - rud )    

        
    bp= - kparonkz*(1.+etaz0*rdown_p)   /( 1.-etaz0*rdown_p)
    cp=   kparonkz*(1.+eta2dz*rup_p)    /( 1.-eta2dz*rup_p)
                             
    
    return As,Ap,Bs,Bp,Cs,Cp,Ds,Dp,bs,bp,cs,cp


###### Dyadic Green function of free space, defined without any angular spectrum representation    
def FreeDyadG(k,rvec):
  
    x=rvec[0,:] 
    y=rvec[1,:] 
    z=rvec[2,:] 
    
    r=np.sqrt(x*x+y*y+z*z) 
    kr=k*r                      #k being wavenumber in the homog medium
    F=np.exp(1.0j*kr)/r       #Essentially G derives from scalar spherical wave by differentiation
    DF=(1.j*kr-1)*F/(r*r)
    DDF=-(kr**2+3.j*kr-3)/(r**4)*F
    
     
    o=(0.0+0.0j)*kr #vectorized nil
    l=1.0+o  # vectorized unity
    aux1=k*k*np.array([[l,o,o,o,o,o],
                       [o,l,o,o,o,o],
                       [o,o,l,o,o,o],
                       #
                       [o,o,o,l,o,o],
                       [o,o,o,o,l,o],
                       [o,o,o,o,o,l]])
 
 
    
    #aux2a=-1.j*k*np.array([[   o,     z,      -y],
    #                        [ -z,      o,       x],
    #                        [  y,     -x,       o]])

    #aux2=np.block([[i3,aux2a],[-aux2a,i3]]) 
    
    aux2=np.array(
         [[l,o,o,   o, -1.j*k*z,1.j*k*y],
          [o,l,o,   1.j*k*z, o, -1.j*k*x ],
          [o,o,l,      -1.j*k*y,1.j*k*x,o],
          #
          [o,1.j*k*z,-1.j*k*y,   l, o, o],
          [-1.j*k*z,o,1.j*k*x,   o,l, o],
          [1.j*k*y,-1.j*k*x,o,    o,o, l ]
          ])
    
    
    
#    aux3a=np.outer(rvec,rvec)
#    aux3a=np.array([[x*x,x*y,x*z],
#                    [x*y,y*y,y*z],
#                    [x*z,y*z,z*z]])  
    
    
    aux3=  np.array(
           [[x*x,x*y,x*z,   o,o,o],
            [x*y,y*y,y*z,   o,o,o],
            [x*z,y*z,z*z,   o,o,o],
            #
            [o,o,o,     x*x,x*y,x*z],
            [o,o,o,     x*y,y*y,y*z],
            [o,o,o,     x*z,y*z,z*z]]) 
     
    
    dyadG=(aux1*F + (aux2*DF+aux3*DDF));   
    
    return dyadG