
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

from Library.Core import Core_ImGLDOS as ImGL



''' Coordinate wrapped interface to core ImG
Routines above are slab centric i.e.,  consider a dipole at height z from slab boundary, where the slab might be layer m out of n=1..N layers
                The base routines then require to enumerate separately the slabs  m+1 ... N on one end of the slab of interest,
                and also the slabs 1 to m-1 in reverse order, counting away from the source
 
User-centric is: consider  a stratified system with first interface at z=0, and then  a set of finite layers of thickness d1,d2,d3, dN
               the source is at a coordinate z in the system
                
The routine below casts the LDOS function in user centric coordinates, using the imported helper function
''' 
def LDOS(k0,zlist,nstack,dstack) :
    '''
    #  wrapper routine. Problem spec is k0 = 2pi/lambda_vac,  a set of z-points, and the geometry dstack, nstack
    #  for thicknesses and indices
    #
    # reshuffles this in coordinates for inputting in Amos and Barnes, and then runs Amos and Barnes
    #  all the points in a given slab are bunched together as one input to Amos and Barnes to benefit from some vectorization
    '''
 
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

    pu,mu=check.checkPuamMu(pu, mu)
    
    rhoE_par,rhoE_per, rhoM_par,rhoM_per,rhoC=LDOS(k0,zlist,nstack,dstack)    
    out=    (np.abs(pu[0])**2+np.abs(pu[1])**2)*rhoE_par+(np.abs(pu[2])**2)*rhoE_per
    out=out+(np.abs(mu[0])**2+np.abs(mu[1])**2)*rhoM_par+(np.abs(mu[2])**2)*rhoM_per
    out=out+2.0*rhoC*np.imag(np.conjugate(pu[1])*mu[0]-pu[0]*np.conjugate(mu[1]))
    norm=(np.abs(pu)**2+np.abs(mu)**2)
    
    return out/norm.sum()


def LDOSintegrandplottrace(k0,kparlist,zlist,nstack,dstack,guidevisible=1):
    '''
    Returns the integrand that when integrated provides LDOS, for 
    plot purposes. Essentially this is equiv to LDOS versus k||
    
    Input: 
        k0 wavenumber in free space
        kparlist, list of kparallels of interest, in units of k0
        zlist, positions of interest
        nstack,dstack the geometry as usual
    
    The routine provides an out array of shape (5,n_z, n_kparlist)
    Elements 1,2,3,4,5 are for Epar, Eperp,Hpar,Hperp, and crossed, as usual
    
    The routine fudges a slight imaginary offset into k: this is required so that 
    truly guided modes actually light up. 
    '''    
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