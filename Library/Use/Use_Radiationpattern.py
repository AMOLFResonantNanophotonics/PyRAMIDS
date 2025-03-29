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

from Library.Util import Util_argumentrewrapper as cc
from Library.Util import Util_argumentchecker as check


import numpy as np
from Library.Core import Core_Radiationpattern as Radp


''' Coordinate wrapped versions 
Routines above are slab centric i.e.,  consider a dipole at height z from slab boundary, where the slab might be layer m out of n=1..N layers
                The base routines then require to enumerate separately the slabs  m+1 ... N on one end of the slab of interest,
                and also the slabs 1 to m-1 in reverse order, counting away from the source
 
User-centric is: consider  a stratified system with first interface at z=0, and then  a set of finite layers of thickness d1,d2,d3, dN
               the source is at a coordinate z in the system
                
The routines below casts into user centric coordinates, using the imported helper function
'''



def TotalRadiatedatanyPandM(pu,mu,k0,zlist,nstack,dstack) :
#  Problem specification is
#  k0 =w/2 is free space wavebumber
#  pu,mu magnetic dipole moment
#  dstack and nstack parameterize geometry of multilayer stack
#  zlist is a list of dipole positions of interest anywhere in the stack 
#  Desired output; radiation pattern integrated over all angle
#  
#  THis routine is a wrapper that   
#  * Reshuffles coordinates for slabcentric input for  core radiation pattern routine 
#  *  all the points in a given slab are bunched together as one input to   vectorized routine TotalRadiatedfromLayer
#   

    #type check the arguments / format massage
    k0,zlist,nstack,dstack=check.checkFullk0znd(k0, zlist, nstack, dstack)
    pu,mu=check.checkPuamMu(pu, mu)
    # classify the points z in terms of in which layer they reside
    zlist=np.array(zlist)
    lib=cc.pinpointdomain(zlist,dstack)
    unilib=list(set(lib))  # library of unique domains that are visited by zlist
    
    out=np.zeros([3,np.size(zlist)]) #allocate room for all outputs
    
    skip=False
    for m in unilib : #loop over the unique domains. Within each domain TotalRadiated is vectorized over z-values (in zz)
         indices=[i for i, x in enumerate(lib) if  x == m]
         
         zz,dslab,nslab,n2,n3,nd2list,nd3list=cc.providecoordinates(m,zlist[indices],nstack,dstack)
            
         if check.checkRealPositive(nslab) :    
                 out[:,indices]=Radp.TotalRadiatedfromlayerPuMu(pu,mu,k0,zz,dslab,nslab,n2,n3,nd2list,nd3list) 
                 
         else :
             skip=True
        
         if skip:
             print('In TotalRadiated: For layers with refractive index not positive or real, 0.0 entered as value')

    norm=(np.abs(pu)**2+np.abs(mu)**2)
    
    return out/norm.sum()  #total of 15 outputs: From Ppar and per, M par and per, and cross,  first over 4pi. then only up, then only down.



''' Integrated far field power,  demanded through nicely wrapped coordinates'''

def TotalRadiated(k0,zlist,nstack,dstack) :
    pu=np.array([1.0,0.0,0.0])
    mu=np.array([0.0,0.0,0.0])
    Ppar=TotalRadiatedatanyPandM(pu,mu,k0,zlist,nstack,dstack)
    
    pu=np.array([0.0,0.0,1.0])
    mu=np.array([0.0,0.0,0.0])
    Pperp=TotalRadiatedatanyPandM(pu,mu,k0,zlist,nstack,dstack)
 
    pu=np.array([0.0,0.0,0.0])
    mu=np.array([1.0,0.0,0.0])
    Mpar=TotalRadiatedatanyPandM(pu,mu,k0,zlist,nstack,dstack)

    pu=np.array([0.0,0.0,0.0])
    mu=np.array([0.0,0.0,1.0])
    Mperp=TotalRadiatedatanyPandM(pu,mu,k0,zlist,nstack,dstack)

    pu=np.array([1.0,0.0,0.0])
    mu=np.array([0.0,1.j,0.0])
    PMplus=TotalRadiatedatanyPandM(pu,mu,k0,zlist,nstack,dstack)

    pu=np.array([1.0,0.0,0.0])
    mu=np.array([0.0,-1.j,0.0])
    PMmin=TotalRadiatedatanyPandM(pu,mu,k0,zlist,nstack,dstack)

    return np.vstack((Ppar[0],Pperp[0],Mpar[0],Mperp[0],0.5*(PMplus[0]-PMmin[0]),
    Ppar[1],Pperp[1],Mpar[1],Mperp[1],0.5*(PMplus[1]-PMmin[1]),
    Ppar[2],Pperp[2],Mpar[2],Mperp[2],0.5*(PMplus[2]-PMmin[2])))

    #total of 15 outputs: From Ppar and per, M par and per, and cross,  first over 4pi. then only up, then only down.




def RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack,rdipxy=[0,0]) :
#  Problem specifcation is
#  k0 =w/c is free space wavebumber
#  dstack and nstack parameterize geometry of multilayer stack
#  z is a  dipole position of interest anywhere in the stack 
#  Dipole moment (pu,mu) electric and magnetic
#  Desired output; radiation pattern resolved for angle the,phi
#
#  THis routine is a wrapper that   
#  * Reshuffles coordinates for slabcentric input for  core radiation pattern routine 
#  *  all the points in a given slab are bunched together as one input to   vectorized routine TotalRadiatedfromLayer
#   
#  Requirements for use:
#  - single z-value
     #type check the arguments / format massage
 
    
    k0,zlist,nstack,dstack=check.checkFullk0znd(k0, z, nstack, dstack)
    thelist,philist,shape=check.checkThetaAndPhi(thelist, philist)
    pu,mu=check.checkPuamMu(pu, mu)
    
    if len(zlist)>1:
        print("In radiation pattern - not vectorized over z. Do just 1st element") 
        zlist=zlist[0]

    
    lib=cc.pinpointdomain(zlist,dstack)
    zz,dslab,nslab,n2,n3,nd2list,nd3list=cc.providecoordinates(lib[0],zlist,nstack,dstack)
    
    P=np.zeros(np.size(thelist),dtype=complex)
    Es=np.zeros(np.size(thelist),dtype=complex)
    Ep=np.zeros(np.size(thelist),dtype=complex)

    up=np.where(np.cos(thelist)>0)
    down=np.where(np.cos(thelist)<=0)
    if check.checkRealPositive(nslab) :  
        if check.checkRealPositive(n2) : #down is into n2
            val = Radp.PEdown(pu,mu,thelist[down],philist[down],k0,zz,dslab,nslab,n2,n3,nd2list,nd3list,np.array(rdipxy))
            P[down]=val[0]
            Es[down]=val[1]
            Ep[down]=val[2]
        else :
            print('Entering 0s for halfspace n2 [down] that has non-real refractive index')
        
        if check.checkRealPositive(n3) : # up is into n3
            val=Radp.PEup(pu,mu,thelist[up],philist[up],k0,zz,dslab,nslab,n2,n3,nd2list,nd3list,np.array(rdipxy))
            P[up]=val[0]
            Es[up]=val[1]
            Ep[up]=val[2]
        else :                        
            print('Entering 0s for halfspace n3 [up] that has non-real refractive index')
    else:
        print('In radiation pattern: z-position is in a slab with non-real refractive index. Returning zeroes.')
    

 
    return np.reshape(np.real(P),shape),[np.reshape(Es,shape),np.reshape(Ep,shape)],[np.reshape(thelist,shape),np.reshape(philist,shape)]


