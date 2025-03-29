#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coordinate wrapping utility
Conversion routine between the natural "user-centric" coordinate and the "slab-centric" coordinate system in the base routines

User centric:  consider  a stratified system with first interface at z=0, and then  a set of finite layers of thickness d1,d2,d3, dN
               the source is at a coordinate z in the system
               
               
Slab centric:   The natural coordinate system for LDOS and radiation pattern routines is "slab centric".
                I.e.,  consider a dipole at height z from slab boundary, where the slab might be layer m out of n=1..N layers
                The base routines then require to enumerate separately the slabs  m+1 ... N on one end of the slab of interest,
                and also the slabs 1 to m-1 in reverse order, counting away from the source
                

The coordinate wrapper translates coordinates and all the arguments from user centric to slab centric. The approach is:
    - pinpointdomain. For a list of z, and geometry,  pinpoint for each z in which domain n=0, 1....N+1 it lies (incl half infinite spaces)
    - provide coordinates. Given the domain has been pinpointed,  take the user centric z and stack definition, and return the slab centric definition
    
@author: dpal,fkoenderink
"""



#%%
 
import numpy as np


    
def basicgeoinfo(z,dstack):
     
    zstack=np.cumsum(np.append(0.0,np.array(dstack))) 
    stacksize=np.sum(dstack)  #total stack thickness
    numdomains=len(zstack)+1 
    tightoverhang=1.0*np.max([-min(z),max(z)-stacksize,0.0]) # for z in half infinite media we define an overhang to define a finite interval in which they fit - the 1.0 makes it tight
    return zstack,stacksize,numdomains,tightoverhang
  
def stackseparator(nstack,dstack):
    n2=nstack[0]
    n3=nstack[-1]
    ninterior=nstack[1:-1]
 
    ndlist = np.transpose([ninterior,dstack]) 
    return n2,n3,ndlist,ninterior
    
def pinpointdomain(zlist,dstack):
#  given a list of z values (real) in zlist, and a set of layer thickness (also real, first interface is at z=0),  returns for each z in zlist 
#  the index m corresponding to the domain that z is in  (m=0 stands for left half space, counting up)    

## the idea is as follows:
#    - create a list of domain boundaries, named zstack
#    - to deal with the half infinite spaces if you want to avoid if statements
#      you create domain boundaries that are in the infinite half spaces, sufficiently far out to to contain all the points. We call this amount "overhang"
#    - then you loop over all the domains and decide if the points are in or out of the domain
#      (you could also loop over zlist instead of zstack, but typically you might have many zlists, and few domains)

    
    zstack,stacksize,numdomains,overhang=basicgeoinfo(zlist,dstack)  
    zstack= np.append([-overhang,0.0],np.cumsum(np.append(dstack,overhang))) # domain boundaries, including fictitious slabs into half infinites, needed to do the sorting
    library=np.zeros_like(zlist,dtype=int)  #    
    for m in range(numdomains): #loop over all domains
        indices=[i for i, x in enumerate(zlist) if zstack[m]<= x <= zstack[m+1]]
        library[indices]=m
      
    return library



####################        Coordinate reshuffle routine                #################
def providecoordinates(m,z,nstack,dstack):
    # m:   assignment given z and geometry, of coordinates to domains, returned from pinpointdomain
    #      this must be a single integer, meaning the data is assumed sliced in small lists of z that are just in one given layer
    # z-values:  list of z-values
    # dstack,nstack geometry 
    # reshuffles layer definitions to match Amos and Barnes routines 
  
 
    zstack,stacksize,numdomains,tightoverhang=basicgeoinfo(z,dstack)  
    
    # parameters of half infinite spaces
    n2,n3,dum,ninterior=stackseparator(nstack,dstack) 

 
    if m==0:
        #points are in z<0 half infinite medium
        # to use slab routine, we define a fictitious finite thickness slab index matched with the n2 half infinite medium
        # within which the points z of interest lie
        dslab=2.0*tightoverhang
        nslab=n2
        zz=dslab+z
        nd2list=np.transpose([[],[]])
        nd3list=np.transpose([ninterior,dstack])
        
      
        
    elif m==numdomains-1:
        #points are in z>0 half infinite medium beyond the stack
        # to use slab routine, we define a fictitious finite thickness slab index matched with the n3 half infinite medium
        # within which the points z of interest lie
        
        dslab=2.0*tightoverhang
        nslab=n3
        zz=z-stacksize
        nd2list=np.transpose([ninterior[::-1],dstack[::-1]]) # note the reverse ordering
        nd3list=np.transpose([[],[]])
        
        
    else :
        #points are in an actual finite thikcness layer, layer m.
        #the m-1 because m=0 labels the half-infinite layer
        dslab=dstack[m-1]
        nslab=ninterior[m-1]
        zz=z-zstack[m-1]
        
        n=ninterior[:m-1]
        d=dstack[:m-1]
        nd2list=np.transpose([n[::-1],d[::-1]]) # note the reverse odering
          
        nd3list=np.transpose([ninterior[m:],dstack[m:]])
    
    return zz,dslab,nslab,n2,n3,nd2list,nd3list
    
def nvalueatzposition(z,nstack,dstack):
    
    m=pinpointdomain(z,dstack)
    return np.array(nstack)[m]


def spherical2cartesian(theta,phi):
    'returns the spherical unit vectors for radiation patterns, and the s- p- unit vectors'
    
    
    cosp=np.cos(phi)
    sinp=np.sin(phi)
    cost=np.cos(theta)
    sint=np.sin(theta)
    

    khat=[cosp*sint, sinp*sint, cost] 
    
    shat=[-sinp,cosp,0.0*cosp]
    phat=[cosp*cost,sinp*cost,-sint]
    


    return khat,shat,phat


    

