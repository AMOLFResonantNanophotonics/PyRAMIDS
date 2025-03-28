#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
- validates the Green's function calculations in both free-space and 
    layered media by comparing analytical solutions, the angular spectrum method, and interface reflections. 

- Ensures self-consistency across methods, checks the correctness of scattered Green’s function components, 
    and verifies expected symmetry properties when flipping the interface.

"""
#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%%

from Library.Use import Use_Green as Gs
from Library.Core import Core_Greenslab as coreG

import numpy as np
from matplotlib import pyplot as plt


def simpledyadGplot():
    
    ''' Computes the dyadic Green’s function in homogeneous free space. '''    
    
    
    k0=6.0
    ns=1.0
 
    poslist=np.linspace(0.3,7.9,39) 
    
    rsource=np.array([0,0,0.1])
    
    rvec=np.array([poslist/3-rsource[0],poslist-rsource[1],poslist/20.0-rsource[2]])
    
    
     
    GGGF=coreG.FreeDyadG(ns*k0,rvec) 
    
     
    #fase factor helps to cycle through phase when plotting
    #GGGF=GGGF*checkfase
     

    
    fignum=4
    offsetx=np.array([0,1,0,1])*3
    offsety=np.array([0,1,1,0])*3
    field=['EE','HH','EH','HE']
    lab=['x','y','z']
    for q in range(fignum):
        fig,axs=plt.subplots(3,3,sharex='col',sharey='row')
        for i in range(3):
            for j in range(3):
      #         print(['$G^{'+field[q]+'}_{'+lab[i]+lab[j]+'}$'])
      #          print(i+offsetx[q],j+offsety[q])
                axs[i,j].plot(poslist,np.real(GGGF[i+offsetx[q],j+offsety[q],:]))
                axs[i,j].legend([lab[i]+lab[j]])
     
        
    
        fig.suptitle('Simple dyad $G^{'+field[q]+'}$')
        print(fig.number)
        plt.show()

    return

def testfreespaceGreendyadplot():
    '''Ensures that the angular spectrum method correctly reconstructs the free-space Green’s function'''
        
    k0=6.0
     
    ns=1.5
    nstack=[ns,ns,ns]
    dstack=[15.0]
    
    
    poslist=np.linspace(0.3,7.9,39)
      
    
    rdetect=np.array([poslist/3,poslist,poslist/20.0])
    rsource=np.array([0,0,0.1])
              
    
    GGGF=Gs.GreenFree(k0,nstack,dstack,rdetect,rsource)  
    GGG= Gs.Green(k0,nstack,dstack,rdetect,rsource)
    
  
    #fase factor helps to cycle through phase when plotting
    #GGGF=GGGF*checkfase
    #GGG=GGG*checkfase
    
    
    
    fignum=4
    offsetx=np.array([0,1,0,1])*3
    offsety=np.array([0,1,1,0])*3
    field=['EE','HH','EH','HE']
    lab=['x','y','z']
    for q in range(fignum):
        fig,axs=plt.subplots(3,3,sharex='col',sharey='row')
        for i in range(3):
            for j in range(3):
     
                axs[i,j].plot(poslist,np.real( GGG[i+offsetx[q],j+offsety[q],:]),
                              poslist,np.real( GGGF[i+offsetx[q],j+offsety[q],:]))
                if i+j==0:
                    axs[i,j].legend(['Angular spec '+lab[i]+lab[j],'analyt'])
                else:
                    axs[i,j].legend([lab[i]+lab[j]])
                        
                                         
        fig.suptitle('Free space, angular spec vs simple -  $G^{'+field[q]+'}$')
        print(fig.number)
        plt.show()
    return


def reflectedGtest():
    
    '''Computes Green’s function for a source above a single interface between two media.'''
    
    
    k0=12.0
    
     
    
    ns=3.0
    na=1.0
    nstack=[ns,na,na]
    dstack=[5.5]
     
    
    poslist=np.linspace(0.1,8.1,81)
      
    
    rdetect=np.array([poslist/6,poslist/2,poslist/2])
    rsource=np.array([0,0,4.05/2.])           
    
    
    
    Gscat,Ghom=Gs.GreenS(k0,nstack,dstack,rdetect,rsource) 
    Gfull=     Gs.Green(k0,nstack,dstack,rdetect,rsource) 
    
    
    #fase factor helps to cycle through phase when plotting
    #GGGF=GGGF*checkfase
    #GGG=GGG*checkfase
    
    GGG=Gscat
    GGGF=Ghom
    
    Gdiff=Gfull-Ghom
    
    fignum=4
    offsetx=np.array([0,1,0,1])*3
    offsety=np.array([0,1,1,0])*3
    field=['EE','HH','EH','HE']
    lab=['x','y','z']
    
    for q in range(fignum):
        fig,axs=plt.subplots(3,3)
        for i in range(3):
            for j in range(3):
     
                axs[i,j].plot(poslist,np.real( GGG[i+offsetx[q],j+offsety[q],:]),
                              poslist,np.real(GGGF[i+offsetx[q],j+offsety[q],:]),
                              poslist,np.real(Gdiff[i+offsetx[q],j+offsety[q],:]),'r:')
          
                
                if i+j==0:
                    axs[i,j].legend(['reflected part ','free '+lab[i]+lab[j],'full min free'])
            
                else:
                    axs[i,j].legend([lab[i]+lab[j]])
        fig.suptitle('Reflected, single interf $G^{'+field[q]+'}$')
        print(fig.number)
        plt.show()

    return

def testflippedgeometry():
    
    
    '''
    Mirrors the layered system, placing the interface above the source instead of below.
    
    Compares Green’s function in both configurations.
    
    Checks if:
    Diagonal components remain unchanged.
    Off-diagonal components (cross terms) change sign, as expected for a mirrored system.
    '''
    
    k0=12.0
    
     
    
    ns=3.0
    na=1.0
    nstack=[ns,na,na]
    dstack=[5.5]
    
    poslist=np.linspace(0.1,8.1,81)
    rdetect=np.array([poslist/6,poslist/2,poslist/2])
    rsource=np.array([0,0,4.05/2.])           
    Gfull=     Gs.Green(    k0,nstack,dstack,rdetect,rsource) 
    Gfree=     Gs.GreenFree(k0,nstack,dstack,rdetect,rsource) 
    
    
    nstack=[na,na,ns]
    
    rdetect=[-rdetect[0],-rdetect[1],dstack[0]-rdetect[2]]
    rsource=[-rsource[0],-rsource[1],dstack[0]-rsource[2]]
    
    Gfullmirror=     Gs.Green(    k0,nstack,dstack,rdetect,rsource) 
    Gfreemirror=     Gs.GreenFree(k0,nstack,dstack,rdetect,rsource) 
    
 
       
    fignum=4
    offsetx=np.array([0,1,0,1])*3
    offsety=np.array([0,1,1,0])*3
    field=['EE','HH','EH','HE']
    lab=['x','y','z']
    
    for q in range(fignum):
        fig,axs=plt.subplots(3,3)
        for i in range(3):
            for j in range(3):
     
                axs[i,j].plot(poslist,np.real( Gfull[i+offsetx[q],j+offsety[q],:]),
                              poslist,np.real( Gfullmirror[i+offsetx[q],j+offsety[q],:]),':',
                              poslist,np.real( Gfree[i+offsetx[q],j+offsety[q],:]), 
                              poslist,np.real( Gfreemirror[i+offsetx[q],j+offsety[q],:]),':')
          
                if i+j==0:
                    axs[i,j].legend(['full '+lab[i]+lab[j],'mirrored', 'free','free mirrored'])
               
                else:
                    axs[i,j].legend([lab[i]+lab[j]])

        fig.suptitle('Single interf - with flipped geometry  $G^{'+field[q]+'}$')
        print(fig.number)
        plt.show() 
    
    return


plt.close('all')

print('Testing of Green function routine (Vectorised version)')

print('(1)  Plotting the dydadic free space Green function that is reference')

simpledyadGplot()

print('(2)  Plotting the dydadic free space Green function according to angular spectrum method')
print('[Full code complexity,  compared to simple analytical]')
print('May appear slow, due to first-time NUMBA compilation')
print('Figures:')

testfreespaceGreendyadplot()

print('(3)  Single interace reflected G. Concerns sources above (z>0) interface')
print('[Full code complexity, compared to simple analytical')
reflectedGtest()

     
print('(4)  Check that if mirroring the geometry, meaning the interface is ABOVE the source, get the same result')
print('Do note that off diag Gs have a minus sign. Free Dyad G plotted for reference')  
testflippedgeometry()
