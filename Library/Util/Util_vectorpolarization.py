"""
@author: dpal,fkoenderink

Transform Far field radiation patterns into 2D back focal plane & Stokes polarimetry 


"""




#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%% 
from Library.Util import Util_argumentchecker as check
from Library.Util import Util_argumentrewrapper as cc


import numpy as np
import matplotlib.pyplot as plt

 

def FarfieldEsp2Ecartesian3D(theta,phi,Es,Ep):

    check.checkBFPinput(theta,phi,Es,Ep)
    
    khat,shat,phat=cc.spherical2cartesian(theta,phi)
    
    Ex=Es*shat[0]+Ep*phat[0]
    Ey=Es*shat[1]+Ep*phat[1]
    Ez=Es*shat[2]+Ep*phat[2]

    return Ex,Ey,Ez


def FarfieldEsp2EcartesianBFP(theta,phi,Es,Ep) :
    check.checkBFPinput(theta,phi,Es,Ep)

    khat,shat,phat=cc.spherical2cartesian(theta,phi)
    # basis vectors for the cylindrical coordinate space of BFPs
    # on basis of the spherical s- and p- basis vectors
    # Note the subtlety that the radial vector points outwards on the upper hemisphere, and downwards on the lower hemisphere
    renormp=np.sqrt(phat[0]**2+phat[1]**2)
    radial=[phat[0]/renormp,phat[1]/renormp]
    tangential=[shat[0],shat[1]]
    
    #Cartesian fields in the back focal plane
    Ex=Es*tangential[0]+Ep*radial[0]
    Ey=Es*tangential[1]+Ep*radial[1]
    return Ex,Ey



def Intensity2Stokes(Ix,Iy,Idp,Idm,Ircp,Ilcp):
    check.checkBFPinput(Ix,Iy,Idp,Idm)
    check.checkBFPinput(Ix,Iy,Ircp,Ilcp)

    S0=Ix+Iy #Stokes 0, total intensity
    S1=Ix-Iy #S1 parameter 
    S2=Idp-Idm  #S2 ,difference of linear diagonal chann
    S3=Ircp-Ilcp  #S3, difference in circular channels
    return S0, S1, S2, S3



def Field2Stokes(Ex,Ey):
    
    #projected on linear diagonal 45, -45

    check.checkequaldimension(Ex,Ey)

    Edp=(Ex+Ey)*np.sqrt(0.5)
    Edm=(Ex-Ey)*np.sqrt(0.5)
    #projected on circular right and left
    ERCP=(Ex-1.j*Ey)*np.sqrt(0.5)
    ELCP=(Ex+1.j*Ey)*np.sqrt(0.5)
    
    
    S0,S1,S2,S3=Intensity2Stokes(np.abs(Ex)**2,np.abs(Ey)**2,
                                 np.abs(Edp)**2,np.abs(Edm)**2,
                                 np.abs(ERCP)**2,np.abs(ELCP)**2)
    
    ellipticity=0.5*np.angle(np.sqrt(S1**2+S2**2)+1.j*S3) #ellipticity
    majororientation=0.5*np.angle(S1+1.j*S2) #ellips major axis orientation

              
    return S0,S1,S2,S3,ellipticity,majororientation
 
    
def BFPcartesianStokes(theta,phi,Es,Ep) :
    check.checkBFPinput(theta,phi,Es,Ep)

    Ex,Ey=FarfieldEsp2EcartesianBFP(theta, phi, Es, Ep)
    return Field2Stokes(Ex,Ey) 
    

def BFPspStokes(theta,phi,Es,Ep) :
    check.checkBFPinput(theta,phi,Es,Ep)
    s=np.sign(np.cos(theta))
    S0,S1,S2,S3,ellipticity,majororientation=Field2Stokes(s*Ep,Es)
    majororientation=0.5*np.angle(np.exp(2.0j*(majororientation+phi)))
    return S0,S1,S2,S3,ellipticity,majororientation


def BFPplotIntensity(theta,phi,Es,Ep,nrefr,title=' ',basis='cartesian') :
    check.checkBFPinput(theta,phi,Es,Ep)
    khat,shat,phat=cc.spherical2cartesian(theta,phi)
    kx = khat[0]*nrefr 
    ky = khat[1]*nrefr
    
    if basis=='sp' :
        S0,S1,S2,S3, epsilon,alpha = BFPspStokes(theta,phi,Es,Ep)
        title=title+'(sp-basis)'
    else:
        S0,S1,S2,S3, epsilon,alpha = BFPcartesianStokes(theta,phi,Es,Ep)
        title=title+'(cartesian analysis)'
    
    fig,ax=plt.subplots(figsize=(6,5))
    s=S0.max()
    pcm = ax.pcolormesh(kx,ky,S0,vmin=0,vmax=s,cmap='jet', shading='gouraud')
    ax.set_title('S0')
    ax.set_aspect('equal')
    ax.set_xlim([np.min(kx)-.05,np.max(kx)+0.05])
    ax.set_ylim([np.min(ky)-.05,np.max(ky)+0.05])
    ax.set_aspect('equal')
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.ax.tick_params(labelsize=16)
    ax.tick_params(axis='both', labelsize=16)   
    ax.set_xlabel('k$_{x}$/k$_{0}$',fontsize = 16)
    ax.set_ylabel('k$_{y}$/k$_{0}$',fontsize = 16)
    
    plt.show()
 
    return S0

def BFPplotpassport(theta,phi,Es,Ep,nrefr,title=' ',basis='cartesian') :
    check.checkBFPinput(theta,phi,Es,Ep)
    khat,shat,phat=cc.spherical2cartesian(theta,phi)
    kx = khat[0]*nrefr 
    ky = khat[1]*nrefr
    
    if basis=='sp' :
        S0,S1,S2,S3, epsilon,alpha = BFPspStokes(theta,phi,Es,Ep)
        title=title+'(sp-basis)'
    else:
        S0,S1,S2,S3, epsilon,alpha = BFPcartesianStokes(theta,phi,Es,Ep)
        title=title+'(cartesian analysis)'
    
    
    fig,ax=plt.subplots(2,2,figsize=(15,12))
    s=S0.max()
    pcm = ax[0][0].pcolormesh(kx,ky,S0,vmin=0,vmax=s,cmap='inferno', shading='gouraud')
    ax[0][0].set_title('S0',fontsize = 18)
    ax[0][0].set_aspect('equal')
    ax[0][0].set_xlim([-nrefr,nrefr])
    ax[0][0].set_ylim([-nrefr,nrefr])
    ax[0][0].set_xlabel('k$_{x}$/k$_{0}$',fontsize = 18)
    ax[0][0].set_ylabel('k$_{y}$/k$_{0}$',fontsize = 18)
    cbar = fig.colorbar(pcm, ax=ax[0][0], shrink=0.9)
    cbar.ax.tick_params(labelsize=16)
    
    s = np.max(S1/S0)
    pcm = ax[0][1].pcolormesh(kx,ky,S1/S0,vmin=-s,vmax=s,cmap='bwr', shading='gouraud')
    ax[0][1].set_title('S1/S0',fontsize = 18) 
    ax[0][1].set_aspect('equal')
    ax[0][1].set_xlim([-nrefr,nrefr])
    ax[0][1].set_ylim([-nrefr,nrefr])
    cbar = fig.colorbar(pcm, ax=ax[0][1], shrink=0.9)
    cbar.ax.tick_params(labelsize=16)

    s = np.max(S2/S0)
    pcm = ax[1][0].pcolormesh(kx,ky,S2/S0,vmin=-s,vmax=s,cmap='bwr', shading='gouraud')
    ax[1][0].set_title('S2/S0',fontsize = 18) 
    ax[1][0].set_aspect('equal')
    ax[1][0].set_xlim([-nrefr,nrefr])
    ax[1][0].set_ylim([-nrefr,nrefr])
    cbar = fig.colorbar(pcm, ax=ax[1][0], shrink=0.9)
    cbar.ax.tick_params(labelsize=16)
    
    s = np.max(S3/S0)
    pcm = ax[1][1].pcolormesh(kx,ky,S3/S0,vmin=-s,vmax=s,cmap='bwr', shading='gouraud')
    ax[1][1].set_title('S3/S0',fontsize = 18) 
    ax[1][1].set_aspect('equal')
    ax[1][1].set_xlim([-nrefr,nrefr])
    ax[1][1].set_ylim([-nrefr,nrefr])
    cbar = fig.colorbar(pcm, ax=ax[1][1], shrink=0.9)
    cbar.ax.tick_params(labelsize=16)

    # ax[0][1].contourf(kx,ky,epsilon,vmin=-0.5,vmax=0.5,cmap='coolwarm',levels=50)
    # ax[0][1].set_title('ellipticity')
    # ax[0][1].set_aspect('equal')
    # ax[0][1].set_xlim([-nrefr,nrefr])
    # ax[0][1].set_ylim([-nrefr,nrefr])

    # ax[0][2].contourf(kx,ky,alpha,vmin=-np.pi/2,vmax=np.pi/2,cmap='hsv',levels=50)
    # ax[0][2].set_title('ellipse orientation')
    # ax[0][2].set_aspect('equal')
    # ax[0][2].set_xlim([-nrefr,nrefr])
    # ax[0][2].set_ylim([-nrefr,nrefr])
    
    fig.suptitle(title,fontsize='large')

    plt.show()
    
    fig,ax=plt.subplots(figsize=(6,5))
    s=S0.max()
    pcm = ax.pcolormesh(kx,ky,S0,vmin=0,vmax=s,cmap='inferno', shading='gouraud')
    ax.set_title('S0')
    ax.set_aspect('equal')
    ax.set_xlim([np.min(kx)-.05,np.max(kx)+0.05])
    ax.set_ylim([np.min(ky)-.05,np.max(ky)+0.05])
    ax.set_aspect('equal')
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.ax.tick_params(labelsize=16)
    ax.tick_params(axis='both', labelsize=16)   
    ax.set_xlabel('k$_{x}$/k$_{0}$',fontsize = 16)
    ax.set_ylabel('k$_{y}$/k$_{0}$',fontsize = 16)
    
    plt.show()
 
    return kx, ky, S0, S1, S2, S3




def BFPplotpassportS(theta,phi,S0,S1,S2,S3,nrefr,title=' ') :
    check.checkBFPinput(theta,phi,S0,S1)
    khat,shat,phat=cc.spherical2cartesian(theta,phi)
    kx = khat[0]*nrefr 
    ky = khat[1]*nrefr
    
    
    
    fig,ax=plt.subplots(2,2,figsize=(15,12))
    s=S0.max()
    pcm = ax[0][0].pcolormesh(kx,ky,S0,vmin=0,vmax=s,cmap='inferno', shading='gouraud')
    ax[0][0].set_title('S0',fontsize = 18)
    ax[0][0].set_aspect('equal')
    ax[0][0].set_xlim([-nrefr,nrefr])
    ax[0][0].set_ylim([-nrefr,nrefr])
    ax[0][0].set_xlabel('k$_{x}$/k$_{0}$',fontsize = 18)
    ax[0][0].set_ylabel('k$_{y}$/k$_{0}$',fontsize = 18)
    cbar = fig.colorbar(pcm, ax=ax[0][0], shrink=0.9)
    cbar.ax.tick_params(labelsize=16)
    
    s = np.max(S1/S0)
    pcm = ax[0][1].pcolormesh(kx,ky,S1/S0,vmin=-s,vmax=s,cmap='bwr', shading='gouraud')
    ax[0][1].set_title('S1/S0',fontsize = 18) 
    ax[0][1].set_aspect('equal')
    ax[0][1].set_xlim([-nrefr,nrefr])
    ax[0][1].set_ylim([-nrefr,nrefr])
    cbar = fig.colorbar(pcm, ax=ax[0][1], shrink=0.9)
    cbar.ax.tick_params(labelsize=16)

    s = np.max(S2/S0)
    pcm = ax[1][0].pcolormesh(kx,ky,S2/S0,vmin=-s,vmax=s,cmap='bwr', shading='gouraud')
    ax[1][0].set_title('S2/S0',fontsize = 18) 
    ax[1][0].set_aspect('equal')
    ax[1][0].set_xlim([-nrefr,nrefr])
    ax[1][0].set_ylim([-nrefr,nrefr])
    cbar = fig.colorbar(pcm, ax=ax[1][0], shrink=0.9)
    cbar.ax.tick_params(labelsize=16)
    
    s = np.max(S3/S0)
    pcm = ax[1][1].pcolormesh(kx,ky,S3,vmin=-s,vmax=s,cmap='bwr', shading='gouraud')
    ax[1][1].set_title('S3/S0',fontsize = 18) 
    ax[1][1].set_aspect('equal')
    ax[1][1].set_xlim([-nrefr,nrefr])
    ax[1][1].set_ylim([-nrefr,nrefr])
    cbar = fig.colorbar(pcm, ax=ax[1][1], shrink=0.9)
    cbar.ax.tick_params(labelsize=16)

    # ax[0][1].contourf(kx,ky,epsilon,vmin=-0.5,vmax=0.5,cmap='coolwarm',levels=50)
    # ax[0][1].set_title('ellipticity')
    # ax[0][1].set_aspect('equal')
    # ax[0][1].set_xlim([-nrefr,nrefr])
    # ax[0][1].set_ylim([-nrefr,nrefr])

    # ax[0][2].contourf(kx,ky,alpha,vmin=-np.pi/2,vmax=np.pi/2,cmap='hsv',levels=50)
    # ax[0][2].set_title('ellipse orientation')
    # ax[0][2].set_aspect('equal')
    # ax[0][2].set_xlim([-nrefr,nrefr])
    # ax[0][2].set_ylim([-nrefr,nrefr])
    
    fig.suptitle(title,fontsize='large')

    plt.show()
 
    return

