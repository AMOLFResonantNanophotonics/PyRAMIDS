#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

print('Making Figure 4')

#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def savefig(folderpath, filename):
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    plt.savefig(os.path.join(folderpath, filename), bbox_inches='tight')
    
    
folder = r"pdfimages/"

#%%

import numpy as np
import matplotlib.pyplot as plt

from Library.Use import Use_LDOS as ImGLDOS
from Library.Use import Use_Radiationpattern as Farfield
from Library.Util import Util_vectorpolarization as vector

def plot(S0_down, S1_down, S2_down, S3_down, folder, title):
    fig,ax=plt.subplots(figsize=(6,5), dpi=400)
    pcm = ax.pcolormesh(kx, ky, S0_down/np.max(S0_down), vmin=0, vmax=1, cmap='inferno', shading='gouraud', rasterized = True)
    ax.set_title('S0')
    ax.set_aspect('equal')
    ax.set_xlim([np.min(kx)-.05,np.max(kx)+0.05])
    ax.set_ylim([np.min(ky)-.05,np.max(ky)+0.05])
    ax.set_aspect('equal')
    cbar = fig.colorbar(pcm, ax=ax, shrink = 0.9)
    cbar.set_ticks([0, 0.5 , 1])  # Set ticks at min and max
    cbar.ax.set_yticklabels([f'{0:.2f}', f'{0.5:.2f}',f'{1:.2f}'])
    cbar.ax.tick_params(labelsize=16)
    
    xticks = [-1, 0, 1]
    yticks = [-1, 0, 1]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f'{tick:.2f}' for tick in xticks])
    ax.set_yticklabels([f'{tick:.2f}' for tick in yticks])
    
    ax.tick_params(axis='both', labelsize=16)   
    ax.set_xlabel('k$_{x}$/k$_{0}$',fontsize = 16)
    ax.set_ylabel('k$_{y}$/k$_{0}$',fontsize = 16)
    ax.tick_params(axis='both', which='major', direction='in', length=4, width=1.)  # Major ticks
    
    file = [folder,title+'_S0'+'.pdf']
    savefig(file[0], file[1])
    plt.show()
    plt.close()
    
    
    s = 1
    fig,ax=plt.subplots(figsize=(6,5), dpi=400)
    pcm = ax.pcolormesh(kx, ky, S1_down/S0_down, vmin=-s, vmax=s, cmap='bwr', shading='gouraud', rasterized = True)
    ax.set_title('S1/S0')
    ax.set_aspect('equal')
    ax.set_xlim([np.min(kx)-.05,np.max(kx)+0.05])
    ax.set_ylim([np.min(ky)-.05,np.max(ky)+0.05])
    ax.set_aspect('equal')
    cbar = fig.colorbar(pcm, ax=ax, shrink = 0.9)
    cbar.set_ticks([-s, 0 , s])  # Set ticks at min and max
    cbar.ax.tick_params(labelsize=16)
    
    
    xticks = [-1, 0, 1]
    yticks = [-1, 0, 1]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f'{tick:.2f}' for tick in xticks])
    ax.set_yticklabels([f'{tick:.2f}' for tick in yticks])
    
    ax.tick_params(axis='both', labelsize=16)   
    ax.set_xlabel('k$_{x}$/k$_{0}$',fontsize = 16)
    ax.set_ylabel('k$_{y}$/k$_{0}$',fontsize = 16)
    ax.tick_params(axis='both', which='major', direction='in', length=4, width=1.)  # Major ticks
    
    
    file = [folder,title+'_S1_S0'+'.pdf']
    savefig(file[0], file[1])
    plt.show()
    plt.close()
    
    
    fig,ax=plt.subplots(figsize=(6,5), dpi=400)
    pcm = ax.pcolormesh(kx, ky, S2_down/S0_down, vmin=-s, vmax=s, cmap='bwr', shading='gouraud', rasterized = True)
    ax.set_title('S2/S0')
    ax.set_aspect('equal')
    ax.set_xlim([np.min(kx)-.05,np.max(kx)+0.05])
    ax.set_ylim([np.min(ky)-.05,np.max(ky)+0.05])
    ax.set_aspect('equal')
    cbar = fig.colorbar(pcm, ax=ax, shrink = 0.9)
    cbar.set_ticks([-s, 0 , s])  # Set ticks at min and max
    cbar.ax.tick_params(labelsize=16)
    
    
    xticks = [-1, 0, 1]
    yticks = [-1, 0, 1]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f'{tick:.2f}' for tick in xticks])
    ax.set_yticklabels([f'{tick:.2f}' for tick in yticks])
    
    ax.tick_params(axis='both', labelsize=16)   
    ax.set_xlabel('k$_{x}$/k$_{0}$',fontsize = 16)
    ax.set_ylabel('k$_{y}$/k$_{0}$',fontsize = 16)
    ax.tick_params(axis='both', which='major', direction='in', length=4, width=1.)  # Major ticks
    
    
    file = [folder,title+'_S2_S0'+'.pdf']
    savefig(file[0], file[1])
    plt.show()
    plt.close()
    
    
    fig,ax=plt.subplots(figsize=(6,5), dpi=400)
    pcm = ax.pcolormesh(kx, ky, S3_down/S0_down, vmin=-s, vmax=s, cmap='bwr', shading='gouraud', rasterized = True)
    ax.set_title('S3/S0')
    ax.set_aspect('equal')
    ax.set_xlim([np.min(kx)-.05,np.max(kx)+0.05])
    ax.set_ylim([np.min(ky)-.05,np.max(ky)+0.05])
    ax.set_aspect('equal')
    cbar = fig.colorbar(pcm, ax=ax, shrink = 0.9)
    cbar.set_ticks([-s, 0 , s])  # Set ticks at min and max
    cbar.ax.tick_params(labelsize=16)
    
    
    xticks = [-1, 0, 1]
    yticks = [-1, 0, 1]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f'{tick:.2f}' for tick in xticks])
    ax.set_yticklabels([f'{tick:.2f}' for tick in yticks])
    
    ax.tick_params(axis='both', labelsize=16)   
    ax.set_xlabel('k$_{x}$/k$_{0}$',fontsize = 16)
    ax.set_ylabel('k$_{y}$/k$_{0}$',fontsize = 16)
    ax.tick_params(axis='both', which='major', direction='in', length=4, width=1.)  # Major ticks
    
    
    file = [folder,title+'_S3_S0'+'.pdf']
    savefig(file[0], file[1])
    plt.show()
    plt.close()



def plot_polar(P, thelist, folder, title):
    
    f,ax = plt.subplots(1,1,figsize=(5,10),subplot_kw=dict(projection="polar"), dpi=200)
    f.subplots_adjust(hspace=0.15)
        
    ax.plot(thelist,P/np.max(P),'r', label = r'$p_z$')
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    
    rmax = 1.  # Manually set maximum radial value
    ax.set_rlim([0, rmax+0.1])  # Set radial axis limits
    ax.set_yticks([rmax / 2,  rmax])  
    
    r_max = ax.get_ylim()[1]  # Get the plot's radial limit
    
    angle = np.arcsin(nstack[2]/nstack[0])
    ax.plot([np.pi - angle, np.pi - angle], [0, r_max], '-', color= 'gray', alpha = 0.4, linewidth=1.)  # Dashed black line
    ax.plot([np.pi + angle, np.pi + angle], [0, r_max], '-', color= 'gray', alpha = 0.4, linewidth=1.)  # Dashed black line
    
    ax.set_yticklabels([])
    
    file = [folder,title+'_polar'+'.pdf']
    savefig(file[0], file[1])
    plt.show()
    plt.close()

#%%

lam=1
k0=2.*np.pi/lam

nstack=[1.5,3.5,1.0]
dstack=[0.1*lam]
kparlist=np.linspace(0,4.0,1500)
zlist=np.linspace(-0.6,0.7,1200)

#%%

p=ImGLDOS.LDOSatanyPandM([1.0,0,0], [0,0,0], k0, zlist, nstack, dstack)
m=ImGLDOS.LDOSatanyPandM([0.0,0,0], [0,1,0], k0, zlist, nstack, dstack)
spinp=ImGLDOS.LDOSatanyPandM([1.0,1j,0], [0,0,0], k0, zlist, nstack, dstack)
spinm=ImGLDOS.LDOSatanyPandM([0,0,0], [1.0,1j,0], k0, zlist, nstack, dstack)

kerker=ImGLDOS.LDOSatanyPandM([1.0,0,0], [0,1,0], k0, zlist, nstack, dstack)
smin=ImGLDOS.LDOSatanyPandM([1.0,0,0], [0,1.j,0], k0, zlist, nstack, dstack)
splus=ImGLDOS.LDOSatanyPandM([1.0,0,0], [0,-1.j,0], k0, zlist, nstack, dstack)

plt.figure(figsize=(4,5))
plt.plot(zlist,p,'violet',zlist,m,'b',zlist[::12],spinp[::12],'r-.',zlist,spinm,'r--', zlist,kerker,'k',zlist,smin,'g',zlist,splus,'g-.')
plt.legend(['LDOS for px','my','Spinning p','Spining m','Kerker','(px,my)=(1,i)','(1,-i)'])
plt.ylabel(r'z / $\lambda$')
plt.ylabel('LDOS over vacuum LDOS')
plt.xlim([0.075,0.6])
file = [folder,'Fig_4_LDOS'+'.pdf']
savefig(file[0], file[1])
plt.show()
plt.show()


#%%

z=+dstack[0]/2


Nthe=201
Nphi = 301
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(+0.01,(np.pi/2)-0.001,Nthe)
thelist=thelist+np.pi/2


#%%px 
'''spining electric dipole dipole'''

pu=[1.0,1j,0.0]  # electric dipole
mu=[0.0,0.0,0.0]  # magnetic dipole
 
P_d,outE,[theta_d,phi_d]=Farfield.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_down_px=outE[0]
Ep_down_px=outE[1]
  
kx, ky, S0_x, S1_x, S2_x, S3_x = vector.BFPplotpassport(theta_d,-phi_d,Es_down_px,Ep_down_px,nstack[0],title=' ',basis='cartesian') 

plot(S0_x, S1_x, S2_x, S3_x, folder, 'Fig_4_spinning_electricdipole')
print(np.max(S0_x))
#%%px 
'''spining magnetic dipole dipole'''

mu=[1.0,1j,0.0]  # electric dipole
pu=[0.0,0.0,0.0]  # magnetic dipole
 
P_d,outE,[theta_d,phi_d]=Farfield.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_down_px=outE[0]
Ep_down_px=outE[1]
  
kx, ky, S0_x, S1_x, S2_x, S3_x = vector.BFPplotpassport(theta_d,-phi_d,Es_down_px,Ep_down_px,nstack[0],title=' ',basis='cartesian') 

plot(S0_x, S1_x, S2_x, S3_x, folder, 'Fig_4_spinning_magneticdipole')
print(np.max(S0_x))



#%% 
'''Kerker dipole'''

pu=[1.0, 0.0, 0.0]  # electric dipole
mu=[0.0, 1.0, 0.0]  # magnetic dipole
 
P_d,outE,[theta_d,phi_d]=Farfield.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_down_px=outE[0]
Ep_down_px=outE[1]
  
kx, ky, S0_x, S1_x, S2_x, S3_x = vector.BFPplotpassport(theta_d,-phi_d,Es_down_px,Ep_down_px,nstack[0],title=' ',basis='cartesian') 

plot(S0_x, S1_x, S2_x, S3_x, folder, 'Fig_4_kerkerdipole')
print(np.max(S0_x))

#%%
'''Magneto-electric Dipole'''
pu=[1.0,0.0,0.0]  # electric dipole
mu=[0.0,1j,0.0]  # magnetic dipole
 
P_d,outE,[theta_d,phi_d]=Farfield.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_down_px=outE[0]
Ep_down_px=outE[1]
  
kx, ky, S0_x, S1_x, S2_x, S3_x = vector.BFPplotpassport(theta_d,-phi_d,Es_down_px,Ep_down_px,nstack[0],title=' ',basis='cartesian') 

plot(S0_x, S1_x, S2_x, S3_x, folder, 'Fig_4_magnetoelectric_pos_m1j_dipole')
print(np.max(S0_x))


#%%

pu=[1.0,0.0,0.0]  # electric dipole
mu=[0, -1j,0.0]  # magnetic dipole
 
P_d,outE,[theta_d,phi_d]=Farfield.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_down_px=outE[0]
Ep_down_px=outE[1]
  
kx, ky, S0_x, S1_x, S2_x, S3_x = vector.BFPplotpassport(theta_d,-phi_d,Es_down_px,Ep_down_px,nstack[0],title=' ',basis='cartesian') 
plot(S0_x, S1_x, S2_x, S3_x, folder, 'Fig_4_magnetoelectric_neg_m1j_dipole')
print(np.max(S0_x))

#%%
'''POLAR PLOTS'''

#%%

# nstack=[1.,1.,1.0]

Nthe=201
thelist=np.linspace(-np.pi/2,3.0*np.pi/2,Nthe)


#%%px 
'''spining electric dipole dipole'''

pu=[1.0,1j,0.0]  # electric dipole
mu=[0.0,0.0,0.0]  # magnetic dipole
 
P_d,outE,[theta_d,phi_d]=Farfield.RadiationpatternPandField(k0,z,pu,mu,thelist,0*thelist,nstack,dstack)

plot_polar(P_d, thelist, folder, 'Fig_4_spinning_electricdipole')

#%%px 
'''spining magnetic dipole dipole'''

mu=[1.0,1j,0.0]  # electric dipole
pu=[0.0,0.0,0.0]  # magnetic dipole
 
P_d,outE,[theta_d,phi_d]=Farfield.RadiationpatternPandField(k0,z,pu,mu,thelist,0*thelist,nstack,dstack)

plot_polar(P_d, thelist, folder, 'Fig_4_spinning_magneticdipole')

#%% 
'''Kerker dipole'''

pu=[1.0, 0.0, 0.0]  # electric dipole
mu=[0.0, 1.0, 0.0]  # magnetic dipole
 
P_d,outE,[theta_d,phi_d]=Farfield.RadiationpatternPandField(k0,z,pu,mu,thelist,0*thelist,nstack,dstack)

plot_polar(P_d, thelist, folder, 'Fig_4_kerkerdipole')


#%%
'''Magneto-electric Dipole'''
pu=[1.0,0.0,0.0]  # electric dipole
mu=[0.0,1j,0.0]  # magnetic dipole
 
P_d,outE,[theta_d,phi_d]=Farfield.RadiationpatternPandField(k0,z,pu,mu,thelist,0*thelist,nstack,dstack)

plot_polar(P_d, thelist, folder, 'Fig_4_magnetoelectric_pos_m1j_dipole')


#%%

pu=[1.0,0.0,0.0]  # electric dipole
mu=[0, -1j,0.0]  # magnetic dipole

P_d,outE,[theta_d,phi_d]=Farfield.RadiationpatternPandField(k0,z,pu,mu,thelist,0*thelist,nstack,dstack)

plot_polar(P_d, thelist, folder, 'Fig_4_magnetoelectric_neg_m1j_dipole')








