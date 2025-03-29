#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print('Making Figure 2')

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

from Library.Use import Use_Radiationpattern as Radpat
from Library.Util import Util_vectorpolarization as vector

import matplotlib.pyplot as plt
import numpy as np

#%%


'''Setting up layer geometry and wavelength'''

plt.close('all')
lam=1
k0=2.*np.pi/lam
nstack=[1.5,2.,1.0]

dstack=[0.1*lam]
z=+dstack[0]/2

Nthe=501
Nphi = 151
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(-np.pi/2,3.0*np.pi/2,Nthe)

###### UPPER HEMISPHERE
 
pu=[1.0,0.0,0.0]
mu=[0.0,0.0,0.0]  

P_u_x,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0*thelist,nstack,dstack)

pu=[0.0,1.0,0.0]
mu=[0.0,0.0,0.0]  
 
P_u_y,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0*thelist,nstack,dstack)

pu=[0.0,0.0,1.0]
mu=[0.0,0.0,0.0]  
 
P_u_z,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0*thelist,nstack,dstack)

P_u = P_u_x + P_u_y + P_u_z


f,ax = plt.subplots(1,1,figsize=(5,10),subplot_kw=dict(projection="polar"), dpi=200)
f.subplots_adjust(hspace=0.15)
    
ax.plot(thelist,P_u/np.max(P_u),'r', label = r'$p_z$')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

rmax = 1.  # Manually set maximum radial value
ax.set_rlim([0, rmax+0.1])  # Set radial axis limits
ax.set_yticks([rmax / 2, rmax / 1.33, rmax / 4, rmax])  

r_max = ax.get_ylim()[1]  # Get the plot's radial limit

angle = np.arcsin(nstack[2]/nstack[0])
ax.plot([np.pi - angle, np.pi - angle], [0, r_max], '-', color= 'gray', alpha = 0.4, linewidth=1.)  # Dashed black line
ax.plot([np.pi + angle, np.pi + angle], [0, r_max], '-', color= 'gray', alpha = 0.4, linewidth=1.)  # Dashed black line

ax.set_yticklabels([])

file = [folder,'Fig_2a_Polar_isotropicDipole'+'.pdf']
savefig(file[0], file[1])
plt.show()





#%%
Nthe=101
Nphi = 151
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(+0.01,(np.pi/2)-0.001,Nthe)


pu=[1.0,0.0,0.0]
mu=[0.0,0.0,0.0]  

P_u_x,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_up_px=outE[0]
Ep_up_px=outE[1]

kx, ky, S0_x, S1_x, S2_x, S3_x = vector.BFPplotpassport(theta_u,phi_u,Es_up_px,Ep_up_px,nstack[-1],title=' ',basis='cartesian') 

 
pu=[0.0,1.0,0.0]
mu=[0.0,0.0,0.0]  
 
P_u_y,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_up_py=outE[0]
Ep_up_py=outE[1]

kx, ky, S0_y, S1_y, S2_y, S3_y = vector.BFPplotpassport(theta_u,phi_u,Es_up_py,Ep_up_py,nstack[-1],title=' ',basis='cartesian') 

pu=[0.0,0.0,1.0]
mu=[0.0,0.0,0.0]  
 
P_u_z,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_up_pz=outE[0]
Ep_up_pz=outE[1]

kx, ky, S0_z, S1_z, S2_z, S3_z = vector.BFPplotpassport(theta_u,phi_u,Es_up_pz,Ep_up_pz,nstack[-1],title=' ',basis='cartesian') 
 

S0_up=S0_x+S0_y+S0_z
S1_up=S1_x+S1_y+S1_z
S2_up=S2_x+S2_y+S2_z
S3_up=S3_x+S3_y+S3_z




#%%

fig,ax=plt.subplots(figsize=(6,5), dpi=400)
pcm = ax.pcolormesh(kx, ky, S0_up/np.max(S0_up), vmin=0, vmax=1, cmap='inferno', shading='gouraud', rasterized = True)
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

file = [folder,'Fig_2b_air_S0'+'.pdf']
savefig(file[0], file[1])
plt.show()
plt.close()


s = 1
fig,ax=plt.subplots(figsize=(6,5), dpi=400)
pcm = ax.pcolormesh(kx, ky, S1_up/S0_up, vmin=-s, vmax=s, cmap='bwr', shading='gouraud', rasterized = True)
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


file = [folder,'Fig_2c_air_S1_S0'+'.pdf']
savefig(file[0], file[1])
plt.show()
plt.close()


fig,ax=plt.subplots(figsize=(6,5), dpi=400)
pcm = ax.pcolormesh(kx, ky, S2_up/S0_up, vmin=-s, vmax=s, cmap='bwr', shading='gouraud', rasterized = True)
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


file = [folder,'Fig_2d_air_S2_S0'+'.pdf']
savefig(file[0], file[1])
plt.show()
plt.close()


fig,ax=plt.subplots(figsize=(6,5), dpi=400)
pcm = ax.pcolormesh(kx, ky, S3_up/S0_up, vmin=-s, vmax=s, cmap='bwr', shading='gouraud', rasterized = True)
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


file = [folder,'Fig_2e_air_S3_S0'+'.pdf']
savefig(file[0], file[1])
plt.show()
plt.close()

#%%
'''LOWER HEMISPHERE'''
thelist=-thelist+np.pi

 
pu=[1.0,0.0,0.0]
 
P_u,outE,[theta_d,phi_d]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_up_px=outE[0]
Ep_up_px=outE[1]

kx, ky, S0_x, S1_x, S2_x, S3_x = vector.BFPplotpassport(theta_d,-phi_d,Es_up_px,Ep_up_px,nstack[0],title=' ',basis='cartesian') 

pu=[0.0,1.0,0.0]
 
P_u,outE,[theta_d,phi_d]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_up_py=outE[0]
Ep_up_py=outE[1]

kx, ky, S0_y, S1_y, S2_y, S3_y = vector.BFPplotpassport(theta_d,-phi_d,Es_up_py,Ep_up_py,nstack[0],title=' ',basis='cartesian') 

pu=[0.0,.0,1.0]
 
P_u,outE,[theta_d,phi_d]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_up_pz=outE[0]
Ep_up_pz=outE[1]

kx, ky, S0_z, S1_z, S2_z, S3_z = vector.BFPplotpassport(theta_d,-phi_d,Es_up_pz,Ep_up_pz,nstack[0],title=' ',basis='cartesian') 


S0_down=S0_x+S0_y+S0_z
S1_down=S1_x+S1_y+S1_z
S2_down=S2_x+S2_y+S2_z
S3_down=S3_x+S3_y+S3_z

#%%

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

file = [folder,'Fig_2f_glass_S0'+'.pdf']
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


file = [folder,'Fig_2g_glass_S1_S0'+'.pdf']
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


file = [folder,'Fig_2h_glass_S2_S0'+'.pdf']
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


file = [folder,'Fig_2i_glass_S3_S0'+'.pdf']
savefig(file[0], file[1])
plt.show()
plt.close()


#%% 
'''WEIRD ANGLE - Novotny type DIPOLE'''

Nthe=501
Nphi = 151
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(-np.pi/2,3.0*np.pi/2,Nthe)

###### UPPER HEMISPHERE
 
pu=[np.sqrt(3.0)/2.0 ,0.0, 0.5]
mu=[0.0,0.0,0.0]  

P_u_x,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0*thelist,nstack,dstack)


P_u = P_u_x 


f,ax = plt.subplots(1,1,figsize=(5,10),subplot_kw=dict(projection="polar"), dpi=200)
f.subplots_adjust(hspace=0.15)
    
ax.plot(thelist,P_u/np.max(P_u),'r', label = r'$p_z$')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

rmax = 1.  # Manually set maximum radial value
ax.set_rlim([0, rmax+0.1])  # Set radial axis limits
ax.set_yticks([rmax / 2, rmax / 1.33, rmax / 4, rmax])  

r_max = ax.get_ylim()[1]  # Get the plot's radial limit

angle = np.arcsin(nstack[2]/nstack[0])
ax.plot([np.pi - angle, np.pi - angle], [0, r_max], '-', color= 'gray', alpha = 0.4, linewidth=1.)  # Dashed black line
ax.plot([np.pi + angle, np.pi + angle], [0, r_max], '-', color= 'gray', alpha = 0.4, linewidth=1.)  # Dashed black line

ax.set_yticklabels([])

file = [folder,'Fig_2j_Polar_NovotnyDipole'+'.pdf']
savefig(file[0], file[1])
plt.show()


#%%


Nthe=101
Nphi = 151
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(+0.01,(np.pi/2)-0.001,Nthe)


'''LOWER HEMISPHERE'''
thelist=-thelist+np.pi

 
 
P_u,outE,[theta_d,phi_d]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_up_px=outE[0]
Ep_up_px=outE[1]


kx, ky, S0_x, S1_x, S2_x, S3_x = vector.BFPplotpassport(theta_d,-phi_d,Es_up_px,Ep_up_px,nstack[0],title=' ',basis='cartesian') 

S0_down=S0_x
S1_down=S1_x
S2_down=S2_x
S3_down=S3_x


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

file = [folder,'Fig_2k_glass_S0'+'.pdf']
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


file = [folder,'Fig_2l_glass_S1_S0'+'.pdf']
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


file = [folder,'Fig_2m_glass_S2_S0'+'.pdf']
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


file = [folder,'Fig_2n_glass_S3_S0'+'.pdf']
savefig(file[0], file[1])
plt.show()
plt.close()
