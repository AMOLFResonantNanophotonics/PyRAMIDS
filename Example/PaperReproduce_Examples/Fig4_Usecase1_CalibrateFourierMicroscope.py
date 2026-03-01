#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

from Library.Use import Use_Radiationpattern as Radpat
from Library.Util import Util_vectorpolarization as vector

import matplotlib.pyplot as plt
import numpy as np

#%%
def plot_kspace_map(kx, ky, Enh, title, filename, nfar = 1, vmin=None, vmax=None, cmap="inferno",
    circle=True, ticks=(-1, 0, 1), cbar_ticks=None):
    
    fig, ax = plt.subplots(figsize=(6,5), dpi=400)
    pcm = ax.pcolormesh(kx, ky, Enh,vmin=vmin, vmax=vmax,cmap=cmap,shading="gouraud",rasterized=True)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim([kx.min() - 0.05, kx.max() + 0.05])
    ax.set_ylim([ky.min() - 0.05, ky.max() + 0.05])

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t:.2f}" for t in ticks])
    ax.set_yticklabels([f"{t:.2f}" for t in ticks])

    ax.tick_params(axis="both",which="major",direction="in",length=4,width=1,labelsize=16,top=True,right=True)

    ax.set_xlabel(r"$k_x/k_0$", fontsize=16)
    ax.set_ylabel(r"$k_y/k_0$", fontsize=16)

    if circle:
        theta = np.linspace(0, 2*np.pi, 400)
        ax.plot(nfar *np.cos(theta), nfar *np.sin(theta), "--", lw=2, color="k", alpha=0.9)

    cbar = fig.colorbar(pcm, ax=ax, shrink=0.9)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
        cbar.ax.set_yticklabels([f"{t:.2f}" for t in cbar_ticks])
    cbar.ax.tick_params(labelsize=16)

    savefig(folder, filename)
    plt.show()
    plt.close()


#%%
'''Setting up layer geometry and wavelength'''
plt.close('all')
lam=1
k0=2.*np.pi/lam
nstack=[1.5,1.7,1.0]

dstack=[0.4*lam]

#%%
Nthe=501
Nphi = 151
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(-np.pi/2,3.0*np.pi/2,Nthe)


###### UPPER HEMISPHERE
zlist = np.linspace(0+0.0001,1.0-0.0001,100)*dstack[0]
P_u=0
for zs in zlist:
 
    pu=[1.0,0.0,0.0]
    mu=[0.0,0.0,0.0]  
    
    P_u_x,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,zs,pu,mu,thelist,0*thelist,nstack,dstack)
    
    pu=[0.0,1.0,0.0]
    mu=[0.0,0.0,0.0]  
     
    P_u_y,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,zs,pu,mu,thelist,0*thelist,nstack,dstack)
    
    pu=[0.0,0.0,1.0]
    mu=[0.0,0.0,0.0]  
     
    P_u_z,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,zs,pu,mu,thelist,0*thelist,nstack,dstack)
    P_u = P_u + P_u_x + P_u_y + P_u_z


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

file = [folder,'Fig_4a_Polar_isotropicDipole'+'.pdf']
savefig(file[0], file[1])
plt.show()
 

#%%
Nthe=101
Nphi = 151
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(+0.001,(np.pi/2)-0.001,Nthe)

zlist = np.linspace(0+0.0001,1.0-0.0001,100)*dstack[0]

S0_up = 0
S1_up = 0
S2_up = 0
S3_up = 0

for zs in zlist:
 
    pu=[1.0,0.0,0.0]
    mu=[0.0,0.0,0.0]  
    
    P_u_x,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,zs,pu,mu,thelist,philist,nstack,dstack)
    Es_up_px=outE[0]
    Ep_up_px=outE[1]
    
    kx, ky, S0_x, S1_x, S2_x, S3_x = vector.BFPplotpassport(theta_u,phi_u,Es_up_px,Ep_up_px,nstack[-1],title=' ',basis='cartesian', plot = False) 
    
     
    pu=[0.0,1.0,0.0]
    mu=[0.0,0.0,0.0]  
     
    P_u_y,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,zs,pu,mu,thelist,philist,nstack,dstack)
    Es_up_py=outE[0]
    Ep_up_py=outE[1]
    
    kx, ky, S0_y, S1_y, S2_y, S3_y = vector.BFPplotpassport(theta_u,phi_u,Es_up_py,Ep_up_py,nstack[-1],title=' ',basis='cartesian', plot = False) 
    
    pu=[0.0,0.0,1.0]
    mu=[0.0,0.0,0.0]  
     
    P_u_z,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,zs,pu,mu,thelist,philist,nstack,dstack)
    Es_up_pz=outE[0]
    Ep_up_pz=outE[1]
    
    kx, ky, S0_z, S1_z, S2_z, S3_z = vector.BFPplotpassport(theta_u,phi_u,Es_up_pz,Ep_up_pz,nstack[-1],title=' ',basis='cartesian', plot = False) 
     
    S0_up= S0_x+S0_y+S0_z
    S1_up= S1_x+S1_y+S1_z
    S2_up= S2_x+S2_y+S2_z
    S3_up= S3_x+S3_y+S3_z

print(np.max(S0_up))

#%%
# S0
plot_kspace_map(kx, ky, S0_up / np.max(S0_up), title="S0",filename="Fig_4b_air_S0.pdf",
    vmin=0, vmax=1,cmap="inferno", cbar_ticks=[0, 0.5, 1])

# S1/S0
s = 1
plot_kspace_map(kx, ky, S1_up / S0_up, title="S1/S0", filename="Fig_4c_air_S1_S0.pdf",
    vmin=-s, vmax=s, cmap="bwr", cbar_ticks=[-s, 0, s])

# S2/S0
plot_kspace_map(kx, ky, S2_up / S0_up,title="S2/S0", filename="Fig_4d_air_S2_S0.pdf",
    vmin=-s, vmax=s, cmap="bwr", cbar_ticks=[-s, 0, s])

# S3/S0
plot_kspace_map(kx, ky, S3_up / S0_up, title="S3/S0",filename="Fig_4e_air_S3_S0.pdf",
    vmin=-s, vmax=s, cmap="bwr",cbar_ticks=[-s, 0, s])

#%%
'''LOWER HEMISPHERE'''
print('Now doing Lower hemisphere')
thelist=thelist+np.pi/2

S0_down = 0
S1_down = 0
S2_down = 0
S3_down = 0

for zs in zlist:
    pu=[1.0,0.0,0.0]
     
    P_u,outE,[theta_d,phi_d]=Radpat.RadiationpatternPandField(k0,zs,pu,mu,thelist,philist,nstack,dstack)
    Es_up_px=outE[0]
    Ep_up_px=outE[1]
    
    kx, ky, S0_x, S1_x, S2_x, S3_x = vector.BFPplotpassport(theta_d,-phi_d,Es_up_px,Ep_up_px,nstack[0],title=' ',basis='cartesian', plot=False) 
    
    pu=[0.0,1.0,0.0]
     
    P_u,outE,[theta_d,phi_d]=Radpat.RadiationpatternPandField(k0,zs,pu,mu,thelist,philist,nstack,dstack)
    Es_up_py=outE[0]
    Ep_up_py=outE[1]
    
    kx, ky, S0_y, S1_y, S2_y, S3_y = vector.BFPplotpassport(theta_d,-phi_d,Es_up_py,Ep_up_py,nstack[0],title=' ',basis='cartesian',  plot=False) 
    
    pu=[0.0,.0,1.0]
     
    P_u,outE,[theta_d,phi_d]=Radpat.RadiationpatternPandField(k0,zs,pu,mu,thelist,philist,nstack,dstack)
    Es_up_pz=outE[0]
    Ep_up_pz=outE[1]
    
    kx, ky, S0_z, S1_z, S2_z, S3_z = vector.BFPplotpassport(theta_d,-phi_d,Es_up_pz,Ep_up_pz,nstack[0],title=' ',basis='cartesian', plot=False) 
    
    
    S0_down=S0_x+S0_y+S0_z
    S1_down=S1_x+S1_y+S1_z
    S2_down=S2_x+S2_y+S2_z
    S3_down=S3_x+S3_y+S3_z

print(np.max(S0_down))

#%%
# S0
plot_kspace_map(kx, ky, S0_down / np.max(S0_down), nfar= nstack[0], title="S0",filename="Fig_4f_glass_S0.pdf",
    vmin=0, vmax=1,cmap="inferno", cbar_ticks=[0, 0.5, 1])

# S1/S0
s = 1
plot_kspace_map(kx, ky, S1_down / S0_down, nfar= nstack[0], title="S1/S0", filename="Fig_4g_glass_S1_S0.pdf",
    vmin=-s, vmax=s, cmap="bwr", cbar_ticks=[-s, 0, s])

# S2/S0
plot_kspace_map(kx, ky, S2_down / S0_down, nfar= nstack[0], title="S2/S0", filename="Fig_4h_glass_S2_S0.pdf",
    vmin=-s, vmax=s, cmap="bwr", cbar_ticks=[-s, 0, s])

# S3/S0
plot_kspace_map(kx, ky, S3_down / S0_down, nfar= nstack[0], title="S3/S0",filename="Fig_4i_glass_S3_S0.pdf",
    vmin=-s, vmax=s, cmap="bwr",cbar_ticks=[-s, 0, s])

#%% 
'''Single dipole in the middle of the layer - Novotny book example DIPOLE'''
print('Single dipole in the middle of the layer - Novotny book example DIPOLE orientation')

Nthe=501
Nphi = 151
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(-np.pi/2,3.0*np.pi/2,Nthe)

###### UPPER HEMISPHERE
pu=[np.sqrt(3.0)/2.0 ,0.0, 0.5]
mu=[0.0,0.0,0.0]  

z=dstack[0]*0.5


P_u,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0*thelist,nstack,dstack)


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

file = [folder,'Fig_4j_Polar_NovotnyDipole'+'.pdf']
savefig(file[0], file[1])
plt.show()


#%%


Nthe=101
Nphi = 151
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(+0.01,(np.pi/2)-0.001,Nthe)


'''LOWER HEMISPHERE'''
thelist=thelist+np.pi/2

P_u,outE,[theta_d,phi_d]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_up_px=outE[0]
Ep_up_px=outE[1]


kx, ky, S0_x, S1_x, S2_x, S3_x = vector.BFPplotpassport(theta_d,-phi_d,Es_up_px,Ep_up_px,nstack[0],title=' ',basis='cartesian', plot=False) 

S0_down=S0_x
S1_down=S1_x
S2_down=S2_x
S3_down=S3_x

print(np.max(S0_down))

#%%
# S0
plot_kspace_map(kx, ky, S0_down / np.max(S0_down), nfar= nstack[0], title="S0",filename="Fig_4k_glass_S0_arbitrary.pdf",
    vmin=0, vmax=1,cmap="inferno", cbar_ticks=[0, 0.5, 1])

# S1/S0
s = 1
plot_kspace_map(kx, ky, S1_down / S0_down, nfar= nstack[0], title="S1/S0", filename="Fig_4l_glass_S1_S0_arbitrary.pdf",
    vmin=-s, vmax=s, cmap="bwr", cbar_ticks=[-s, 0, s])

# S2/S0
plot_kspace_map(kx, ky, S2_down / S0_down, nfar= nstack[0], title="S2/S0", filename="Fig_4m_glass_S2_S0_arbitrary.pdf",
    vmin=-s, vmax=s, cmap="bwr", cbar_ticks=[-s, 0, s])

# S3/S0
plot_kspace_map(kx, ky, S3_down / S0_down, nfar= nstack[0], title="S3/S0",filename="Fig_4n_glass_S3_S0_arbitrary.pdf",
    vmin=-s, vmax=s, cmap="bwr",cbar_ticks=[-s, 0, s])

