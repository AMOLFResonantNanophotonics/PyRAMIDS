#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
print('Doing Figure 6b')

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

from Library.Use import Use_Multiplescattering as ms
from Library.Util import Util_vectorpolarization as radplot
from Library.Util import Util_argumentchecker as check

import numpy as np
from matplotlib import pyplot as plt


#%%
# Drude Model for Gold
def DrudeLorentz(omega, omega_p, gamma):
    return (omega_p**2) / (omega**2 - 1j * gamma * omega)

#%%

lam = 0.6
k0 = 2*np.pi/ lam
om = 2.0 * np.pi*3E14 / lam  # Convert wavelength to angular frequency

wp_Au = 8E15
g_Au = 8.5E13  
epsilon_Au = DrudeLorentz(om, wp_Au, g_Au)
nAu = np.sqrt(epsilon_Au)  # Convert permittivity to refractive index
r_particle = 0.05
V = (4/3) * np.pi * (r_particle) ** 3

nstack = [1.46, 1.7, 1]
dstack = [0.2]

#%%

pitch = 0.5
N=10
x=np.arange(-N,N+1,1)
y=np.arange(-N,N+1,1)
x,y=np.meshgrid(x,y)
x=x.flatten()
y=y.flatten()

rdip=np.array([x,y,x*0.00+r_particle])*pitch
rdip=check.checkr(rdip)

diplayer, Ndip=ms.dipolelayerchecker(rdip ,nstack,dstack)

alpha=ms.Rayleighspherepolarizability(nAu,nstack[diplayer], V)
alphalist = np.tile(alpha, (Ndip, 1, 1))

invalpha = ms.invalphadynamicfromstatic(alphalist, rdip, diplayer, k0, nstack, dstack)

M = ms.SetupandSolveCouplingmatrix(invalpha, rdip, k0, nstack, dstack)

#%%
Nthe=301
Nphi = 201
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(+0.001,(np.pi/2)-0.001,Nthe)

drive='source'

rsource=np.array([0.0,0.0,dstack[0]*0.5])
#%% 'px'
pnmsource=np.array([1.0,0.0,0.0, 0.0, 0.0, 0.0])

driving = ms.Emitterdriving(pnmsource,rsource,rdip,k0,nstack,dstack)

pnm=ms.Solvedipolemoments(M, driving)

rparticles= rdip
pnmparticles= pnm

rtotal=np.hstack([rdip,np.reshape(rsource,(3,1))])  
pnmtotal =np.hstack([pnm,np.reshape(pnmsource,(6,1))])   


# Esu,Epu,thetau,phiu=ms.Differentialradiatedfieldmanydipoles(thelist, philist, pnmtotal, rtotal, diplayer,k0,nstack, dstack)



thelist=-thelist+np.pi
Esd,Epd,thetad,phid=ms.Differentialradiatedfieldmanydipoles(thelist, philist, pnmtotal, rtotal, diplayer, k0,nstack, dstack)
kx, ky, S0_x, S1_x, S2_x, S3_x = radplot.BFPplotpassport(thetad,-phid,Esd,Epd,nstack[0],title=' ',basis='cartesian') 


Esd_no,Epd_no,thetad,phid=ms.Differentialradiatedfieldmanydipoles(thelist, philist, np.reshape(pnmsource,(6,1)), np.reshape(rsource,(3,1)), diplayer, k0,nstack, dstack)
kx, ky, S0_x_no, S1_x_no, S2_x_no, S3_x_no = radplot.BFPplotpassport(thetad,-phid,Esd_no,Epd_no,nstack[0],title=' ',basis='cartesian') 

#%% 'py'
pnmsource=np.array([0.0,1.0,0.0, 0.0, 0.0, 0.0])

driving = ms.Emitterdriving(pnmsource,rsource,rdip,k0,nstack,dstack)

pnm=ms.Solvedipolemoments(M, driving)

rparticles=rdip
pnmparticles=pnm

rtotal=np.hstack([rdip,np.reshape(rsource,(3,1))])  
pnmtotal =np.hstack([pnm,np.reshape(pnmsource,(6,1))])   


# Esu,Epu,thetau,phiu=ms.Differentialradiatedfieldmanydipoles(thelist, philist, pnmtotal, rtotal, diplayer,k0,nstack, dstack)

# kx_u, ky_u, S0_y_u, S1_y_u, S2_y_u, S3_y_u = radplot.BFPplotpassport(thetau,phiu,Esu,Epu,nstack[-1],title=' ',basis='cartesian') 


# thelist=-thelist+np.pi
Esd,Epd,thetad,phid=ms.Differentialradiatedfieldmanydipoles(thelist, philist, pnmtotal, rtotal, diplayer, k0,nstack, dstack)

# Esd_no,Epd_no,thetad,phid=ms.Differentialradiatedfieldmanydipoles(thelist, -philist, np.reshape(pnmsource,(6,1)), np.reshape(rsource,(3,1)), diplayer, k0,nstack, dstack)

kx, ky, S0_y, S1_y, S2_y, S3_y = radplot.BFPplotpassport(thetad,-phid,Esd,Epd,nstack[0],title=' ',basis='cartesian') 

Esd_no,Epd_no,thetad,phid=ms.Differentialradiatedfieldmanydipoles(thelist, philist, np.reshape(pnmsource,(6,1)), np.reshape(rsource,(3,1)), diplayer, k0,nstack, dstack)
kx, ky, S0_y_no, S1_y_no, S2_y_no, S3_y_no = radplot.BFPplotpassport(thetad,-phid,Esd_no, Epd_no,nstack[0],title=' ',basis='cartesian') 


#%% 'pz'
pnmsource=np.array([0.0,0.0,1.0, 0.0, 0.0, 0.0])

driving = ms.Emitterdriving(pnmsource,rsource,rdip,k0,nstack,dstack)

pnm=ms.Solvedipolemoments(M, driving)

rparticles=rdip
pnmparticles=pnm

rtotal=np.hstack([rdip,np.reshape(rsource,(3,1))])  
pnmtotal =np.hstack([pnm,np.reshape(pnmsource,(6,1))])   


# Esu,Epu,thetau,phiu=ms.Differentialradiatedfieldmanydipoles(thelist, philist, pnmtotal, rtotal, diplayer,k0,nstack, dstack)

# thelist=-thelist+np.pi
Esd,Epd,thetad,phid=ms.Differentialradiatedfieldmanydipoles(thelist, philist, pnmtotal, rtotal, diplayer, k0,nstack, dstack)

kx, ky, S0_z, S1_z, S2_z, S3_z = radplot.BFPplotpassport(thetad,-phid,Esd,Epd,nstack[0],title=' ',basis='cartesian') 


Esd_no,Epd_no,thetad,phid=ms.Differentialradiatedfieldmanydipoles(thelist, philist, np.reshape(pnmsource,(6,1)), np.reshape(rsource,(3,1)), diplayer, k0,nstack, dstack)
kx, ky, S0_z_no, S1_z_no, S2_z_no, S3_z_no = radplot.BFPplotpassport(thetad,-phid,Esd_no, Epd_no,nstack[0],title=' ',basis='cartesian') 

#%%

S0_down = S0_x + S0_y + S0_z
S1_down = S1_x + S1_y + S1_z
S2_down = S2_x + S2_y + S2_z
S3_down = S3_x + S3_y + S3_z


S0_down_no = S0_x_no + S0_y_no + S0_z_no
S1_down_no = S1_x_no + S1_y_no + S1_z_no
S2_down_no = S2_x_no + S2_y_no + S2_z_no
S3_down_no = S3_x_no + S3_y_no + S3_z_no


#%%

s= S0_down/S0_down_no
fig,ax=plt.subplots(figsize=(6,5), dpi=400)
pcm = ax.pcolormesh(kx, ky, S0_down/S0_down_no, vmin=0, vmax= 8, cmap='inferno', shading='gouraud', rasterized = True)
ax.set_title('S0')
ax.set_aspect('equal')
ax.set_xlim([np.min(kx)-.05,np.max(kx)+0.05])
ax.set_ylim([np.min(ky)-.05,np.max(ky)+0.05])
ax.set_aspect('equal')
cbar = fig.colorbar(pcm, ax=ax, shrink = 0.9)
# cbar.set_ticks([0, 0.5 , 1])  # Set ticks at min and max
# cbar.ax.set_yticklabels([f'{0:.2f}', f'{0.5:.2f}',f'{1:.2f}'])
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

file = [folder,'Fig_6b_glass_S0_Enh'+'.pdf']
savefig(file[0], file[1])
plt.show()
plt.close()

#%%
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


file = [folder,'Fig_6c_glass_S1_S0'+'.pdf']
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


file = [folder,'Fig_6d_glass_S2_S0'+'.pdf']
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


file = [folder,'Fig_6e_glass_S3_S0'+'.pdf']
savefig(file[0], file[1])
plt.show()
plt.close()
