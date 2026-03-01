#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
print('Doing Figure 8b-d')

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

from Library.Use import Use_Multiplescattering as ms
from Library.Util import Util_vectorpolarization as radplot
from Library.Util import Util_argumentchecker as check

import numpy as np
from matplotlib import pyplot as plt


#%%
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
V = (r_particle) ** 3

nstack = [1.5, 1.75, 1]
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

invalpha = ms.invalphadynamicfromstatic(alphalist, rdip, k0, nstack, dstack)

M = ms.SetupandSolveCouplingmatrix(invalpha, rdip, k0, nstack, dstack)

#%%
Nthe=151
Nphi = 121
philist=np.linspace(0.001, (2*np.pi)-0.001, Nphi)
thelist=np.linspace(+0.001, (np.pi/2)-0.001, Nthe)

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

#if upper hemisphere....
# Esu,Epu,thetau,phiu=ms.Differentialradiatedfieldmanydipoles(thelist, philist, pnmtotal, rtotal,k0,nstack, dstack)

thelist=-thelist+np.pi
Esd,Epd,thetad,phid=ms.Differentialradiatedfieldmanydipoles(thelist, philist, pnmtotal, rtotal, k0,nstack, dstack)
kx, ky, S0_x, S1_x, S2_x, S3_x = radplot.BFPplotpassport(thetad,-phid,Esd,Epd,nstack[0],title=' ',basis='cartesian', plot = False) 


Esd_no,Epd_no,thetad,phid=ms.Differentialradiatedfieldmanydipoles(thelist, philist, np.reshape(pnmsource,(6,1)), np.reshape(rsource,(3,1)), k0,nstack, dstack)
kx, ky, S0_x_no, S1_x_no, S2_x_no, S3_x_no = radplot.BFPplotpassport(thetad,-phid,Esd_no,Epd_no,nstack[0],title=' ',basis='cartesian', plot = False) 

#%% 'py'
pnmsource=np.array([0.0,1.0,0.0, 0.0, 0.0, 0.0])

driving = ms.Emitterdriving(pnmsource,rsource,rdip,k0,nstack,dstack)

pnm=ms.Solvedipolemoments(M, driving)

rparticles=rdip
pnmparticles=pnm

rtotal=np.hstack([rdip,np.reshape(rsource,(3,1))])  
pnmtotal =np.hstack([pnm,np.reshape(pnmsource,(6,1))])   


Esd,Epd,thetad,phid=ms.Differentialradiatedfieldmanydipoles(thelist, philist, pnmtotal, rtotal, k0,nstack, dstack)
kx, ky, S0_y, S1_y, S2_y, S3_y = radplot.BFPplotpassport(thetad,-phid,Esd,Epd,nstack[0],title=' ',basis='cartesian', plot = False) 

Esd_no,Epd_no,thetad,phid=ms.Differentialradiatedfieldmanydipoles(thelist, philist, np.reshape(pnmsource,(6,1)), np.reshape(rsource,(3,1)), k0,nstack, dstack)
kx, ky, S0_y_no, S1_y_no, S2_y_no, S3_y_no = radplot.BFPplotpassport(thetad,-phid,Esd_no, Epd_no,nstack[0],title=' ',basis='cartesian', plot = False) 


#%% 'pz'
pnmsource=np.array([0.0,0.0,1.0, 0.0, 0.0, 0.0])

driving = ms.Emitterdriving(pnmsource,rsource,rdip,k0,nstack,dstack)

pnm=ms.Solvedipolemoments(M, driving)

rparticles=rdip
pnmparticles=pnm

rtotal=np.hstack([rdip,np.reshape(rsource,(3,1))])  
pnmtotal =np.hstack([pnm,np.reshape(pnmsource,(6,1))])   


Esd,Epd,thetad,phid=ms.Differentialradiatedfieldmanydipoles(thelist, philist, pnmtotal, rtotal, k0,nstack, dstack)
kx, ky, S0_z, S1_z, S2_z, S3_z = radplot.BFPplotpassport(thetad,-phid,Esd,Epd,nstack[0],title=' ',basis='cartesian', plot = False) 


Esd_no,Epd_no,thetad,phid=ms.Differentialradiatedfieldmanydipoles(thelist, philist, np.reshape(pnmsource,(6,1)), np.reshape(rsource,(3,1)),  k0,nstack, dstack)
kx, ky, S0_z_no, S1_z_no, S2_z_no, S3_z_no = radplot.BFPplotpassport(thetad,-phid,Esd_no, Epd_no,nstack[0],title=' ',basis='cartesian', plot = False) 

#%%
#incoherent sum
S0_down = S0_x + S0_y + S0_z
S1_down = S1_x + S1_y + S1_z
S2_down = S2_x + S2_y + S2_z
S3_down = S3_x + S3_y + S3_z


S0_down_no = S0_x_no + S0_y_no + S0_z_no
S1_down_no = S1_x_no + S1_y_no + S1_z_no
S2_down_no = S2_x_no + S2_y_no + S2_z_no
S3_down_no = S3_x_no + S3_y_no + S3_z_no

#%%
# S0/S0_down .. enhancement
plot_kspace_map(kx, ky, S0_down / S0_down_no, nfar= nstack[-1], ticks=(-nstack[0], 0, nstack[0]), title="S0 (Enhancement)", circle=True, filename="Fig_8b_glass_S0.pdf",
    vmin=0.7, vmax=2.5,cmap="inferno", cbar_ticks=[0.7, 1.6, 2.5])
print(np.max(S0_down / S0_down_no))

#%%
# S1/S0
s = 0.7
plot_kspace_map(kx, ky, S1_down / S0_down, nfar= nstack[-1], ticks=(-nstack[0], 0, nstack[0]), title="S1/S0", circle=True, filename="Fig_8c_glass_S1_S0.pdf",
    vmin=-s, vmax=s, cmap="bwr", cbar_ticks=[-s, 0, s])

# S2/S0
plot_kspace_map(kx, ky, S2_down / S0_down, nfar= nstack[-1], ticks=(-nstack[0], 0, nstack[0]), title="S2/S0", circle=True, filename="Fig_8d_glass_S2_S0.pdf",
    vmin=-s, vmax=s, cmap="bwr", cbar_ticks=[-s, 0, s])

s= 0.25
# S3/S0
plot_kspace_map(kx, ky, S3_down / S0_down, nfar= nstack[-1], ticks=(-nstack[0], 0, nstack[0]), title="S3/S0", circle=True, filename="Fig_8e_glass_S3_S0.pdf",
    vmin=-s, vmax=s, cmap="bwr",cbar_ticks=[-s, 0, s])