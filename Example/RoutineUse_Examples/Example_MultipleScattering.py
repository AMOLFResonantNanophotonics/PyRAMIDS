#!/usr/bin/env python3
"""
#Routine-use example: multiple scattering from dipole particles in stacks
#
#Runs a wavelength sweep for a dipole particle and compares extinction/scattering
#cross-sections in homogeneous versus layered surrounding media under plane-wave
#excitation.
#
#Demonstrates core coupled-dipole workflow:
#polarizability -> dynamic inverse polarizability -> driving -> dipole moments
#-> extracted optical cross-sections.

@author: dpal,fkoenderink
"""

#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#%%
from Library.Use import Use_Multiplescattering as ms
from Library.Util import Util_argumentchecker as check
from Library.Util import Util_vectorpolarization as radplot

import numpy as np
from matplotlib import pyplot as plt

def DrudeLorentz(omega, omega_p, gamma):
    return (omega_p**2) / (omega**2 + 1j * gamma * omega)

def Drude(w,drudeparam):
    wSP=drudeparam[0]
    g=drudeparam[1]
    epsinf=drudeparam[2]
    return epsinf-wSP*wSP/(w*(w+1.0j*g))

Nk = 41
lamlist = np.linspace(500, 800, Nk)  # Wavelengths in nm
om = 2.0 * np.pi*2.99792458e17 / lamlist  # angular frequency
klist = 2.0 * np.pi / lamlist

#%% material properties
wp_Au = 1.35E15
g_Au = 1.25E14
eps_inf = 9.54
epsilon_Au = DrudeLorentz(om, wp_Au, g_Au)
drudeparamAu = [wp_Au,g_Au,eps_inf]
eps_Au=Drude(om,drudeparamAu)
nAu = np.sqrt(epsilon_Au) 

r_particle = 50 # nm
V = (r_particle)**3 # in CGS units.. we drop 4pi.. and our rayleigh polz. definition also dont ghave factor of 3.

rdip = np.zeros((3,1))
rdip[2]  = r_particle
rdip=check.checkr(rdip)

#%%normal incidence illumination
theta=np.array([0.])
phi=np.array([0.])

s=np.array([1]) #y-polarization
p=np.array([0])

nstack_list = [
    [1.6, 1.6],        # homogeneous
    [1.45, 1.7, 1.0]   # layered with middle layer 400 nm thick
]

Scat_cs = np.zeros((len(nstack_list), len(lamlist)))
Ext_cs  = np.zeros((len(nstack_list), len(lamlist)))

for j, nstack in enumerate(nstack_list):
    dstack = [] if len(nstack)==2 else [400] # layer definition goes here... with 400nm thickness
    diplayer, Ndip = ms.dipolelayerchecker(rdip ,nstack,dstack)
    
    for i, lam in enumerate(lamlist):
        k0 = 2*np.pi/ lam
        
        alpha=ms.Rayleighspherepolarizability(nAu[i],nstack[diplayer], V)    
        alphalist = np.tile(alpha, (Ndip, 1, 1))
        invalpha = ms.invalphadynamicfromstatic(alphalist, rdip, k0, nstack, dstack)
        driving, intensity =ms.Planewavedriving(theta,phi,s,p,rdip,k0,nstack,dstack)
        M = invalpha
        
        pnm=ms.Solvedipolemoments(M, driving)
        pdipvecs = pnm.reshape(6,1)
            
        work=ms.Work(pnm, rdip, driving, k0,nstack,dstack)  
        Pup, Pdown = ms.TotalfarfieldpowerManydipoles(pdipvecs, rdip, k0, nstack, dstack)
        
        Scat_cs[j,i] = (Pdown + Pup)/np.real(intensity)[0]
        Ext_cs[j,i]  = np.sum(work)/np.real(intensity)[0]
        
#%%
fig, ax = plt.subplots(figsize=(7,5), dpi=300)
for j, nstack in enumerate(nstack_list):
    # ax.plot(om/1E15, Scat_cs[j]*1e-4, ls = '--',label=f" Scat. CS.; Layers RI={nstack}")
    ax.plot(om/1E15, Ext_cs[j]*1e-4, ls = '-',label=f" Ext. CS.; Layers RI={nstack}")

ax.set_xlabel(r"$\omega\ \,[\times 10^{15}\ \mathrm{rad/s}]$")
ax.set_ylabel(r"$\sigma_{\mathrm{ext}}\ \,[\times 10^{-2}\ \mu\mathrm{m}^2]$")
ax.legend()
ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
lam_ticks = np.array([500, 625, 800])  # nm
omega_ticks = 2.0 * np.pi * 2.99792458e17  / lam_ticks / 1e15   
ax_top.set_xticks(omega_ticks)
ax_top.set_xticklabels(lam_ticks.astype(int))
ax_top.set_xlabel("Wavelength (nm)")
plt.title(f"1 Gold sphere, radius = {r_particle} nm in different environments")
plt.tight_layout()
plt.show()

#%%
k0 = 2*np.pi/ 600 # in nm-1
nstack=nstack_list[1]
# nstack = [1.5]*len(nstack_list[1]) # just homogeneous at n=1.5 for example

dstack=np.array([400])

# --- dipole grid ---
N=1.5
x=np.arange(-N,N+1,1)
y=np.arange(-N,N+1,1)

x,y=np.meshgrid(x,y)
x=x.flatten()
y=y.flatten()

rdip=np.array([x,y,x*0+r_particle])
rdip=check.checkr(rdip)
zlist=rdip[2,:]

# --- drive ---
# drive='planewave'
drive='source'

if drive =='planewave':
    theta=np.array([np.pi/6]) # some random angle... circular polz.
    phi=np.array([0.1])
    s=np.array([1])
    p=np.array([1.j])

if drive =='source':
    rsource=np.array([0.0,0.0,200]) # 1 dipole at origin at the height of the middle of the layer at z = 200
    pnmsource=np.array([1.0,0.0,0.0, 0.0, 0.0, 0.0]) # only x oriented E dipole


diplayer, Ndip=ms.dipolelayerchecker(rdip ,nstack,dstack)
alpha=ms.Rayleighspherepolarizability(3.0,nstack[diplayer], V)
alphalist=np.zeros([Ndip,6,6],dtype=complex)
for ii in range(Ndip):
    alphalist[ii,:,:]=alpha

invalpha=ms.invalphadynamicfromstatic(alphalist, rdip, k0, nstack, dstack)    
M=ms.SetupandSolveCouplingmatrix(invalpha,rdip,k0, nstack, dstack)


if drive == 'planewave':
    driving, intensity =ms.Planewavedriving(theta,phi,s,p,rdip,k0,nstack,dstack)


if drive == 'source': 
    driving = ms.Emitterdriving(pnmsource,rsource,rdip,k0,nstack,dstack)
    
pnm=ms.Solvedipolemoments(M, driving)        
work=ms.Work(pnm, rdip, driving, k0, nstack, dstack)  

if drive =='planewave':
    extinction=work/intensity
    print('Extinction [area, in nm^2 if k0  and V are in SI units]:', extinction)
    print('Extinction relative to lambda squared: ', extinction*(0.5*k0*nstack[0]/np.pi)**2)


if drive == 'source':
    rparticles=rdip
    pnmparticles=pnm
    rdip=np.hstack([rdip,np.reshape(rsource,(3,1))])  
    pnm =np.hstack([pnm,np.reshape(pnmsource,(6,1))])   
    

Nthe=101
Nphi = 151
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(+0.001,(np.pi/2)-0.001,Nthe)
Esu,Epu,thetau,phiu=ms.Differentialradiatedfieldmanydipoles(thelist, philist, pnm, rdip,k0,nstack, dstack)

thelist=-thelist+np.pi
Esd,Epd,thetad,phid=ms.Differentialradiatedfieldmanydipoles(thelist, philist, pnm, rdip, k0,nstack, dstack)

radplot.BFPplotpassport(thetau,phiu,Esu,Epu,nstack[-1],'Upper hemisphere')
radplot.BFPplotpassport(thetad,-phid,Esd,Epd,nstack[0],'Lower hemisphere')

Pup, Pdown=ms.TotalfarfieldpowerManydipoles(pnm, rdip, k0, nstack, dstack)

if drive == 'planewave':
    print('nstack:', nstack)
    print('Optical theorem match[should be 1.0 for plane wave input, in absence of guided modes]')
    print((Pup+Pdown)/np.sum(work))



    
