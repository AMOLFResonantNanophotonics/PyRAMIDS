#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print('PRB 88, 045407 ')
print('Making Figure 6')

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
from Library.Use import Use_Planewaves as pw
from Library.Use import Use_LDOS as ImGLDOS
import numpy as np
from matplotlib import pyplot as plt
from Library.Use import Use_Radiationpattern as Farfield
from matplotlib.colors import LogNorm


plt.close('all')

def Ag(w):
    # Optics Express 18 5124 (2010), fitting silver
    # w is in eV
    eps1=2.1485
    wp=9.1821
    gp=0.0210
    
    w1=4.180
    f1=0.1227
    g1=0.2659
    
    w2=4.5309
    f2=0.2167
    g2=0.4269
    
    w3=5.0094
    f3=0.2925
    g3=0.6929
    
    w4=5.7530
    f4=0.4305
    g4=1.1210
    
    w5=6.9104
    f5=0.6943
    g5=1.3410

    eps= eps1-wp*wp/(w*w+1.0j*gp*w) + f1*w1*w1/(w1*w1-w*w-1.0j*g1*w) +f2*w2*w2/(w2*w2-w*w-1.0j*g2*w) +f3*w3*w3/(w3*w3-w*w-1.0j*g3*w)+f4*w4*w4/(w4*w4-w*w-1.0j*g4*w)+f5*w5*w5/(w5*w5-w*w-1.0j*g5*w)
    return eps


#%%
lammax=0.500   ## range of vac wavelengths in micron
lammin=0.360

dutc=np.array([0.015, 0.015])  # thicknesses of layers in micron
nTiO2=2.83  # 'PRB 88, 045407 does not list actual optical constants used, so guessing both Tio2 and Ag

N=8 # number of layers
 
# plot range
kparlist=np.linspace(0,10.0,201) 
k0list=np.linspace(2.0*np.pi/lammax,2.0*np.pi/lammin,251)

# declare variables
out=np.zeros([5,np.size(k0list),np.size(kparlist)])
p=0*k0list
m=0*k0list
splus=0*k0list
smin=0*k0list

for index,k0 in enumerate(k0list):
    
 
    # in elegant loop over omega, because LDOS routines by themselves don't with dispersive media ...
    lam=2.0*np.pi/k0
    eV=1.2398/lam
    nutc=np.array([np.sqrt(Ag(eV)),nTiO2])

    # sets the unit cell replicates it N times   
    nmulti=np.tile(nutc,N)
    dstack=np.tile(dutc,N)
    
    #if you want to end with a silver layer (half a unit cell extra). Uncleear if PRB 88, 045407 does that
    nmulti=np.append(nmulti,nutc[0])
    dstack=np.append(dstack,dutc[0])

    # pre and postpend half inifinite media as air
    nstack=np.insert(nmulti,0,1.0)
    nstack=np.append(nstack,1.0)
    
    # source 15 nm above stack, expressed in micron
    zsource=np.sum(dstack)+0.015

    # wave vector resolved LDOS integrand
    out[:,index,:]=np.squeeze(ImGLDOS.LDOSintegrandplotdispersion([k0],kparlist,np.array(zsource),nstack,dstack))
 
    # LDOS normalized to vacuum for salient dipole orientations
    p[index]=ImGLDOS.LDOSatanyPandM([1.0,0,0], [0,0,0], k0, np.array(zsource), nstack, dstack)[0]
    m[index]=ImGLDOS.LDOSatanyPandM([0.0,0,0], [0,1,0], k0, np.array(zsource), nstack, dstack)[0]
    splus[index]=ImGLDOS.LDOSatanyPandM([1.0,0,0], [0,1.0j,0], k0, np.array(zsource), nstack, dstack)[0]
    smin[index]=ImGLDOS.LDOSatanyPandM([1.0,0,0], [0,-1.0j,0], k0, np.array(zsource), nstack, dstack)[0]
    
 #%%
# plot equivalent of PRB 88, 045407 Figure 3b
fig, ax = plt.subplots(figsize=(9, 7), dpi=350)
pcm = ax.pcolor(kparlist, k0list, np.abs(out[0,:,:]),  norm=LogNorm(vmin=1e-3, vmax=10**1.2), cmap = 'inferno',rasterized= True)
cbar = fig.colorbar(pcm, ax=ax, shrink=0.9)
cbar.set_label(r'$\log_{10}$ LDOS_E integrand - x dipole', fontsize=16)
cbar.ax.tick_params(labelsize=16)

# pcm.set_clim([-3, 1.8])
special = np.array([490.0, 450.0, 410.0,370.0])
special=2.*np.pi/special*1000
labels=['490','450','410','370']
ax.set_yticks(special,labels)

ax.set_ylabel('Wavelength (nm)', fontsize=16)
ax.set_xlabel('$k_\\parallel / k_0$', fontsize=16)
ax.tick_params(axis="both",which="major",direction="in",length=4,width=1,labelsize=16,top=True,right=True)

file = [folder,'Fig_6_2DLDOS'+'.pdf']
savefig(file[0], file[1])
plt.show()

# plot equivalent of PRB 88, 045407 Figure 2, but for different dipole choices
fig, ax = plt.subplots(figsize=(8, 6), dpi=350)
l=2.0*np.pi/k0list*1000.0
pcm = ax.plot(l,p,'r',l,m,'b',l,splus,'g',l,smin,'-.g')
ax.set_xlabel('Wavelength (nm)', fontsize=16)
ax.set_ylabel('LDOS over vacuum', fontsize=16)
ax.legend(['$p_x$','$m_y$','$p_x+im_y$','$p_x-im_y$'], fontsize=16)
ax.tick_params(axis="both",which="major",direction="in",length=4,width=1,labelsize=16,top=True,right=True)
special = np.array([490.0, 450.0, 410.0,370.0])
labels=['490','450','410','370']
ax.set_xticks(special,labels)

file = [folder,'Fig_6_LDOS_cuts'+'.pdf']
savefig(file[0], file[1])
plt.show()

 
#%%
# now evaluate radiation patterns at some salient wavelength
lam=0.45    
k0=2.0*np.pi/lam
eV=1.2398/lam


#unit cell
nutc=np.array([np.sqrt(Ag(eV)),nTiO2])
# stack
nmulti=np.tile(nutc,N)
dstack=np.tile(dutc,N)
# postpend extra silver lauer
nmulti=np.append(nmulti,nutc[0])
dstack=np.append(dstack,dutc[0])
# pre and postpend air as embedding medium
nstack=np.insert(nmulti,0,1.0)
nstack=np.append(nstack,1.0)
#locate source
zsource=np.sum(dstack)+0.015 
 
#plot coordinates
Nthe=501
Nphi = 161
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(0+0.001,np.pi/2-0.001,Nthe)

# evaluate radiation patterns
pu=[1.0,0.0,0.0]
mu=[0.0,0.0,0.0]  
    
P_u_px,outE,[theta_u,phi_u]=Farfield.RadiationpatternPandField(k0,zsource,pu,mu,thelist,philist,nstack,dstack)
    
pu=[1.0,0.0,0.0]
mu=[0.0,1.0,0.0]  
P_u_my,outE,[theta_u,phi_u]=Farfield.RadiationpatternPandField(k0,zsource,pu,mu,thelist,philist,nstack,dstack)

pu=[1.0,0.0,0.0]
mu=[0.0,1.0j,0.0]  
P_u_splus,outE,[theta_u,phi_u]=Farfield.RadiationpatternPandField(k0,zsource,pu,mu,thelist,philist,nstack,dstack)
     
pu=[1.0,0.0,0.0]
mu=[0.0,-1.0j,0.0]  
P_u_smin,outE,[theta_u,phi_u]=Farfield.RadiationpatternPandField(k0,zsource,pu,mu,thelist,philist,nstack,dstack)

# plot just some polar cross sections
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'},
                        layout='constrained', dpi = 400)
pion2=int((Nphi-1)/4)
pi=int((Nphi-1)/2)
plt.plot(thelist,P_u_px[pion2,:],'r',-thelist,P_u_px[pion2+pi,:],'r')
plt.plot(thelist,P_u_my[pion2,:],'b',-thelist,P_u_my[pion2+pi,:],'b')
plt.plot(thelist,P_u_splus[pion2,:],'g',-thelist,P_u_splus[pion2+pi,:],'g')
plt.plot(thelist,P_u_smin[pion2,:],'-.g',-thelist,P_u_smin[pion2+pi,:],'-.g')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

rmax = 0.5  # Manually set maximum radial value
ax.set_rlim([0, rmax+0.1])  # Set radial axis limits
ax.set_yticks([rmax / 2,  rmax])  

r_lim = ax.get_ylim()[1]  # Get the plot's radial limit

ax.set_thetamin(-90)
ax.set_thetamax(90)
file = [folder,'Fig_6_radiation_patterns'+'.pdf']
savefig(file[0], file[1])
plt.show()


# # supposing you want 2D Fourier image plots
# kx=np.cos(phi_u)*np.sin(theta_u)
# ky=np.sin(phi_u)*np.sin(theta_u)

# fig, ax=plt.subplots(figsize=(6,6))

# plt.pcolor(kx,ky,P_u_my)
# plt.show()
 