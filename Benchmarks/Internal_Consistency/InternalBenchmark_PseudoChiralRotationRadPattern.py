
"""
For magnetochiral dipole, ImG LDOS definition constraints the definition of the rotation of pseudo chiral dipoles
if (px = 1, my= 1.0j) is lefthanded, then the rotated lefthanded version is (py=1.0, mx=-1.0j)

Consequently, the Stokes parameter S3 (which represents circular polarization) should yield the same radiation pattern
 in the same hemisphere for both dipole configurations
"""

#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
    
#%%
import numpy as np
import matplotlib.pyplot as plt

from Library.Use import Use_Radiationpattern as Radpat
from Library.Util import Util_vectorpolarization as vector
from Library.Util import Util_argumentrewrapper as cc


#%%

'''in microns - so here the length unit of your code is defined'''
lam = 1
k0 = 2*np.pi/lam

'''layers definition'''
nstack = [1.5, 1.5]
dstack = []

z = 0.1*lam

'''calculate for far field angles'''
Nthe = 201
Nphi = 151

thelist=np.linspace(+0.0001,(90*np.pi/180)-0.001,Nthe)
philist=np.linspace(0+0.001,(2*np.pi),Nphi)

the, ph = np.meshgrid(thelist, philist)

#%%
pu = [0, 0, 0]
mu = [0, 0, 0]

#%%

'''px = 1j my'''

pu[0] = 1
mu[1] = 1j


# Compute radiation pattern
P_u, outE, angle = Radpat.RadiationpatternPandField(
    k0, z, pu, mu, thelist, philist, nstack, dstack)

Es = outE[0]
Ep = outE[1]

S0,S1,S2,S3,ellipticity,majororientation = vector.BFPcartesianStokes(the, ph, Es, Ep)

khat,shat,phat=cc.spherical2cartesian(angle[0],angle[1])

kx = khat[0]*nstack[0] 
ky = khat[1]*nstack[0] 


s = S3/S0

fig,ax=plt.subplots(figsize=(6,5), dpi=400)
pcm = ax.pcolormesh(kx, ky, s, vmin= - np.max(s), vmax=np.max(s), cmap='bwr', shading='gouraud', rasterized = True)
ax.set_title(r'$p_x$ = $1j$ $m_y$ Towards upper hemisphere')
ax.set_aspect('equal')
ax.set_xlim([np.min(kx)-.05,np.max(kx)+0.05])
ax.set_ylim([np.min(ky)-.05,np.max(ky)+0.05])
ax.set_aspect('equal')
cbar = fig.colorbar(pcm, ax=ax, shrink = 0.9)
cbar.set_ticks([-1, 0. , 1])
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
plt.show()
plt.close()



''''''
print('The lower and upper hemispheres exhibit exactly rotated due to the perspective change when viewed from the origin')

thelist = thelist + np.pi/2

# Compute radiation pattern
P_u, outE, angle = Radpat.RadiationpatternPandField(
    k0, z, pu, mu, thelist, philist, nstack, dstack)

Es = outE[0]
Ep = outE[1]

S0,S1,S2,S3,ellipticity,majororientation = vector.BFPcartesianStokes(the, -ph, Es, Ep)

khat,shat,phat=cc.spherical2cartesian(angle[0],angle[1])

kx = khat[0]*nstack[0] 
ky = khat[1]*nstack[0] 


s = S3/S0

fig,ax=plt.subplots(figsize=(6,5), dpi=400)
pcm = ax.pcolormesh(kx, ky, s, vmin= - np.max(s), vmax=np.max(s), cmap='bwr', shading='gouraud', rasterized = True)
ax.set_title(r'$p_x$ = $1j$ $m_y$ Towards lower hemisphere')
ax.set_aspect('equal')
ax.set_xlim([np.min(kx)-.05,np.max(kx)+0.05])
ax.set_ylim([np.min(ky)-.05,np.max(ky)+0.05])
ax.set_aspect('equal')
cbar = fig.colorbar(pcm, ax=ax, shrink = 0.9)
cbar.set_ticks([-1, 0. , 1])
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
plt.show()
plt.close()



#%%
#%%
thelist=np.linspace(+0.0001,(90*np.pi/180)-0.001,Nthe)
philist=np.linspace(0+0.001,(2*np.pi),Nphi)

the, ph = np.meshgrid(thelist, philist)


pu = [0, 0, 0]
mu = [0, 0, 0]

#%%

pu[1] = 1
mu[0] = -1j

# Compute radiation pattern
P_u, outE, angle = Radpat.RadiationpatternPandField(
    k0, z, pu, mu, thelist, philist, nstack, dstack)

Es = outE[0]
Ep = outE[1]

S0,S1,S2,S3,ellipticity,majororientation = vector.BFPcartesianStokes(the, ph, Es, Ep)

khat,shat,phat=cc.spherical2cartesian(angle[0],angle[1])

kx = khat[0]*nstack[0] 
ky = khat[1]*nstack[0] 


s = S3/S0

fig,ax=plt.subplots(figsize=(6,5), dpi=400)
pcm = ax.pcolormesh(kx, ky, s, vmin= - np.max(s), vmax=np.max(s), cmap='bwr', shading='gouraud', rasterized = True)
ax.set_title(r'$p_y$ = -$1j$ $m_x$ Towards upper hemisphere')
ax.set_aspect('equal')
ax.set_xlim([np.min(kx)-.05,np.max(kx)+0.05])
ax.set_ylim([np.min(ky)-.05,np.max(ky)+0.05])
ax.set_aspect('equal')
cbar = fig.colorbar(pcm, ax=ax, shrink = 0.9)
cbar.set_ticks([-1, 0. , 1])
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
plt.show()
plt.close()



thelist = thelist + np.pi/2

# Compute radiation pattern
P_u, outE, angle = Radpat.RadiationpatternPandField(
    k0, z, pu, mu, thelist, philist, nstack, dstack)

Es = outE[0]
Ep = outE[1]

S0,S1,S2,S3,ellipticity,majororientation = vector.BFPcartesianStokes(the, -ph, Es, Ep)

khat,shat,phat=cc.spherical2cartesian(angle[0],angle[1])

kx = khat[0]*nstack[0] 
ky = khat[1]*nstack[0] 


s = S3/S0

fig,ax=plt.subplots(figsize=(6,5), dpi=400)
pcm = ax.pcolormesh(kx, ky, s, vmin= - np.max(s), vmax=np.max(s), cmap='bwr', shading='gouraud', rasterized = True)
ax.set_title(r'$p_y$ = -$1j$ $m_x$ Towards lower hemisphere')
ax.set_aspect('equal')
ax.set_xlim([np.min(kx)-.05,np.max(kx)+0.05])
ax.set_ylim([np.min(ky)-.05,np.max(ky)+0.05])
ax.set_aspect('equal')
cbar = fig.colorbar(pcm, ax=ax, shrink = 0.9)
cbar.set_ticks([-1, 0. , 1])
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
plt.show()
plt.close()
