#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculates and shows all the plane wave capabilities
- Reflectance, Transmittance, Absorptance
- On axis Electric field, Absorption
"""

#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%%

from Library.Use import Use_Planewaves as pw
import numpy as np
from matplotlib import pyplot as plt

plt.close('all')


lam = 0.7
k0= 2*np.pi/ lam

## Arbitrary stack

nstack=np.array([1.0, 2.+0.01j, 1.6+0.05j, 2.5+0.01j])
dstack=np.array([0.5,0.6])

kparlist=np.arange(0.0,1.0,0.01)*k0*nstack[0].real




#%%
'''Calculating R,T,A of the total stack'''



RTAs, RTAp=pw.IntensityRT(k0, kparlist, nstack, dstack)

plt.figure(1,figsize=(4,4), dpi=300)
plt.plot(kparlist/k0,RTAs[0],kparlist/k0,RTAp[0])
plt.xlabel('kpar')
plt.ylabel('intensity R')
plt.ylim([0, 1.])
plt.legend(['Rs','Rp'])
plt.show()

'''Calculating Absoprtion per layer'''

plt.figure(2,figsize=(4,4), dpi=300)
As_perlayer,Ap_perlayer=pw.PerLayerAbsorption(k0, kparlist, nstack, dstack)
plt.plot(kparlist/k0,np.transpose(As_perlayer),kparlist/k0,np.transpose(Ap_perlayer),':')
plt.title('Absorp. per layer (p = solid, s= dashed)')
plt.xlabel('kpar')
plt.ylabel('Absorption')
plt.show()



#%% 
'''On axis field, and absorption'''

kpar=k0*nstack[0]*np.sin(np.pi/6)

zlist=np.arange(-1,2,0.02)

Sfield,Pfield,AbsS,AbsP = pw.OnAxisLocalFieldandAbsorption(k0, kpar, nstack, dstack,zlist)
 

fig,ax=plt.subplots(4,2,sharex='col',sharey='row',figsize=(10,15), dpi=300)

ax[0][0].plot(zlist,Sfield[0,0,:].real,'-o',zlist,np.abs(Sfield[0,0,:])**2)
ax[0][0].legend([r'real(Es)','$|Es|^2$'])

ax[0][0].axvline(0, color='gray', linestyle='--', linewidth=1.5)
ax[0][0].axvline(dstack[0], color='gray', linestyle='--', linewidth=1.5)  
ax[0][0].axvline(sum(dstack), color='gray', linestyle='--', linewidth=1.5)



ax[1][0].plot(zlist,Sfield[1,0,:].real,'-o',zlist,np.abs(Sfield[1,0,:])**2)
ax[1][0].legend([r'real(Hsx)','$|Hsx|^2$'])



ax[2][0].plot(zlist,Sfield[2,0,:].real,'-o',zlist,np.abs(Sfield[2,0,:])**2)
ax[2][0].legend([r'real(Hsz)','$|Hsz|^2$'])



ax[3][0].plot(zlist,AbsS[0,:])
ax[3][0].legend(['Absorption S'])

ax[3][0].axvline(0, color='gray', linestyle='--', linewidth=1.5)
ax[3][0].axvline(dstack[0], color='gray', linestyle='--', linewidth=1.5)  
ax[3][0].axvline(sum(dstack), color='gray', linestyle='--', linewidth=1.5)


ax[0][1].plot(zlist,Pfield[0,0,:].real,'-o',zlist,np.abs(Pfield[0,0,:])**2)
ax[0][1].legend([r'real(Epx','$|Epx|^2$'])

ax[0][1].axvline(0, color='gray', linestyle='--', linewidth=1.5)
ax[0][1].axvline(dstack[0], color='gray', linestyle='--', linewidth=1.5)  
ax[0][1].axvline(sum(dstack), color='gray', linestyle='--', linewidth=1.5)



ax[1][1].plot(zlist,Pfield[1,0,:].real,'-o', zlist,np.abs(Pfield[1,0,:])**2)
ax[1][1].legend([r'real(Epz) DISCONTINUOUS','$|Epz|^2$'])



ax[2][1].plot(zlist,Pfield[2,0,:].real,'-o',zlist,np.abs(Pfield[2,0,:])**2)
ax[2][1].legend([r'real(Hpy)','$|Hpy|^2$'])


ax[3][1].plot(zlist,AbsP[0,:])
ax[3][1].legend(['Absorption P'])

ax[3][1].axvline(0, color='gray', linestyle='--', linewidth=1.5)
ax[3][1].axvline(dstack[0], color='gray', linestyle='--', linewidth=1.5)  
ax[3][1].axvline(sum(dstack), color='gray', linestyle='--', linewidth=1.5)


plt.subplots_adjust(hspace=0.2, bottom=0.08)
fig.text(0.25, 0.04, r'z [in $\mu$ m]', ha='center', fontsize=14)
fig.text(0.75, 0.04, r'z [in $\mu$ m]', ha='center', fontsize=14)
plt.show()


##########################

#%%
'''Calculating and plot field'''

x=np.arange(-1.5,2.5,0.02)
z=np.arange(-1.5,2.5,0.02)

x,z=np.meshgrid(x,z)
xx=x.flatten()
zz=z.flatten()
yy=0*zz

rlist=np.transpose(np.array([xx,yy,zz]))


k0=6.0


theta=np.array([np.pi/6])
phi=0.0

EH=pw.CartesianField(theta, phi, 1j*theta,   theta, rlist, k0, nstack, dstack)

Ez=np.reshape(EH[0,0,:],x.shape)
Ey=np.reshape(EH[1,0,:],x.shape)
Ez=np.reshape(EH[2,0,:],x.shape)


plt.figure(figsize = (4,4), dpi=300)
pcm = plt.pcolormesh(x,z,Ey.real,vmin=-np.max(np.abs(EH)),vmax=np.max(np.abs(EH)),cmap='bwr', shading = 'auto')
plt.xlabel('x')
plt.ylabel('z')

plt.axhline(0, color='gray', linestyle='--', linewidth=1.5)
plt.axhline(dstack[0], color='gray', linestyle='--', linewidth=1.5)
plt.axhline(sum(dstack), color='gray', linestyle='--', linewidth=1.5)

cbar = plt.colorbar(pcm)
cbar.set_label("Re(Ey)", fontsize=12)  # Label for clarity
plt.show()


#%%

print('###  sample a random point and check transversality and Poynting vector, for a homogeneous medium  ###')


nstack=np.array([2.0,2.0,2.0, 2.0])
dstack=np.array([0.5, 0.5])
 

k0=6.0


theta=0.0*np.array([np.pi/3, np.pi/3,np.pi/3,np.pi/3])
phi=0.0

EH=pw.CartesianField(theta, phi, np.array([1,0,1,1]),   np.array([0,1,1,1.j])+0*theta, rlist, k0, nstack, dstack)
 

Esample=EH[0:3,3,500]
Hsample=EH[3:6,3,500]
khat=np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta), np.cos(theta)])[:,3]

print('Transversility: Inner product E k = ',np.dot(Esample,khat))
print('Transversility: Inner product H k = ',np.dot(Hsample,khat))
print('Orthogonality: Inner product E H = ',np.dot((Esample), (Hsample)))
print('Outer product k x E = ', np.cross(khat,Esample))
print('Outer product k x H = ', np.cross(khat,Hsample))
print('Poynting vector in z direction: Outer product Re[E* x H] = ', np.real(np.cross(np.conj(Esample), Hsample)))
print('Compare: k = ', khat)