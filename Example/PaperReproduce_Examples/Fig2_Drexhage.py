#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
print('Doing Figure 2a, b')

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
from Library.Use import Use_LDOS as ImGLDOS
import numpy as np
from matplotlib import pyplot as plt
from Library.Use import Use_Multiplescattering as ms
from Library.Use import Use_Green as Green
from Library.Use import Use_LDOS as LDOS
from Library.Use import Use_Radiationpattern as radpat

## image  dipole construction
k0=2.0*np.pi # choose any k you want
zlist=np.arange(0.001,10,0.15)/k0 #plotrange scaled to k0
nstack=[1.0,1.0] # vacuum is air above and air below
dstack=[] # no layer inbetween


# hard code source and image, for two orientation cases
ppar=np.array([[1.0,-1.0], #px
              [0,0], #py
              [0,0], #pz
              [0,0], #mx
              [0,0], #my
              [0,0]]) #mz
pper=np.array([[0.0,0.0], #px
              [0,0], #py
              [1.0,1.0], #pz
              [0,0], #mx
              [0,0], #my
              [0,0]]) #mz
 

# set up variable to hold result 
Pparup=0*zlist
Pperup=0*zlist
Wxx=0*zlist
Wzz=0*zlist

# The multiple scattering library is used to evaluate the response of multiple phased dipoles.
# This requires a loop over configurations  

larmor=(k0**4*4/3*np.pi)
freeldos=2/3*k0**3

for idx,z in enumerate(zlist):
    
    rdip=np.array([[0,0],[0,0],[0,2*z]]) # distance between source and image is twice the height
    
    ## radiation pattern integral
    Pup,Pdown=ms.TotalfarfieldpowerManydipoles(ppar,rdip,k0,nstack,dstack)
    Pparup[idx]=Pup/larmor # Image charge analysis predicts the field, that only exists in the upper half space
    Pup,Pdown=ms.TotalfarfieldpowerManydipoles(pper,rdip,k0,nstack,dstack)
    Pperup[idx]=Pup/larmor
    
    ## instead you can also look at the work that the real source is doing against the field of its image
    G=Green.GreenFree(k0, nstack, dstack, rdip[:,1], rdip[:,0])
    Wxx[idx]=1-np.imag(np.squeeze(G[0,0]))/freeldos  ## image charge opposite sign, enters here as prefactor
    Wzz[idx]=1+np.imag(np.squeeze(G[2,2]))/freeldos
    
## plot
fig, ax = plt.subplots(figsize=(6, 5), dpi=350)
ax.plot(k0*zlist/(2.0*np.pi),Wxx,'kx',k0*zlist/(2.0*np.pi),Wzz,'ko',k0*zlist/(2.0*np.pi),Pparup,'r',k0*zlist/(2.0*np.pi),Pperup,'b')
ax.set_xlabel(r'Distance to mirror $z/\lambda$')
ax.set_ylabel(r'LDOS, image dipole model')
ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
ax.set_xticks([0.0, 0.5, 1.0, 1.5])
plt.legend([r'From work done by image $p_{||}$',r'From work done by image $p_{\perp}$','Far field total, $p_{||}$',r'Far field total, $p_{\perp}$'])
ax.tick_params(axis='both', direction='in', length=4, width=1)
fig.tight_layout()
file = [folder,'Fig_2a_LDOS_work_freespace'+'.pdf']
savefig(file[0], file[1])
plt.show()
 

#%% ##
'''now analyze the case of an actual Ag mirror'''
nstack=np.array([0.052+4.41j,1.0])
zlist2=np.arange(0.001,10,0.05)/k0 #plotrange scaled to k0

## total LDOS, from Im G
ldepar,ldeperp,ldmpar,ldmper,ldc=LDOS.LDOS(k0, zlist2, nstack, dstack)

## Radiated part only
out=radpat.TotalRadiated(k0, zlist2, nstack, dstack)
ldeparr=out[0,:]
ldeperpr=out[1,:]

 
totaliso=0.667*ldepar+0.333*ldeperp
radiatiso=0.667*ldeparr+0.333*ldeperpr
absiso=totaliso-radiatiso
imageiso=0.667*Wxx+0.333*Wzz

fig, ax = plt.subplots(figsize=(6, 5), dpi=350)
ax.plot(k0*zlist2/(2.0*np.pi),totaliso,'r',k0*zlist2/(2.0*np.pi),radiatiso,'b',k0*zlist2/(2.0*np.pi),absiso,'b--',k0*zlist/(2.0*np.pi),imageiso,'k:')
ax.set_ylim(0.0,3.5)
ax.set_xlabel(r'Distance to mirror $z/\lambda$')
ax.set_ylabel(r'LDOS')
ax.set_yticks([0.0, 1.0,  2.0, 3.0])
ax.set_xticks([0.0, 0.5, 1.0, 1.5])
plt.legend([r'Ag mirror @ 650 nm (J+C), isotropic',
            r'Ag mirror $p_{||}$ radiative part only',
            r'Ag mirror $p_{||}$ nonradiative part only',
            r'Perfect mirror, image charge'])

ax.tick_params(axis='both', direction='in', length=4, width=1)
fig.tight_layout()
file = [folder,'Fig_2b_LDOS_infront_mirror'+'.pdf']
savefig(file[0], file[1])
plt.show()

