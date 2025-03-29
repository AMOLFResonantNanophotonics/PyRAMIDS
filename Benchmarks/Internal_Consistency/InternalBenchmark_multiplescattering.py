#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Optical Theorem check for plane wave incidence in no guided mode situation'''
print('## Optical Theorem check ##')

#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#%%

from Library.Use import Use_Multiplescattering as ms
from Library.Util import Util_argumentchecker as check

import numpy as np
from matplotlib import pyplot as plt


import time

plt.close('all')

k0=2*np.pi/600E-9


nstack=np.array([1., 1., 1.])
dstack=np.array([0.5E-6])

n_dipole = 3.5
r_dipole = 50E-9
V=4/3*np.pi*(r_dipole)**3


'''Periodic: 7 x 7'''
N=3
x=np.arange(-N,N+1,1)
y=np.arange(-N,N+1,1)
x,y=np.meshgrid(x,y)
x=x.flatten()
y=y.flatten()
rdip=np.array([x*500E-9,y*500E-9,r_dipole+0*x])


'''2 particles seperated in x direction'''
# rdip = np.zeros((3,2))
# rdip[2, 0] = r_dipole
# rdip[2, 1] = r_dipole
# rdip[0, 1] = 100E-9


rdip=check.checkr(rdip)
zlist=rdip[2,:]


drive='planewave'

print('Problem setting = ', drive)
print('Number of dipoles = ', rdip.shape[1])

theta=np.array([np.pi/6])
phi=np.array([0.1])
s=np.array([1])
p=np.array([1j])

diplayer, Ndip=ms.dipolelayerchecker(rdip ,nstack,dstack)
alpha=ms.Rayleighspherepolarizability(n_dipole,nstack[diplayer], V)

t0=time.time()
alphalist=np.zeros([Ndip,6,6],dtype=complex)
for ii in range(Ndip):
    alphalist[ii,:,:]=alpha

invalpha=ms.invalphadynamicfromstatic(alphalist, rdip,diplayer, k0, nstack, dstack)
talpha=time.time()-t0

t0=time.time()
M=ms.SetupandSolveCouplingmatrix(invalpha,rdip,k0, nstack, dstack)
tG=time.time()-t0

if drive == 'planewave':
    driving, intensity = ms.Planewavedriving(theta,phi,s,p,rdip,k0,nstack,dstack)

t0=time.time()
pnm=ms.Solvedipolemoments(M, driving)
tsolve=time.time()-t0

work=ms.Work(pnm, diplayer, driving, k0,nstack)  

if drive =='planewave':
    extinction=work/intensity
    print('Extinction [area, in m2 if k0  and V are in SI units]:', extinction)
    print('Extinction relative to lambda squared: ', extinction*(0.5*k0*nstack[0]/np.pi)**2)


Nthe=101
Nphi = 151
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(+0.001,(np.pi/2)-0.01,Nthe)


t0=time.time()
Pup, Pdown=ms.TotalfarfieldpowerManydipoles(pnm, rdip, diplayer,k0, nstack, dstack) 
tfarf=time.time()-t0


print('Optical theorem match[should be 1.0 for plane wave input, in absence of guided modes]. Should be < 1 for plane waves, in presence of guided modes or absorption. Has no meaning for local source driving')
print((Pup+Pdown)/np.sum(work))

print('Required time to dress alpha (LDOS)', talpha)
print('Required time to get all the G functions',tG)
print('Required time to solve',tsolve)
