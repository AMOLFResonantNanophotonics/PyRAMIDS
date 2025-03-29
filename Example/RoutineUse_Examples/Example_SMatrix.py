#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''This example use file calculating amplitude and phase; with how the core functions work 

- simple air/glass 
- air/n=1.6(400nm)/ glass 

- shows the guided modes beyond k0
'''


#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#%%
import numpy as np
import matplotlib.pyplot as plt


from Library.Core import Core_Smatrix as Smatrix
from Library.Util import Util_argumentrewrapper as cc
from Library.Use import Use_Planewaves as planewave


#%%

print('Doing Air-glass interface')

nstack = [1,1.465]
dstack = []
lam = 0.6
k0 = 2*np.pi/lam

kparlist = k0*np.linspace(0,1.465,350)


nin,nout,ndlist,dum=cc.stackseparator(nstack, dstack)

amp = np.zeros((4,len(kparlist))) # r in [s,p], t in [s,p]
phase = np.zeros((4,len(kparlist)))


for i, kpar in enumerate(kparlist):
    rts=Smatrix.rt_s(k0,kpar,nin,nout,ndlist)
    rtp=Smatrix.rt_p(k0,kpar,nin,nout,ndlist)
    amp[:,i] = [np.abs(rts)[0], np.abs(rtp)[0], np.abs(rts)[1], np.abs(rtp)[1]]
    phase[:,i] = [np.angle(rts)[0], np.angle(rtp)[0], np.angle(rts)[1], np.angle(rtp)[1] ]


fig, ax = plt.subplots(figsize=(5,4))
ax.plot(kparlist/k0,amp[2,:], label = 's')
ax.plot(kparlist/k0,amp[3,:], label = 'p')
ax.set_xlabel('$k_{||}$/$k_{0}$', fontsize=16)
ax.set_ylabel('|t|', fontsize=16)
ax.tick_params(axis='both', labelsize=16) 
ax.axvline(x=1, color='k', linestyle='--')
plt.legend(fontsize=16)
plt.show()

fig, ax = plt.subplots(figsize=(5,4))
ax.plot(kparlist/k0,phase[2,:], label = 's')
ax.plot(kparlist/k0,phase[3,:], label = 'p')
ax.set_xlabel('$k_{||}$/$k_{0}$', fontsize=16)
ax.set_ylabel('$phase$ [in rad]', fontsize=16)
ax.tick_params(axis='both', labelsize=16) 
ax.axvline(x=1, color='k', linestyle='--')
plt.legend(fontsize=16)
plt.show()


#%%

ampsq = np.zeros((4,len(kparlist))) # r in [s,p], t in [s,p]

for i, kpar in enumerate(kparlist):
    sq = planewave.IntensityRT(k0, 0, nstack, dstack)
    
    ampsq[:,i] = [sq[0][0][0], sq[1][0][0], sq[0][1][0], sq[1][1][0] ]
    
    
fig, ax = plt.subplots(figsize=(5,4))
ax.plot(kparlist/k0,amp[0,:], label = 'Rs')
ax.plot(kparlist/k0,amp[1,:], label = 'Rp')
ax.plot(kparlist/k0,amp[2,:], label = 'Ts')
ax.plot(kparlist/k0,amp[3,:], label = 'Tp')
ax.set_xlabel('$k_{||}$/$k_{0}$', fontsize=16)
ax.set_ylabel('R or T', fontsize=16)
ax.tick_params(axis='both', labelsize=16) 
ax.axvline(x=1, color='k', linestyle='--')
plt.legend(fontsize=16)
plt.show()



#%%
#%%
print('Doing Air-waveguide-glass 3 layer sandwich geometry')

nstack = [1,1.6, 1.465]
dstack = [0.4]
lam = 0.6
k0 = 2*np.pi/lam

kparlist = k0*np.linspace(0,2,500)
nin,nout,ndlist,dum=cc.stackseparator(nstack, dstack)

amp = np.zeros((4,len(kparlist))) # r in [s,p], t in [s,p]
phase = np.zeros((4,len(kparlist)))


for i, kpar in enumerate(kparlist):
    rts=Smatrix.rt_s(k0,kpar,nin,nout,ndlist)
    rtp=Smatrix.rt_p(k0,kpar,nin,nout,ndlist)
    amp[:,i] = [np.abs(rts)[0], np.abs(rtp)[0], np.abs(rts)[1], np.abs(rtp)[1]]
    phase[:,i] = [np.angle(rts)[0], np.angle(rtp)[0], np.angle(rts)[1], np.angle(rtp)[1] ]

fig, ax = plt.subplots(figsize=(5,4))
ax.plot(kparlist/k0,amp[0,:], label = 's')
ax.plot(kparlist/k0,amp[1,:], label = 'p')
ax.set_xlabel('$k_{||}$/$k_{0}$', fontsize=16)
ax.set_ylabel('|r|', fontsize=16)
ax.tick_params(axis='both', labelsize=16) 
ax.axvline(x=1, color='k', linestyle='--')
ax.set_ylim([0,6])
plt.legend(fontsize=16)
plt.show()

print('See the guided mode appears as pole beyond k0')



fig, ax = plt.subplots(figsize=(5,4))
ax.plot(kparlist/k0,amp[2,:], label = 's')
ax.plot(kparlist/k0,amp[3,:], label = 'p')
ax.set_xlabel('$k_{||}$/$k_{0}$', fontsize=16)
ax.set_ylabel('|t|', fontsize=16)
ax.tick_params(axis='both', labelsize=16) 
ax.axvline(x=1, color='k', linestyle='--')
ax.set_ylim([0,6])
plt.legend(fontsize=16)
plt.show()

fig, ax = plt.subplots(figsize=(5,4))
ax.plot(kparlist/k0,phase[0,:], label = 's')
ax.plot(kparlist/k0,phase[1,:], label = 'p')
ax.set_xlabel('$k_{||}$/$k_{0}$', fontsize=16)
ax.set_ylabel('phase (r) [in rad]', fontsize=16)
ax.tick_params(axis='both', labelsize=16) 
ax.axvline(x=1, color='k', linestyle='--')
plt.legend(fontsize=16)
plt.show()


fig, ax = plt.subplots(figsize=(5,4))
ax.plot(kparlist/k0,phase[2,:], label = 's')
ax.plot(kparlist/k0,phase[3,:], label = 'p')
ax.set_xlabel('$k_{||}$/$k_{0}$', fontsize=16)
ax.set_ylabel('phase (t) [in rad]', fontsize=16)
ax.tick_params(axis='both', labelsize=16) 
ax.axvline(x=1, color='k', linestyle='--')
plt.legend(fontsize=16)
plt.show()
