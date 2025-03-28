# -*- coding: utf-8 -*-
"""
Reproducing Barnes et. al. 2020 J. Opt. 22, 073501
"""
print('###  Literature Benchmark: Barnes et. al. 2020, J. Opt. 22, 073501  ###')

#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

#%%

import numpy as np
import matplotlib.pyplot as plt
from Library.Use import Use_LDOS as ImGLDOS


def DrudeLorentz(omega, omega_p, gamma):
    return 1 - (omega_p**2)/(omega**2 + 1j*gamma*omega)


Nk = 700
om = np.linspace(0.5E15,8E15,Nk)
k0 = om/3E14

wp_Au = 1.29E16
g_Au = 1.01E14
nAu = np.sqrt(DrudeLorentz(om,wp_Au,g_Au))

dstack=[0.2]

Nq = 700
kparlist=np.linspace(0.001,20.0,Nq)
zlist=0.04
outaverage = np.zeros([Nk,Nq])
relLDOS = np.zeros([Nk])


X,Y = np.meshgrid(kparlist,om)
for i in range(Nk):
    nstack=[nAu[i],np.sqrt(2.49),nAu[i]]
    out=ImGLDOS.LDOSintegrandplottrace(k0[i],kparlist,zlist,nstack,dstack, guidevisible=0)
    average = (2*out[0,0,:] + out[1,0,:])/3
    outaverage[i,:] = average
    
    outpar1fromP,outperp1fromP,outparfromM,outperpfromM,outcross=ImGLDOS.LDOS(k0[i],zlist,nstack,dstack)
    relLDOS[i] = ((outpar1fromP[0]*2.+outperp1fromP[0])/3.)/np.sqrt(2.49)
    
    X[i,:] = X[i,:]*k0[i]#*nstack[1]

#%%
fig, ax = plt.subplots(dpi=400)
pcm = ax.pcolormesh(X,Y*1E-15,np.log10(outaverage),
                    shading='gouraud', vmin = -9, vmax = 2.5,
                    edgecolors = None, cmap='viridis')
fig.colorbar(pcm, ax=ax,label = 'log$_{10}P(k_{||})$')
ax.set_xlabel(r"in-plane wavevector $k_{||}$ ($\mu m^{-1})$")
ax.set_ylabel(r"angular frequency ($10^{15}$ in rad $s^{-1}$)")
ax.set_xlim([0,50.])
ax.set_ylim([0.8,8.])

plt.title("Power dissipated, William L Barnes et al 2020 J. Opt. 22 Fig 13")
plt.show()
#######################################

plt.figure(dpi=400)
plt.plot(om*1E-15,relLDOS,'k')
plt.ylabel('LDOS relative to bulk')
plt.xlabel(r'angular frequency ($10^{15}$ in rad $s^{-1}$)')
plt.axhline(y = 1, color = 'k', linestyle = '--')
plt.show()

#######################################
om_sel = 2.72E15
k0 = om_sel/3E14

nAu = np.sqrt(DrudeLorentz(om_sel,wp_Au,g_Au))

nstack=[nAu,np.sqrt(2.49),nAu]
out=ImGLDOS.LDOSintegrandplottrace(k0,kparlist,zlist,nstack,dstack)
average = (2*out[0,0,:] + out[1,0,:])/3

plt.figure(dpi=400)
plt.plot(kparlist*k0,(average),'k')
plt.xlabel(r'in-plane wavevector $k_{||}$ ($\mu m^{-1})$')
plt.ylabel(r'dissipated power $P(k_{||})$')
plt.yscale('log')
plt.xlim([0,50])
plt.ylim([1E-4,1E4])

plt.show()