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


def savefig(folderpath, filename):
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    plt.savefig(os.path.join(folderpath, filename), bbox_inches='tight')
    
    
folder = r"pdfimages/"
#%%

import numpy as np
import matplotlib.pyplot as plt
from Library.Use import Use_LDOS as ImGLDOS


def DrudeLorentz(omega, omega_p, gamma):
    return 1 - (omega_p**2)/(omega**2 + 1j*gamma*omega)


Nk = 701
om = np.linspace(0.2E15,8E15,Nk)
k0 = om/3E14

wp_Au = 1.29E16
g_Au = 1.01E14
nAu = np.sqrt(DrudeLorentz(om,wp_Au,g_Au))

dstack=[0.2]

Nq = 701
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
fig, ax = plt.subplots(figsize=(6,4),dpi=400)
pcm = ax.pcolormesh(X,Y*1E-15,np.log10(outaverage),
                    shading='gouraud', vmin = -9, vmax = 2.5,
                    edgecolors = None, cmap='viridis')
fig.colorbar(pcm, ax=ax,label = 'log$_{10}P(k_{||})$')
ax.set_xlabel(r"in-plane wavevector $k_{||}$ ($\mu m^{-1})$")
ax.set_ylabel(r"angular frequency ($10^{15}$ in rad $s^{-1}$)")
ax.set_xlim([0,50.])
ax.set_ylim([0.8,8.])
plt.title("W. L. Barnes et al 2020 J. Opt. 22/Fig 13c," "\n"
          "Power dissipated")

file = [folder,'LitBenchmark_Barnes_etal_JOpt2020_Fig13c'+' .pdf']
savefig(file[0], file[1])
plt.show()


#%%
plt.figure(figsize=(5,5), dpi=400)
plt.plot(relLDOS, om*1E-15, 'k')
plt.xlabel(r'LDOS relative to bulk ($\rho_t / \rho$)')
plt.ylabel(r'angular frequency ($10^{15}$ in rad s$^{-1}$)')
plt.axvline(x=1, color='k', linestyle='--')   # bulk LDOS reference
plt.gca().invert_xaxis()
plt.title("W. L. Barnes et al., J. Opt. 22 (2020)/Fig. 13d")
plt.xlim([4.5,0])
plt.tight_layout()

file = [folder,'LitBenchmark_Barnes_etal_JOpt2020_Fig13d'+' .pdf']
savefig(file[0], file[1])
plt.show()


#%%
#######################################
om_sel = 2.72E15
k0 = om_sel/3E14
nAu = np.sqrt(DrudeLorentz(om_sel,wp_Au,g_Au))
nstack=[nAu,np.sqrt(2.49),nAu]

Nq = 2001
kparlist=np.linspace(0.001,20.0,Nq)

out=ImGLDOS.LDOSintegrandplottrace(k0,kparlist,zlist,nstack,dstack)
average = (2*out[0,0,:] + out[1,0,:])/3
plt.figure(figsize = (6,4),dpi=400)
plt.plot(kparlist*k0,(average),'k')
plt.xlabel(r'in-plane wavevector $k_{||}$ ($\mu m^{-1})$')
plt.ylabel(r'dissipated power $P(k_{||})$')
plt.yscale('log')
plt.xlim([0,50])
plt.ylim([1E-4,1E4])
plt.title(r"W. L. Barnes et al., J. Opt. 22 (2020)/Fig. 13b" "\n"
    rf"Cross-cut at $\omega = {om_sel/1e15:.2f}\times10^{{15}}\,\mathrm{{rad\,s^{{-1}}}}$")
file = [folder,'LitBenchmark_Barnes_etal_JOpt2020_Fig13b'+' .pdf']
savefig(file[0], file[1])
plt.show()