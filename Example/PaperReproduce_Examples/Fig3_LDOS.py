#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print('Making Figure 3')

#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%%

from Library.Use import Use_Planewaves as pw
from Library.Use import Use_LDOS as ImGLDOS
import numpy as np
from matplotlib import pyplot as plt
from Library.Use import Use_Radiationpattern as Farfield


plt.close('all')


def savefig(folderpath, filename):
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    plt.savefig(os.path.join(folderpath, filename), bbox_inches='tight')
    
    
folder = r"pdfimages/"
#%%
lam = 1
k0= 2*np.pi/ lam

nstack = [1.5, 3.5, 1]
dstack = [0.1*lam]

#%%
'''On axis field, and absorption'''

kparlist=np.linspace(0,4.0,1500)

zlist=np.linspace(-0.6,0.7,1200)



#%%
'''LDOS Calculation'''
ld=ImGLDOS.LDOS(k0,zlist,nstack,dstack)
ld_avg = (2*ld[0] + ld[1])/3

data=Farfield.TotalRadiated(k0,zlist,nstack,dstack)
out = (2*data[0] + data[1])/3


fig, ax = plt.subplots(figsize=(7, 5), dpi=350)

ax.plot(zlist, ld[0], 'g-', label=r'Total $\parallel$ LDOS from E')
ax.plot(zlist, data[0], 'r-', label=r'$\parallel$ Radiated LDOS from E')

ax.plot([-0.6, 0, 0, dstack[0], dstack[0], 0.7],[nstack[0], nstack[0], nstack[1], 
                                                 nstack[1], nstack[2], nstack[2]],'k', label="Refractive Index Profile")

ax.set_xlabel(r'z in units of vac.  $\lambda$')
ax.set_ylabel('LDOS / vac LDOS')

yticks = [0, 1.5, 3, 4.5]
ax.set_yticks(yticks)

xticks = [-0.5, -0.25, 0, 0.25, 0.5]
ax.set_xticks(xticks)

ax.tick_params(axis='both', direction='in', length=4, width=1)
ax.set_xlim([-0.6, 0.7])
ax.set_ylim([-0.1, 5.1])
ax.legend(['total', 'radiated', 'refractiveindex'])
file = [folder,'Fig_3a_LDOS_parallel'+'.pdf']
savefig(file[0], file[1])

plt.show()



fig, ax = plt.subplots(figsize=(7, 5), dpi=350)


ax.plot(zlist, ld[1], 'g--', label=r'Total $\perp$ LDOS from E')
ax.plot(zlist, data[1], 'r--', label=r'$\perp$ Radiated LDOS from E')

ax.plot([-0.6, 0, 0, dstack[0], dstack[0], 0.7],[nstack[0], nstack[0], nstack[1], 
                                                 nstack[1], nstack[2], nstack[2]],'k', label="Refractive Index Profile")

ax.set_xlabel(r'z in units of vac.  $\lambda$')
ax.set_ylabel('LDOS / vac LDOS')

yticks = [0, 1.5, 3, 4.5]
ax.set_yticks(yticks)

xticks = [-0.5, -0.25, 0, 0.25, 0.5]
ax.set_xticks(xticks)
ax.tick_params(axis='both', direction='in', length=4, width=1)
ax.set_xlim([-0.6, 0.7])
ax.set_ylim([-0.1, 5.1])
ax.legend(['total', 'radiated', 'refractiveindex'])
file = [folder,'Fig_3b_LDOS_perpendicular'+'.pdf']
savefig(file[0], file[1])

plt.show()



#%%


out=ImGLDOS.LDOSintegrandplottrace(k0,kparlist,zlist,nstack,dstack)

#%%
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
pcm = ax.pcolor(kparlist, zlist, np.log10(np.abs(out[0,:,:])), cmap = 'inferno',rasterized= True)
cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(r'$\log_{10}$ LDOS_E integrand', fontsize=12)

ax.plot(kparlist, 0 * kparlist, 'w')  
ax.plot(kparlist, 0 * kparlist + dstack[0], 'w') 

ax.plot([nstack[0], nstack[0]], [-1.0, 0.0], 'w--')
ax.plot([nstack[1], nstack[1]], [0, dstack[0]], 'w--')
ax.plot([nstack[2], nstack[2]], [dstack[0], 1.0], 'w--')

pcm.set_clim([-2, 2])
ax.set_xlabel(r'$k_{\parallel}$/$k_0$')
ax.set_ylabel(r'z / $\lambda$')
ax.set_title(r'log10 LDOS E $\parallel$ integrand')
ax.set_ylim([-0.5, 0.6])


yticks = [-0.4, -0.2, 0, 0.2, 0.4]
ax.set_yticks(yticks)

xticks = [0, 1, 1.5, 3.5]
ax.set_xticks(xticks)

ax.tick_params(axis='both', direction='in', length=4, width=1)
file = [folder,'Fig_3b_LDOStrace_parallel'+'.pdf']
savefig(file[0], file[1])
plt.show()



fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
pcm = ax.pcolor(kparlist, zlist, np.log10(np.abs(out[1,:,:])), cmap = 'inferno', rasterized = True)
cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(r'$\log_{10}$ LDOS_E integrand', fontsize=12)

ax.plot(kparlist, 0 * kparlist, 'w')  
ax.plot(kparlist, 0 * kparlist + dstack[0], 'w') 

ax.plot([nstack[0], nstack[0]], [-1.0, 0.0], 'w--')
ax.plot([nstack[1], nstack[1]], [0, dstack[0]], 'w--')
ax.plot([nstack[2], nstack[2]], [dstack[0], 1.0], 'w--')

pcm.set_clim([-2, 2])
ax.set_xlabel(r'$k_{\parallel}$/$k_0$')
ax.set_ylabel(r'z / $\lambda$')
ax.set_title(r'log10 LDOS E $\perp$ integrand')
ax.set_ylim([-0.5, 0.6])


yticks = [-0.4, -0.2, 0, 0.2, 0.4]
ax.set_yticks(yticks)

xticks = [0, 1, 1.5, 3.5]
ax.set_xticks(xticks)

ax.tick_params(axis='both', direction='in', length=4, width=1)
file = [folder,'Fig_3b_LDOStrace_perpendicular'+'.pdf']
savefig(file[0], file[1])
plt.show()

