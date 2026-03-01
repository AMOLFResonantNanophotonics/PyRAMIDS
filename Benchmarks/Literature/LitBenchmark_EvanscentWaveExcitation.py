
"""
Literarture Benchmark -

Interferometric Evanescent Wave Excitation of a Nanoantenna
for Ultrasensitive Displacement and Phase Metrology

Lei Wei et. al, PRL 121 193901 (2018)
"""
#%%

print('### Literature Benchmark:   Lei Wei et. al. Phys. Rev. Lett. 121 193901, 2018 Fig 1b ###')
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

import numpy as np
from matplotlib import pyplot as plt
        
#%%



c0 = 3e17
lam = 600
k0 = 2*np.pi/lam
om = 2*np.pi*c0/ lam


n_glass = 1.5
n_air = 1

nstack= [n_glass,n_air]
dstack= []

NA_illum = 1.3
theta_illum = np.arcsin(NA_illum/n_glass)
theta_crit = np.arcsin(n_air/n_glass)

th_ = np.array([theta_illum, theta_illum])
ph_ = np.array([0, np.pi])
s_ = np.array([0,0]) 
p_ = np.array([1,-1]) #Out of phase p-polarization


#%%

x = np.linspace(-300, 300, 301) 
z = np.linspace(0.1, 200, 201)
x, z = np.meshgrid(x, z)

xx = x.flatten()
zz = z.flatten()
yy = np.zeros_like(xx)
rlist = np.array([xx, yy, zz]) 


#%%
# Compute Plane Wave Fields
EH = pw.CartesianField(th_, ph_, s_, p_, rlist, k0, nstack, dstack)
EH_sum = np.sum(EH, axis=1)


#%%
Poynting_ = np.zeros((3, *x.shape), dtype="complex")

Poynting_ = 0.5 * np.cross(EH_sum[0:3, :], np.conj(EH_sum[3:6, :]), axis=0)

Poynting_x = Poynting_[0, :].reshape(*x.shape)
Poynting_z = Poynting_[2, :].reshape(*x.shape)

Poynting_magnitude = np.sqrt(np.abs(Poynting_[0,:]**2) + np.abs(Poynting_[1,:]**2) + np.abs(Poynting_[2,:]**2)).reshape(*x.shape)


 
#%%
fig, ax = plt.subplots(figsize=(8, 4), dpi =400)

cmap_background = "viridis"
pcm = ax.pcolor(NA_illum*x/lam, NA_illum*z/lam,  Poynting_magnitude, vmin = 0, vmax = 1.9, shading='auto', cmap=cmap_background, rasterized = True)

cbar = fig.colorbar(pcm, ax=ax, shrink = 0.9)
cbar.set_ticks([0, 1.9])
cbar.ax.tick_params(labelsize=16)

subsample = 11
quiver = ax.quiver((NA_illum*x/lam)[::subsample, ::subsample], (NA_illum*z/lam)[::subsample, ::subsample],
                    np.imag(Poynting_x)[::subsample, ::subsample], np.imag(Poynting_z)[::subsample, ::subsample], 
                    color="white", scale=25, width=0.005, headwidth=4)

ax.set_xlabel(r"$k_{\parallel}y / 2\pi$", fontsize=14)
ax.set_ylabel(r"$k_{\parallel}z / 2\pi$", fontsize=14)

ax.set_aspect("auto")
ax.set_xlim([-0.5, 0.5])  
ax.set_ylim([0, 0.3])

xticks = [-0.5, 0, 0.5]
yticks = [0, 0.3]
ax.set_xticks(xticks)
ax.set_yticks(yticks)


ax.set_xticklabels([f'{tick:.2f}' for tick in xticks])
ax.set_yticklabels([f'{tick:.2f}' for tick in yticks])

ax.set_title(r'L. Wei et al. PRL 2018/Fig1b'
             '\n'
             r'$\Im\{S\}$')
file = [folder,'LitBenchmark_LWei_etal_PRL2018_Fig1b'+' .pdf']
savefig(file[0], file[1])
plt.show()


