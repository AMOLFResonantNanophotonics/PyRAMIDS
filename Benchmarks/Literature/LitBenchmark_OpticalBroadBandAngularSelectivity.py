# -*- coding: utf-8 -*-

#%%
'''Reproducing Shen Soljcacic et. al. Science 343, 1499-1501 (2014)'''


print('###  Literature Benchmark: Shen at. al. Science 343, 1499-1501 (2014)  ###')
#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%%

import numpy as np
import matplotlib.pyplot as plt
from Library.Use import Use_Planewaves as planewave

nSiO2 = np.sqrt(2.18)
nTa2O5 = np.sqrt(4.33)

num_layers = 42
nstack = [nSiO2] + [nTa2O5, nSiO2] * num_layers + [nSiO2]

def thickeachlayer(m):
    return 140 * (1.165**(m-1)) / 2  # nm

# Generate thickness values for 6 repeating patterns
dstack = np.concatenate([
    np.full(14, thickeachlayer(1)),
    np.full(14, thickeachlayer(2)),
    np.full(14, thickeachlayer(3)),
    np.full(14, thickeachlayer(4)),
    np.full(14, thickeachlayer(5)),
    np.full(14, thickeachlayer(6))
])

# Define incident angle range (in degrees) and wavelength range (in nm)
thetalist = np.linspace(0, 80, 151)
lam = np.linspace(380, 720, 301)

# Initialize output matrix
output = np.zeros((len(lam), len(thetalist)))

# transmission values
for i, wavelength in enumerate(lam):
    k0 = 2 * np.pi / wavelength  # Free-space wavevector
    for j, theta in enumerate(thetalist):
        kpar = nstack[0] * k0 * np.sin(np.radians(theta))  
        output[i, j] = planewave.IntensityRT(k0, kpar, nstack, dstack)[1][1][0]  

fig, ax = plt.subplots(figsize=(9, 6),dpi=300)

pcm = ax.pcolormesh(thetalist, lam, output, shading='gouraud', vmin=0, vmax=1, cmap='jet')

cbar = fig.colorbar(pcm, ax=ax, label='Tp')
cbar.ax.tick_params(labelsize=14)

ax.set_xlabel("Incident Angle (°)", fontsize=14)
ax.set_ylabel("Wavelength (nm)", fontsize=14)
ax.set_title("Optical Broadband Angular Selectivity (Science 2014, Fig. 4)", fontsize=12)

ax.tick_params(axis='both', labelsize=14)
plt.show()