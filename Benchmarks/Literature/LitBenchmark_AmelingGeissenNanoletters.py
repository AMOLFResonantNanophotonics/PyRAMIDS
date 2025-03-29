
#%%
'''Reproducing Ameling et. al. Nano Lett. 10, 4394-4398 (2010)'''

#%%
print('### Literature Benchmark:   Ameling et. al. Nano Lett. 10, 4394-4398, 2010 ###')

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

# Drude Model for Gold
def DrudeLorentz(omega, omega_p, gamma):
    return (omega_p**2) / (omega**2 - 1j * gamma * omega)

Nk = 400
lam = np.linspace(700, 3000, Nk)  # Wavelengths in nm
om = 2.0 * np.pi*3E17 / lam  # Convert wavelength to angular frequency
klist = 2.0 * np.pi / lam  # Free-space wavevector

wp_Au = 1.37E16  # Gold plasma frequency (rad/s)
g_Au = 8E13  # Gold damping parameter (rad/s)
epsilon_Au = DrudeLorentz(om, wp_Au, g_Au)
nAu = np.sqrt(epsilon_Au)  # Convert permittivity to refractive index



nSpacer = 1.4   # Define Spacer Refractive Index

# Spacer Thickness Range
Nq = 400
dSpacer = np.linspace(100, 3000, Nq)  # Spacer thickness in nm

# Initialize RT Matrix
RT = np.zeros([Nk, Nq])

for m in range(Nk):
    for q in range(Nq):
        # Material Stack (Air | Gold | Spacer | Gold | Air)
        nstack = [1, nAu[m], nSpacer, nAu[m], 1]
        dstack = [20, dSpacer[q], 20]  # Thicknesses (nm)
        
        k0 = 2 * np.pi / lam[m]  # Free-space wavevector

        Rp = planewave.IntensityRT(k0, 0, nstack, dstack)[0][0][0] 
        Rs = planewave.IntensityRT(k0, 0, nstack, dstack)[1][0][0] 

        RT[m, q] = (Rp + Rs) / 2
#%%
X, Y = np.meshgrid(dSpacer, lam)

fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
img = ax1.pcolormesh(X, Y/1e3, RT, vmin = 0, vmax= 1, cmap='jet', shading='gouraud')
cbar = fig.colorbar(img, ax=ax1)
cbar.ax.tick_params(labelsize=14) 
ax1.set_xlabel("Spacer Thickness (nm)", fontsize = 14)
ax1.set_ylabel(r"Wavelength ($\mu$m)", fontsize = 14)
ax1.set_title("Reflection for. Spacer Thickness vs Wavelength")
ax1.tick_params(axis='both', labelsize=14)

plt.show()