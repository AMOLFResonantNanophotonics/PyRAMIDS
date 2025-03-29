
#%%
'''Reproducing Joulian et. al. Phys. Rev. B. 68, 245405 (2003)'''

print('###  Literature Benchmark: Joulian et. al. Phys. Rev. B. 68, 245405 (2003)  ###')
#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%%

import numpy as np
from Library.Use import Use_LDOS as ImGLDOS
import matplotlib.pyplot as plt


def epsilon(omega, omega_p=1.747e16, gamma=7.596e13):
    """
    Calculate the dielectric function epsilon based on Drudes.
    Returns:
        The dielectric function
    """
    return 1 - (omega_p**2) / (omega * (omega + 1j * gamma))


# Define frequency range from 10^14 to 10^17 rad/s (log scale)
omega_values = np.logspace(14, 17, 500)

epsilon_values = epsilon(omega_values)

n_Al_values = np.sqrt(epsilon_values)  # Complex refractive index

#%%

'''Figure 2'''

zlist = [1e-9, 10e-9, 100e-9, 1000e-9]  # Distance from the aluminum surface (in meters)
dstack = []  # Semi-infinite substrate, so no additional layers
rho = np.zeros((len(omega_values), 4))  # To store LDOS values


for j, z in enumerate(zlist):
    for i, omega in enumerate(omega_values):
        k0 = omega / 3e8  # Free-space wavevector (k0 = omega / c)
        nstack = [n_Al_values[i], 1] 
        
        rho_0 = omega**2 / (3 * np.pi**2 * 3e8**3)  # Free-space LDOS

        
        rhoE_par, rhoE_perp, rhoM_par, rhoM_perp, rhoC = ImGLDOS.LDOS(k0, z, nstack, dstack)
        
        # Compute total electric and magnetic LDOS contributions
        rhoE_total = (rhoE_par[0] * 2 + rhoE_perp[0]) / 3  # Electric dipole LDOS
        rhoM_total = (rhoM_par[0] * 2 + rhoM_perp[0]) / 3  # Magnetic dipole LDOS
        rho[i, j] = (rhoE_total + rhoM_total ) *rho_0 # Total LDOS

# Plot LDOS
plt.figure(dpi=400)
plt.loglog(omega_values, rho[:, 0], 'r-', label=r"z = 1nm")  # Electric LDOS
plt.loglog(omega_values, rho[:, 1], 'b--', label=r"z = 10nm")  # Magnetic LDOS
plt.loglog(omega_values, rho[:, 2], 'g--', label=r"z = 100nm")  # Total LDOS
plt.loglog(omega_values, rho[:, 3], 'k--', label=r"z = 1000nm")  # Total LDOS


plt.xlabel(r"$\omega$ (rad s$^{-1}$)")
plt.ylabel("Density of states")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()



#%% 
'''Figure 3'''

# LDOS calculation at z = 10 nm
z = 10e-9  # Distance from the aluminum surface (in meters)
dstack = []  # Semi-infinite substrate, so no additional layers
rho = np.zeros((len(omega_values), 3))  # To store LDOS values

for i, omega in enumerate(omega_values):
    k0 = omega / 3e8  # Free-space wavevector (k0 = omega / c)
    nstack = [n_Al_values[i], 1] 
    
    rho_0 = omega**2 / (3 * np.pi**2 * 3e8**3)  # Free-space LDOS

    
    rhoE_par, rhoE_perp, rhoM_par, rhoM_perp, rhoC = ImGLDOS.LDOS(k0, z, nstack, dstack)
    
    # Compute total electric and magnetic LDOS contributions
    rho[i, 0] = rho_0*(rhoE_par[0] * 2 + rhoE_perp[0]) / 3  # Electric dipole LDOS
    rho[i, 1] = rho_0*(rhoM_par[0] * 2 + rhoM_perp[0]) / 3  # Magnetic dipole LDOS
    rho[i, 2] = rho[i, 0] + rho[i, 1]  # Total LDOS

# Plot LDOS
plt.figure(dpi=400)
plt.loglog(omega_values, rho[:, 0], 'b--', label=r"$\rho_E(\omega)$")  # Electric LDOS
plt.loglog(omega_values, rho[:, 1], 'g-.', label=r"$\rho_M(\omega)$")  # Magnetic LDOS
plt.loglog(omega_values, rho[:, 2], 'k-', label=r"Density of states")  # Total LDOS


plt.xlabel(r"$\omega$ (rad s$^{-1}$)")
plt.ylabel("Density of states")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
