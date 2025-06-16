#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
'''Reproducing Sersic et. al. Phys. Rev. B 83, 245102 (2011)'''

#%%
print('### Literature Benchmark:   Sersic et. al. PRB 83, 2011 Figure 3f ###')

#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%%

import numpy as np
import matplotlib.pyplot as plt

from Library.Use import Use_Multiplescattering as ms



def magnetoelectric_polarizability(omega, omega0, gamma, V, etaE, etaH, etaC, nsur):
    c0 = 3e17  # [nm/s]
    k = omega / (c0 / nsur)
    Lorentz = (omega0**2 * V) / (omega0**2 - omega**2 - 1j * omega * gamma)

    alpha0 = np.zeros((6, 6), dtype=complex)
    alpha0[0, 0] = etaE * Lorentz
    alpha0[5, 5] = etaH * Lorentz       
    alpha0[0, 5] = 1j * etaC * Lorentz  
    alpha0[5, 0] = -alpha0[0, 5]  

    # Fudge to ensure invertibility
    alpha0 += 1e-12 * Lorentz * np.eye(6)
    return alpha0



def rotate_polarizability(alpha_tensor, psi_deg):
    """
    Rotate a 6x6 polarizability tensor by psi_deg around the z-axis.
    """
    psi = np.deg2rad(psi_deg)
    R3 = np.array([
        [np.cos(psi), np.sin(psi), 0],
        [-np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])
    
    R6 = np.block([
        [R3, np.zeros_like(R3)],
        [np.zeros_like(R3), R3]
    ])
    return R6.T @ alpha_tensor @ R6



def stereodimer_positions(d=150):
    return np.array([[0, 0, 0], [0, 0, d]])


#%%

V = 200 * 200 * 30  # nm^3
etaE, etaH, etaC = 0.7, 0.3, 0.4
omega0 = 1.26E15 #2 * np.pi * 200e12  # ~200 THz resonance
gamma = 1.25e14/10  # gold
nsur = 1.0

twist_angles = np.linspace(0, 180, 61)
lamlist = np.linspace(900, 3500, 101)
omlist = 2.0 * np.pi * 3E17 / lamlist

rdip = stereodimer_positions(d=150)

ext_vs_twist = np.zeros((len(twist_angles), len(omlist)))

for j, psi in enumerate(twist_angles):
    for i, omega in enumerate(omlist):
        
        
        alpha1 = magnetoelectric_polarizability(omega, omega0, gamma, V, etaE, etaH, etaC, nsur)
        alpha2 = rotate_polarizability(alpha1.copy(), psi)
        
        
        alphalist = np.zeros((2, 6, 6), dtype=complex)
        alphalist[0] = alpha1
        alphalist[1] = alpha2

        k0 = omega / 3E17
        
        nstack = [1.0, 1.0]
        dstack = []
        
        diplayer, Ndip = ms.dipolelayerchecker(rdip, nstack, dstack)
        invalpha = ms.invalphadynamicfromstatic(alphalist, rdip, diplayer, k0, nstack, dstack)

        theta, phi = np.array([0.0]), np.array([0.0])
        s, p = np.array([0]), np.array([1])
        driving, intensity = ms.Planewavedriving(theta, phi, s, p, rdip, k0, nstack, dstack)

        M = ms.SetupandSolveCouplingmatrix(invalpha, rdip, k0, nstack, dstack)
        pnm = ms.Solvedipolemoments(M, driving)
        work = ms.Work(pnm, diplayer, driving, k0, nstack)
        ext_cs = np.sum(work) / np.sum(intensity)
        ext_vs_twist[j, i] = ext_cs
        
        

#%%

frequencies_THz = omlist / (2 * np.pi * 1e12)

fig, ax = plt.subplots(figsize=(7, 5),dpi=450)

cmax = 0.8#np.max(ext_vs_twist.T*1e-6)

pcm = ax.pcolormesh(twist_angles, frequencies_THz, ext_vs_twist.T*1e-6, vmin = 0., vmax = cmax, shading='gouraud', cmap='hot', rasterized = True)

cbar = fig.colorbar(pcm, ax=ax)
        
ticks = [0, cmax/2, cmax]
cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
cbar.set_label(r'Extinction $\sigma$ [$\mu$m$^2$]', fontsize=14)
cbar.ax.tick_params(labelsize=14)  

ax.set_xticks([0, 90, 180])
ax.set_xlabel(r"Twist angle ($^{o}$)", fontsize=14)

ax.set_ylabel("Frequency [THz]", fontsize=14)
ax.set_title("", fontsize=12)
ax.set_ylim([90,320])
ax.tick_params(axis='both', labelsize=14)
plt.title('Sersic et. al. PRB 83, 2011, Figure 3f')
plt.show()
