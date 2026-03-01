#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
print('Doing Figure XX')

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

from Library.Use import Use_Multiplescattering as ms

import numpy as np
from matplotlib import pyplot as plt


#%%
from scipy.linalg import inv

c = 299792458 * 1e9 # [nm/s]


def alpha_radiative_tensor(omega, alpha_static, n_refractive):
    v = c / n_refractive
    k = omega / v  # 1/nm
    inv_alpha_rad = inv(alpha_static) - np.eye(6, dtype=complex) * (1j * (2/3) * k**3)
    return inv(inv_alpha_rad)

class DrudeSigar:
    """Returns the polarizability tensor of a prolate 
    #     spheroid with long axis at an angle phi w.r.t 
    #     the x-axis 
    
    #     nsur: refractive index of surrounding
    #     a: ellipsoid long axis in m
    #     b: ellipsoid short axis in m
    #     angle: angle w.r.t x-axis in radians
    #     omega: driving frequency
    """
    
    def __init__(self, eps_inf, omegap, gamma, nsur, a, b, angle):
        self.eps_inf = eps_inf
        self.omega_plasma = omegap
        self.gamma_plasma = gamma
        self.nsur = nsur
        self.a = a/2  # meters
        self.b = b/2  # meters
        self.angle = angle
        self.ex = np.sqrt(1 - (self.b/self.a)**2)
        self.eps_sur = self.nsur**2
        
        if abs(self.a - self.b) < 1e-12:
            
            # exact sphere
            self.ex = 0.0
            self.prolateL1 = 1/3
            self.prolateL2 = 1/3
        else:
            self.ex = np.sqrt(1 - (self.b/self.a)**2)
            e = self.ex
            self.prolateL1 = ((1-e**2)/e**2) * (-1 + (1/(2*e))*np.log((1+e)/(1-e)))
            self.prolateL2 = (1 - self.prolateL1)/2

    def rotation(self, angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle),  np.cos(angle), 0],
                         [0,              0,             1]], dtype=float)

    def rotation_6x6(self, angle):
        R3 = self.rotation(angle)
        Z  = np.zeros((3,3))
        return np.block([[R3, Z],
                         [Z,  R3]])

    def epsilon(self, omega):
        return self.eps_inf - self.omega_plasma**2 / (omega**2 + 1j*omega*self.gamma_plasma)

    def static_polarizability_tensor(self, omega):  
        V = (1/3)*self.a*self.b*self.b     # 4 pi dropped for convention match with ms  rayleigh...
        
        alpha1 = V* ((self.epsilon(omega) - self.eps_sur) / (self.eps_sur + self.prolateL1*(self.epsilon(omega)-self.eps_sur)))
        alpha2 = V* ((self.epsilon(omega) - self.eps_sur) / (self.eps_sur + self.prolateL2*(self.epsilon(omega)-self.eps_sur)))


        P = np.zeros((6, 6), dtype=complex)

        P[0,0] = alpha1
        P[1,1] = alpha2
        P[2,2] = alpha2
        
        P[3,3] = V * 1e-20
        P[4,4] = V * 1e-20
        P[5,5] = V * 1e-20
        
        return P

    def dynamic_polarizability_tensor(self, omega):
        A0 = self.static_polarizability_tensor(omega)
        A  = alpha_radiative_tensor(omega, A0, self.nsur)
        R6 = self.rotation_6x6(self.angle)
        return R6 @ A @ inv(R6)

# Drude Model for Gold
def DrudeLorentz(omega, omega_p, gamma):
    return (omega_p**2) / (omega**2 - 1j * gamma * omega)

def Drude(w,drudeparam):
    wSP=drudeparam[0]
    g=drudeparam[1]
    epsinf=drudeparam[2]
    return epsinf-wSP*wSP/(w*(w+1.0j*g))


def generate_heptamer_rdip(radius, z_position):

    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points for hexagon
    x_hex = radius * np.cos(angles)
    y_hex = radius * np.sin(angles)
    
    rdip = [(0, 0, z_position)]  # Central particle
    rdip += [(x, y, z_position) for x, y in zip(x_hex, y_hex)]  # Outer particles
    return rdip


#%%
#%%



Nk = 151
lamlist = np.linspace(520, 660, Nk)  # Wavelengths in nm

om = 2.0 * np.pi*3E17 / lamlist  # Convert wavelength to angular frequency
klist = 2.0 * np.pi / lamlist  # Free-space wavevector

wp_Au = 5E15
g_Au = 6E13
eps_inf = 1 
epsilon_Au = DrudeLorentz(om, wp_Au, g_Au)
drudeparamAu=[wp_Au,g_Au,eps_inf]
eps_Au=Drude(om,drudeparamAu)
nAu = np.sqrt(epsilon_Au) 

nstack = [1., 1.]
dstack = []

rdip = generate_heptamer_rdip(80, 0)
diplayer, Ndip=ms.dipolelayerchecker(rdip ,nstack,dstack)


theta=np.array([0.00])
phi=np.array([0.00])

s=np.array([1]) 
p=np.array([0])

Scat_cs = 0.*lamlist
Ext_cs = 0.*lamlist


for i, lam in enumerate(lamlist):
    
    k0 = 2*np.pi/ lam

    alphalist = 0.0j*np.tile(np.eye(6), (np.shape(rdip)[0], 1, 1))
    alphalist[0,:] = DrudeSigar(eps_inf, wp_Au, g_Au, 1, 50, 30, 0).static_polarizability_tensor(om[i])   
    alphalist[1:,:] = DrudeSigar(eps_inf, wp_Au, g_Au, 1, 75, 30, 0).static_polarizability_tensor(om[i])   

    
    invalpha = ms.invalphadynamicfromstatic(alphalist, rdip,k0, nstack, dstack)
    driving, intensity =ms.Planewavedriving(theta,phi,s,p,rdip,k0,nstack,dstack)
    M = ms.SetupandSolveCouplingmatrix(invalpha, rdip, k0, nstack, dstack)    
    
    pnm=ms.Solvedipolemoments(M, driving)
    work=ms.Work(pnm, rdip,driving, k0,nstack,dstack)  
    
    Pup, Pdown=ms.TotalfarfieldpowerManydipoles(pnm, rdip, k0, nstack, dstack)
    
    Scat_cs[i] =  (Pdown + Pup)/np.sum(intensity)
    Ext_cs[i] = np.sum(work)/ np.sum(intensity)
    
#%%
fig, ax = plt.subplots(figsize=(6, 4), dpi = 400)

ax.plot(lamlist / 1000, Ext_cs * 1E-6, label='Extinction')

ax.set_xlabel(r'Wavelength [$\mu$m]')
ax.set_ylabel(r'Cross Sec. [$\mu m^2$]')
ax.legend()
ax.axvline(x=0.594, color='black', linestyle='--', linewidth=1)
ax.tick_params(direction='in', which='both')
plt.show()

# %%
import scipy.linalg as LA

nlam = len(lamlist)
Ne = Ndip * 3

M_elec_all = np.zeros((nlam, Ne, Ne), dtype=complex)

for cl, lam in enumerate(lamlist):

    om_target = 2*np.pi*3E17 / lam
    k0 = 2*np.pi / lam

    alphalist = 0.0j*np.tile(np.eye(6), (Ndip,1,1))
    alphalist[0,:]  = DrudeSigar(eps_inf, wp_Au, g_Au, 1, 50, 30, 0).static_polarizability_tensor(om_target)
    alphalist[1:,:] = DrudeSigar(eps_inf, wp_Au, g_Au, 1, 75, 30, 0).static_polarizability_tensor(om_target)

    invalpha = ms.invalphadynamicfromstatic(alphalist, rdip, k0, nstack, dstack)
    M = ms.SetupandSolveCouplingmatrix(invalpha, rdip, k0, nstack, dstack)

    M_elec_all[cl,:,:] = M[:Ne,:Ne]
    
#%%
all_evals = np.zeros((nlam, Ne), dtype=complex)
all_evecs = np.zeros((nlam, Ne, Ne), dtype=complex)

for cl in range(nlam):
    eigvals, eigvecs = LA.eig(M_elec_all[cl])
    all_evals[cl,:] = eigvals
    all_evecs[cl,:,:] = eigvecs
    
    
    
#%%

super_idx = np.zeros(nlam, dtype=int)
sub_idx   = np.zeros(nlam, dtype=int)

for cl, lam in enumerate(lamlist):
    eigvals = all_evals[cl]
    eigvecs = all_evecs[cl]

    K = 6
    cand = np.argsort(np.abs(eigvals))[:K]

    B = np.zeros(K)
    for j, m in enumerate(cand):
        v = eigvecs[:, m]
        px = v[0::3]; py = v[1::3]
        B[j] = (np.abs(px.sum())**2 + np.abs(py.sum())**2).real

    super_idx[cl] = cand[np.argmax(B)]
    sub_idx[cl]   = cand[np.argmin(B)]

cl_super = np.argmin(np.abs(all_evals[np.arange(nlam), super_idx]))
cl_sub   = np.argmin(np.abs(all_evals[np.arange(nlam), sub_idx]))

lam_super = lamlist[cl_super]
lam_sub   = lamlist[cl_sub]

vec_super = all_evecs[cl_super, :, super_idx[cl_super]]
vec_sub   = all_evecs[cl_sub,   :, sub_idx[cl_sub]]
#%%
def quiver_mode(vec, rdip_arr, title=""):
    x = rdip_arr[0,:]
    y = rdip_arr[1,:]

    px = vec[0::3]
    py = vec[1::3]

    plt.figure(figsize=(4,4))
    plt.quiver(x, y, np.real(px), np.real(py))
    plt.scatter(x, y, s=20)
    plt.axis("equal")
    plt.title(title)
    plt.xlabel("x (nm)")
    plt.ylabel("y (nm)")
    plt.show()

rdip_arr = np.array(rdip).T

quiver_mode(vec_super, rdip_arr, f"Bright @ {lam_super:.1f} nm")
quiver_mode(vec_sub,    rdip_arr, f"Subradiant @ {lam_sub:.1f} nm")