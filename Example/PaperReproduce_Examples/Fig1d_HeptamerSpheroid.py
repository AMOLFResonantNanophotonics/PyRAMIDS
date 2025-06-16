#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
print('Doing Figure 1d')

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

c = 299792458 * 1e9 # [m/s]

def alpha_radiative(omega, alpha_0, n_refractive):
    """
    EH symmetric unit system
    :param omega:
    :param alpha_0:
    :param n_refractive:
    :param a:
    :return:
    """
    v = c/n_refractive
    k = omega/v
    return alpha_0 / (1 - 1j * k ** 3 * 2 * alpha_0 / 3)






def alpha_radiative_tensor(omega, alpha_static, n_refractive):
    """
    EH symmetric unit system
    :param omega:
    :param alpha_0:
    :param n_refractive:
    :param a:
    :return:
    """
    v = c/n_refractive
    k = omega/v
    
    inv_alpha_rad =  inv(alpha_static) - np.eye(6)*1j * k**3 * 2 /3
    return inv(inv_alpha_rad)


    
class DrudeSigar(): 
    """Returns the polarizability tensor of a prolate 
    spheroid with long axis at an angle phi w.r.t 
    the x-axis 
    
    nsur: refractive index of surrounding
    a: ellipsoid long axis in m
    b: ellipsoid short axis in m
    angle: angle w.r.t x-axis in radians
    omega: driving frequency
    
    """
    
    
    def __init__(self,eps_inf, omegap, gamma,nsur,a ,b, angle):
        self.eps_inf = eps_inf
        self.omega_plasma = omegap
        self.gamma_plasma = gamma
        self.nsur = nsur
        self.a = a/2 #Careful, input is length of the whole sigar, self.a is only half!
        self.b = b/2
        self.angle = angle
        self.ex = np.sqrt(1-((self.b)/(self.a))**2) #Exentricity
        self.eps_sur = self.nsur**2
    
        ##Geometrical factors from Bohren Huffman (5.33)##
        self.prolateL1 = (((1-self.ex**2))/(self.ex**2)) * (-1 + 1/(2*self.ex)*np.log((1+self.ex)/(1-self.ex))) ##Long axis
        self.prolateL2 = (1-self.prolateL1)/2 ##Short axis
        
    def rotation(self, angle):
        return np.array([[np.cos(angle) ,-np.sin(angle) ,0],[np.sin(angle), np.cos(angle) ,0],[0, 0, 1]]) #Rotation operator for rotations around z-axis
    
    def rotation_6x6(self, angle):    
        """ Returns a 6x6 block diagonal rotation matrix to operate on the full polarizability tensor. """
        # R3 = np.array([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle), 0],[0, 0, 1]])  # 3x3 Rotation matrix
        return np.block([[self.rotation(self.angle), np.zeros((3,3))],  # Upper left 3x3 is R3, upper right is 0s
                         [np.zeros((3,3)), self.rotation(self.angle)]])  # Lower left is 0s, lower right is R3
    
    def epsilon(self, omega):
        return self.eps_inf - self.omega_plasma ** 2 / (omega ** 2 + 1j * omega * self.gamma_plasma)

    def static_polarizability_tensor(self, omega):
        ##Following Bohrne, Huffman, eq. 5.32
        
        ##Here, the factor 4pi is included, following bohren and huffman. This factor is NOT present in the polarizability of a sphere
        alpha1 = 4*np.pi/3*self.a*self.b*self.b*((self.epsilon(omega) - self.eps_sur)/(self.eps_sur + self.prolateL1 * (self.epsilon(omega) - self.eps_sur))) #Without depolarization; along a --> long
        alpha2 = 4*np.pi/3*self.a*self.b*self.b*((self.epsilon(omega) - self.eps_sur)/(self.eps_sur + self.prolateL2 * (self.epsilon(omega) - self.eps_sur))) #Without depolarization; b --> short
        
        Pxx = alpha1
        Pyy = alpha2
        Pzz = alpha2
        
        P = np.eye(6)*1j
        
        P[0,0] = Pxx
        P[1,1] = Pyy
        P[2,2] = Pzz
        
        return P
    
   
    def dynamic_polarizability_tensor(self, omega):
        
        ##Add dynamic correction, no depolarizations##
        # c0 = 299792458
        # k = omega*self.nsur/c0
        static_polarizability = self.static_polarizability_tensor(omega)
        

        # return np.matmul(self.rotation_6x6(self.angle),np.matmul(alpha_radiative_tensor(omega,static_polarizability,self.nsur),inv(self.rotation_6x6(self.angle))))
        return np.matmul(self.rotation_6x6(self.angle), np.matmul(static_polarizability, inv(self.rotation_6x6(self.angle))))



#%%



# Drude Model for Gold
def DrudeLorentz(omega, omega_p, gamma):
    return (omega_p**2) / (omega**2 - 1j * gamma * omega)

def Drude(w,drudeparam):
    # Drude model with a bound electron offset
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


Nk = 151
lamlist = np.linspace(450, 700, Nk)  # Wavelengths in nm

om = 2.0 * np.pi*3E17 / lamlist  # Convert wavelength to angular frequency
klist = 2.0 * np.pi / lamlist  # Free-space wavevector

wp_Au = 5E15
g_Au = 7E13
eps_inf = 1  
epsilon_Au = DrudeLorentz(om, wp_Au, g_Au)
drudeparamAu=[wp_Au,g_Au,eps_inf]
eps_Au=Drude(om,drudeparamAu)
nAu = np.sqrt(epsilon_Au) 


height_mid = 20
rdip = generate_heptamer_rdip(62, 0)


theta=np.array([0.00])
phi=np.array([0.00])

s=np.array([1]) 
p=np.array([0])

Scat_cs = 0.*lamlist
Ext_cs = 0.*lamlist


for i, lam in enumerate(lamlist):
    
    k0 = 2*np.pi/ lam
    
    nstack = [1., 1.]
    dstack = []

    diplayer, Ndip=ms.dipolelayerchecker(rdip ,nstack,dstack)
    
    alphalist = 0.0j*np.tile(np.eye(6), (Ndip, 1, 1))
    
    alphalist[0,:] = DrudeSigar(eps_inf, wp_Au, g_Au, 1, 25, 10, 0).dynamic_polarizability_tensor(om[i])
    alphalist[1:,:] = DrudeSigar(eps_inf, wp_Au, g_Au, 1, 40, 10, 0).dynamic_polarizability_tensor(om[i])

    invalpha = ms.invalphadynamicfromstatic(alphalist, rdip, diplayer, k0, nstack, dstack)
    
    driving, intensity =ms.Planewavedriving(theta,phi,s,p,rdip,k0,nstack,dstack)
    
    M = ms.SetupandSolveCouplingmatrix(invalpha, rdip, k0, nstack, dstack)
    
    pnm=ms.Solvedipolemoments(M, driving)
    work=ms.Work(pnm, diplayer, driving, k0,nstack)  
    
    Pup, Pdown=ms.TotalfarfieldpowerManydipoles(pnm, rdip, diplayer, k0, nstack, dstack)
    
    Scat_cs[i] =  (Pdown + Pup)/np.sum(intensity)
    Ext_cs[i] = np.sum(work)/ np.sum(intensity)
    
#%%
fig, ax = plt.subplots(figsize=(6, 4), dpi = 400)

ax.plot(lamlist / 1000, Scat_cs * 1E-6, label='Scat. CS')

ax.set_xlabel(r'Wavelength [$\mu$m]')
ax.set_ylabel(r'Cross Sec. [$\mu m^2$]')
ax.legend()
# ax.axvline(x=0.572, color='black', linestyle='--', linewidth=1)

ax.tick_params(direction='in', which='both')
file = [folder,'Fig_1d_Heptamer'+' .pdf']
savefig(file[0], file[1])
plt.show()

# %%
# import scipy.linalg as LA

# lam = 572
# om = 2.0 * np.pi*3E17 / lam  # Convert wavelength to angular frequency

# k0 = 2*np.pi/ lam

# alphalist = 0.0j*np.tile(np.eye(6), (Ndip, 1, 1))

# alphalist[0,:] = DrudeSigar(eps_inf, wp_Au, g_Au, 1, 25, 10, 0).dynamic_polarizability_tensor(om)
# alphalist[1:,:] = DrudeSigar(eps_inf, wp_Au, g_Au, 1, 40, 10, 0).dynamic_polarizability_tensor(om)
# driving, intensity = ms.Planewavedriving(theta,phi,s,p,rdip,k0,nstack,dstack)

# M = ms.SetupandSolveCouplingmatrix(invalpha, rdip, k0, nstack, dstack)

# # Diagonalize the Coupling Matrix to find dressed states
# eigenvalues, eigenvectors = LA.eigh(M)  # Eigen decomposition

# # Compute Dressed Polarizability
# alpha_dressed = np.zeros_like(M, dtype=complex)
# for i in range(len(eigenvalues)):
#     alpha_dressed += np.outer(eigenvectors[:, i], eigenvectors[:, i]) / (eigenvalues[i] - om)


# pnm=ms.Solvedipolemoments(M, driving) # driven response
# # pnm=ms.Solvedipolemoments(alpha_dressed, driving)

# px = pnm[0,:]
# py = pnm[1,:]
# rdips = np.array(rdip)
# dipx = rdips[:,0]
# dipy = rdips[:,1]

# magnitude = np.sqrt(px**2 + py**2)
# px_normalized = px / magnitude
# py_normalized = py / magnitude


# plt.figure()
# plt.quiver(dipx, dipy, px, py, color='k')

# plt.show()



# eigenvalues, eigenvectors = LA.eigh(M)  # Eigen decomposition

# # Step 2: Select dominant eigenmodes (basis vectors)
# # Select first few eigenmodes (e.g., first three)
# modes_to_plot = [-1, -2, -3]  # Index of eigenmodes to visualize

# # Extract dipole positions
# rdips = np.array(rdip)
# dipx = rdips[:, 0]
# dipy = rdips[:, 1]

# # Step 3: Plot Quiver Plots for Selected Eigenmodes
# fig, axes = plt.subplots(1, len(modes_to_plot), figsize=(12, 4))

# for i, mode in enumerate(modes_to_plot):
#     mode_vector = eigenvectors[:, mode]  # Get eigenvector (basis mode)
    
#     # Extract x and y components
#     px_mode = mode_vector[0::6]  # Select every 6th component for px
#     py_mode = mode_vector[1::6]  # Select every 6th component for py

#     # Normalize vectors for visualization
#     # magnitude = np.sqrt(px_mode**2 + py_mode**2)
#     # px_mode /= magnitude
#     # py_mode /= magnitude

#     # Quiver plot for each mode
#     ax = axes[i]
#     ax.quiver(dipx, dipy, px_mode, py_mode, color='b')
#     ax.set_xlabel('x-position (nm)')
#     ax.set_ylabel('y-position (nm)')
#     ax.set_title(f'Eigenmode {mode} (λ={lam} nm)')
#     ax.axis('equal')

# plt.tight_layout()
# plt.show()





