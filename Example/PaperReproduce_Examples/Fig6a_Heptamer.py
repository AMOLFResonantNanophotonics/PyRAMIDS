#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
print('Doing Figure 6a')

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


class DrudePancake(): 
    """Returns the polarizability tensor of a pancake
    with permitivitty following the drude model
    with the short axis parallel to the z-axis 

    nsur: refractive index of surrounding
    a: ellipsoid long axis in m
    b: ellipsoid short axis in m
    angle: angle w.r.t x-axis in radians
    omega: driving frequency
    
    """
    
    
    def __init__(self,eps_inf, omegap, gamma,nsur,a ,b):
        self.eps_inf = eps_inf
        self.omega_plasma = omegap
        self.gamma_plasma = gamma
        self.nsur = nsur
        self.a = a/2 #Careful, input is length of the whole pancake, self.a is only half!
        self.b = b/2
        self.ex = np.sqrt(1-((self.b)/(self.a))**2) #Exentricity
        self.eps_sur = self.nsur**2
    
        ##Geometrical factors from Bohren Huffman (5.34)##
        self.ge = np.sqrt((1-self.ex**2)/self.ex**2)
        self.oblateL1 = self.ge/(2*self.ex**2) * (np.pi/2 - np.arctan(self.ge))-self.ge**2/2
        self.oblateL2 = 1-2*self.oblateL1
        
        
    def epsilon(self, omega):
        return self.eps_inf - self.omega_plasma ** 2 / (omega ** 2 + 1j * omega * self.gamma_plasma)
        # return self.omega_plasma ** 2 / (omega ** 2 - 1j * omega * self.gamma_plasma)

    
    def static_polarizability_tensor(self, omega):
       ##Following Bohrne, Huffman, eq. 5.34
       
       ##Here, the factor 4pi is included, following bohren and huffman. This factor is NOT present in the original polarizability of a sphere in Nelson's code, I included it here as well
       alpha1 = 4*np.pi/3*self.a*self.a*self.b*((self.epsilon(omega) - self.eps_sur)/(self.eps_sur + self.oblateL1 * (self.epsilon(omega) - self.eps_sur))) #Without depolarization; along a --> long
       alpha2 = 4*np.pi/3*self.a*self.a*self.b*((self.epsilon(omega) - self.eps_sur)/(self.eps_sur + self.oblateL2 * (self.epsilon(omega) - self.eps_sur))) #Without depolarization; b --> short
       
       Pxx = alpha1
       Pyy = alpha1
       Pzz = alpha2
       
       P = np.eye(6)*1j
       
       P[0,0] = Pxx
       P[1,1] = Pyy
       P[2,2] = Pzz
       return P


    def dynamic_polarizability_tensor(self, omega):
       
       ##Add dynamic correction, no depolarizations##
       static_polarizability = self.static_polarizability_tensor(omega)
      
       return alpha_radiative_tensor(omega,static_polarizability,self.nsur)


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


Nk = 201
lamlist = np.linspace(250, 850, Nk)  # Wavelengths in nm

om = 2.0 * np.pi*3E17 / lamlist  # Convert wavelength to angular frequency
klist = 2.0 * np.pi / lamlist  # Free-space wavevector

wp_Au = 8E15
g_Au = 7E13
eps_inf = 1  
epsilon_Au = DrudeLorentz(om, wp_Au, g_Au)
drudeparamAu=[wp_Au,g_Au,eps_inf]
eps_Au=Drude(om,drudeparamAu)
nAu = np.sqrt(epsilon_Au)  # Convert permittivity to refractive index



'''This is actually SILVER'''
# eps_inf=5.43+0.55j #background eps for Ag, Palik fit. See PRB Hinke Schokker in 2014
# wp_Au=1.39E16 #Drude wSPP for gold, Palik fit  
# g_Au=8.21E13 # Drude damping gold
# drudeparamAg=[wp_Au,g_Au,eps_inf]
# eps_Ag=Drude(om,drudeparamAg)
# nAu = np.sqrt(eps_Ag) 


# r_outside = 45
# r_centre = 45

# V_outside = (4/3) * np.pi * (r_outside) ** 3
# V_centre = (4/3) * np.pi * (r_centre) ** 3

# # rdip = generate_heptamer_rdip(100, -r_outside)

height = 25
rdip = generate_heptamer_rdip(100, -height/2)


theta=np.array([0.00])
phi=np.array([0.00])

s=np.array([0]) 
p=np.array([1])

Scat_cs = 0.*lamlist
Ext_cs = 0.*lamlist


Scat_cs_nospacer = 0.*lamlist
Ext_cs_nospacer = 0.*lamlist

Scat_cs_free = 0.*lamlist
Ext_cs_free = 0.*lamlist

for i, lam in enumerate(lamlist):
    
    k0 = 2*np.pi/ lam
    
    nstack = [1., 1.5, nAu[i],  1.5]
    dstack = [50, 60]

    diplayer, Ndip=ms.dipolelayerchecker(rdip ,nstack,dstack)

    # alpha=ms.Rayleighspherepolarizability(nAu[i],nstack[diplayer], V_outside)
    # alphalist = np.tile(alpha, (Ndip, 1, 1))
    # alphalist[0,:] = ms.Rayleighspherepolarizability(nAu[i],nstack[diplayer], V_centre)
    
    alphalist = 0.0j*np.tile(np.eye(6), (Ndip, 1, 1))
    alphalist[0,:] = DrudePancake(eps_inf, wp_Au, g_Au, 1, 50, height).dynamic_polarizability_tensor(om[i])
    alphalist[1:,:] = DrudePancake(eps_inf, wp_Au, g_Au, 1, 35, height).dynamic_polarizability_tensor(om[i])

    invalpha = ms.invalphadynamicfromstatic(alphalist, rdip, diplayer, k0, nstack, dstack)
    
    driving, intensity =ms.Planewavedriving(theta,phi,s,p,rdip,k0,nstack,dstack)
    
    M = ms.SetupandSolveCouplingmatrix(invalpha, rdip, k0, nstack, dstack)
    
    pnm=ms.Solvedipolemoments(M, driving)
    work=ms.Work(pnm, diplayer, driving, k0,nstack)  
    
    Pup, Pdown=ms.TotalfarfieldpowerManydipoles(pnm, rdip, diplayer, k0, nstack, dstack)
    
    Scat_cs[i] =  (Pdown + Pup)/np.sum(intensity)
    Ext_cs[i] = np.sum(work)/ np.sum(intensity)
    
    
    nstack = [1., 1.5, nAu[i],  1.5]
    dstack = [50, 0]

    diplayer, Ndip=ms.dipolelayerchecker(rdip ,nstack,dstack)

    invalpha = ms.invalphadynamicfromstatic(alphalist, rdip, diplayer, k0, nstack, dstack)
    driving, intensity =ms.Planewavedriving(theta,phi,s,p,rdip,k0,nstack,dstack)
    M = ms.SetupandSolveCouplingmatrix(invalpha, rdip, k0, nstack, dstack)
    
    pnm=ms.Solvedipolemoments(M, driving)
    work=ms.Work(pnm, diplayer, driving, k0,nstack)  
    
    Pup, Pdown=ms.TotalfarfieldpowerManydipoles(pnm, rdip, diplayer, k0, nstack, dstack)
    Scat_cs_nospacer[i] =  (Pdown + Pup)/np.sum(intensity)
    Ext_cs_nospacer[i] = np.sum(work)/ np.sum(intensity)
    
    nstack = [1., 1., 1,  1.]
    dstack = [50, 0]

    diplayer, Ndip=ms.dipolelayerchecker(rdip ,nstack,dstack)

    invalpha = ms.invalphadynamicfromstatic(alphalist, rdip, diplayer, k0, nstack, dstack)
    driving, intensity =ms.Planewavedriving(theta,phi,s,p,rdip,k0,nstack,dstack)
    M = ms.SetupandSolveCouplingmatrix(invalpha, rdip, k0, nstack, dstack)
    
    pnm=ms.Solvedipolemoments(M, driving)
    work=ms.Work(pnm, diplayer, driving, k0,nstack)  
    
    Pup, Pdown=ms.TotalfarfieldpowerManydipoles(pnm, rdip, diplayer, k0, nstack, dstack)
    Scat_cs_free[i] =  (Pdown + Pup)/np.sum(intensity)
    Ext_cs_free[i] = np.sum(work)/ np.sum(intensity)
    
#%%
fig, ax = plt.subplots(figsize=(6, 4), dpi = 400)

# Plot the data
ax.plot(lamlist / 1000, Scat_cs * 1E-6, label='Scat. CS')
ax.plot(lamlist / 1000, Ext_cs * 1E-6, label='Ext. CS')
ax.plot(lamlist / 1000, (Ext_cs - Scat_cs) * 1E-6, label='Abs. CS')

# Labels and legend
ax.set_xlabel(r'Wavelength [$\mu$m]')
ax.set_ylabel(r'Cross Sec. [$\mu m^2$]')
ax.legend()
# ax.axvline(x=0.565, color='black', linestyle='--', linewidth=1)

# Set ticks inside
ax.tick_params(direction='in', which='both')  # 'both' ensures ticks are applied to major & minor
file = [folder,'z_Fig_6a_Heptamer'+' .pdf']
savefig(file[0], file[1])
# Show plot
plt.show()


#%%
#%%
fig, ax = plt.subplots(figsize=(6, 4), dpi = 400)

# Plot the data
ax.plot(lamlist / 1000, Scat_cs * 1E-6, 'b-',label='Scat. CS')
# ax.plot(lamlist / 1000, Ext_cs * 1E-6, label='Ext. CS')
ax.plot(lamlist / 1000, (Ext_cs - Scat_cs) * 1E-6, 'r-', label='Abs. CS')

ax.plot(lamlist / 1000, Scat_cs_nospacer * 1E-6, 'b--',label='Scat. CS no mirror')

ax.plot(lamlist / 1000, Scat_cs_free * 1E-6, 'gray', label='Scat. CS free')

# ax.plot(lamlist / 1000, Ext_cs * 1E-6, label='Ext. CS')
ax.plot(lamlist / 1000, (Ext_cs_nospacer - Scat_cs_nospacer) * 1E-6, 'r--',label='Abs. CS no mirror')
ax.plot(lamlist / 1000, (Ext_cs_free - Scat_cs_free) * 1E-6, 'k--',label='Abs. CS free')

# Labels and legend
ax.set_xlabel(r'Wavelength [$\mu$m]')
ax.set_ylabel(r'Cross Sec. [$\mu m^2$]')
ax.legend()
# ax.axvline(x=0.565, color='black', linestyle='--', linewidth=1)

# Set ticks inside
ax.tick_params(direction='in', which='both')  # 'both' ensures ticks are applied to major & minor
file = [folder,'Fig_6a_Heptamer'+' .pdf']
savefig(file[0], file[1])
# Show plot
plt.show()






