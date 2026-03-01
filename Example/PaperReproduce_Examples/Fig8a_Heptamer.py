#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
print('Doing Figure 8a')

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
from scipy.linalg import inv

#%%
c = 299792458 * 1e9 # [nm/s]

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
    only in case of homogeneous surrounding!!
    
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
    
    def __init__(self,eps_inf, omegap, gamma, nsur, a ,b):
        self.eps_inf = eps_inf
        self.omega_plasma = omegap
        self.gamma_plasma = gamma
        self.nsur = nsur
        self.eps_sur = self.nsur**2

        self.a = a/2 ## interpret input as full axis lengths; store semi-axes
        self.b = b/2
        
        ratio = self.b / self.a
        ratio = min(ratio, 1.0)
        self.ex = np.sqrt(max(0.0, 1.0 - ratio**2)) # eccentricity
    
        ##Geometrical factors from Bohren Huffman Book (5.33-5.34)##
        
        # Oblate spheroid depolarization factors (stable near sphere)
        if self.ex < 1e-12:
            self.oblateL1 = 1/3
            self.oblateL2 = 1/3
        else:
            ge = np.sqrt((1 - self.ex**2) / (self.ex**2))
            self.oblateL1 = (ge/(2*self.ex**2)) * (np.pi/2 - np.arctan(ge)) - (ge**2)/2
            self.oblateL2 = 1 - 2*self.oblateL1
        
        
    def epsilon(self, omega):
        return self.eps_inf - self.omega_plasma ** 2 / (omega**2 + 1j * omega * self.gamma_plasma)

    
    def static_polarizability_tensor(self, omega):
        ##Following Bohrne, Huffman, eq. 5.34
        eps = self.epsilon(omega)
        deps = eps - self.eps_sur
         
        # geometric volume of spheroid with semi-axes (a,a,b)
        V = self.a *self.a *self.b/3        
        #divide by 3 so sphere limit matches Rayleighspherepolarizability from ms
        alpha1 = (V) * deps / (self.eps_sur + self.oblateL1 * deps)
        alpha2 = (V) * deps / (self.eps_sur + self.oblateL2 * deps)
        
        Pxx = alpha1
        Pyy = alpha1
        Pzz = alpha2
        
        P = np.zeros((6, 6), dtype=complex)

        P[0,0] = Pxx
        P[1,1] = Pyy
        P[2,2] = Pzz
        
        P[3,3] = V * 1e-20
        P[4,4] = V * 1e-20
        P[5,5] = V * 1e-20
        return P


    def dynamic_polarizability_tensor(self, omega):       
       
        ##Add dynamic correction, no depolarizations##
        ## only in case of homogeneous surrounding!!
        
       static_polarizability = self.static_polarizability_tensor(omega)
       return alpha_radiative_tensor(omega,static_polarizability,self.nsur)


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
Nk = 81
lamlist = np.linspace(450, 700, Nk)  # Wavelengths in nm

om = 2.0 * np.pi*3E17 / lamlist  # Convert wavelength to angular frequency
klist = 2.0 * np.pi / lamlist  # Free-space wavevector

wp_Au = 7E15
g_Au = 1E14
eps_inf = 1 
epsilon_Au = DrudeLorentz(om, wp_Au, g_Au)
drudeparamAu = [wp_Au,g_Au,eps_inf]
eps_Au=Drude(om,drudeparamAu)
nAu = np.sqrt(epsilon_Au)  # Convert permittivity to refractive index

height = 30
rdip = generate_heptamer_rdip(90, -height/2) # in air.

#%%
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
    alphalist = 0.0j*np.tile(np.eye(6), (np.shape(rdip)[0], 1, 1))
    alphalist[0,:] = DrudePancake(eps_inf, wp_Au, g_Au, 1, 70, height).static_polarizability_tensor(om[i])   #central particle
    alphalist[1:,:] = DrudePancake(eps_inf, wp_Au, g_Au, 1, 50, height).static_polarizability_tensor(om[i])
    
    
    '''Doing the mirror + spacer situation'''
    nstack = [1., 1.5, nAu[i],  1.5]
    dstack = [80, 60]

    invalpha = ms.invalphadynamicfromstatic(alphalist, rdip,k0, nstack, dstack)
    driving, intensity =ms.Planewavedriving(theta,phi,s,p,rdip,k0,nstack,dstack)
    M = ms.SetupandSolveCouplingmatrix(invalpha, rdip, k0, nstack, dstack)    
    
    pnm=ms.Solvedipolemoments(M, driving)
    work=ms.Work(pnm, rdip,driving, k0,nstack,dstack)  
    
    Pup, Pdown=ms.TotalfarfieldpowerManydipoles(pnm, rdip, k0, nstack, dstack)
    Scat_cs[i] =  (Pdown + Pup)/np.sum(intensity)
    Ext_cs[i] = np.sum(work)/ np.sum(intensity)
    
    
    '''Directly on the mirror with no spacer'''
    nstack = [1., nAu[i],  1.5]
    dstack = [60]

    invalpha = ms.invalphadynamicfromstatic(alphalist, rdip, k0, nstack, dstack)
    driving, intensity =ms.Planewavedriving(theta,phi,s,p,rdip,k0,nstack,dstack)
    M = ms.SetupandSolveCouplingmatrix(invalpha, rdip, k0, nstack, dstack)
    
    pnm=ms.Solvedipolemoments(M, driving)
    work=ms.Work(pnm, rdip, driving, k0,nstack,dstack)  
    
    Pup, Pdown=ms.TotalfarfieldpowerManydipoles(pnm, rdip,k0, nstack, dstack)
    Scat_cs_nospacer[i] =  (Pdown + Pup)/np.sum(intensity)
    Ext_cs_nospacer[i] = np.sum(work)/ np.sum(intensity)
    
    
    '''In free space'''
    nstack = [1., 1.]
    dstack = []

    invalpha = ms.invalphadynamicfromstatic(alphalist, rdip, k0, nstack, dstack)
    driving, intensity =ms.Planewavedriving(theta,phi,s,p,rdip,k0,nstack,dstack)
    M = ms.SetupandSolveCouplingmatrix(invalpha, rdip, k0, nstack, dstack)
    
    pnm=ms.Solvedipolemoments(M, driving)
    work=ms.Work(pnm, rdip, driving, k0,nstack,dstack)  
    
    Pup, Pdown=ms.TotalfarfieldpowerManydipoles(pnm, rdip, k0, nstack, dstack)
    Scat_cs_free[i] =  (Pdown + Pup)/np.sum(intensity)
    Ext_cs_free[i] = np.sum(work)/ np.sum(intensity)
      
#%%
fig, ax = plt.subplots(figsize=(6, 4), dpi = 400)

ax.plot(lamlist, Scat_cs * 1E-6, 'b-',label='Scat. CS')
ax.plot(lamlist, Scat_cs_nospacer * 1E-6, 'r-',label='Scat. CS on mirror')
ax.plot(lamlist, Scat_cs_free * 1E-6, 'gray',ls = '-',  label='Scat. CS free')

# ax.plot(lamlist, Ext_cs * 1E-6, 'b--',label='Ext. CS')
# ax.plot(lamlist, Ext_cs_nospacer* 1E-6, 'r--', label='Ext. CS on mirror')
# ax.plot(lamlist, Ext_cs_free * 1E-6, 'gray', ls = '--', label='Ext. CS free')

ax.plot(lamlist, (Ext_cs - Scat_cs) * 1E-6, 'b--',label='Abs. CS')
ax.plot(lamlist, (Ext_cs_nospacer - Scat_cs_nospacer)  * 1E-6, 'r--', label='Abs. CS on mirror')
ax.plot(lamlist, (Ext_cs_free - Scat_cs_free) * 1E-6, 'gray', ls = '--', label='Abs. CS free')

ax.set_xlabel(r'Wavelength [nm]')
ax.set_ylabel(r'Cross Sec. [$\mu m^2$]')
ax.legend()

ax.tick_params(direction='in', which='both')  #
file = [folder,'Fig_8a_Heptamer_mirror'+' .pdf']
savefig(file[0], file[1])
plt.show()

