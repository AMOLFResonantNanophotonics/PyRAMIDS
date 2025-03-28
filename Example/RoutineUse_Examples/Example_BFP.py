#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This script shows how to plot the far field radiation pattern 
including Stokes polarimetry for different dipoles
'''


#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%%

import numpy as np
from Library.Util import Util_vectorpolarization as vector
 
from Library.Use import Use_Radiationpattern as Radpat
from matplotlib import pyplot as plt


'''Setting up layer geometry and wavelength'''

plt.close('all')
lam=0.6
k0=2.*np.pi/lam
nstack=[1.5,1.6,1.0]

z=+0.05
dstack=[0.1] 



''''''
### scatterer, px
''''''

pu=[1.0,0.0,0.0]  # electric dipole
mu=[0.0,0.0,0.0]  # magnetic dipole
 
Nthe=101
Nphi = 151
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(+0.01,(np.pi/2)-0.01,Nthe)


P_u,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_up_px=outE[0]
Ep_up_px=outE[1]


thelist=thelist+np.pi/2
P_d,outE,[theta_d,phi_d]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_down_px=outE[0]
Ep_down_px=outE[1]
  
print('Linear p_x dipole')

vector.BFPplotpassport(theta_u,phi_u,Es_up_px,Ep_up_px,nstack[-1],'p_x Upper hemisphere')
vector.BFPplotpassport(theta_d,-phi_d,Es_down_px,Ep_down_px,nstack[0],'p_x Lower hemisphere') 





''''''
### scatterer, py
''''''

pu=[0.0,1.0,0.0]    
mu=[0.0,0.0,0.0]  
 
Nthe=101
Nphi = 151
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(+0.01,(np.pi/2)-0.01,Nthe)


P_u,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_up_px=outE[0]
Ep_up_px=outE[1]


thelist=thelist+np.pi/2
P_d,outE,[theta_d,phi_d]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_down_px=outE[0]
Ep_down_px=outE[1]
  
print('Linear p_y dipole')

vector.BFPplotpassport(theta_u,phi_u,Es_up_px,Ep_up_px,nstack[-1],'p_y  Upper hemisphere')
vector.BFPplotpassport(theta_d,-phi_d,Es_down_px,Ep_down_px,nstack[0],'p_y Lower hemisphere') 





''''''
### scatterer, pz
''''''

pu=[0.0,0.0,1.0]    
mu=[0.0,0.0,0.0]  
 
Nthe=101
Nphi = 151
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(+0.01,(np.pi/2)-0.01,Nthe)


P_u,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_up_px=outE[0]
Ep_up_px=outE[1]


thelist=thelist+np.pi/2
P_d,outE,[theta_d,phi_d]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_down_px=outE[0]
Ep_down_px=outE[1]
  
print('Linear p_z dipole')

vector.BFPplotpassport(theta_u,phi_u,Es_up_px,Ep_up_px,nstack[-1],'p_z  Upper hemisphere')
vector.BFPplotpassport(theta_d,-phi_d,Es_down_px,Ep_down_px,nstack[0],'p_z Lower hemisphere') 




''''''
### scatterer, px+ipy
''''''

pu=[1.0,1.0j,0.0]
mu=[0.0,0.0,0.0]  
 

Nthe=101
Nphi = 151
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(+0.01,(np.pi/2)-0.01,Nthe)

P_u,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_up_pxiy=outE[0]
Ep_up_pxiy=outE[1]


thelist=thelist+np.pi/2
P_d,outE,[theta_d,phi_d]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_down_pxiy=outE[0]
Ep_down_pxiy=outE[1]
  

print('Circular in plane electric dipole')


vector.BFPplotpassport(theta_u,phi_u,Es_up_pxiy,Ep_up_pxiy,nstack[-1],'p_x+ip_y,  Upper hemisphere')
vector.BFPplotpassport(theta_d,-phi_d,Es_down_pxiy,Ep_down_pxiy,nstack[0],'p_x+ip_y Lower hemisphere') 




''''''
### scatterer, my
''''''

pu=[0.0,0.0,0.0]    
mu=[0.0,1.0,0.0]  
 
Nthe=101
Nphi = 151
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(+0.01,(np.pi/2)-0.01,Nthe)


P_u,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_up_px=outE[0]
Ep_up_px=outE[1]


thelist=thelist+np.pi/2
P_d,outE,[theta_d,phi_d]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_down_px=outE[0]
Ep_down_px=outE[1]
  
print('Linear m_y dipole')

vector.BFPplotpassport(theta_u,phi_u,Es_up_px,Ep_up_px,nstack[-1],'m_y  Upper hemisphere')
vector.BFPplotpassport(theta_d,-phi_d,Es_down_px,Ep_down_px,nstack[0],'m_y Lower hemisphere') 




''''''
### scatterer, px+imy
''''''

pu=[1.0,0,0.0]
mu=[0.0,1.j,0.0]  
 

Nthe=101
Nphi = 151
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(+0.01,(np.pi/2)-0.01,Nthe)

P_u,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_up_pximy=outE[0]
Ep_up_pximy=outE[1]


thelist=thelist+np.pi/2
P_d,outE,[theta_d,phi_d]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_down_pximy=outE[0]
Ep_down_pximy=outE[1]
  

print('In plane +handed electric-magnetic p_x+im_y')

vector.BFPplotpassport(theta_u,phi_u,   Es_up_pximy,  Ep_up_pximy,nstack[-1],'p_x+im_y,  Upper hemisphere')
vector.BFPplotpassport(theta_d,-phi_d,Es_down_pximy,Ep_down_pximy,nstack[0],'p_x+im_y Lower hemisphere') 


''''''
### scatterer, px-imy
''''''

pu=[1.0,0,0.0]
mu=[0.0,-1.j,0.0]  
 


Nthe=101
Nphi = 151
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(+0.01,(np.pi/2)-0.01,Nthe)

P_u,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_up_pxmimy=outE[0]
Ep_up_pxmimy=outE[1]


thelist=thelist+np.pi/2
P_d,outE,[theta_d,phi_d]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_down_pxmimy=outE[0]
Ep_down_pxmimy=outE[1]
  
print('In plane - handed electric-magnetic p_x-im_y')


vector.BFPplotpassport(theta_u,phi_u,   Es_up_pxmimy,  Ep_up_pxmimy,nstack[-1],'p_x-im_y  Upper hemisphere')
vector.BFPplotpassport(theta_d,-phi_d,Es_down_pxmimy,Ep_down_pxmimy,nstack[0],'p_x-im_y Lower hemisphere') 



''''''
### Fluorescence, incoherent sum
# 
# Based on following two mathematical insights.

# (1) The incoherent sum over all dipole orientations when it comes to intensities in the far field, is equivalent to summing three orthogonal orientations. This is certainly true for classical fixed current (not fixed power) sources.
# (2) Intensities are additive when incoherent. This means Stokes parameters are additive too.
''''''


###### UPPER HEMISPHERE
 
pu=[1.0,0.0,0.0]
mu=[0.0,0.0,0.0]  
 
Nthe=101
Nphi = 151
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(+0.01,(np.pi/2)-0.01,Nthe)

P_u,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_up_px=outE[0]
Ep_up_px=outE[1]

Ex,Ey=vector.FarfieldEsp2EcartesianBFP(theta_u, phi_u, Es_up_px, Ep_up_px)
S0_up_px,S1_up_px,S2_up_px,S3_up_px, eps,alph=vector.Field2Stokes(Ex, Ey)


pu=[0.0,1.0,0.0]
P_u,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_up_py=outE[0]
Ep_up_py=outE[1]

Ex,Ey=vector.FarfieldEsp2EcartesianBFP(theta_u, phi_u, Es_up_py, Ep_up_py)
S0_up_py,S1_up_py,S2_up_py,S3_up_py, eps,alph=vector.Field2Stokes(Ex, Ey)

pu=[0.0,0.0,1.0]

P_u,outE,[theta_u,phi_u]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_up_pz=outE[0]
Ep_up_pz=outE[1]
Ex,Ey=vector.FarfieldEsp2EcartesianBFP(theta_u, phi_u, Es_up_pz, Ep_up_pz)
S0_up_pz,S1_up_pz,S2_up_pz,S3_up_pz, eps,alph=vector.Field2Stokes(Ex, Ey)



###### LOWER  HEMISPHERE

pu=[1.0,0.0,0.0]
thelist=thelist+np.pi/2
P_d,outE,[theta_d,phi_d]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_down_px=outE[0]
Ep_down_px=outE[1]

Ex,Ey=vector.FarfieldEsp2EcartesianBFP(theta_d, -phi_d, Es_down_px, Ep_down_px)
S0_down_px,S1_down_px,S2_down_px,S3_down_px, eps,alph=vector.Field2Stokes(Ex, Ey)

pu=[0.0,1.0,0.0]
P_d,outE,[theta_d,phi_d]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_down_py=outE[0]
Ep_down_py=outE[1]

Ex,Ey=vector.FarfieldEsp2EcartesianBFP(theta_d, -phi_d, Es_down_py, Ep_down_py)
S0_down_py,S1_down_py,S2_down_py,S3_down_py, eps,alph=vector.Field2Stokes(Ex, Ey)

pu=[0.0,0.0,1.0]
P_d,outE,[theta_d,phi_d]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_down_pz=outE[0]
Ep_down_pz=outE[1]

Ex,Ey=vector.FarfieldEsp2EcartesianBFP(theta_d, -phi_d, Es_down_pz, Ep_down_pz)
S0_down_pz,S1_down_pz,S2_down_pz,S3_down_pz, eps,alph=vector.Field2Stokes(Ex, Ey)

print('Fluorescence doing incoherent sum')

S0_up=S0_up_px+S0_up_py+S0_up_pz
S1_up=S1_up_px+S1_up_py+S1_up_pz
S2_up=S2_up_px+S2_up_py+S2_up_pz
S3_up=S3_up_px+S3_up_py+S3_up_pz


S0_down=S0_down_px+S0_down_py+S0_down_pz
S1_down=S1_down_px+S1_down_py+S1_down_pz
S2_down=S2_down_px+S2_down_py+S2_down_pz
S3_down=S3_down_px+S3_down_py+S3_down_pz

#%%
vector.BFPplotpassportS(theta_u,phi_u, S0_up,  S1_up,  S2_up,  S3_up,   nstack[-1],'fluorescence isotropic')
vector.BFPplotpassportS(theta_d,-phi_d,S0_down,S1_down,S2_down,S3_down, nstack[0], 'fluorescence isotropic') 


print('The lower and upper hemispheres exhibit exactly rotated due to the perspective change when viewed from the origin')