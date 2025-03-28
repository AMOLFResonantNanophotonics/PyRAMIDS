#!/usr/bin/env python3
# -*- coding: utf-8 -*-
print('making Figure 5')

#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%%

from Library.Use import Use_Planewaves as pw
from Library.Use import Use_LDOS as ImGLDOS
import numpy as np
from matplotlib import pyplot as plt
from Library.Use import Use_Radiationpattern as Farfield


plt.close('all')


def savefig(folderpath, filename):
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    plt.savefig(os.path.join(folderpath, filename), bbox_inches='tight')
    
    
folder = r"pdfimages/"
#%%


lam = 0.6
k0= 2*np.pi/ lam

## Arbitrary stack


'''Substrate (glass) / ITO (conductive layer, Anode) /Hole Transport Layer, HTL (PVK)/
Solgel (efficiency)/ Perovskite (Active layer)/ TPBI_LiF (ETL)/ Cathode (Al, reflector)/ Air'''


nstack=np.array([1.47, 1.75, 1.56, 1.42, 2.15 + 0.15*1j, 1.8, 0.912 + 6.55*1j, 1])
dstack=np.array([0.01, 0.02, 0.06, 0.150, 0.04, 0.08])


kparlist=np.arange(0.0,1.0,0.01)*k0*nstack[0].real

# kparlist= k0*np.sin(10*np.pi/180)*nstack[0]


'''Calculating Absoprtion per layer'''

plt.figure(2,figsize=(4,4), dpi=300)
As_perlayer,Ap_perlayer=pw.PerLayerAbsorption(k0, kparlist, nstack, dstack)
plt.plot(kparlist/k0,np.transpose(As_perlayer[3,:]),kparlist/k0,np.transpose(Ap_perlayer[3,:]),':')
plt.title('Absorp. per layer (p = solid, s= dashed)')
plt.xlabel(r'$k_{||}/k_0$')
plt.ylabel('Absorption')
plt.show()


#%%
'''On axis field, and absorption'''

kpar=k0*nstack[0]*np.sin(np.pi/3)

zlist=np.arange(-0.1,sum(dstack) + 0.1,0.001)

Sfield,Pfield,AbsS,AbsP = pw.OnAxisLocalFieldandAbsorption(k0, kpar, nstack, dstack,zlist)
 


#%%
fig, ax = plt.subplots(figsize=(4, 8), dpi=450)
ax.plot(AbsP[0, :], zlist, 'r-', label='Absorption P') 
ax.set_xlabel('Absorption P', color='r')
ax.tick_params(axis='x', labelcolor='r', direction='in', length=4, width=1)
# ax.set_xticks([0, 2.5, 5, 7.5])


ax_top = ax.twiny()  
ax_top.plot(AbsS[0, :], zlist, 'b:', label='Absorption S')
ax_top.set_xlabel('Absorption S', color='b')
ax_top.tick_params(axis='x', labelcolor='b', direction='in', length=4, width=1)
# ax_top.set_xticks([0, 2.5, 5, 7.5])

ax.set_ylabel(r'z [$\mu$m]')
yticks = np.insert(dstack.cumsum(), 0, 0)  # Add 0 at the beginning
ax.set_yticks(yticks)
# ax.set_yticks(dstack.cumsum()) 
ax.set_ylim([-0.13, sum(dstack) + 0.13])
ax.invert_yaxis()
dstack_cumsum = np.cumsum([0] + list(dstack))
for d in dstack_cumsum:
    ax.axhline(d, color='gray', linestyle='--', linewidth=1.5)

colors = ['grey', 'lightcoral', 'lightskyblue', 'orange', 'blueviolet', 'tan']
alphas = [0.5, 0.5, 0.6, 0.4, 0.3, 0.8]
for i in range(len(dstack_cumsum) - 1):
    ax.axhspan(dstack_cumsum[i], dstack_cumsum[i+1], color=colors[i % len(colors)], alpha=alphas[i], rasterized=True)

ax.axhspan(-0.2, 0, color='gray', alpha=0.2)  # Substrate region
ax.axhspan(dstack_cumsum[-1], dstack_cumsum[-1] + 0.05, color='white', alpha=0.2, rasterized=True)  # Top layer
ax.tick_params(axis='both', direction='in', length=4, width=1)

# plt.title("Absorption S and P vs z")
file = [folder,'Fig_5a_AbsorptiononAxis'+'.pdf']
savefig(file[0], file[1])
plt.show()
plt.close()




#%%
'''LDOS Calculation'''

rhoE_par,rhoE_perp,rhoM_par,rhoM_perp,rhoC = ImGLDOS.LDOS(k0,zlist,np.real(nstack),dstack)
Total_LDOS = (2*rhoE_par + rhoE_perp)/3

data=Farfield.TotalRadiated(k0,zlist,np.real(nstack),dstack)   
outpar_all4pi_fromP = data[0]
outperp_all4pi_fromP = data[1]
Rad_LDOS = (2*outpar_all4pi_fromP + outperp_all4pi_fromP)/3

outpar_guide_fromP = np.subtract(rhoE_par, outpar_all4pi_fromP)
outperp_guide_fromP = np.subtract(rhoE_perp, outperp_all4pi_fromP)
Guided_LDOS = (2*outpar_guide_fromP + outperp_guide_fromP)/3



fig, ax = plt.subplots(figsize=(4, 8), dpi=450)
ax.plot(Total_LDOS, zlist,'k:', label = 'Total LDOS')
ax.plot(Rad_LDOS, zlist, 'g--', label = 'Radiative LDOS')
ax.plot(Guided_LDOS, zlist, 'brown', label = 'Guided LDOS')
ax.legend()

for d in dstack_cumsum:
    ax.axhline(d, color='gray', linestyle='--', linewidth=1.5)

colors = ['grey', 'lightcoral', 'lightskyblue', 'orange', 'blueviolet', 'tan']
alphas = [0.5, 0.5, 0.6, 0.4, 0.3, 0.8]
for i in range(len(dstack_cumsum) - 1):
    ax.axhspan(dstack_cumsum[i], dstack_cumsum[i+1], color=colors[i % len(colors)], alpha=alphas[i], rasterized=True)

ax.axhspan(-0.2, 0, color='gray', alpha=0.2)  # Substrate region
ax.axhspan(dstack_cumsum[-1], dstack_cumsum[-1] + 0.05, color='white', alpha=0.2, rasterized=True)  # Top layer
ax.set_ylim([-0.13, sum(dstack) + 0.13])
ax.set_ylabel(r'z [$\mu$m]')
yticks = np.insert(dstack.cumsum(), 0, 0)  # Add 0 at the beginning
ax.set_yticks(yticks)
ax.invert_yaxis()

ax.set_xticks([0, 1.4, 2.8])

ax.tick_params(axis='both', direction='in', length=4, width=1)
file = [folder,'Fig_5b_LDOSAxis'+'.pdf']
savefig(file[0], file[1])
plt.show()

#%%
'''Radiation Pattern for a random dipole in the middle of the perovskite layer'''

nstack=np.array([1.47, 1.75, 1.56, 1.42, np.real(2.15 + 0.15*1j), 1.8, 0.912 + 6.55*1j, 1])

z = sum(dstack[0:3]) + dstack[3]/2

pux=np.array([1.0,0.0, 0.0])
puy=np.array([0.0,1.0, 0.0])
puz=np.array([0.0,0.0, 1.0])

mu=np.array([0,0,0])

Nthe = 1001
# thelist=np.append(np.linspace(-np.pi/2+0.0001,np.pi/2-0.0001,Nthe),np.linspace(np.pi/2+0.0001,3*np.pi/2-0.0001,Nthe))
thelist= np.linspace(np.pi/2+0.0001,3*np.pi/2-0.0001,Nthe)


Pkx,Ex,[th,ph]=Farfield.RadiationpatternPandField(k0, z, pux, mu, thelist, 0.0*thelist, nstack, dstack)
Pky,Ey,[th,ph]=Farfield.RadiationpatternPandField(k0, z, puy, mu, thelist, 0.0*thelist, nstack, dstack)
Pkz,Ez,[th,ph]=Farfield.RadiationpatternPandField(k0, z, puz, mu, thelist, 0.0*thelist, nstack, dstack)

rhoE_par,rhoE_perp,rhoM_par,rhoM_perp,rhoC = ImGLDOS.LDOS(k0,z, nstack,dstack)
Total_LDOS = (2*rhoE_par + rhoE_perp)/3

outFarField = (1/3)*(Pkx + Pky + Pkz)/Total_LDOS


f,ax = plt.subplots(figsize=(4, 4),subplot_kw=dict(projection="polar"), dpi=500)
ax.plot(thelist,outFarField/ np.max(outFarField),'r', linestyle = '-', linewidth=2.)
ax.set_theta_zero_location("S")
ax.set_theta_direction(-1)
ax.tick_params(axis='x', pad=10)  # Move angular labels farther from the plot
ax.tick_params(axis='y', pad=10)  # Move radial labels farther from the plot

ax.set_rticks([0.4,0.8]) 
ax.set_rmax(1.2)


plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

theta_shade = np.linspace(np.pi/2, 3*np.pi/2, 100)
r_shade = np.linspace(0, 1.2, 100)
Theta, R = np.meshgrid(theta_shade, r_shade)
ax.contourf(Theta, R, np.ones_like(Theta), levels=1, colors='grey', alpha=0.2)

ax.set_thetamin(90)
ax.set_thetamax(270)

file = [folder,'Fig_5c_RadiationPattern'+'.pdf']
savefig(file[0], file[1])

plt.show()





 