
print('Making Figure 1c')

#%%

'''
Benchmark routine replicating
Polarization-controlled directional scattering for nanoscopic position sensing
Martin Neugebauer  Paweł Woźniak  Ankan Bag Gerd Leuchs  Peter Banzer 
Nat. Commun. 2016 20 7 11286
'''
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

from Library.Use import Use_Radiationpattern as Radpat
from Library.Util import Util_vectorpolarization as vector

import matplotlib.pyplot as plt
import numpy as np


#plot range for angles
Nthe=2021;
#%% Only into glass
thelist=np.linspace(-np.pi/2,3.0*np.pi/2,Nthe)


nstack=[1.5,1.0]
dstack=[]
z=42  #  dipole is assumed at centre of sphere than z  = 42 nm.



#%%
lam=550.0 #wvlength in vacuum
k0=2.0*np.pi/lam

#electric z-dipole only
pu=np.array([0.0 ,0.0, 1])
mu=np.array([0,0,0])

Pk_pz,E_pz,tf=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0.0*thelist,nstack,dstack)


f,ax = plt.subplots(2,1,figsize=(5,10),subplot_kw=dict(projection="polar"), dpi=200)
f.subplots_adjust(hspace=0.15)
    
ax[0].plot(thelist,Pk_pz,'r--', label = r'$p_z$')
ax[0].set_theta_zero_location("N")
ax[0].set_theta_direction(-1)

ax[0].set_yticklabels([])

#magnetic y-dipole only
lam=670.0 #wvlength in vacuum
k0=2.0*np.pi/lam


pu=np.array([0.0, 0.0, 0.])
mu=np.array([0., 1., 0.])

Pk_my,E_my,tf=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0.0*thelist,nstack,dstack)

ax[0].plot(thelist,Pk_my,'k',label = r'$m_y$')
ax[0].set_theta_zero_location("N")
ax[0].set_theta_direction(-1)
ax[0].set_title("Fig 2b") 
ax[0].legend(loc="best")
ax[0].set_thetamin(90)
ax[0].set_thetamax(270)

r_max = ax[0].get_ylim()[1]  # Get the plot's radial limit

angle = np.arcsin(nstack[1]/nstack[0])
ax[0].plot([np.pi - angle, np.pi - angle], [0, r_max], ':', color= 'gray', linewidth=1.)  # Dashed black line
ax[0].plot([np.pi + angle, np.pi + angle], [0, r_max], ':', color= 'gray', linewidth=1.)  # Dashed black line


#magnetic + electric 
lam=520.0 #wvlength in vacuum
k0=2.0*np.pi/lam


pu=np.array([0.0, 0.0, 1.])
mu=np.array([0., -1., 0.])

Pk_mp,E_mp,tf=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0.0*thelist,nstack,dstack)

ax[1].plot(thelist,Pk_mp,'b',label = r'$m_y + p_z$')
ax[1].set_theta_zero_location("N")
ax[1].set_theta_direction(-1)
ax[1].set_title("Fig 2c") 
ax[1].legend(loc="best")
ax[1].set_yticklabels([])

ax[1].set_thetamin(90)
ax[1].set_thetamax(270)

r_max = ax[1].get_ylim()[1] 

angle = np.arcsin(nstack[1]/nstack[0])
ax[1].plot([np.pi - angle, np.pi - angle], [0, r_max], ':', color= 'gray', linewidth=1.5)  # Dashed black line
ax[1].plot([np.pi + angle, np.pi + angle], [0, r_max], ':', color= 'gray', linewidth=1.5)  # Dashed black line

plt.show()

#%%
f,ax = plt.subplots(1,1,figsize=(5,5),subplot_kw=dict(projection="polar"), dpi=400)
f.subplots_adjust(hspace=0.15)

ax.plot(thelist,Pk_pz/np.max(Pk_pz),'r--', label = r'$p_z$')
ax.plot(thelist,Pk_my/np.max(Pk_my),'k--', alpha = 0.6,label = r'$m_y$')
ax.plot(thelist,Pk_mp/np.max(Pk_mp),'b',label = r'$m_y + p_z$')

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_title("Fig 2b") 
ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1))  # Moves legend outside
# ax.set_thetamin(90)
# ax.set_thetamax(270)
# Define custom rmax
rmax = 1.  # Manually set maximum radial value
ax.set_rlim([0, rmax+0.1])  # Set radial axis limits

# Set radial tick positions manually (e.g., rmax/2, rmax/3)
ax.set_yticks([rmax / 2, rmax / 1.33, rmax / 4, rmax])  
# ax.set_yticklabels([r'$r_{max}/2$', r'$r_{max}/3$', r'$r_{max}$'])  # Custom tick labels

angle = np.arcsin(nstack[1]/nstack[0])
ax.plot([np.pi - angle, np.pi - angle], [0, r_max], '-', color= 'gray', alpha = 0.4, linewidth=1.)  # Dashed black line
ax.plot([np.pi + angle, np.pi + angle], [0, r_max], '-', color= 'gray', alpha = 0.4, linewidth=1.)  # Dashed black line


file = [folder,'Fig_1c_a_BanzerPolar'+'.pdf']
savefig(file[0], file[1])
plt.show()

#%%

NAmin, NAmax  = 0., 1.4
thetamin, thetamax = np.arcsin(NAmin/nstack[0]), np.arcsin(NAmax/nstack[0]) 



Nthe=101
Nphi = 151
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(np.pi - thetamin,np.pi - thetamax,Nthe)


P_d,outE,[theta_d,phi_d]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_down_mp=outE[0]
Ep_down_mp=outE[1]

kx, ky, S0, S1, S2, S3 = vector.BFPplotpassport(theta_d,-phi_d,Es_down_mp,Ep_down_mp,nstack[0],title=' ',basis='cartesian') 


# vector.BFPplotIntensity(theta_d,-phi_d,Es_down_mp,Ep_down_mp,nstack[0],'p_x Lower hemisphere') 

#%%

fig,ax=plt.subplots(figsize=(6,5), dpi=400)
pcm = ax.pcolormesh(kx, ky, S0/np.max(S0), vmin=0, vmax=1, cmap='inferno', shading='gouraud', rasterized = True)
ax.set_title('S0')
ax.set_aspect('equal')
ax.set_xlim([np.min(kx)-.05,np.max(kx)+0.05])
ax.set_ylim([np.min(ky)-.05,np.max(ky)+0.05])
ax.set_aspect('equal')
cbar = fig.colorbar(pcm, ax=ax, shrink = 0.9)
cbar.set_ticks([0, 0.5 , 1])  # Set ticks at min and max
cbar.ax.set_yticklabels([f'{0:.2f}', f'{0.5:.2f}',f'{1:.2f}'])
cbar.ax.tick_params(labelsize=16)



# # Add axis labels with extremes and 0
xticks = [-NAmax, 0, NAmax]
yticks = [-NAmax, 0, NAmax]
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels([f'{tick:.2f}' for tick in xticks])
ax.set_yticklabels([f'{tick:.2f}' for tick in yticks])

ax.tick_params(axis='both', labelsize=16)   
ax.set_xlabel('k$_{x}$/k$_{0}$',fontsize = 16)
ax.set_ylabel('k$_{y}$/k$_{0}$',fontsize = 16)
ax.tick_params(axis='both', which='major', direction='in', length=4, width=1.)  # Major ticks

circle = plt.Circle((0, 0), 1, color='white', linewidth=2, fill=False)
ax.add_patch(circle)

file = [folder,'Fig_1c_b_Banzerglass'+'.pdf']
savefig(file[0], file[1])
plt.show()

plt.show(block=False)
# plt.pause(1)
plt.close()
