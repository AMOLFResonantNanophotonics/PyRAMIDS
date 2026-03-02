#%%
'''Reproducing Neugebauer et al., Nat. Commun. 7, 11286 (2016)'''

#%%
print('### Literature Benchmark: Neugebauer et al., Nat. Commun. 7, 11286 (2016) ###')
#%%
import os
import sys
import matplotlib.pyplot as plt

plt.close('all')

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
thelist=np.linspace(np.pi/2,3.0*np.pi/2,Nthe)


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


f,ax = plt.subplots(2,1,figsize=(6,10),subplot_kw=dict(projection="polar"), dpi=300)
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
ax[0].set_title("Fig. 2b") 
ax[0].legend(loc="best")
ax[0].set_thetamin(90)
ax[0].set_thetamax(270)

r_max = ax[0].get_ylim()[1]  # Get the plot's radial limit

angle = np.arcsin(nstack[1]/nstack[0])
ax[0].plot([np.pi - angle, np.pi - angle], [0, r_max], ':', color= 'gray', linewidth=1.)  # Dashed black line
ax[0].plot([np.pi + angle, np.pi + angle], [0, r_max], ':', color= 'gray', linewidth=1.)  # Dashed black line

#magnetic y-dipole only
lam=520.0 #wvlength in vacuum
k0=2.0*np.pi/lam


pu=np.array([0.0, 0.0, 1.])
mu=np.array([0., -1., 0.])

Pk_mp,E_mp,tf=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0.0*thelist,nstack,dstack)

ax[1].plot(thelist,Pk_mp,'b',label = r'$m_y + p_z$')
ax[1].set_theta_zero_location("N")
ax[1].set_theta_direction(-1)
ax[1].set_title("Fig. 2c") 
ax[1].legend(loc="best")
ax[1].set_yticklabels([])
ax[1].set_thetamin(90)
ax[1].set_thetamax(270)
r_max = ax[1].get_ylim()[1] 
angle = np.arcsin(nstack[1]/nstack[0])
ax[1].plot([np.pi - angle, np.pi - angle], [0, r_max], ':', color= 'gray', linewidth=1.5)  # Dashed black line
ax[1].plot([np.pi + angle, np.pi + angle], [0, r_max], ':', color= 'gray', linewidth=1.5)  # Dashed black line
f.suptitle('Neugebauer et al., Nat. Commun. 7, 11286 (2016), Fig. 2b,c')
file = [folder,'LitBenchmark_Neugebauer_etal_NatCommun2016_Fig2'+' .pdf']
savefig(file[0], file[1])
plt.show()


#%%

NAmin, NAmax  = 0.95, 1.3
thetamin, thetamax = np.arcsin(NAmin/nstack[0]), np.arcsin(NAmax/nstack[0]) 


Nthe=101
Nphi = 151
philist=np.linspace(0+0.001,(2*np.pi),Nphi)
thelist=np.linspace(np.pi - thetamin,np.pi - thetamax,Nthe)


P_d,outE,[theta_d,phi_d]=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,philist,nstack,dstack)
Es_down_mp=outE[0]
Ep_down_mp=outE[1]

S0, fig = vector.BFPplotIntensity(theta_d,-phi_d,Es_down_mp,Ep_down_mp,nstack[0],
                                  title = 'Neugebauer et al. Nature Commun. 20 7 11286 (2016), Fig. 2e \n', show=False) 

fig.savefig(folder + 'LitBenchmark_Neugebauer_etal_NatCommun2016_Fig2e.pdf',
            dpi=300, bbox_inches='tight')
plt.show()
