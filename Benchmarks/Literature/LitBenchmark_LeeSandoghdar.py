'''
Benchmark routine replicating
K.G. Lee et al. Nature Photonics volume 5, pages 166–169 (2011)
'''

print('###  Literature Benchmark: Lee et. al. Nature Photonics 5, 166-169 (2011)  ###')

#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%%

from Library.Use import Use_Radiationpattern as Radpat

import matplotlib.pyplot as plt
import numpy as np

#-----------------------------------------------
#-----------------------------------------------
#
#Lee, Vahid Sandoghdar et al. Nature Photonics '11 - Figure 1
#
#-----------------------------------------------
#-----------------------------------------------
lam=580.0 #wvlength in vacuum
k0=2.0*np.pi/lam

#electric z-dipole only
pu=np.array([0.0 ,0.0, 1])
mu=np.array([0,0,0])

#plot range for angles
Nthe=2021;
thelist=np.linspace(-np.pi/2,3.0*np.pi/2,Nthe)

#run through the 3 cases
nstack=[1.78,1.0]
dstack=[]
z=5

Pk,E,tf=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0.0*thelist,nstack,dstack)
f,ax = plt.subplots(3,1,figsize=(5,15),subplot_kw=dict(projection="polar"))
f.subplots_adjust(hspace=0.35)
    
scaler=(np.cos(thelist)>0)*9+1 #Lee multiplies upper hemisphere by 10
ax[2].plot(thelist,scaler*Pk,'k')
ax[2].set_theta_zero_location("N")
ax[2].set_theta_direction(-1)
ax[2].set_title("Lee-Fig1, z-dipole 5 nm above sapphire") 

nstack=[1.78, 1.5, 1.0] 
dstack=[350.0] 
z=200 #position of the dipole from the interface
Pr,E,tf=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0.0*thelist,nstack,dstack)


scaler=(np.cos(thelist)>0)*9+1 #Lee multiplies upper hemisphere by 10
ax[0].plot(thelist,scaler*Pr,'r')
ax[0].set_theta_zero_location("N")
ax[0].set_theta_direction(-1)
ax[0].set_title("Lee, t=350nm, h=200 nm") 




nstack=[1.78, 1.5, 1.0] 
dstack=[600.0] 
z=200 #position of the dipole from the interface 

Pg,E,tf=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0.0*thelist,nstack,dstack)

scaler=(np.cos(thelist)>0)*9+1 #Lee multiplies upper hemisphere by 10
ax[1].plot(thelist,scaler*Pg,'g')
ax[1].set_theta_zero_location("N")
ax[1].set_theta_direction(-1)
ax[1].set_title("Lee, t=600nm, h=200 nm")
plt.show()
 

plt.figure(figsize=(8,5))
plt.plot(thelist-np.pi,Pk/1.5**2,'k',thelist-np.pi,Pr,'r',thelist-np.pi,Pg,'g')
plt.xlim([0,np.pi/2.]) 
plt.xlabel('Angle (lower hemisphere, radians)')
plt.ylabel('Power/steradian (normalization is in a.u in Lee et al.)')
plt.title('K.G. Lee et al. Nature Photonics volume 5, pages 166–169 (2011), Fig 1')
# not 100% sure how KG Lee introduced the normalization /1/eps_host. Probably
# to normalize the total power of the source to unity.
plt.show()
