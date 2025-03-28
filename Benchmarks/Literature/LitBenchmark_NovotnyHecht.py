 
"""
Test radiation pattern routine by plotting the electric dipole radiation patterns
reported in Novotny & Hecht, Principles of Nano-optics,  1st edition, figures 10.7 and 10.9
"""

print('### Literature Benchmark: Novotny & Hecht, Principles of Nano-optics,  1st edition, figures 10.7 and 10.9 ###')

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

#---------------------------------------------------#
#---------------------------------------------------#
#                                                   #
# Novotny Chapter 10 - Fig. 10.7 Radiation Patterns #
#                                                   #
#---------------------------------------------------#
#---------------------------------------------------#

lam=488.0
k0=2.0*np.pi/lam

#geometry of the system
nstack=[1.50, np.sqrt(5.0), 1.0]  # refr index
dstack=[80.0]  # thickness (nm) of waveguide layer
pu=np.array([np.sqrt(3.0)/2.0 ,0.0, 0.5]) #electric dipole
mu=np.array([0,0,0]) #no magnetic dipole

#plotpoints
Nthe=4021;
thelist=np.linspace(-np.pi/2,3.0*np.pi/2,Nthe)


# Novotny & Hecht plot 4 heights, labelled as lambda /x, with x in heightlist
heightlist=[100,10,1,1/5]

#auxiliary, for the for loop
plotnum=[[0,0],[0,1],[1,0],[1,1]] # to run through the subplots
numh=len(heightlist)
fig,axs=plt.subplots(2,2,figsize=(10,10),subplot_kw=dict(projection="polar"))

for m in range(numh):
    z=lam/heightlist[m] + 80.0 

    P,E,tf=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0.0*thelist,nstack,dstack)


    axs[plotnum[m][0],plotnum[m][1]].plot(thelist,P)
    axs[plotnum[m][0],plotnum[m][1]].set_theta_zero_location("N")
    axs[plotnum[m][0],plotnum[m][1]].set_theta_direction(-1)
    axs[plotnum[m][0],plotnum[m][1]].set_title('Novotny&Hecht 1st edit, Fig 10.7. h=lamba/'+str(heightlist[m]))
    
                                                                            
plt.show()

#####################

plt.figure()
lam=633.0
k0=2.0*np.pi/lam


nstack=[1.50, 1.0] 
dstack=[] 
pu=np.array([np.sqrt(3.0)/2.0 ,0.0, 0.5])
mu=np.array([0,0,0])

Nthe=4021;
thelist=np.linspace(-np.pi/2,3.0*np.pi/2,Nthe)+np.pi/2
z=20 

P,E,tf=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0.0*thelist,nstack,dstack)

plt.plot(thelist*180/(np.pi),P)
plt.xlabel('theta (degrees)')
plt.ylabel('P')
plt.title('Novotny & Hecht - Figure 10.9c (1st edition)')
plt.show()