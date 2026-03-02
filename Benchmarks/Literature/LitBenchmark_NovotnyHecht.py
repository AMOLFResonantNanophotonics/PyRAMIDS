#%%
'''Reproducing Novotny and Hecht, Principles of Nano-Optics (1st ed.), Figs. 10.7 and 10.9'''

#%%
print('### Literature Benchmark: Novotny & Hecht, Principles of Nano-Optics (1st ed.), Figs. 10.7 and 10.9 ###')

#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def savefig(folderpath, filename):
    os.makedirs(folderpath, exist_ok=True)
    plt.savefig(os.path.join(folderpath, filename), bbox_inches="tight")


folder = "pdfimages/"
#%%

from Library.Use import Use_Radiationpattern as Radpat

import matplotlib.pyplot as plt
import numpy as np


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


fig,axs=plt.subplots(2,2,figsize=(12,12),subplot_kw=dict(projection="polar"), dpi=400)
for m in range(numh):
    z=lam/heightlist[m] + 80.0 

    P,E,tf=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0.0*thelist,nstack,dstack)


    axs[plotnum[m][0],plotnum[m][1]].plot(thelist,P, 'k' )
    axs[plotnum[m][0],plotnum[m][1]].set_theta_zero_location("N")
    axs[plotnum[m][0],plotnum[m][1]].set_theta_direction(-1)
    axs[plotnum[m][0],plotnum[m][1]].set_title(r'h=$\lambda$/'+str(heightlist[m]))

fig.suptitle('Novotny & Hecht, Fig. 10.7 (1st ed.)', fontsize=16)

file = [folder,'LitBenchmark_NovotnyHecht_1stEdBook_Fig10_7'+' .pdf']
savefig(file[0], file[1])
plt.show()



#####################

plt.figure(dpi=300)
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
plt.plot(thelist*180/(np.pi),P,'k')
plt.xlabel(r'$\theta\ (^\circ)$')
plt.ylabel(r'$P/P_0$')
plt.title('Novotny & Hecht, Fig. 10.9c (1st ed.)')
file = [folder,'LitBenchmark_NovotnyHecht_1stEdBook_Fig10_9'+' .pdf']
savefig(file[0], file[1])
plt.show()
