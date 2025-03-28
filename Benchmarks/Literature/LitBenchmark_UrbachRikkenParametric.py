"""
Reproduces Urbach & Rikken, Phys Rev A 57, 3913 (1998)
"""

print('### Literature Benchmark: Urbach & Rikken, Phys. Rev. A 57 3913 (1998) ###')

#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
#%%


from Library.Use import Use_LDOS as ImGLDOS
from Library.Use import Use_Radiationpattern as Farfield

import matplotlib.pyplot as plt
import numpy as np
 
 
 

#Urbach & Rikken figure 10
lam=1.0  #wvlen in micron
k0=2*np.pi/lam
nslab=3.6
dlist=[0.25,3.0] #thicknesses of the thin layer to loop over (there are just 2)
nlist=np.linspace(1.,5.,75)  #plot points for the substrate refractive index that is scanned
n3=1.0   #superstrate

numd=len(dlist)
numn=len(nlist)

ldavav=np.zeros([numd,numn])
ldravav=np.zeros([numd,numn])

for m in range(numd):
    for mm in range(numn):
        d=dlist[m]
        zlist=np.linspace(0,d,100)
        nstack=np.array([nlist[mm],nslab,n3])
        dstack=np.array([d])
        
        rhoE_par,rhoE_perp,rhoM_par,rhoM_perp,rhoC=ImGLDOS.LDOS(k0,zlist,nstack,dstack)
        ldav=(rhoE_par*2.0+rhoE_perp)/3.0
        
        
        data=Farfield.TotalRadiated(k0,zlist,nstack,dstack)
        ldrav=(data[0]*2.0+data[1])/3.0
        
        ldavav[m,mm]=np.mean(ldav)
        ldravav[m,mm]=np.mean(ldrav)
        


fig,ax=plt.subplots(1,2,figsize=(10,5))
ax[0].plot(nlist,ldavav[0,:],'k:',nlist,ldavav[1,:],'k-.', nlist,ldravav[0,:],'k',nlist,ldravav[1,:],'k--')
ax[0].set_xlabel('n of substrate')
ax[0].set_ylabel('mean LDOS')
ax[0].set_xlim([1.0,5.0])
ax[0].set_ylim([0.0,4.1])
ax[0].legend(['d=lam/4 all modes','d=3lam all modes','d=lam/4; rad. modes','d=3lam; rad modes'])
ax[1].plot(nlist,ldavav[0,:]/nlist,'k:',nlist,ldavav[1,:]/(nlist**2+1),'k-.', nlist,ldravav[0,:]/nlist,'k',nlist,ldravav[1,:]/(nlist**2+1),'k--')
ax[1].set_xlabel('n of substrate')
 
ax[1].set_xlim([1.0,5.0])
ax[1].set_ylim([0.0,2.1])
ax[1].legend(['d=lam/4 all modes','d=3lam all modes','d=lam/4; rad. modes','d=3lam; rad modes'])
plt.show()



#Urbach & Rikken figure 10lda
lam=0.611  #wvlen in micron
k0=2*np.pi/lam
nZnSe=2.61
nPS=1.585
nglass=1.585
nair=1.0
nsuperstrlist=[nair,nglass]

 
dPS=0.07
dZnSelist=np.linspace(0,0.550,100) #number of plotpoints for ZnSe thickness


numd=len(dZnSelist)
numn=len(nsuperstrlist)



ldavav=np.zeros([numd,numn])
ldravav=np.zeros([numd,numn])

for m in range(numn):
    for mm in range(numd):
        nstack=np.array([nglass,nPS,nZnSe,nsuperstrlist[m]])
        dstack=np.array([dPS,dZnSelist[mm]])
        zlist=np.linspace(0,dPS,25)
        
        
        rhoE_par,rhoE_perp,rhoM_par,rhoM_perp,rhoC=ImGLDOS.LDOS(k0,zlist,nstack,dstack)
        ldav=(rhoE_par*2.0+rhoE_perp)/3.0
            
            
        data=Farfield.TotalRadiated(k0,zlist,nstack,dstack)
        ldrav=(data[0]*2.0+data[1])/3.0
        
        
        ldavav[mm,m]=np.mean(ldav)
        ldravav[mm,m]=np.mean(ldrav)

plt.figure(figsize=(6,6))
plt.plot(dZnSelist*1000,ldavav[:,0],'k',dZnSelist*1000,ldavav[:,1],'k:',dZnSelist*1000,ldravav[:,0],'k--',dZnSelist*1000,ldravav[:,1],'k-.')
plt.xlabel('Thickness (nm) of ZnSe')
plt.ylabel('Averaged LDOS normaliz to vac')
plt.ylim([0,2.5])
plt.legend(['Symmetric total','Asymm total','Symmetric radiative','Asymmetric radiative'])

plt.show()