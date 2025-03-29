 
"""
Reproduce Electric and magnetic LDOS in Li, Karaveli et al. PRL 121, 227403 (2018), Fig 1
Note that to reproduce the other figures there is not quite enough info in the paper.
Therefore this routine provides a "replica with caveats" for Figure 3a as example
 
"""

print('###  Literature Benchmark: Li et. al. Phys. Rev. Lett. 121 227403 (2018)  ###')

#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%%

from Library.Use import Use_LDOS as ImGLDOS

import matplotlib.pyplot as plt
import numpy as np
 
 
## Figure 1 Karaveli
plt.figure()
lam=1.0
k0=2.*np.pi/lam
epshost=2.25
epsmir=-97+11.5j
nstack=[np.sqrt(epsmir),np.sqrt(epshost)]
dstack=[]

zlist=np.linspace(1.E-3,0.5,150)
rhoE_par, rhoE_perp,rhoM_par,rhoM_perp,rhoC=ImGLDOS.LDOS(k0,zlist,nstack,dstack)    

ldE_iso = (rhoE_par*2.0+rhoE_perp)/3.
ldM_iso = (rhoM_par*2.0+rhoM_perp)/3.

plt.plot(zlist,ldE_iso/nstack[1],'r',zlist,ldM_iso/nstack[1],'b',zlist,(ldE_iso+ldM_iso)/nstack[1],'k')
plt.ylim([0.5, 3])
plt.xlabel('d/lambda')
plt.ylabel('LDOS over that in host')
plt.legend(['Electric (isotropic)','Magnetic','sum']) 
plt.xlim([0,0.5])
plt.title('Li, Karaveli et al. PRL 121, 227403 (2018), Fig 1a')
plt.show()



plt.figure()
plt.plot(zlist,1./(ldE_iso/nstack[1]),'r',
         zlist,1./(ldM_iso/nstack[1]),'b',
         zlist,1./((ldE_iso+ldM_iso)/(2.0*nstack[1])),'k')
plt.plot(zlist,1/(0.8+0.2*ldE_iso/nstack[1]),'r:',
         zlist,1/(0.8+0.2*ldM_iso/nstack[1]),'b:')
         
plt.ylim([0.5, 3])
plt.xlabel('d/lambda')
plt.ylabel('Apparent lifetime')
plt.legend(['ED','MD','ED+MD av','ED (QE=0.2)','MD (QE=0.2']) 
plt.xlim([0.0,0.5])
plt.ylim([0.6,1.4])
plt.title('Li, Karaveli et al. PRL 121, 227403 (2018), Fig 1b')

plt.show()




plt.figure()

## attempt to reproduce Figure 3a
nMgO=1.66 #stated in supplement. Refr index of MgO
nQrtz=1.447
nAu=np.sqrt(epsmir)
lam=1.4 # approx emission wvln. Precise bandwidth not listed in paper
k0=2.*np.pi/lam
dAu=0.200
dspacer=np.linspace(1.E-3,0.5,100)
nstack=[nQrtz,nMgO,nMgO,nMgO,nAu,1.0] #guessing that Fig1 and Fig3 match eps_Au

#variable allocation to hold result
ldisoE=np.zeros(np.shape(dspacer))
ldisoM=np.zeros(np.shape(dspacer))

for m in range(len(dspacer)): #scan over spacer height
    dspac=dspacer[m]
    dstack=[0.018,0.018,dspac,dAu] #emitters are in 2nd layer of 18 nm thick
    zlist=0.018+0.018*np.linspace(0.001,0.99,5) #emitter heights to average over
    
    rhoE_par, rhoE_perp,rhoM_par,rhoM_perp,rhoC=ImGLDOS.LDOS(k0,zlist,nstack,dstack)    

    ldisoE[m]=np.mean((2.0*rhoE_par+rhoE_perp)/3.)
    ldisoM[m]=np.mean((2.0*rhoM_par+rhoM_perp)/3.)

plt.plot(1000*dspacer,nMgO/ldisoE,'r--',1000*dspacer,nMgO/ldisoM,'b--',1000*dspacer,nMgO*2.0/(ldisoE+ldisoM),'k--')
plt.xlim([0,500])
plt.ylim([0.42,1.68])
plt.xlabel('distance (nm)')
plt.ylabel('Inverse of LDOS rel. to LDOS in MgO')
plt.legend(['ED','MD','mean'])
plt.title('Approximation to Fig. 3 in Li, Karaveli et al. PRL 121, 227403 (2018),')
plt.show()