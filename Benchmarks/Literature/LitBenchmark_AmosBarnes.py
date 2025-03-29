
"""
# Example script that reproduces the calculations in  AMOS & Barnes PRB 55, 7249 1997   #
"""

print('###  Literature Benchmark: AMOS & Barnes: PRB 55 7249 1997  ###' )

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
 
 
##### Figure 2

lam=0.614  
k0=2*np.pi/lam

nstack=[1.0,np.sqrt(2.49),np.sqrt(2.49),np.sqrt(-16.0+0.4j),1.456] #air, emitter layer, spacer, silver, glass
dspacer = np.linspace(0,0.5,100)

dAg=0.2 #200 nm silver thickbess
demit=0.0026

inverserate= np.empty(len(dspacer))

QE=0.7

for i in range(len(dspacer)):
    dstack = [demit,dspacer[i],dAg] 
    zlist=demit/2.0 #just one point, in the middle of the emitter layer
    rhoE_par,rhoE_perp,rhoM_par,rhoM_perp,rhoC=ImGLDOS.LDOS(k0,zlist,nstack,dstack)    
  
    outTotal_iso = (rhoE_par*2.0+rhoE_perp)/3.
    rate = QE*np.mean(outTotal_iso) + (1.0-QE)
    inverserate[i] = 1./rate
# Total LDOS

plt.figure(dpi=400)

plt.plot(1000*dspacer,inverserate,'-')    

plt.xlabel("Distance (in nm)")
plt.ylabel("1/(LDOS x QE + (1-QE))")
plt.title("AMOS and Barnes PRB'97 - 200nm mirror")
plt.legend(['Assuming 70% QE. Note: for [ms] you need Eu3+ lifetime'])
plt.xlim([0,1000*np.max(dspacer)])
plt.ylim([0,1.2])
plt.show()

################## Figure 3

lam=0.614  
k0=2*np.pi/lam

#nstack=[1.465,0.049+4.081j,1.58,1.58,1.0] #glass, silver, spacer, emitter layer, air
nstack=[1.0,np.sqrt(2.49),np.sqrt(2.49),np.sqrt(-9.1+16.0j),1.456] #air, emitter layer, spacer, silver, glass
dspacer = np.linspace(0,0.5,150)

dAg=0.0133 #200 nm silver thickbess
demit=0.0026

inverserate= np.empty(len(dspacer))

QE=0.7

for i in range(len(dspacer)):
    dstack = [demit,dspacer[i],dAg] 
    zlist=demit/2.0 #just one point, in the middle of the emitter layer
    rhoE_par,rhoE_perp,rhoM_par,rhoM_perp,rhoC=ImGLDOS.LDOS(k0,zlist,nstack,dstack)    
  
    outTotal_iso = (rhoE_par*2.0+rhoE_perp)/3.
    rate = QE*np.mean(outTotal_iso) + (1.0-QE)
    inverserate[i] = 1./rate
# Total LDOS

plt.figure(dpi=400)

plt.plot(1000*dspacer,inverserate,'-')    

plt.xlabel("Distance (in nm)")
plt.ylabel("1/(LDOS x QE + (1-QE))")
plt.title("AMOS and Barnes PRB'97 - 14 nm mirror/Fig 3")
plt.legend(['Assuming 70% QE. Note: for [ms] you need Eu3+ lifetime'])
plt.xlim([0,1000*np.max(dspacer)])
plt.ylim([0,1.2])
plt.show()



################## Figure 4

plt.figure(dpi=400)

lam=0.614  
k0=2*np.pi/lam

dspacer = np.linspace(0,0.08,50)
dAglist=[0.2,0.0667, 0.0461,0.0384,0.0267,0.0133] 
epsAglist=[-16.0+0.4j,-15.5+0.4j,-16.0+0.4j,-14.2+0.7j,-15.0+0.7j,-9.1+16.j]

demit=0.0026

QE=0.7
numgeom=len(dAglist)
inverserate= np.empty([numgeom,len(dspacer)])
for p in range(numgeom):
    nstack=[1.0,np.sqrt(2.49),np.sqrt(2.49),np.sqrt(epsAglist[p]),1.456] #air, emitter layer, spacer, silver, glass
    for i in range(len(dspacer)):
        dstack = [demit,dspacer[i],dAglist[p]] 
        zlist=demit/2.0 #just one point, in the middle of the emitter layer
        rhoE_par,rhoE_perp,rhoM_par,rhoM_perp,rhoC=ImGLDOS.LDOS(k0,zlist,nstack,dstack)    
  
        outTotal_iso = (rhoE_par*2.0+rhoE_perp)/3.
        rate = QE*np.mean(outTotal_iso) + (1.0-QE)
        inverserate[p,i] = 1./rate
    # Total LDOS
plt.plot(1000*dspacer,np.transpose(inverserate))
plt.xlabel("Distance (in nm)")
plt.ylabel("1/(LDOS x QE + (1-QE))")
plt.title("AMOS and Barnes PRB'97  Fig 4. Assuming 70% QE")
plt.legend(['200 nm Ag','66.7nm','46.1 nm','38.4 nm','26.7nm','13.3 nm'])


################## Figure 5


lam=0.614  
k0=2*np.pi/lam

nstack=[1.0,np.sqrt(2.49),np.sqrt(2.49),np.sqrt(-16.0+0.4j),1.456] #air, emitter layer, spacer, silver, glass

dAg=0.2 #200 nm silver thickbess
demit=0.0026

dspacer=[0.2,0.08,0.02]

kparlist=np.linspace(0.001,4.0,1500)


f,ax = plt.subplots(3,1,figsize=(5,10),dpi=400)
leg=[]
for i in range(len(dspacer)):
    dstack = [demit,dspacer[i],dAg] 
    zlist=demit/2.0 #just one point, in the middle of the emitter layer
    
    out=ImGLDOS.LDOSintegrandplottrace(k0,kparlist,zlist,nstack,dstack, guidevisible=1)
    ax[i].plot(kparlist,np.log10((1.E-20+2.*out[0,0,:]+out[1,0,:])/3.0))#little offset avoids log10 error message at 0
    ax[i].set_xlim([0,3])
    ax[i].set_ylim([-3.5,2.5])
    ax[i].set_xlabel('kpar / k0')
    ax[i].set_ylabel('Decay rate contrib.')
    ax[i].legend(['Distance (nm):'+str(1000*dspacer[i])])
    
ax[0].set_title('PRB 55, 7249 (1997) Fig 5)')
plt.show()
