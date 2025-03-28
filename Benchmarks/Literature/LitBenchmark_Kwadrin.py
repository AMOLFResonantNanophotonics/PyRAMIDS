#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 18:01:14 2023

Reproducing
Kwadrin, PHYSICAL REVIEW B 87, 125123 (2013)
"""

print('###  Literature Benchmark: Kwadrin et. al. Phys. Rev. B. 87, 125123 (2013)  ###')

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

lam=1.5  
k0=2*np.pi/lam

epsAg= -121.53 + 3.10j 
epsSi= 12.11
nstack1=[np.sqrt(epsAg),1.0]
nstack2=[np.sqrt(epsSi),1.0]
dstack=[]
zlist= np.linspace(0.001,1.5,500)

rhoE_par_1,rhoE_perp_1,rhoM_par_1,rhoM_perp_1,rhoC=ImGLDOS.LDOS(k0,zlist,nstack1,dstack)
rhoE_par_2,rhoE_perp_2,rhoM_par_2,rhoM_perp_2,rhoC=ImGLDOS.LDOS(k0,zlist,nstack2,dstack)


plt.plot(zlist,rhoE_par_1,'b--',zlist,rhoM_perp_1,'r--',zlist,rhoE_par_2,'b',zlist,rhoM_perp_2,'r') 
plt.ylim([0,2])
plt.xlim([0,1.5])
plt.xlabel('d (um)')
plt.ylabel('LDOS/LDOS_vac')
plt.legend(['Ag px','Ag mz','Si px','Si mz'])
plt.show()

plt.plot(zlist,rhoE_par_1,'b--',zlist,rhoM_perp_1,'r--',zlist,rhoE_par_2,'b',zlist,rhoM_perp_2,'r') 
plt.ylim([0,16.5])
plt.xlim([0,0.3])
plt.xlabel('d (um)')
plt.ylabel('LDOS/LDOS_vac')
plt.legend(['Ag px','Ag mz','Si px','Si mz'])
plt.show()

