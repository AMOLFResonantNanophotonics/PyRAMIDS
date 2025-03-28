#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computes Green's function for all components'
"""
#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%%

from Library.Use import Use_Green as Gs
import numpy as np
from matplotlib import pyplot as plt

plt.close('all')

k0=2*np.pi/0.6

 
ng=1.5
ns=2.0
na=1.0
nstack=[ng,ns,na]
dstack=[0.4]
 

poslist=np.linspace(0.1,8.1,81)
  

rdetect=np.array([poslist/6,poslist/2,poslist/31])
rsource=np.array([0,0,0.1])           


Gscat,Ghom=     Gs.GreenS(k0,nstack,dstack,rdetect,rsource) 
Gfull=Gscat+Ghom
 

fignum=4
offsetx=np.array([0,1,0,1])*3
offsety=np.array([0,1,1,0])*3
field=['EE','HH','EH','HE']
lab=['x','y','z']

for q in range(fignum):
    fig,axs=plt.subplots(3,3, dpi=300)
    for i in range(3):
        for j in range(3):
 
            axs[i,j].plot( 
                          poslist,np.real(Gfull[i+offsetx[q],j+offsety[q],:]))
      
            
            if i+j==0:
                axs[i,j].legend(['In-slab Green function'])
        
            else:
                axs[i,j].legend([lab[i]+lab[j]])
    fig.suptitle('$G^{'+field[q]+'}$')
    print(fig.number)
    plt.show()