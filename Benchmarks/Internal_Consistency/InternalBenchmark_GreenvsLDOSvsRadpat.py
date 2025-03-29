#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
- Consistency check of our LDOS implementation for standard dipole orientations in non-absorbing, no guided mode
system from all the routines (Integration of radiation pattern, ImG LDOS, Green's function.


"""

#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%%
from Library.Use import Use_Radiationpattern as Radpat
from Library.Use import Use_Green as Gs
from Library.Use import Use_LDOS as ImG
from Library.Util import Util_argumentrewrapper as aux



import numpy as np

from matplotlib import pyplot as plt


lam=2.0
k0=2.*np.pi/lam
nstack=[3.5,1.5,1.0]
dstack=[0.2]
zlist=np.linspace(-0.5,0.75,250)

ldallRadpat=Radpat.TotalRadiated(k0,zlist,nstack,dstack)

ldEpar,ldEperp,ldHpar,ldHperp,ldC=ImG.LDOS(k0,zlist,nstack,dstack)  

Gscatt,Gfree=Gs.GreenS(k0,nstack,dstack,np.array([0*zlist,0*zlist,zlist]),np.array([1E-5+0*zlist,0*zlist,zlist]))

Gscatt, Gfree = Gscatt.T, Gfree.T
nversusz=aux.nvalueatzposition(zlist,nstack,dstack)

plt.close('all')
fig,ax=plt.subplots(ncols=3,figsize=(26,8))
ax[0].plot(zlist,ldallRadpat[0,:],'r--', zlist,ldEpar,'r-', zlist,nversusz*(1+np.imag(Gscatt[:,0,0])/(2/3*k0*k0*k0)),'ro-')
ax[0].plot(zlist,ldallRadpat[1,:],'b--', zlist,ldEperp,'b-',zlist,nversusz*(1+np.imag(Gscatt[:,2,2])/(2/3*k0*k0*k0)),'bo-')
ax[0].legend(['Parallelp from Radpat','Dito  from ImG', 'Dito from full Gscat', 'Perpendicular p from Radpat', 'Dito from Im G', 'Dito from full Gscat'])
ax[0].set_title('Electric')
ax[0].set_xlabel('z ')
ax[0].set_ylabel('LDOS over free space') 

 
ax[1].plot(zlist,ldallRadpat[2,:],'r--', zlist,ldHpar,'r-',zlist,nversusz*(1+np.imag(Gscatt[:,3,3])/(2/3*k0*k0*k0)),'ro-')
ax[1].plot(zlist,ldallRadpat[3,:],'b--', zlist,ldHperp,'b-',zlist,nversusz*(1+np.imag(Gscatt[:,5,5])/(2/3*k0*k0*k0)),'bo-')
ax[1].legend(['Parallel m from Radpat','Dito  from ImG', 'Dito from full Gscat', 'Perpendicular m from Radpat', 'Dito from Im G', 'Dito from full Gscat'])
ax[1].set_title('Magnetic')
ax[1].set_xlabel('z ')
ax[1].set_ylabel('LDOS over free space') 


ax[2].plot(zlist,ldallRadpat[4,:],'r--', zlist,ldC,'r-',zlist,nversusz*(np.real(Gscatt[:,4,0])/(2/3*k0*k0*k0)),'ro-')

ax[2].legend(['Pseudochiral term from Radpat', 'Dito from ImG','Dito from full scatt G'])
ax[2].set_title('magnetoelectric')
ax[2].set_xlabel('z ')
ax[2].set_ylabel('LDOS over free space') 
plt.show()