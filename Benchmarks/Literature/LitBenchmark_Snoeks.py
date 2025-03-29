
"""
# Reproduces Snoeks, Lagendijk and Polman, PRL 74, 2459 (1995)
"""

print('### Literature Benchmark: Snoeks et. al. Phys. Rev. Lett. 74, 2459, 1995 ###')
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
 
 
##### Figure 4

lam=1.5317      
k0=2*np.pi/lam
nstack1=[1.0,1.5]
nstack2=[3.0,1.5]
dstack=[]
zlist=lam/(2.*np.pi)*np.linspace(-4.3,4.3,100)

outpar1fromP,outperp1fromP,outparfromM,outperpfromM,outcross=ImGLDOS.LDOS(k0,zlist,nstack1,dstack)
outpar2fromP,outperp2fromP,outparfromM,outperpfromM,outcross=ImGLDOS.LDOS(k0,zlist,nstack2,dstack)

plt.figure()
plt.plot(zlist/(lam/(2.*np.pi)),((outpar1fromP*2.+outperp1fromP)/3.)/1.5,'k--',zlist/(lam/(2.*np.pi)),((outpar2fromP*2.+outperp2fromP)/3.)/1.5,'k')
plt.plot([-5,-1],[1./1.5,1./1.5],'k:',[-5,5],[1.,1.],'k:',[-5,-1],[3./1.5,3./1.5],'k:')
plt.plot([0,0],[0.5,2.5],'k--')
plt.xlabel('Position z (lam/2pi)')
plt.ylabel('LDOS (isotropic) normalized to n=1.5')
plt.xlim([-4,4])
plt.ylim([0.5,2.5])
plt.title('Snoeks, Lagendijk & Polman, PRL 74 2459 (1995)')
plt.show()