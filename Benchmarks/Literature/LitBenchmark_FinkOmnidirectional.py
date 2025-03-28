
#%%
'''Reproducing Fink & Thomas et. al. Science 282, 1679-1998 (1998) Figure 2'''


print('###  Literature Benchmark: Fink et. al. Science 282, 1679-1998 (1998) Figure 2')
#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#%%

import numpy as np
import matplotlib.pyplot as plt
from Library.Use import Use_Planewaves as planewave  


#%%

# n1, n2 = 1, 2.2
# d1, d2 = 2.2, 1.7

n1, n2 = 4.6, 1.6
d1, d2 = 0.8, 1.6
a = d1 + d2

n = [n1,n2]
d = [d1, d2]

N = 10
nstack = [1] + list(np.tile(n, N)) + [1]
dstack = list(np.tile(d, N))

freqlist = np.linspace(0.001, 0.6, 600) *(2*np.pi/a)

kparlist = np.linspace(0, 0.8, 600)  *(2*np.pi/a)

kpar, FREQ = np.meshgrid(kparlist, freqlist, indexing='ij')

rs = np.zeros_like(kpar)
rp = np.zeros_like(kpar)


for i, kp in enumerate(kparlist):
        for j, freq in enumerate(freqlist):
            
            k0 = freq * (2*np.pi/a)
            kx =  kp  * (2*np.pi/a)
            
            Rs = planewave.IntensityRT(k0, kx, nstack, dstack)[0][1][0] 
            Rp = planewave.IntensityRT(k0, kx, nstack, dstack)[1][1][0] 
            
            rs[j, i] = Rs  
            rp[j, i] = Rp

#%%
kpar_full = np.concatenate((-kparlist[::-1], kparlist)) 
reflectance_full = np.hstack((np.flip(rp, axis=1), rs)) 

KX_FULL, FREQ_FULL = np.meshgrid(kpar_full, freqlist, indexing='ij')

light_line = freqlist


fig, ax = plt.subplots(figsize=(9, 6), dpi=400)

pcm = ax.pcolormesh(KX_FULL, FREQ_FULL, reflectance_full.T, cmap="gray_r", shading='auto', vmin=0, vmax=1)

ax.plot(light_line, freqlist, 'k-', label="Light Line")
ax.plot(-light_line, freqlist, 'k-', label="Light Line")


fig.colorbar(pcm, label="Reflectance")
plt.ylim([0,0.5])
plt.xlim([-1.5,1.5])

ax.set_xlabel(r"Parallel Wave Vector ($k_x 2\pi/a$)", fontsize=14)
ax.set_ylabel(r"Frequency ($\omega a / 2\pi c$)", fontsize=14)
ax.set_title("Omnidirectional Reflector - Reflection Spectrum", fontsize=14)

plt.show()


