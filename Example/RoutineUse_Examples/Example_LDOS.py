''' Plot example - insight into LDOS contributions from plotting LDOS integrand
    Example for a dielectric waveguide. Method proposed by  Amos & Barnes PRB 55 7249 (1997) Figure 5 
  
    Example: waveguide of n=3.5 on glass, with thickness d=lam/10
    
    Returns plots for LDOS_E (par, perp), LDOS_M (par,perp) and LDOS_C (relates to chirality)

    What you are suppose to find:
        - Gentle LDOS(k||) oscillations in both half spaces below the light lines, i.e., below kpar/k0 is 1.5 and 1.0
        - Distinct guided mode contributions between kpar/k0=1.5 and 3.5
        - These appear as vertical stripes, and show the mode profile    
 '''

#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%%

import numpy as np
import matplotlib.pyplot as plt

from Library.Use import Use_LDOS as ImGLDOS
from Library.Use import Use_Radiationpattern as Farfield

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


lam=1.0
k0=2.*np.pi/lam

nstack=[1.5,3.5,1.0]
dstack=[lam/10.]

kparlist=np.linspace(0,4.0,1500)
zlist=np.linspace(-0.6,0.7,1200)

 
#%%

p=ImGLDOS.LDOSatanyPandM([1.0,0,0], [0,0,0], k0, zlist, nstack, dstack)
m=ImGLDOS.LDOSatanyPandM([0.0,0,0], [0,1,0], k0, zlist, nstack, dstack)
spinp=ImGLDOS.LDOSatanyPandM([1.0,1j,0], [0,0,0], k0, zlist, nstack, dstack)
kerker=ImGLDOS.LDOSatanyPandM([1.0,0,0], [0,1,0], k0, zlist, nstack, dstack)
smin=ImGLDOS.LDOSatanyPandM([1.0,0,0], [0,1.j,0], k0, zlist, nstack, dstack)
splus=ImGLDOS.LDOSatanyPandM([1.0,0,0], [0,-1.j,0], k0, zlist, nstack, dstack)

plt.figure(figsize=(4,5))
plt.plot(zlist,p,'r',zlist,m,'b',zlist[::12],spinp[::12],'r.',zlist,kerker,'k',zlist,smin,'g',zlist,splus,'g-.')
plt.legend(['LDOS for px','my','Spinning p','Kerker','(px,my)=(1,i)','(1,-i)'])
plt.ylabel(r'z / $\lambda$')
plt.ylabel('LDOS over vacuum LDOS')
plt.xlim([0.07,0.7])
plt.show()


#%%

ld=ImGLDOS.LDOS(k0,zlist,nstack,dstack)
data=Farfield.TotalRadiated(k0,zlist,nstack,dstack)
   
plt.figure(figsize=(7,7),dpi = 300)

plt.plot(zlist, ld[0] +ld[1], 'g--', label='Total LDOS from E')      # Red dotted line

plt.plot(zlist, ld[1], 'r', label=r'$\perp$ Total LDOS from E')   # Red solid line
plt.plot(zlist, data[1], 'r:', label=r'$\perp$ Radiated LDOS from E')      # Red dotted line
plt.plot(zlist, ld[1] - data[1], 'r--', label=r'$\perp$ Guided LDOS from E')      # Red dotted line

plt.plot(zlist, ld[0], 'b', label=r'$\parallel$ Total LDOS from E')   # Blue solid line
plt.plot(zlist, data[0], 'b:', label=r'$\parallel$ Radiated LDOS from E')      # Blue dotted line
plt.plot(zlist, ld[0] - data[0], 'b--', label=r'$\parallel$ Guided LDOS from E')      # Red dotted line


# Plot black step-like structure
plt.plot([-0.6, 0, 0, dstack[0], dstack[0], 0.7], 
         [nstack[0], nstack[0], nstack[1], nstack[1], nstack[2], nstack[2]], 
         'k', label="Refractive Index Profile")  
plt.xlabel(r'z in units of vac.  $\lambda$')
plt.ylabel('LDOS/vac LDOS')
plt.xlim([-0.6,0.7])
plt.legend()

plt.show()


print('Guided mode appear as poles')

plt.figure(figsize=(7,7),dpi = 300)

plt.plot(zlist, ld[2] +ld[3], 'g--', label='Total LDOS from H')      # Red dotted line

plt.plot(zlist, ld[2], 'r', label=r'$\perp$ Total LDOS from H')   # Red solid line
plt.plot(zlist, data[2], 'r:', label=r'$\perp$ Radiated LDOS from H')      # Red dotted line
plt.plot(zlist, ld[3], 'b', label=r'$\parallel$ Total LDOS from H')   # Blue solid line
plt.plot(zlist, data[3], 'b:', label=r'$\parallel$ Radiated LDOS from H')      # Blue dotted line

# Plot black step-like structure
plt.plot([-0.6, 0, 0, dstack[0], dstack[0], 0.7], 
         [nstack[0], nstack[0], nstack[1], nstack[1], nstack[2], nstack[2]], 
         'k', label="Refractive Index Profile")  
plt.xlabel(r'z in units of vac.  $\lambda$')
plt.ylabel('LDOS/vac LDOS')
plt.xlim([-0.6,0.7])
plt.legend()

plt.show()


plt.figure(figsize=(7,7),dpi = 300)
plt.plot(zlist, ld[1], 'r', label=r'$\perp$ Total LDOS from E')   # Red solid line
plt.plot(zlist, data[1], 'r:', label=r'$\perp$ Radiated LDOS from E')      # Red dotted line
plt.plot(zlist, ld[4], 'b', label=r'Total LDOS cross')   # Red solid line
plt.plot(zlist, data[4], 'b:', label='Radiated LDOS cross')      # Red dotted line

# Plot black step-like structure
plt.plot([-0.6, 0, 0, dstack[0], dstack[0], 0.7], 
         [nstack[0], nstack[0], nstack[1], nstack[1], nstack[2], nstack[2]], 
         'k', label="Refractive Index Profile")  
plt.xlabel(r'z in units of vac.  $\lambda$')
plt.ylabel('LDOS/vac LDOS')
plt.xlim([-0.6,0.7])
plt.legend()

plt.show()



plt.figure(figsize=(7,7),dpi = 300)

plt.plot(zlist, ld[2] +ld[3], 'g--', label='Total LDOS from E')      # Red dotted line

plt.plot(zlist, ld[0], 'r', label=r'$\perp$ Total LDOS from E')   # Red solid line
plt.plot(zlist, ld[1], 'r:', label=r'$\perp$ Radiated LDOS from E')      # Red dotted line
plt.plot(zlist, ld[2], 'b', label=r'Total LDOS cross')   # Red solid line

# Plot black step-like structure
plt.plot([-0.6, 0, 0, dstack[0], dstack[0], 0.7], 
         [nstack[0], nstack[0], nstack[1], nstack[1], nstack[2], nstack[2]], 
         'k', label="Refractive Index Profile")  
plt.xlabel(r'z in units of vac.  $\lambda$')
plt.ylabel('LDOS/vac LDOS')
plt.xlim([-0.6,0.7])
plt.legend()

plt.show()

#%%
'''LDOS integrand trace'''

out=ImGLDOS.LDOSintegrandplottrace(k0,kparlist,zlist,nstack,dstack)

#%%
pcm = plt.pcolor(kparlist,zlist,np.log10(np.abs(out[0,:,:])+1E-23))

cbar = plt.colorbar(pcm)
cbar.set_label(r'$\log_{10}$ LDOS_E integrand', fontsize=12)  # Set colorbar label

plt.plot(kparlist,0*kparlist,'w')
plt.plot(kparlist,0*kparlist,'w')
plt.plot(kparlist,0*kparlist+dstack[0],'w')
plt.plot([nstack[0],nstack[0]],[-1.0, 0.0],'w--')
plt.plot([nstack[1],nstack[1]],[0,dstack[0]],'w--')
plt.plot([nstack[2],nstack[2]],[dstack[0], 1.0],'w--')
plt.clim([-2, 2])
plt.xlabel(r'$k_{\parallel}$/$k_0$')
plt.ylabel(r'z / $\lambda$')
plt.ylim([-0.5,0.7])
plt.title(r'log10 LDOS E $\parallel$ integrand')
plt.show()


pcm = plt.pcolor(kparlist,zlist,np.log10(np.abs(out[1,:,:])+1E-23))

cbar = plt.colorbar(pcm)
cbar.set_label(r'$\log_{10}$ LDOS_E integrand', fontsize=12)  # Set colorbar label

plt.plot(kparlist,0*kparlist,'w')
plt.plot(kparlist,0*kparlist,'w')
plt.plot(kparlist,0*kparlist+dstack[0],'w')
plt.plot([nstack[0],nstack[0]],[-1.0, 0.0],'w--')
plt.plot([nstack[1],nstack[1]],[0,dstack[0]],'w--')
plt.plot([nstack[2],nstack[2]],[dstack[0], 1.0],'w--')
plt.clim([-2, 2])
plt.xlabel(r'$k_{\parallel}$/$k_0$')
plt.ylabel(r'z / $\lambda$')
plt.title(r'log10 LDOS E $\perp$ integrand')
plt.ylim([-0.5,0.7])
plt.show()


pcm = plt.pcolor(kparlist,zlist,np.log10(np.abs(out[2,:,:])+1E-23))

cbar = plt.colorbar(pcm)
cbar.set_label(r'$\log_{10}$ LDOS_H integrand', fontsize=12)  # Set colorbar label

plt.plot(kparlist,0*kparlist,'w')
plt.plot(kparlist,0*kparlist,'w')
plt.plot(kparlist,0*kparlist+dstack[0],'w')
plt.plot([nstack[0],nstack[0]],[-1.0, 0.0],'w--')
plt.plot([nstack[1],nstack[1]],[0,dstack[0]],'w--')
plt.plot([nstack[2],nstack[2]],[dstack[0], 1.0],'w--')
plt.clim([-2, 2])
plt.xlabel(r'$k_{\parallel}$/$k_0$')
plt.ylabel(r'z / $\lambda$')
plt.title('log10 LDOS_H integrand (parallel)')
plt.ylim([-0.5,0.7])
plt.show()


pcm = plt.pcolor(kparlist,zlist,np.log10(np.abs(out[3,:,:])+1E-23))

cbar = plt.colorbar(pcm)
cbar.set_label(r'$\log_{10}$ LDOS_H integrand', fontsize=12)  # Set colorbar label

plt.plot(kparlist,0*kparlist,'w')
plt.plot(kparlist,0*kparlist,'w')
plt.plot(kparlist,0*kparlist+dstack[0],'w')
plt.plot([nstack[0],nstack[0]],[-1.0, 0.0],'w--')
plt.plot([nstack[1],nstack[1]],[0,dstack[0]],'w--')
plt.plot([nstack[2],nstack[2]],[dstack[0], 1.0],'w--')
plt.ylim([-0.5,0.7])
plt.clim([-2, 2])
plt.xlabel(r'$k_{\parallel}$/$k_0$')
plt.ylabel(r'z / $\lambda$')
plt.title('log10 LDOS_H integrand (perpendicular)')
plt.show()


pcm = plt.pcolor(kparlist,zlist,np.sign(out[4,:,:])*np.log10(np.abs(out[4,:,:]*1E-4)+1E-23))

cbar = plt.colorbar(pcm)
cbar.set_label(r'$\log_{10}$ LDOS_cross', fontsize=12)  # Set colorbar label

plt.plot(kparlist,0*kparlist,'w')
plt.plot(kparlist,0*kparlist,'w')
plt.plot(kparlist,0*kparlist+dstack[0],'w')
plt.plot([nstack[0],nstack[0]],[-1.0, 0.0],'w--')
plt.plot([nstack[1],nstack[1]],[0,dstack[0]],'w--')
plt.plot([nstack[2],nstack[2]],[dstack[0], 1.0],'w--')
plt.ylim([-0.5,0.7])
plt.clim([-6, 7])
plt.xlabel(r'$k_{\parallel}$/$k_0$')
plt.ylabel(r'z / $\lambda$')
plt.title('log10 LDOS_C integrand')
plt.show()

#%%

rhoCplus=np.squeeze(out[0,:,:]+out[2,:,:]+2.0*out[4,:,:])
rhoCmin=np.squeeze(out[0,:,:]+out[2,:,:]-2.0*out[4,:,:])

rhomean=0.5*((2.*out[0,:,:]+out[1,:,:])/3+(2.*out[2,:,:]+out[3,:,:])/3.) 

#%%
def signedclippedlog(data,lowerclip):
       log10dat=np.log10(np.abs(data)+1.E-23)-lowerclip
       log10dat[log10dat<0.]=0.0
       return np.multiply(log10dat,np.sign(np.real(data)) )

pcm = plt.pcolor(kparlist,zlist,signedclippedlog(rhomean,-2))

cbar = plt.colorbar(pcm)
cbar.set_label(r'$\log_{10}$ LDOS_mean', fontsize=12)  # Set colorbar label

plt.plot(kparlist,0*kparlist,'w')
plt.plot(kparlist,0*kparlist,'w')
plt.plot(kparlist,0*kparlist+dstack[0],'w')
plt.plot([nstack[0],nstack[0]],[-1.0, 0.0],'w--')
plt.plot([nstack[1],nstack[1]],[0,dstack[0]],'w--')
plt.plot([nstack[2],nstack[2]],[dstack[0], 1.0],'w--')
plt.ylim([-0.5,0.7])
plt.clim([-2, 4])
plt.xlabel(r'$k_{\parallel}$/$k_0$')
plt.ylabel(r'z / $\lambda$')
plt.title('log10 LDOS, mean (p and m, all orientations)')
plt.show()
