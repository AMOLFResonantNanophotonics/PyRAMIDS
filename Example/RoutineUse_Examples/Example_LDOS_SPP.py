''' Plot example SPP - insight into LDOS contributions from plotting LDOS integrand
    Example for a simple Ag Drude Glass interface.
    
    Source in glass at various heights away from metal.

 '''

#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%%

import numpy as np
import matplotlib.pyplot as plt

from Library.Use import Use_LDOS as ImGLDOS


plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300



def drude(k,kp,g):
    eps=1-kp*kp/(k*(k+1.j*g))
    return np.sqrt(eps)   


def spp(k0,nstack):
    epsm=nstack[0]**2
    epsd=nstack[1]**2
    return np.real(k0*np.sqrt((epsm*epsd)/(epsm+epsd)))
    

lammax=2.;
lammin=0.3;

kmax=2.*np.pi/lammin
kmin=2*np.pi/lammax


nk=250
k0list=np.linspace(kmin,kmax,nk)

kp=2*np.pi/0.2
g=0.02*kp


kparlistSI=np.linspace(0.001,4.5*kmax,300)

k0plot,kparplot=np.meshgrid(k0list,kparlistSI)


ImGEx5=0.0*k0plot
ImGEz5=0.0*k0plot

ImGEx20=0.0*k0plot
ImGEz20=0.0*k0plot


ImGEx50=0.0*k0plot
ImGEz50=0.0*k0plot


ImGEx100=0.0*k0plot
ImGEz100=0.0*k0plot

kSPP=0*k0list

for n in range(nk):
    k0=k0list[n]
    nstack=[drude(k0,kp,g),1.5]
    dstack=[]



    zlist=np.array([0.005])
    
    out=ImGLDOS.LDOSintegrandplottrace(k0,kparlistSI/k0,zlist,nstack,dstack)

    ImGEx5[:,n]=out[0,:,:]
    ImGEz5[:,n]=out[1,:,:]



    
    zlist=np.array([0.020])
    out=ImGLDOS.LDOSintegrandplottrace(k0,kparlistSI/k0,zlist,nstack,dstack)

    ImGEx20[:,n]=out[0,:,:]
    ImGEz20[:,n]=out[1,:,:]

    zlist=np.array([0.050])
    out=ImGLDOS.LDOSintegrandplottrace(k0,kparlistSI/k0,zlist,nstack,dstack)

    ImGEx50[:,n]=out[0,:,:]
    ImGEz50[:,n]=out[1,:,:]

    zlist=np.array([0.100])
    out=ImGLDOS.LDOSintegrandplottrace(k0,kparlistSI/k0,zlist,nstack,dstack)

    ImGEx100[:,n]=out[0,:,:]
    ImGEz100[:,n]=out[1,:,:]
    
    kSPP[n]=spp(k0,nstack)



plt.figure(figsize=(3.5,5))
pcm = plt.pcolormesh(kparplot,k0plot/(2.*np.pi*1.240),np.log10(ImGEx5*0.667+ImGEz5*0.333 + 1E-23),vmin=-3,vmax=2, shading='auto')

cbar = plt.colorbar(pcm)
cbar.set_label(r'$\log_{10}$ LDOS', fontsize=12)  # Set colorbar label


plt.plot(kSPP,k0list/(2.*np.pi*1.240),'r:')
plt.xlabel(r'$k_{||}\ (\mu m^{-1})$')
plt.ylabel(r'$\omega\ (eV)$')
plt.title('5 nm from Ag')
plt.show()



plt.figure(figsize=(3.5,5))
pcm = plt.pcolormesh(kparplot,k0plot/(2.*np.pi*1.240),np.log10(ImGEx20*0.667+ImGEz20*0.333 + 1E-23),vmin=-3,vmax=2, shading='auto')

cbar = plt.colorbar(pcm)
cbar.set_label(r'$\log_{10}$ LDOS', fontsize=12)  # Set colorbar label

plt.plot(kSPP,k0list/(2.*np.pi*1.240),'r:')
plt.xlabel(r'$k_{||}\ (\mu m^{-1})$')
plt.ylabel(r'$\omega\ (eV)$')
plt.title('20 nm from Ag')
plt.show()



plt.figure(figsize=(3.5,5))
pcm = plt.pcolormesh(kparplot,k0plot/(2.*np.pi*1.240),np.log10(ImGEx50*0.667+ImGEz50*0.333 + 1E-23),vmin=-3,vmax=2, shading='auto')

cbar = plt.colorbar(pcm)
cbar.set_label(r'$\log_{10}$ LDOS', fontsize=12)  # Set colorbar label

plt.plot(kSPP,k0list/(2.*np.pi*1.240),'r:')
plt.xlabel(r'$k_{||}\ (\mu m^{-1})$')
plt.ylabel(r'$\omega\ (eV)$')
plt.title('50 nm from Ag')
plt.show()



plt.figure(figsize=(3.5,5))
pcm = plt.pcolormesh(kparplot,k0plot/(2.*np.pi*1.240),np.log10(ImGEx100*0.667+ImGEz100*0.333 + 1E-23),vmin=-3,vmax=2, shading='auto')

cbar = plt.colorbar(pcm)
cbar.set_label(r'$\log_{10}$ LDOS', fontsize=12)  # Set colorbar label

plt.plot(kSPP,k0list/(2.*np.pi*1.240),'r:')
plt.xlabel(r'$k_{||}\ (\mu m^{-1})$')
plt.ylabel(r'$\omega\ (eV)$')
plt.title('100 nm from Ag')
plt.show()

