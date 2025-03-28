''' Plot example MIM - insight into LDOS contributions from plotting LDOS integrand
    Example for a simple Metal-Insulator-Meta structure  in insultator surrounding.. 
    with gap  between metal layer increasing.
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


kparlistSI=np.linspace(0,4.5*kmax,300)

k0plot,kparplot=np.meshgrid(k0list,kparlistSI)


ImGEx10=0.0*k0plot
ImGEz10=0.0*k0plot

ImGEx20=0.0*k0plot
ImGEz20=0.0*k0plot


ImGEx50=0.0*k0plot
ImGEz50=0.0*k0plot


ImGEx500=0.0*k0plot
ImGEz500=0.0*k0plot

kSPP=0*k0list

for n in range(nk):
    k0=k0list[n]

    nstack=[1.5,drude(k0,kp,g),1.5,drude(k0,kp,g),1.5]
    dstack=[0.05,0.010,0.05]




    zlist=np.array([0.055])
    out=ImGLDOS.LDOSintegrandplottrace(k0,kparlistSI/k0,zlist,nstack,dstack)

    ImGEx10[:,n]=out[0,:,:]
    ImGEz10[:,n]=out[1,:,:]



    dstack=[0.05,0.020,0.05]
    zlist=np.array([0.06])
    out=ImGLDOS.LDOSintegrandplottrace(k0,kparlistSI/k0,zlist,nstack,dstack)

    ImGEx20[:,n]=out[0,:,:]
    ImGEz20[:,n]=out[1,:,:]



    dstack=[0.05,0.050,0.05]
    zlist=np.array([0.075])
    out=ImGLDOS.LDOSintegrandplottrace(k0,kparlistSI/k0,zlist,nstack,dstack)

    ImGEx50[:,n]=out[0,:,:]
    ImGEz50[:,n]=out[1,:,:]



    dstack=[0.05,0.500,0.05]
    zlist=np.array([0.26])
    out=ImGLDOS.LDOSintegrandplottrace(k0,kparlistSI/k0,zlist,nstack,dstack)

    ImGEx500[:,n]=out[0,:,:]
    ImGEz500[:,n]=out[1,:,:]
    
    kSPP[n]=spp(k0,nstack)



plt.figure(figsize=(2.5,5))
plt.pcolor(kparplot,k0plot/(2.*np.pi*1.240),np.log10(ImGEx10*0.667+ImGEz10*0.333))
plt.xlabel('$k_{||}\ (\mu m^{-1})$')
plt.ylabel('$\omega\ (eV)$')
plt.title('MIM (10 nm gap)')
plt.show()


plt.figure(figsize=(2.5,5))
plt.pcolor(kparplot,k0plot/(2.*np.pi*1.240),np.log10(ImGEx20*0.667+ImGEz20*0.333))
plt.xlabel('$k_{||}\ (\mu m^{-1})$')
plt.ylabel('$\omega\ (eV)$')
plt.title('MIM (20 nm gap)')
plt.show()



plt.figure(figsize=(2.5,5))
plt.pcolor(kparplot,k0plot/(2.*np.pi*1.240),np.log10(ImGEx50*0.667+ImGEz50*0.333),vmin=-3,vmax=2)
plt.xlabel('$k_{||}\ (\mu m^{-1})$')
plt.ylabel('$\omega\ (eV)$')
plt.title('MIM (50 nm gap)')
plt.show()



plt.figure(figsize=(2.5,5))
plt.pcolor(kparplot,k0plot/(2.*np.pi*1.240),np.log10(ImGEx500*0.667+ImGEz500*0.333),vmin=-3,vmax=2)
plt.xlabel('$k_{||}\ (\mu m^{-1})$')
plt.ylabel('$\omega\ (eV)$')
plt.title('MIM (500 nm gap)')
plt.show()

#%%  

ldpar10=0*k0list
ldper10=0*k0list
ldpar20=0*k0list
ldper20=0*k0list
ldpar50=0*k0list
ldper50=0*k0list
ldpar500=0*k0list
ldper500=0*k0list

ldmir10=0*k0list
ldmir20=0*k0list
ldmir50=0*k0list
ldmir500=0*k0list

for n in range(nk):
    k0=k0list[n]
    
    nstack=[1.5,drude(k0,kp,g),1.5,drude(k0,kp,g),1.5]

    dstack=[0.05,0.010,0.05]
    zlist=np.array([0.05+0.005])
    ld=ImGLDOS.LDOS(k0,zlist,nstack,dstack)
    ldpar10[n]=ld[0][0]
    ldper10[n]=ld[1][0]
 

    dstack=[0.05,0.020,0.05]
    zlist=np.array([0.05+0.01])
    ld=ImGLDOS.LDOS(k0,zlist,nstack,dstack)
    ldpar20[n]=ld[0][0]
    ldper20[n]=ld[1][0]
 
    dstack=[0.05,0.050,0.05]
    zlist=np.array([0.05+0.025])
    ld=ImGLDOS.LDOS(k0,zlist,nstack,dstack)
    ldpar50[n]=ld[0][0]
    ldper50[n]=ld[1][0]

    dstack=[0.05,0.500,0.05]
    zlist=np.array([0.05+0.25])
    ld=ImGLDOS.LDOS(k0,zlist,nstack,dstack)
    ldpar500[n]=ld[0][0]
    ldper500[n]=ld[1][0]
    
    
    nstack=[drude(k0,kp,g),1.5]
    dstack=[]
    zlist=np.array([0.005,0.01,0.025,0.25])
    ld=ImGLDOS.LDOS(k0,zlist,nstack,dstack)
    ldmir10[n]=ld[0][0]*0.6667+0.3333*ld[1][0]    
    ldmir20[n]=ld[0][1]*0.6667+0.3333*ld[1][1]    
    ldmir50[n]=ld[0][2]*0.6667+0.3333*ld[1][2]    
    ldmir500[n]=ld[0][3]*0.6667+0.3333*ld[1][3]    
    
    

plt.figure(figsize=(5,4))
plt.semilogy(k0list/(2.*np.pi*1.240),(0.6667*ldpar10+0.333*ldper10),'b',k0list/(2.*np.pi*1.240),ldmir10,'b:',
             k0list/(2.*np.pi*1.240),(0.6667*ldpar20+0.333*ldper20),'r',k0list/(2.*np.pi*1.240),ldmir20,'r:',
             k0list/(2.*np.pi*1.240),(0.6667*ldpar50+0.333*ldper50),'g',k0list/(2.*np.pi*1.240),ldmir50,'g:',
             k0list/(2.*np.pi*1.240),(0.6667*ldpar500+0.333*ldper500),'k',k0list/(2.*np.pi*1.240),ldmir500,'k:',)


plt.xlabel(r'$\omega\ (eV)$')
plt.ylabel('LDOS / vacuum LDOS (middle of MIM)')
plt.title('Solid - IMIMI, Dashed - infront of mirror')
plt.show()











