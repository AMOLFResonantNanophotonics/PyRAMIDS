
"""
Reproduces Urbach & Rikken, Phys Rev A 57, 3913 (1998)
Figures 4-9, 15,16 (all the z-dependent LDOS cross cuts)
"""
#%%

print('### Literature Benchmark: Urbach & Rikken, Phys. Rev. A 57 3913 (1998) ###')


#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
#%%


from Library.Use import Use_LDOS as ImGLDOS
from Library.Use import Use_Radiationpattern as Farfield

import matplotlib.pyplot as plt
import numpy as np
 
 
     
# this file contains one function - the rest is calling scripts
# this function for repeated plot tasks
def BaseUrbachplotter(lam,d,n1,n3,nslab,zlist,plotz,xlab,yrange,titleaddition):
    #Urbach & Rikken for their plots specify wavelength,  d, the n1 and n3 of the halfspaces, and scan over several nslab values. Hence nslab is expected to be a list.
    #zlist is the set of z-sample points. Since Urbach skill their plot axis, plotz is requested as plot cooridnate. xlab is put on the x-axis as label. The plot is clipped at yrange. Here a 
    #list is expected to clip the total, radiative, and guided ldos
    k0=2.*np.pi/lam
    dstack=np.array([d]) # change here for Fig 1,2 d = 0.1 or Fig 3,4 d = 1
    
    fig1,ax=plt.subplots(3,2,figsize=(10,15)) 
    fig1.subplots_adjust(hspace=0.3)
    
    for ns in nslab: 
            
        nstack=np.array([n1,ns,n3])
        
        
        
        
        # Total LDOS
        rhoE_par,rhoE_perp,rhoM_par,rhoM_perp,rhoC=ImGLDOS.LDOS(k0,zlist,nstack,dstack)    
        
        # Radiative LDOS from 4pi integral
        data=Farfield.TotalRadiated(k0,zlist,nstack,dstack)   
        outpar_all4pi_fromP = data[0]
        outperp_all4pi_fromP = data[1]
        
        
        
        # Guided LDOS by subtracting  
        outpar_guide_fromP = np.subtract(rhoE_par, outpar_all4pi_fromP)
        outperp_guide_fromP = np.subtract(rhoE_perp, outperp_all4pi_fromP)
        
        
     
        ax[0][0].plot(plotz,rhoE_par)
        ax[0][1].plot(plotz,rhoE_perp)
        ax[1][0].plot(plotz,outpar_all4pi_fromP) 
        ax[1][1].plot(plotz,outperp_all4pi_fromP)
        ax[2][0].plot(plotz,outpar_guide_fromP)
        ax[2][1].plot(plotz,outperp_guide_fromP)
    
    fig1.suptitle(titleaddition)
    ax[0][0].set_title("Total LDOS par ")
    ax[0][0].set_xlabel('2z/d')
    ax[0][0].set_ylabel('LDOS/vac')
    ax[0][0].set_ylim([0.0,yrange[0]])
    ax[0][0].legend(['n='+str(n) for n in nslab])
     
    ax[0][1].set_title("Total LDOS perp ")
    ax[0][1].set_xlabel(xlab)
    ax[0][1].set_ylabel('LDOS/vac')
    ax[0][1].set_ylim([0.0,yrange[0]])
    ax[0][1].legend(['n='+str(n) for n in nslab])
    
    ax[1][0].set_title("Radiative LDOS ") 
    ax[1][0].set_xlabel(xlab)
    ax[1][0].set_ylabel('LDOS/vac')
    ax[1][0].set_ylim([0.0,yrange[1]])
    ax[1][0].legend(['n='+str(n) for n in nslab])

    ax[1][1].set_title("Radiative LDOS perp ")  
    ax[1][1].set_xlabel(xlab)
    ax[1][1].set_ylabel('LDOS/vac')
    ax[1][1].set_ylim([0.0,yrange[1]])
    ax[1][1].legend(['n='+str(n) for n in nslab])

    ax[2][0].set_title("Guided LDOS ") 
    ax[2][0].set_xlabel(xlab)
    ax[2][0].set_ylabel('LDOS/vac')
    ax[2][0].set_ylim([0.0,yrange[2]])
    ax[2][0].legend(['n='+str(n) for n in nslab])

    ax[2][1].set_title("Guided LDOS per ") 
    ax[2][1].set_xlabel(xlab)
    ax[2][1].set_ylabel('LDOS/vac')
    ax[2][1].set_ylim([0.0,yrange[2]])
    ax[2][1].legend(['n='+str(n) for n in nslab])
                
                
       
    return
plt.show()

#Urbach & Rikken figure 4,5
lam=1.0  #pins the units, everything normalized to this wavelength
d=0.1  
nslab=[1.5, 2.0, 4.0]  # routine will loop over the nslab refractive index
zlist=np.linspace(-lam*0.5,lam*0.5,200)+d/2.
plotz=2.*(zlist-d/2.0)/d
yrange=[5,1.25,5.5] # y-axi range for plotting total / radiative / guided
 
BaseUrbachplotter(lam,d,1.0,1.0,nslab,zlist,plotz,'2z/d',yrange, 'PRA 59 3913, Fig 4,5')


#Urbach & Rikken figure 6,7
lam=1.0
d=1.0
nslab=[1.5, 2.0, 4.0]
zlist=np.linspace(-lam*2.0,lam*2.0,200)+d/2.
plotz=2.*(zlist-d/2.0)/d
 
BaseUrbachplotter(lam,d,1.0,1.0,nslab,zlist,plotz,'2z/d',yrange, 'PRA 59 3913, Fig 6,7')


#Urbach & Rikken figure 8
lam=1.0
d=0.1
nslab=[1.0, 1.5, 1.8]
zlist=np.linspace(-lam*0.5,lam*0.5,200)+d/2.
plotz=2.*(zlist-d/2.0)/d
yrange=[6.0,6.0,6.0] 
BaseUrbachplotter(lam,d,2.0,2.0,nslab,zlist,plotz,'2z/d',yrange, 'PRA 59 3913, Fig 8')


#Urbach & Rikken figure 9
lam=1.0
d=1.0
nslab=[1.0, 1.5, 1.8]
zlist=np.linspace(-lam*2.0,lam*2.0,200)+d/2.
plotz=2.*(zlist-d/2.0)/d
yrange=[6.0,6.0,6.0] 
BaseUrbachplotter(lam,d,2.0,2.0,nslab,zlist,plotz,'2z/d',yrange, 'PRA 59 3913, Fig 9')




#Urbach & Rikken figure 14a
lam=0.611  #wvlen in micron
dlist=[0.02,0.15,0.5] #thicknesses of the thin layer to loop over
n1=1.465 #substrate
n2=1.585 #slab
n3=1.0   #superstrate

fig,ax=plt.subplots(3,1,figsize=(5,15))


for m in range(len(dlist)):
    d=dlist[m]    
    zlist=np.linspace(-d*5.0,d*5.0,200)+d/2.
    plotz=(zlist-d/2.0)/d
    rhoE_par,rhoE_perp,rhoM_par,rhoM_perp,rhoC=ImGLDOS.LDOS(2.0*np.pi/lam,zlist,[n1,n2,n3],[d])    
     
    ax[m].plot(plotz,(rhoE_par*2+rhoE_perp)/3)
    ax[m].plot([-0.5,-0.5],[0,yrange[m]],'k:',[0.5,0.5],[0,yrange[m]],'k:',[-5,3],[n1,n1],'k:',[-5,3],[n2,n2],'k:',[-5,3],[n3,n3],'k:')
    ax[m].set_ylim([0, 2.1])
    ax[m].set_xlabel('z/d')
    ax[m].set_ylabel('Isotropic LDOS')
    ax[m].legend(['d='+str(1000*d)+' nm, at lambda='+str(lam*1000)+' nm'])
    
plt.show()


#Urbach & Rikken figure 16
lam=0.611  #wvlen in micron
d=  0.170  #thicknesses of the thin layer to loop over
n1list=[1.385,1.585,3.920] #substrate
n2=1.585 #slab
n3=1.0   #superstrate
yrange=[2.1, 2.1, 5.1]
fig,ax=plt.subplots(3,1,figsize=(5,15))


for m in range(len(n1list)):
    n1=n1list[m]   
    zlist=np.linspace(-d*4,d*4,200)+d/2.
    plotz=(zlist-d/2.0)/d
    rhoE_par,rhoE_perp,rhoM_par,rhoM_perp,rhoC=ImGLDOS.LDOS(2.0*np.pi/lam,zlist,[n1,n2,n3],[d])    
     
    ax[m].plot(plotz,(rhoE_par*2+rhoE_perp)/3)
    ax[m].plot([-0.5,-0.5],[0,yrange[m]],'k:',[0.5,0.5],[0,yrange[m]],'k:',[-5,2],[n1,n1],'k:',[-5,2],[n2,n2],'k:',[-5,2],[n3,n3],'k:')
    ax[m].set_ylim([0, yrange[m]])
    ax[m].set_xlim([-4,4]) 
    ax[m].set_xlabel('z/d')
    ax[m].set_ylabel('Isotropic LDOS')
    ax[m].legend(['n2='+str(n1)+', at n1='+str(n2)+' and n3='+str(n3)])
plt.show()