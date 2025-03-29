'''
########################
Perpendicular electric and magnetic dipoles / radiation patterns
Lukosz and Kunz JOSA 67 1615'
Fig 2 and 4 (right on interface, different refractive indices)
########################
'''


#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print('###  Literature Benchmark: Lukosz & Kunz JOSA 67, 1615-1619 (1977)  ###')

#%%

from Library.Use import Use_Radiationpattern as Radpat

import matplotlib.pyplot as plt
import numpy as np


lam=1.0 #wvlength in vacuum
k0=2.0*np.pi/lam


#plot range for angles
Nthe=2021;
thelist=np.linspace(-np.pi/2,3.0*np.pi/2,Nthe)

#realizations in Fig 2 to loop over, with linespec for plot
nlist=[2.0,np.sqrt(2.),1.01,1.]
plotspec=['k:','k--','k','k-.']
#electric parallel-dipole only


f,ax = plt.subplots(2,1,figsize=(5,10),gridspec_kw={'height_ratios': [3, 1]})
f,ax2 = plt.subplots(2,1,figsize=(5,10),gridspec_kw={'height_ratios': [1, 1]})
leg=[]

for m in range(len(nlist)):

    nstack=[1.0,nlist[m]]
    dstack=[]
    z=-1.E-5*lam #located just inside medium of index 1
    
    
    pu=np.array([0.0 ,0.0, 1.0])
    mu=np.array([0,0,0])
    P1,E,f=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0.0,nstack,dstack)
    ax[0].plot(180/np.pi*thelist,P1,plotspec[m])
    
    pu=np.array([0.0 ,0.0, 0.0])
    mu=np.array([0,0,1.0])
    P2,E,f=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0.0*thelist,nstack,dstack)
    ax[1].plot(180/np.pi*thelist,P2,plotspec[m])
    
    nstack=[1.0, 1/nlist[m]]
    pu=np.array([0.0 ,0.0, 1.0])
    mu=np.array([0,0,0])
    P1,E,f=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0.0*thelist,nstack,dstack)
    ax2[0].plot(180/np.pi*thelist,P1,plotspec[m])
    
    pu=np.array([0.0 ,0.0, 0.0])
    mu=np.array([0,0,1.0])
    P2,E,f=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0.0*thelist,nstack,dstack)
    ax2[1].plot(180/np.pi*thelist,P2,plotspec[m])
    
    leg.append('n='+str(nlist[m]))
    
ax[0].set_ylim(bottom=0.0)
ax[0].legend(leg)
ax[1].set_ylim(bottom=0.0)
ax[0].set_xlim([0,180])
ax[1].set_xlim([0,180])
ax[1].set_xlabel('Theta')
ax[0].set_ylabel('Flux/steradian')
ax[1].set_ylabel('Flux/steradian')
ax[0].set_title('Perp electric, Fig 2 JOSA 67 1615')
ax[1].set_title('Perp magnetic, Fig 2 JOSA 67 1615]')

ax2[0].set_ylim(bottom=0.0)
ax2[0].legend(leg)
ax2[1].set_ylim(bottom=0.0)
ax2[0].set_xlim([0,180])
ax2[1].set_xlim([0,180])
ax2[1].set_xlabel('Theta')
ax2[0].set_ylabel('Flux/steradian')
ax2[1].set_ylabel('Flux/steradian')
ax2[0].set_title('Perp electric, Fig 4 JOSA 67 1615')
ax2[1].set_title('Perp magnetic, Fig 4 JOSA 67 1615]')

plt.show()

'''
########################
Perpendicular electric and magnetic dipoles / radiation patterns
Lukosz and Kunz JOSA 67 1615'
Fig 3 (different heights)
########################
'''

f,ax = plt.subplots(2,1,figsize=(5,10),gridspec_kw={'height_ratios': [2, 1]})
leg=[]

zlist=-lam*np.array([1.E-5,1/(4.*np.pi),1.0]) #note the minus sign. source is in air
plotspec=['k','k--','k:']
plotname=['0','1/4pi','1']

nsubs=np.sqrt(2.0)
for m in range(len(zlist)): 
    
    z=zlist[m]
    nstack=[1.0,nsubs]
    dstack=[]
   
    pu=np.array([0.0 ,0.0, 1.0])
    mu=np.array([0,0,0])
    P1,E,f=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0.0*thelist,nstack,dstack)
    ax[0].plot(180/np.pi*thelist,P1,plotspec[m])
    
    pu=np.array([0.0 ,0.0, 0.0])
    mu=np.array([0,0,1.0])
    P2,E,f=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0.0*thelist,nstack,dstack)
    ax[1].plot(180/np.pi*thelist,P2,plotspec[m])
    
    leg.append('z/lam='+plotname[m])
    
ax[0].set_ylim(bottom=0.0)
ax[0].legend(leg)
ax[1].set_ylim(bottom=0.0)
ax[0].set_xlim([0,180])
ax[1].set_xlim([0,180])
ax[1].set_xlabel('Theta')
ax[0].set_ylabel('Flux/steradian')
ax[1].set_ylabel('Flux/steradian')
ax[0].set_title('Perp electric, Fig 3 JOSA 67 1615]')
ax[1].set_title('Perp magnetic, Fig 3 JOSA 67 1615]')
plt.show()


'''
########################
Perpendicular electric and magnetic dipoles / radiation patterns
Lukosz and Kunz JOSA 67 1615'
Fig 5 (polar plots at different heights)
########################
'''

f,ax = plt.subplots(1,2,subplot_kw=dict(projection="polar"),gridspec_kw={'width_ratios': [2.5, 1]})
leg=[]

zlist=-lam*np.array([1.E-5,1.0]) #note the minus sign. source is in air
plotspec=['k','k--',]
plotname=['0','1']
#plot range for angles
Nthe=2021;
thelist=np.linspace(0,np.pi,Nthe)


nsubs=1.5
for m in range(len(zlist)): 
    
    z=zlist[m]
    nstack=[1.0,nsubs]
    dstack=[]
   
    pu=np.array([0.0 ,0.0, 1.0])
    mu=np.array([0,0,0])
    P1,E,f=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0.0*thelist,nstack,dstack)
    ax[0].plot(np.pi-thelist,P1,plotspec[m])
    
    pu=np.array([0.0 ,0.0, 0.0])
    mu=np.array([0,0,1.0])
    P2,E,f=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0.0*thelist,nstack,dstack)
    ax[1].plot(np.pi-thelist,P2,plotspec[m])
    
    leg.append('z/lam='+plotname[m])
ax[0].set_title('Fig 5 JOSA 67 1615 ')
ax[0].set_thetamin(0)
ax[0].set_thetamax(180)
ax[1].set_thetamin(0)
ax[1].set_thetamax(180)
ax[1].legend(leg)

plt.show()



'''
#-----------------------------------------------
##  Lukosz JOSA 69 1495 (1979)
##  Radiation patterns of electric and magnetic dipoles above a single interface
##

'''


lam=1.0 #wvlength in vacuum
k0=2.0*np.pi/lam


#plot range for angles
Nthe=2021;
thelist=np.linspace(-np.pi/2,3.0*np.pi/2,Nthe)
nlist=[1./np.sqrt(2),np.sqrt(2.0),0.5,2.0,1.01,0.99]


for nsubs in nlist:

    nstack=[1.0,nsubs]
    dstack=[]
    z=-1.E-5*lam #located just inside medium of index 1
    
    
    #electric parallel-dipole only
    pu=np.array([1.0 ,0.0, 0.0])
    mu=np.array([0,0,0])
    P1,E,f=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0.0*thelist,nstack,dstack)
    P2,E,f=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,np.pi/2.+0.0*thelist,nstack,dstack)
    
 
    plt.figure()
    
    plt.plot(180/np.pi*thelist,0.5*P1,'k:',
              180/np.pi*thelist,0.5*P2,'k--',
              180/np.pi*thelist,0.5*(P1+P2),'k')
    plt.ylim(bottom=0.0)
    plt.xlim([0,180])
    plt.xlabel('Theta')
    plt.ylabel('Flux/steradian')
    plt.legend(['p','s','s+p'])
    plt.title('Parallel electric, n='+str(nsubs)+'[JOSA 69 1495]')
    plt.show()
    #magnetic parallel dipole only
    pu=np.array([0.0 ,0.0, 0.0])
    mu=np.array([0.0,1.0,0])
    P1,E,f=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,0.0*thelist,nstack,dstack)
    P2,E,f=Radpat.RadiationpatternPandField(k0,z,pu,mu,thelist,np.pi/2.+0.0*thelist,nstack,dstack)
    
    # prefactor 0.5 in all the plots results from definition of averaging in 
    # Lukosz
    plt.figure()
    plt.plot(180/np.pi*thelist,0.5*P1,'k:',
              180/np.pi*thelist,0.5*P2,'k--',
              180/np.pi*thelist,0.5*(P1+P2),'k')
    plt.xlim([0,180])
    plt.ylim(bottom=0.0)
    plt.xlabel('Theta')
    plt.ylabel('Flux/steradian')
    plt.legend(['p','s','s+p'])
    plt.title('Parallel magnetic, n='+str(nsubs)+' [JOSA 69 1495]')
    plt.show()

