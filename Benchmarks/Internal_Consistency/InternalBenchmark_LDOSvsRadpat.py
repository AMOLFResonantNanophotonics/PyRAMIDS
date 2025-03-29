#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A consistent LDOS implementation must include the following attributes:

  For any non-absorbing system without guided modes, the radiation pattern integral should
  equal the ImG based LDOS for any dipole orientation    

Note that this is a nontrivial check, since
    Im G is calculated for specific dipole orientations. Is the reconstruction for arbitrary dipole moment correct ?
    The core radiation pattern to LDOS routine similarly uses "favorite orientations"
    Yet radiation patterns are coherent field superpositions, and so getting equality when summing absolute value-squared
    is a nontrivial exercise.
    
This routine is written to do the following:
    Take a test case, here a single interface with a high-dielectric.
    Verify the LDOS-at-any-p-and-m routine against the radiation-pattern-at-any-p-and-m routine 
    For
    - "canonical dipoles" (just px, py, pz, and magnetic equivalent)
    - Off-kilter purely electric and magnetic dipoles
    - Mixed dipoles that are both p and m
    - Mixed dipoles that are both p and m with a phase slip

To our knowledge this is the only real benchmark for the "magnetoelectric LDOS", as there are no literature benchmarks. 
    
"""

#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%%


from Library.Use import Use_Radiationpattern as Radpat
from Library.Use import Use_LDOS as ImGLDOS


import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

lam=1.0
k0=2.*np.pi/lam
nstack=[3.5,1.5,1.0]
dstack=[0.2]
zlist=np.linspace(-0.5,0.75,250)



## all of the Cartesian electric dipoles
print('Doing benchmark run electric dipoles')
pu=[1.0,0.0,0.0]
mu=[0.0,0.0,0.0]
 
ldx=ImGLDOS.LDOSatanyPandM(pu, mu, k0, zlist, nstack, dstack)
ldxR=Radpat.TotalRadiatedatanyPandM(pu, mu, k0, zlist, nstack, dstack)[0]


pu=[0.0,1.0,0.0]
ldy=ImGLDOS.LDOSatanyPandM(pu, mu, k0, zlist, nstack, dstack)
ldyR=Radpat.TotalRadiatedatanyPandM(pu, mu, k0, zlist, nstack, dstack)[0]
pu=[0.0,0.0,1.0]
ldz=ImGLDOS.LDOSatanyPandM(pu, mu, k0, zlist, nstack, dstack)
ldzR=Radpat.TotalRadiatedatanyPandM(pu, mu, k0, zlist, nstack, dstack)[0]

#abbreviated routines

ldallImG=ImGLDOS.LDOS(k0,zlist,nstack,dstack)
ldallRadpat=Radpat.TotalRadiated(k0,zlist,nstack,dstack)


#diagonal dipoles
pu=[1.0, 1.0,0.0]
lddiag1=ImGLDOS.LDOSatanyPandM(pu, mu, k0, zlist, nstack, dstack)
lddiagR1=Radpat.TotalRadiatedatanyPandM(pu, mu, k0, zlist, nstack, dstack)[0]

pu=[1.0, 0.0,-1.0]
lddiag2=ImGLDOS.LDOSatanyPandM(pu, mu, k0, zlist, nstack, dstack)
lddiagR2=Radpat.TotalRadiatedatanyPandM(pu, mu, k0, zlist, nstack, dstack)[0]

pu=[1.0, 1.j ,1.]
lddiag3=ImGLDOS.LDOSatanyPandM(pu, mu, k0, zlist, nstack, dstack)
lddiagR3=Radpat.TotalRadiatedatanyPandM(pu, mu, k0, zlist, nstack, dstack)[0]


print('Point by point error between rad pat and ImG by integration in abbrev. routine is below:')
print(np.abs(ldallImG[0:1]-ldallRadpat[0:1,:]).max())


print('Point by point error between rad pat and ImG by integration for diagonal dipole (green case in last plot')
print(np.abs(lddiagR3-lddiag3).max())




fig,ax=plt.subplots(ncols=3,figsize=(26,8))
ax[0].plot(zlist,ldallImG[0],'b',zlist,ldallImG[1],'g',
         zlist,ldallRadpat[0],'bo',zlist,ldallRadpat[1],'go') 
ax[0].legend(['p|| from ImG','pz from ImG','p|| integrated radiated','dito pz'])
ax[0].set_title('Electric only, cartesian dipoles, abbreviated (fast) routines')
ax[0].set_xlabel('z/lambda')
ax[0].set_ylabel('LDOS over free space') 
ax[1].plot(zlist,ldxR,'ro',zlist,ldxR,'r-',
         zlist,ldyR,'bo',zlist,ldy,'b',
         zlist,ldzR,'go',zlist,ldz,'g-')
ax[1].legend(['px from radpat','px from ImG','py (obscures px)','py','pz','pz'])
ax[1].set_title('Cartesian, using "arbitrary p-mu" routines')
ax[1].set_xlabel('z/lambda')
ax[1].set_ylabel('LDOS over free space') 


ax[2].plot(zlist,lddiagR1,'ro',zlist,lddiagR2,'bo', zlist,lddiagR3,'go',zlist,lddiag1,'r-',
         zlist,lddiag2,'b',
        zlist,lddiag3,'g-')
ax[2].set_title('Some off kilter examples , using "arbitrary p-mu" routines')
ax[2].set_xlabel('z/lambda')
ax[2].set_ylabel('LDOS over free space') 
ax[2].legend(['In-plane diag (px=py at pz=0)','Out of plane diag (px=-pz at py =0)',
                 'Phase slip py=ipx at pz=px','Agreement between points and lines']) 
plt.show()




## all of the Cartesian magnetic dipoles
print('Doing benchmark run magnetic dipoles')
pu=[0.0,0.0,0.0]
mu=[1.0,0.0,0.0]
 
ldx=ImGLDOS.LDOSatanyPandM(pu, mu, k0, zlist, nstack, dstack)
ldxR=Radpat.TotalRadiatedatanyPandM(pu, mu, k0, zlist, nstack, dstack)[0]


mu=[0.0,1.0,0.0]
ldy=ImGLDOS.LDOSatanyPandM(pu, mu, k0, zlist, nstack, dstack)
ldyR=Radpat.TotalRadiatedatanyPandM(pu, mu, k0, zlist, nstack, dstack)[0]
mu=[0.0,0.0,1.0]
ldz=ImGLDOS.LDOSatanyPandM(pu, mu, k0, zlist, nstack, dstack)
ldzR=Radpat.TotalRadiatedatanyPandM(pu, mu, k0, zlist, nstack, dstack)[0]

 

#diagonal dipoles
mu=[1.0, 1.0,0.0]
lddiag1=ImGLDOS.LDOSatanyPandM(pu, mu, k0, zlist, nstack, dstack)
lddiagR1=Radpat.TotalRadiatedatanyPandM(pu, mu, k0, zlist, nstack, dstack)[0]

mu=[1.0, 0.0,-1.0]
lddiag2=ImGLDOS.LDOSatanyPandM(pu, mu, k0, zlist, nstack, dstack)
lddiagR2=Radpat.TotalRadiatedatanyPandM(pu, mu, k0, zlist, nstack, dstack)[0]

mu=[1.0, 1.j ,1.]
lddiag3=ImGLDOS.LDOSatanyPandM(pu, mu, k0, zlist, nstack, dstack)
lddiagR3=Radpat.TotalRadiatedatanyPandM(pu, mu, k0, zlist, nstack, dstack)[0]


print('Point by point error between rad pat and ImG by integration in abbrev. routine is below:')
print(np.abs(ldallImG[2:3]-ldallRadpat[2:3,:]).max())


print('Point by point error between rad pat and ImG by integration for diagonal dipole (green case in last plot')
print(np.abs(lddiagR3-lddiag3).max())




fig,ax=plt.subplots(ncols=3,figsize=(26,8))
ax[0].plot(zlist,ldallImG[2],'b',zlist,ldallImG[3],'g',
         zlist,ldallRadpat[2],'bo',zlist,ldallRadpat[3],'go') 
ax[0].legend(['m|| from ImG','mz from ImG','m|| integrated radiated','dito mz'])
ax[0].set_title('Magnetic only, Cartesian dipoles, abbreviated (fast) routines')
ax[0].set_xlabel('z/lambda')
ax[0].set_ylabel('LDOS over free space') 
ax[1].plot(zlist,ldxR,'ro',zlist,ldxR,'r-',
         zlist,ldyR,'bo',zlist,ldy,'b',
         zlist,ldzR,'go',zlist,ldz,'g-')
ax[1].legend(['mx from radpat','mx from ImG','my (obscures mx)','my','mz','mz'])
ax[1].set_title('Cartesian, using "arbitrary p-mu" routines')
ax[1].set_xlabel('z/lambda')
ax[1].set_ylabel('LDOS over free space') 


ax[2].plot(zlist,lddiagR1,'ro',zlist,lddiagR2,'bo', zlist,lddiagR3,'go',zlist,lddiag1,'r-',
         zlist,lddiag2,'b',
        zlist,lddiag3,'g-')
ax[2].set_title('Some off kilter examples , using "arbitrary p-mu" routines')
ax[2].set_xlabel('z/lambda')
ax[2].set_ylabel('LDOS over free space') 
ax[2].legend(['In-plane diag (mx=my at mz=0)','Out of plane diag (mx=-mz at my =0)',
                 'Phase slip my=imx at mz=mx','Agreement between points and lines']) 
plt.show()



print('Doing mixed magnetic-electric dipoles')

pu=[1.0,0.0,0.0]
mu=[0.0,1,0.0]
ld11=ImGLDOS.LDOSatanyPandM(pu, mu, k0, zlist, nstack, dstack)
ld11R=Radpat.TotalRadiatedatanyPandM(pu, mu, k0, zlist, nstack, dstack)[0]

pu=[1.0,0.0,0.0]
mu=[0.0,-1.0,0.0]
ld1m1=ImGLDOS.LDOSatanyPandM(pu, mu, k0, zlist, nstack, dstack)
ld1m1R=Radpat.TotalRadiatedatanyPandM(pu, mu, k0, zlist, nstack, dstack)[0]




pu=[1.0,0.0,0.0]
mu=[0.0,1.j,0.0]
ld1C=ImGLDOS.LDOSatanyPandM(pu, mu, k0, zlist, nstack, dstack)
ld1CR=Radpat.TotalRadiatedatanyPandM(pu, mu, k0, zlist, nstack, dstack)[0]

pu=[1.0,.0,0.0]
mu=[0.0,0-1.j,0.0]
ld1Cm=ImGLDOS.LDOSatanyPandM(pu, mu, k0, zlist, nstack, dstack)
ld1CmR=Radpat.TotalRadiatedatanyPandM(pu, mu, k0, zlist, nstack, dstack)[0]




fig,ax=plt.subplots(ncols=3,figsize=(12,8))
ax[0].plot(zlist,ld11,'ro',zlist,ld11R,'r',zlist,ld1m1,'bo',zlist,ld1m1R,'b')  
ax[0].set_xlabel('z/lambda')
ax[0].set_ylabel('LDOS over free space')      
ax[0].legend(['px=my from ImG','dito from Radpat','px=-my from ImG','dito from radpat'])
ax[0].set_title('Kerker dipoles (px=+/- my)') 
       
ax[1].plot(zlist,ld1C,'ro',zlist,ld1CR,'r',zlist,ld1Cm,'bo',zlist,ld1CmR,'b')  
ax[1].set_xlabel('z/lambda')
ax[1].set_ylabel('LDOS over free space')      
ax[1].legend(['px=imy from ImG','dito from Radpat','px=-imy from ImG','dito from radpat'])
ax[1].set_title('Pseudochiral dipoles (px=+/- i my)') 

 
ax[2].plot(zlist,(ld1CR-ld1CmR),'ro',zlist,2.*ldallImG[4],zlist,2.*ldallRadpat[4])
ax[2].set_xlabel('z/lambda')
ax[2].legend(['From radpat: P+ - P- ','2 rho_C from abbrev. ImG','dito from abbrev. radpat'])


plt.show()
 

#%%

print('Some off kilter magnetoelectric example')
fig,ax=plt.subplots(ncols=3,figsize=(26,8))


pu=[1.0,0.5,0.0]
mu=[1.0, 0.5j,1.0]
lddiag1=ImGLDOS.LDOSatanyPandM(pu, mu, k0, zlist, nstack, dstack)
lddiagR1=Radpat.TotalRadiatedatanyPandM(pu, mu, k0, zlist, nstack, dstack)[0]
 

pu=[1.0,0.5j,0.0]
mu=[1.0, 1.0j,0.0]
lddiag2=ImGLDOS.LDOSatanyPandM(pu, mu, k0, zlist, nstack, dstack)
lddiagR2=Radpat.TotalRadiatedatanyPandM(pu, mu, k0, zlist, nstack, dstack)[0]
 
pu=[1.0,-0.5j,1.0]
mu=[1.0, 0.4,0.3j]
lddiag3=ImGLDOS.LDOSatanyPandM(pu, mu, k0, zlist, nstack, dstack)
lddiagR3=Radpat.TotalRadiatedatanyPandM(pu, mu, k0, zlist, nstack, dstack)[0]
  


ax[0].plot(zlist,lddiagR1,'ro',zlist,lddiag1,'r-')
ax[0].set_title('Some off kilter examples , magnetoelectric ones')
ax[0].set_xlabel('z/lambda')
ax[0].set_ylabel('LDOS over free space') 
ax[0].legend(['Off kilter magnetoelectric, integrated radpat','Dito LDOS']) 
plt.show()

ax[1].plot(zlist,lddiagR2,'ro',zlist,lddiag2,'r-')
ax[1].set_title('Some off kilter examples , magnetoelectric ones')
ax[1].set_xlabel('z/lambda')
ax[1].set_ylabel('LDOS over free space') 
ax[1].legend(['Off kilter magnetoelectric, integrated radpat','Dito LDOS']) 
plt.show()

ax[2].plot(zlist,lddiagR3,'ro',zlist,lddiag3,'r-')
ax[2].set_title('Off kilter, magnetoelectric, chiral')
ax[2].set_xlabel('z/lambda')
ax[2].set_ylabel('LDOS over free space') 
ax[2].legend(['Off kilter magnetoelectric, integrated radpat','Dito LDOS']) 
plt.show()

 