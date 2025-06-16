"""
@author: dpal,fkoenderink
"""



#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#%%
import numpy as np
from numpy.linalg import inv
import scipy.integrate as integrate


from Library.Use import Use_LDOS as ImG
from Library.Use import Use_Green as Gs
from Library.Use import Use_Radiationpattern as Radpat
from Library.Use import Use_Planewaves as pw
from Library.Util import Util_argumentchecker as check
from Library.Util import Util_argumentrewrapper as cc 



def dipolelayerchecker(rdipvecs,nstack,dstack):
    #checks if all dipoles in the problem are in the same layer, and returns the refr index of that layer
    rdipvecs=check.checkr(rdipvecs)
    z=rdipvecs[2,:]
    lib=cc.pinpointdomain(z,dstack)
    if np.unique(lib).size !=1:
        raise(ValueError('multiple scattering - dipoles spread over multiple layers are not supported'))
    return lib[0],z.size


def Planewavedriving(theta,phi,s,p,rdip,k0,nstack,dstack):
    EH=pw.CartesianField(theta, phi, s, p, rdip, k0, nstack, dstack)
    inputintensity=(np.array(np.abs(s))**2+np.array(np.abs(p))**2)*nstack[0]*0.5
    return np.squeeze(EH), inputintensity


def Emitterdriving(pnmsource,rsource,rdipvecs,k0,nstack,dstack):
    
    GG=Gs.Greensafe(k0, nstack, dstack, rdipvecs, rsource)
    Efield=np.transpose(np.dot(GG.T,pnmsource))
    return Efield


def Lorentzscalar(k0,omegaSPP,gamma,V):
    # Simple Lorentzian polarizability mode as appropriate for a Drude sphere plasmon particle, or similar.
    # Polarizability is in units of volume 
   
    c0=2.99792458E8
    omega=k0*c0
    return omegaSPP**2/(omegaSPP**2-omega**2-1.j*omega*gamma)*V;
   
def Lorentzplasmonspherestatic(k0,omegaSPP,gamma,V):
    # Lorentzian polarizability tensor for an electric-only sphere - static
    Lorentz=Lorentzscalar(k0,omegaSPP,gamma,V)
    alpha=Lorentz*np.identity(3)
    return alpha

   
def Rayleighspherepolarizability(nsph,nslab,V):
    # Simple Rayleigh polarizability
    alpha=np.zeros([6,6],dtype=complex)
    alpha[0:3,0:3]=V*(nsph**2-nslab**2)/(nsph**2+2.0*nslab**2)*np.identity(3)
    
    alpha[3:6,3:6]=V*1E-20*np.identity(3)    
    return alpha


def invalphadynamicfromstatic(alphalist,rdip,diplayer,k0,nstack,dstack):
    # Dressed polarizability, meaning including radiation damping
    # tested using the following logic
    
    # 1/alp = 1/alp0 - G  basic
    # 1/alp = 1/alp0 - i 2/3 k^3 ldos also well known
    #
    #
    # Generalization means that on the diagonal blocks 
    # you replace G by Im G = 2/3 k*3*LDOS
    # For the magnetoelectric blocks instead you should compare to Re G
    #
    # In practice, a test routine works where you compare G/ (2/3 k^3)
    # To the ldos assignment below.
    # INcidentally, you COULD also just calculate this using only the Green function routine.
    # The purely scattering Green fucntion is not singular, and perfectly happy to evaluate at r=r'
    
    
    rdip=check.checkr(rdip)
    
    zlist=rdip[2,:]
    Ndip=zlist.size
    invalphadyn=np.zeros([Ndip,6,6],dtype=complex)
    
    ldEpar,ldEperp,ldHpar,ldHperp,ldC=ImG.LDOS(k0,zlist,nstack,dstack)  
    ldC=1.j*ldC
    
    kslab=k0*nstack[diplayer]
    for ii in range(Ndip):
                        
        GG=1.0j*np.diag([ldEpar[ii],ldEpar[ii],ldEperp[ii],ldHpar[ii],ldHpar[ii],ldHperp[ii]]) # radiation damping
        
        GG[0,4]=-ldC[ii]
        GG[1,3]=-GG[0,4]  
       
        GG[4,0]=-GG[0,4]
        GG[3,1]= GG[0,4]          
 
        invalphadyn[ii,:,:]=inv(alphalist[ii,:,:]) - 2.0/3.0*kslab**2*k0*GG
    
    return invalphadyn
 
def SetupandSolveCouplingmatrix(invalphadyn,rdip,k0,nstack,dstack):
    rdip=check.checkr(rdip)
    Ndip=rdip.shape[1]
    
    M=np.zeros([6*Ndip,6*Ndip],dtype=complex)
    
    ## Create a library of just the unique positions in the rdip difference list
    
    ## allocate the memory
    dx=np.zeros(Ndip*Ndip)
    dy=np.zeros(Ndip*Ndip)
    z1=np.zeros(Ndip*Ndip)
    z2=np.zeros(Ndip*Ndip)

    #  assemble the Delta x, Delta y, z1, z2
    for ii in range(Ndip):
        for jj in range(Ndip):
            p=np.ravel_multi_index((ii,jj),(Ndip,Ndip))
            dx[p]=rdip[0,ii]-rdip[0,jj]
            dy[p]=rdip[1,ii]-rdip[1,jj]
            z1[p]=rdip[2,ii]
            z2[p]=rdip[2,jj]
    
    #  Find the unique entries, create database of indices
    drdipuniq,idx,invlist=np.unique(np.transpose(np.array([dx,dy,z1,z2])),axis=0,return_inverse=True,return_index=True)

    #  Replace the zero-distance element, you don't need it, and it gives overflow.
    idx=np.where((drdipuniq[:,0]==0)*(drdipuniq[:,1]==0)*(drdipuniq[:,2]==drdipuniq[:,3]))
    drdipuniq[idx,0]=drdipuniq[idx,0]+1e-10 
        
    # Now create the Green function library
    rdip1=np.array([drdipuniq[:,0],drdipuniq[:,1],drdipuniq[:,2]])
    rdip2=np.array([0.0*drdipuniq[:,0],0.0*drdipuniq[:,1],drdipuniq[:,3]])
    Glibrary=Gs.Greensafe(k0,nstack,dstack,rdip1,rdip2).T
    
    for ii in range(Ndip):
        for jj in range(Ndip):
            offsetx=ii*6
            offsety=jj*6
            
            if ii==jj:
                M[offsetx+0:offsetx+6,offsety+0:offsety+6]=invalphadyn[ii,:,:]

            if ii!=jj:
                # G=-Gs.Greensafe(k0,nstack,dstack,rdip[:,ii],rdip[:,jj])
                
                # Look up in library
                p=np.ravel_multi_index((ii,jj),(Ndip,Ndip))
                M[offsetx+0:offsetx+6,offsety+0:offsety+6]=-np.squeeze(Glibrary[invlist[p],:,:]) 
             
    return M 


def Solvedipolemoments(M,drivingfield):
    # if M is the (1/a - G) matrix, and drivingfield is given, returns dipole moments
    M=inv(M)
    pm=np.matmul(M,np.transpose(drivingfield).flatten())
    return np.transpose(np.reshape(pm,np.transpose(drivingfield).shape))



def Work(pdipvecs,diplayer,drivingfield,k0,nstack):
    # work done by driving field on induced dipoles
    # note that this is not yet extinction, for which one need to divide by an intensity
    kslab=nstack[diplayer]*k0
    work= 2.*np.pi*kslab*(np.real(drivingfield)*np.imag(pdipvecs)-np.imag(drivingfield)*np.real(pdipvecs))
    return np.sum(work,0)



def Differentialradiatedfieldmanydipoles(theta,phi,pdipvecs,rdip,diplayer,k0,nstack,dstack):
    # differential radiated E-field
    rdip=check.checkr(rdip)
    thelist,philist,shape=check.checkThetaAndPhi(theta, phi)
    Ndip=rdip.shape[1]
    
    EEs=np.zeros(shape,dtype=complex);     EEp=np.zeros(shape,dtype=complex)
     
    
    for ii in range(Ndip):
        z=rdip[2,ii]
        pumu=pdipvecs[:,ii]
    
        dum,outE,f=Radpat.RadiationpatternPandField(k0, z, pumu[0:3], pumu[3:6], theta, phi, nstack, dstack,rdip[0:2,ii])
        
        EEs=EEs+outE[0]
        EEp=EEp+outE[1] 

    pre=(k0*nstack[diplayer])**2
    return np.reshape(pre*EEs,shape),np.reshape(pre*EEp,shape),np.reshape(f[0],shape),np.reshape(f[1],shape) 
 
 
 
def TotalfarfieldpowerManydipoles(pdipvecs,rdip,diplayer,k0,nstack,dstack):
     #  total upward and downward far field radiation, by integration. Not adaptive
     rdip=check.checkr(rdip)

     # guessing the resolution
     L=np.max(np.array([(np.max(rdip[0,:])-np.min(rdip[0,:])), (np.max(rdip[1,:])-np.min(rdip[1,:]))]))
     nmax=np.max(np.real(np.array([nstack[0],nstack[-1]])))
     Nf=2**(np.ceil(np.log2(k0*L*nmax+1)+4).astype('int'))  
     
     # defining the sampling points
     philist=np.arange(0,Nf)/Nf*(2.0*np.pi)
     
     def phiintegratedUP(theta,philist,pdipvecs,rdip,diplayer,k0,nstack,dstack):
         Esu,Epu,thetau,phiu=Differentialradiatedfieldmanydipoles(theta, philist, pdipvecs,rdip,diplayer,k0,nstack,dstack)
         jacobian=np.sin(theta)
         Pup=np.sum((np.abs(Esu)**2+np.abs(Epu)**2))*jacobian 
         
         return Pup

     def phiintegratedDOWN(theta,philist,pdipvecs,rdip,diplayer,k0,nstack,dstack):
         Esu,Epu,thetau,phiu=Differentialradiatedfieldmanydipoles(np.pi-theta, philist, pdipvecs,rdip,diplayer,k0,nstack,dstack)
         jacobian=np.sin(theta) 
         Pdown=np.sum((np.abs(Esu)**2+np.abs(Epu)**2))*jacobian 
         return Pdown
     
     Pu=integrate.quad(lambda x: phiintegratedUP(x,philist,pdipvecs,rdip,diplayer,k0,nstack,dstack),0,np.pi/2)     
     Pd=integrate.quad(lambda x: phiintegratedDOWN(x,philist,pdipvecs,rdip,diplayer,k0,nstack,dstack),0,np.pi/2)     
     
     pre=np.pi/Nf
     return pre*Pu[0], pre*Pd[0]



