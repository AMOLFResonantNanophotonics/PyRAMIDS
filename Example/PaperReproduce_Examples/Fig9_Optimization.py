#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
print('Doing Figure 9')

#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#%%
import numpy as np
import matplotlib.patches as patches

from matplotlib import pyplot as plt
from Library.Use import Use_Multiplescattering as ms
from Library.Util import Util_vectorpolarization as radplot
from Library.Util import Util_argumentchecker as check
from Library.Util import Util_differential_evolution_custom_constraints as de # Import differential evolution custom package.

from joblib import Parallel, delayed
from IPython.utils import io
from scipy.spatial.distance import pdist

#%%
# Number of CPU cores to use (1 = 100% of available cores)
num_cores = max(1, int(os.cpu_count() * 0.75))
print(f"Using {num_cores} cores.")


#%%
# Define folder paths
'''DATA .npz file containing the optimization result SAVE folder and file'''
DATA_DIR = "Fig9_OptimizationData"
os.makedirs(DATA_DIR, exist_ok=True) # Create folders if they don't exist
file_path = os.path.join(DATA_DIR, 'DE_data.npz')

'''SAVE GENERATED IMAGES containing optimized results'''
def savefig(folderpath, filename):
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    plt.savefig(os.path.join(folderpath, filename), bbox_inches='tight')
    
folder = r"pdfimages/"


#%%
# ---------- SIMPLE USER SWITCH ----------
DEFAULT_GEN = 5000
DEFAULT_SCATTERERS = 60

ans = None
try:
    ans = input("Select mode:\n"
    "  y = run optimization (slow)\n"
    "  n = load saved data & make plots\n"
    "Choice/ Type [y/n]?: ").strip().lower()
except Exception:
    pass


if ans == "y":
    RUN_OPT = True
    RUN_SAVED = False

elif ans == "n":
    RUN_OPT = False
    RUN_SAVED = True

else:
    raise ValueError("Invalid input. Please enter only 'y' or 'n'.")

    
# ---- optimization path ----
if RUN_OPT:
    
    try:
        s = input(
            f"How many generations? "
            f"[default={DEFAULT_GEN} — WARNING: long runtime, use small number for testing]: ").strip()
        max_gen = int(s) if s else DEFAULT_GEN
        
        MIN_SCATTERERS = 20
        while True:
            ss = input(
                f"How many scatterers? "
                f"[default={DEFAULT_SCATTERERS} in paper, min={MIN_SCATTERERS}]: ").strip()
            Nscatterer = int(ss) if ss else DEFAULT_SCATTERERS
        
            if Nscatterer >= MIN_SCATTERERS:
                break
            print(f"Too small: Nscatterers={Nscatterer}. Must be >= {MIN_SCATTERERS}. Try again.")
                
    except Exception:
        max_gen = DEFAULT_GEN
        Nscatterer = DEFAULT_SCATTERERS
        
    print(f"OK: running optimization for max_gen={max_gen}, scatterer number={Nscatterer}")

# ---- read data file path ----
elif RUN_SAVED:
    print("OK: not running optimization.")
    
    # ---- ask user which N ----
    s = input("Number of scatterers? (options: 40 / 60 / 80) [default in paper=60]: ").strip()
    
    Nscatterer = int(s) if s else 60

    if Nscatterer not in [40, 60, 80]:
        raise ValueError("Invalid choice. Use 40, 60, or 80.")

    print("Loading saved data and running analysis…")


    # ---- modify file path accordingly ----
    file_path = os.path.join(DATA_DIR, f"N{Nscatterer}", "DE_data.npz")


    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results file not found: {file_path}\n" f"Check that data/N{Nscatterer}/DE_data.npz exists.")
        
    data = np.load(file_path, allow_pickle=True)

    fom_hist = data["fom_hist"]     # shape (G, pop)
    pop_hist = data["pop_hist"]     # shape (G, pop, dim)
    initial_pop = data['initial_pop']
    max_idx = np.argmax( -fom_hist[-1] ) # get the population index of the best FOM...in the last generation

    
    '''# %% ''get the coordinates of the best population'''
    
    result = pop_hist[-1, max_idx, :]
    Np   = len(result)// 2 # Number of scatterers is len(x)/2, as each point x has both x,y coordinates.
    xy_coordinate_best = np.asarray(result).reshape((Np,2)) 

    '''#%% initial population'''
    
    initial_pop = initial_pop[0,:]
    Np   = len(initial_pop)// 2
    xy_coordinate_initial = np.asarray(initial_pop).reshape((Np,2)) 

    Generations = np.arange(fom_hist.shape[0]) 
    
    Merit = -fom_hist[-1, max_idx] # whats the FOM
        
    print("fom_hist shape:", fom_hist.shape)
    print("pop_hist shape:", pop_hist.shape)
    print(f"Best score (min) = {Merit:.6g}")  
    #%%
    plt.figure(dpi=500, figsize = (5,4))
    for i in range(fom_hist.shape[1]):
        plt.plot(Generations, -fom_hist[:, i], lw = 0.55, color='gray', alpha=0.7)
    plt.plot(Generations, -fom_hist[:, max_idx],color='r', label='Best Individual')

    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("FoM", fontsize=14)
    plt.legend(fontsize=14)
    plt.title("DE best score vs generation")
    plt.tick_params(axis='both', labelsize=14, direction='in')
    
    if Nscatterer == 60:
        
        plt.xlim([-20, 1020])
        plt.ylim([np.min(-fom_hist)-1, np.max(-fom_hist)+1])
        plt.tight_layout() 
        file = [folder,'Fig_9a_FOM_generations'+'.pdf']
        savefig(file[0], file[1])
        
    
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()  

#%%
plot_initial = True

#%%
'''How many scatterers'''
total_scatterers = Nscatterer
    
#%%
'''Length in microns'''
lam = 0.6 # emission wavelength
k0 = 2 * np.pi / lam

#%%
'''Layer definition'''
nstack = np.array([ 1, 0.06+ 4.3j , 1.456, 1.6, 1.0])   # 0.4 um emitter layer with one dipole (n=1.6) on lam/4 thickness glass away from 0.2 um Ag mirror in air
dstack = np.array([0.2, lam/4, 0.4])

#%%
'''Particle arrangement & properties'''
n_particle = 0.06 + 4.3j        # Silver refractive index
r_particle = 0.05

z = dstack[0] + dstack[1] + r_particle      # basically centre of sphere is r away from glass-phosphor interface

pitch = 0.4     # pitch in microns
L = 3           # Total Patch size in microns ... thats basically square patch going from -L/2 to L/2 across centre
centre_square_L = 0.15 

## this basically decides a fabrication constraint.. 50 nm (academic E-beam resolution sort of) + 2* radius of particle
min_separation = 0.05 + 2*r_particle
resolution = 0.005
print(f'Patch size is {-L/2} to {L/2} microns in one dimension, so total footprint is {L} microns x {L} microns, with centre forbidden square length of {centre_square_L} microns')

#%%
# Problem setting
drive = 'source'
print('Problem setting =', drive)
#%%
''' Define theta and phi ranges for far field output calculations '''
Nthe = 46
Nphi = 91

thelist = np.linspace(0.001 * np.pi / 180, 90 * (np.pi / 180) - 0.001, Nthe, dtype=float)
philist = np.linspace(0.001 * np.pi / 180, (2 * np.pi) - 0.001, Nphi, dtype=float)

''' Beaming within Desired Cone Angle - basically I used roughly the NA of a lensed fiber '''
cone_angle = np.arcsin(0.15)
print(f'Beaming within Desired half cone angle: {cone_angle*180/np.pi} degrees')
      
idx = thelist<= cone_angle  ## indices to be used later!!!

#%%
''' 3 perpendicular dipoles SOURCES at Origin '''  

pnmsource = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])


rsource = np.array([[0.0, 0.0, sum(dstack) - dstack[-1]/2]])
Ndipoles = rsource.shape[0]

#%% all functions for particle arrangements geometry
'''Particle Arrangement definitions'''


def mask_exclude_center_square(x, y, sq_exclude_L=0.15, margin=0.0):
    """
    Returns a boolean mask that EXCLUDES points inside the central square
    of side (sq_exclude_L + 2*margin), centered at origin.

    If margin = r_particle, then you are excluding points whose centers are within
    a square expanded by r_particle on each side (i.e., keeping particle surfaces out).
    """
    half = 0.5 * sq_exclude_L + margin
    return ~((np.abs(x) <= half) & (np.abs(y) <= half))



'''Square or Rectangle'''
def periodicarrangement_particle(half_L, z_position, pitch, sq_exclude_L=0.15, margin=0.0, shift=(0.0, 0.0)):
    """
    Generate positions for particles arranged in a square lattice,
    with x and y positions ranging from -half_L to half_L. 
    half_L should be then total length/2.. then it will be centred around origin

    Parameters:
    ----------
    L : float
        The limit for x and y coordinates (grid will cover from -L to L).
    z_position : float
        The z-coordinate for all particles.
    pitch : float
        The distance between neighboring particles.

    Returns:
    -------
    rdip : np.ndarray
        3 x N_particles array containing x, y, z positions of particles.
    """
    
    sx, sy = shift
    
    # Determine the range of x and y indices
    # Compute the indices
    x_indices = np.arange(np.floor(-half_L / pitch), np.ceil(half_L / pitch) + 1)
    y_indices = np.arange(np.floor(-half_L / pitch), np.ceil(half_L / pitch) + 1)

    # Generate grid points
    X_indices, Y_indices = np.meshgrid(x_indices, y_indices, indexing='ij')

    
    x = (X_indices * pitch + sx).ravel()
    y = (Y_indices * pitch + sy).ravel()

    # Bound mask.........Apply mask to keep positions within -L to L, keep positions within -L to L
    mask = (x >= -half_L) & (x <= half_L) & (y >= -half_L) & (y <= half_L)
    
    # exclude central square (optional)
    if sq_exclude_L is not None and sq_exclude_L > 0:
        mask &= mask_exclude_center_square(x, y, sq_exclude_L= sq_exclude_L, margin=margin)

    x = x[mask]
    y = y[mask]
    z = np.full_like(x, z_position, dtype=float)

    rdip = np.array([x, y, z])
    return rdip

'''Hexagonal'''
def hexagonalarrangement_particle(half_L, z_position, pitch, sq_exclude_L=0.15, margin=0.0, shift=(0.0, 0.0)):
    """
    Hexagonal lattice in [-L, L]x[-L, L] excluding a central square.
    """
    
    sx, sy = shift
    
    row_height = pitch * np.sqrt(3) / 2

    J_min = int(np.floor(-half_L / row_height))
    J_max = int(np.ceil(half_L / row_height))
    J_indices = np.arange(J_min, J_max + 1)

    x_shift_max = pitch / 2
    x_max = half_L + x_shift_max
    I_min = int(np.floor(-x_max / pitch))
    I_max = int(np.ceil(x_max / pitch))
    I_indices = np.arange(I_min, I_max + 1)

    I, J = np.meshgrid(I_indices, J_indices, indexing='ij')

    x_shift = (J % 2) * (pitch / 2)
    x = (pitch * I + x_shift + sx).ravel()
    y = (row_height * J + sy).ravel()

    # bounds mask
    mask = (x >= -half_L) & (x <= half_L) & (y >= -half_L) & (y <= half_L)

    # exclude central square (optional)
    if sq_exclude_L is not None and sq_exclude_L > 0:
        mask &= mask_exclude_center_square(x, y, sq_exclude_L= sq_exclude_L, margin=margin)

    x = x[mask]
    y = y[mask]
    z = np.full_like(x, z_position, dtype=float)

    return np.array([x, y, z])


'''Random arrangment'''
def random_arrangement(N, z, half_L, minseparation, sq_exclude_L=0.15, margin=0.0):
    """
    Random points in [-L,L]^2 with min separation, excluding central square.
    """
    points = []
    max_attempts = 5000 * N
    attempts = 0

    xmin, xmax, ymin, ymax = -half_L, half_L, -half_L, half_L

    # expanded exclusion half-width
    half = 0.5 * sq_exclude_L + margin

    while len(points) < N and attempts < max_attempts:
        x_rand = np.random.uniform(xmin, xmax)
        y_rand = np.random.uniform(ymin, ymax)

        # exclude central square
        if (abs(x_rand) <= half) and (abs(y_rand) <= half):
            attempts += 1
            continue

        new_point = np.array([x_rand, y_rand])

        if points:
            existing = np.array(points)
            if np.all(np.linalg.norm(existing - new_point, axis=1) >= minseparation):
                points.append(new_point)
        else:
            points.append(new_point)

        attempts += 1

    if len(points) < N:
        raise ValueError("Could not place all points with the given constraints.")

    coords = np.array(points).T
    return np.array([coords[0], coords[1], np.full(N, z, dtype=float)])


'''given xy coordinate of particle positions at a fixed z_position'''
def population_to_xy_particle(xy_coordinate_list, z_position):
    
    Np   = len(xy_coordinate_list)// 2 # Number of scatterers is len(x)/2, as each point x has both x,y coordinates.
    xy_coordinate = np.asarray(xy_coordinate_list).reshape((Np,2)) 
    
    z = np.full(Np, z_position, dtype=float) # fixed z positions here...
    
    rdip = np.array([xy_coordinate[:,0], xy_coordinate[:,1], z])      
    return check.checkr(rdip)

#%% all functions for coupling Matrix and incoherent summation with parallel execution over orientations

def compute_coupling_matrix(rdip, nstack, dstack, n_particle, radius_particle, k0):
    """
    Helper function that computes the particle dipole layer, alphalist, invalpha,
    and coupling matrix based on the particle positions (rdip).
    
    rdip: particle positions
    nstack, dstack: refractive index, and thickness of layered geometry
    n_particle, radius_particle: refractive index and radius of spherical particle
    """
    
    rdip = check.checkr(rdip)

    # Layer checker
    diplayer, Ndip = ms.dipolelayerchecker(rdip, nstack, dstack)
    
    # Compute polarizability
    V = (radius_particle) ** 3
    alpha = ms.Rayleighspherepolarizability(n_particle, nstack[diplayer], V)

    # Set alphalist for all dipoles
    alphalist = np.tile(alpha, (Ndip, 1, 1))

    # Precompute invalpha
    invalpha = ms.invalphadynamicfromstatic(alphalist, rdip, k0, nstack, dstack)
    
    # Compute coupling matrix
    couplingMatrix = ms.SetupandSolveCouplingmatrix(invalpha, rdip, k0, nstack, dstack)
    
    return rdip, Ndip, diplayer, couplingMatrix

def compute_intensity(j, Sourcedip_Orientations, Sourcedip_coords, Particledip, couplingMatrix, thetalist, philist, k0, nstack, dstack, substrate = False):

    
    '''    Compute far-field intensity for ONE source dipole position and ONE dipole orientation.

    random_coords: dipole coordinate positions
    pops intensity -
      - S0 - with particles
      - S0_no - without particles (layered system only)
      Parameters
    ----------
    j : int
        Index of the source dipole position in Sourcedip_coords.
    Sourcedip_Orientations : (6,) array-like
        Source dipole moment [px, py, pz, mx, my, mz].
        Here you use purely electric dipoles: px / py / pz.
    Sourcedip_coords : (Nsrc, 3) ndarray
        Cartesian coordinates of source dipoles.
    Particledip : (3, Np) ndarray
        Positions of all scattering particles.
    couplingMatrix : ndarray
        Precomputed particle–particle coupling matrix (independent of source).
    thetalist, philist : 1D arrays
        Polar and azimuthal angles for far-field sampling.
    k0 : float
        Free-space wave number.
    nstack, dstack : arrays
        Refractive indices and thicknesses of layered medium.
    substrate : bool
        If True, also compute emission into substrate hemisphere.
    
    Returns
    -------
    S0 : 2D ndarray
        Far-field intensity with particles.
    S0_no : 2D ndarray
        Far-field intensity without particles (layered background only).
    S0_subs, S0_no_subs : 2D ndarrays
        Same quantities for substrate hemisphere (if requested).

     
    '''
    
    try:
        rsource = Sourcedip_coords[j]
        pnmsource = np.array(Sourcedip_Orientations)

        driving = ms.Emitterdriving(pnmsource, rsource, Particledip, k0, nstack, dstack)
        pnm = ms.Solvedipolemoments(couplingMatrix, driving)
        
        rdiptotal = np.hstack([Particledip, rsource.reshape((3, 1))])  
        pnmtotal = np.hstack([pnm, np.reshape(pnmsource,(6,1))]) 
        
        
        thet, phi = np.meshgrid(thetalist, philist) # making 2d grid for theta and phi for far field intensity calculations...
        
        Esu, Epu, _, _ = ms.Differentialradiatedfieldmanydipoles(thetalist, philist, pnmtotal, rdiptotal, k0, nstack, dstack)
        S0, _, _, _, _, _ = radplot.BFPcartesianStokes(thet, phi, Esu, Epu)

        Esu_no, Epu_no, _, _ = ms.Differentialradiatedfieldmanydipoles(thetalist, philist, pnmsource.reshape((6, 1)), rsource.reshape((3, 1)), k0, nstack, dstack)
        S0_no, _, _, _, _, _ = radplot.BFPcartesianStokes(thet, phi, Esu_no, Epu_no)
        
        S0_subs = np.ones_like(thet)
        S0_no_subs = np.ones_like(thet)
        
        if substrate == True:
            thetalistd = -thetalist + np.pi
            thetd, phi = np.meshgrid(thetalistd, philist) # making 2d grid for theta and phi for far field intensity calculations...
            
            Esu_subs, Epu_subs, _, _ = ms.Differentialradiatedfieldmanydipoles(thetalistd, philist, pnmtotal, rdiptotal, k0, nstack, dstack)
            S0_subs, _, _, _, _, _ = radplot.BFPcartesianStokes(thetd, phi, Esu_subs, Epu_subs)

            Esu_no_subs, Epu_no_subs, _, _ = ms.Differentialradiatedfieldmanydipoles(thetalistd, philist, pnmsource.reshape((6, 1)), rsource.reshape((3, 1)), k0, nstack, dstack)
            S0_no_subs, _, _, _, _, _ = radplot.BFPcartesianStokes(thetd, phi, Esu_no_subs, Epu_no_subs)

        return S0, S0_no, S0_subs, S0_no_subs

    except Exception as e:
        print(f"Error in process with dipole {j}: {e}")
        return None

def incoherent_summation(Sourcedip_num, Sourcedip_Orientations, Sourcedip_coords, Particledip, CouplingMatrix, thetalist, philist, k0, nstack, dstack, CPUcores, substrate = False):
    
    """
    Perform incoherent summation over:
      (i) multiple source dipole positions, and
      (ii) multiple dipole orientations (px, py, pz).

    The summation is done at the INTENSITY (S0) level, not at the field level.

    Returns
    -------
    Enh : 2D ndarray
        Enhancement map = (sum S0_with_particles) / (sum S0_no_particles)
    """

    thet, _ = np.meshgrid(thetalist, philist)

    # Initialize total fields
    S0_total = np.zeros_like(thet)
    S0_no_total = np.zeros_like(thet)

    S0_subs_total = np.zeros_like(thet)
    S0_no_subs_total = np.zeros_like(thet)

    
    for dip_orientation in Sourcedip_Orientations:
        results = Parallel(n_jobs= CPUcores)(
            delayed(compute_intensity)(j, dip_orientation, Sourcedip_coords, Particledip, CouplingMatrix, thetalist, philist, k0, nstack, dstack, substrate = substrate) for j in range(Sourcedip_num)
        )
        # Combine results for the current orientation
        for result in results:
            if result is not None:
                S0, S0_no, S0_subs, S0_no_subs = result
                
                S0_total += S0
                S0_no_total += S0_no
                
                S0_subs_total += S0_subs
                S0_no_subs_total += S0_no_subs
                
            
    # Calculate enhancement (S0_total / S0_no_total)
    Enh = S0_total / S0_no_total
    
    if substrate == True:
        Enh_subs = S0_subs_total / S0_no_subs_total
    else: 
        Enh_subs = 0.0*S0_subs_total
        
        
    return Enh, Enh_subs, S0_total, S0_no_total, S0_subs_total, S0_no_subs_total


#%%

print('Doing Initial SQUARE Arrangement')
rdip = periodicarrangement_particle((L-2*r_particle)/2, z, pitch, sq_exclude_L= centre_square_L, margin = r_particle, shift = (0.5*pitch, 0.5*pitch) )
rdip, Ndip, diplayer, M = compute_coupling_matrix(rdip, nstack, dstack, n_particle, r_particle, k0)

print(f'Number of particles in square arrangement: {rdip.shape[1]}')

plt.figure(dpi=300, figsize = (4,4))
plt.plot(rdip[:1], rdip[1:2], 'ko')

# ---- particles not within a box of 0.15 x 0.15 um centred around origin ----
half = centre_square_L / 2
ax = plt.gca()
square = patches.Rectangle(
    (-half, -half),  # bottom-left corner
    centre_square_L, centre_square_L,
    linewidth=1.5,edgecolor='red',facecolor='none')
ax.add_patch(square)

# ---- dipole at origin ----
plt.plot(0, 0, marker='x', color='blue', markersize=6, mew=2)
# --------------------------

plt.xlabel("$x$ [$\\mu$m]", fontsize = 14)
plt.ylabel("$y$ [$\\mu$m]", fontsize = 14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title('Reference square arrangement')
plt.axis('equal')
plt.xlim([-L/2, L/2])
plt.ylim([-L/2, L/2])

xticks = [-L/2, 0, L/2]
yticks = [-L/2, 0, L/2]

ax = plt.gca()  # Get the current Axes object
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels([f'{tick:.1f}' for tick in xticks])
ax.set_yticklabels([f'{tick:.1f}' for tick in yticks])

ax.tick_params(axis='both', labelsize=14, direction='in')
plt.tight_layout() 

if Nscatterer == 60:        
    file = [folder,' Fig_9b_square_arrangement_ref'+'.pdf']
    savefig(file[0], file[1])

plt.show(block=False)
plt.pause(0.5)
plt.close()

# Parallelize over dipole positions (j) for each orientation
Enh_up, _, _, _, _, _ = incoherent_summation(Ndipoles, pnmsource, rsource, rdip, M, thelist, philist, 
                                                      k0, nstack, dstack, num_cores, substrate = False)

P_total_up_square = np.sum(Enh_up * np.sin(thelist)  * np.diff(thelist)[0] * np.diff(philist)[0])
P_cone_up_square = np.sum(Enh_up[:,idx] * np.sin(thelist[idx])  * np.diff(thelist)[0] * np.diff(philist)[0])

if plot_initial:
    
    the, ph = np.meshgrid(thelist, philist)
    kx = np.cos(ph)*np.sin(the)*np.real(nstack[-1])
    ky = np.sin(ph)*np.sin(the)*np.real(nstack[-1])
    
    fig,ax=plt.subplots(figsize=(6,5))
    s=Enh_up.max()
    pcm = ax.pcolormesh(kx, ky, Enh_up, vmin=0, vmax=s, cmap='inferno', shading='gouraud')
    ax.set_title('Square Arrangement: Enhancement towards Air')
    ax.set_aspect('equal')
    ax.set_xlim([np.min(kx)-.05,np.max(kx)+0.05])
    ax.set_ylim([np.min(ky)-.05,np.max(ky)+0.05])
    ax.set_aspect('equal')
    
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_ticks([0, 0.5*s, s])
    cbar.set_ticklabels([f'{0:.1f}', f'{0.5*s:.2f}', f'{s:.2f}'])
    cbar.ax.tick_params(labelsize=16)

    ax.set_xlabel('k$_{x}$/k$_{0}$',fontsize = 16)
    ax.set_ylabel('k$_{y}$/k$_{0}$',fontsize = 16)
  
    xticks = [-1, 0, 1]
    yticks = [-1, 0, 1]

    ax = plt.gca()  # Get the current Axes object
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f'{tick:.1f}' for tick in xticks])
    ax.set_yticklabels([f'{tick:.1f}' for tick in yticks])
    
    ax.tick_params(axis='both', labelsize=14, direction='in')
    plt.tight_layout() 

    if Nscatterer == 60:    
        file = [folder,' Fig_9c_square_arrangement_emission_ref'+'.pdf']
        savefig(file[0], file[1])
    
    plt.show(block=False)
    plt.close()
    
#%%
print('Doing Initial HEXAGONAL Arrangement')
rdip = hexagonalarrangement_particle((L-2*r_particle)/2, z, pitch, sq_exclude_L=centre_square_L, margin = r_particle, shift= (0.5*pitch, 0))
rdip, Ndip, diplayer, M = compute_coupling_matrix(rdip, nstack, dstack, n_particle, r_particle, k0)


print(f'Number of particles in hexagonal arrangement: {rdip.shape[1]}')

plt.figure(dpi=300, figsize = (4,4))
# plt.plot(rdip[:2,:].T[:,0], rdip[:2,:].T[:,1], 'ko')
plt.plot(rdip[:1], rdip[1:2], 'ko')

# ---- particles not within a box of 0.15 x 0.15 um centred around origin ----
half = centre_square_L / 2
ax = plt.gca()
square = patches.Rectangle(
    (-half, -half),  # bottom-left corner
    centre_square_L, centre_square_L,
    linewidth=1.5, edgecolor='red',facecolor='none')
ax.add_patch(square)

# ---- dipole at origin ----
plt.plot(0, 0, marker='x', color='blue', markersize=6, mew=2)
# --------------------------

ax.add_patch(square)
plt.xlabel("$x$ [$\\mu$m]", fontsize = 14)
plt.ylabel("$y$ [$\\mu$m]", fontsize = 14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title('Reference hexagonal arrangement')
plt.axis('equal')
plt.xlim([-L/2, L/2])
plt.ylim([-L/2, L/2])

xticks = [-L/2, 0, L/2]
yticks = [-L/2, 0, L/2]

ax = plt.gca()  # Get the current Axes object
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels([f'{tick:.1f}' for tick in xticks])
ax.set_yticklabels([f'{tick:.1f}' for tick in yticks])

ax.tick_params(axis='both', labelsize=14, direction='in')
plt.tight_layout() 

if Nscatterer == 60:    
    file = [folder,' Fig_9d_hexagonal_arrangement_ref'+'.pdf']
    savefig(file[0], file[1])

plt.show(block=False)
plt.pause(0.5)
plt.close()
# #%%

# Parallelize over dipole positions (j) for each orientation
Enh_up, _, _, _, _, _ = incoherent_summation(Ndipoles, pnmsource, rsource, rdip, M, thelist, philist, 
                                                      k0, nstack, dstack, num_cores, substrate = False)


P_total_up_hexagonal = np.sum(Enh_up * np.sin(thelist)  * np.diff(thelist)[0] * np.diff(philist)[0])
P_cone_up_hexagonal = np.sum(Enh_up[:,idx] * np.sin(thelist[idx])  * np.diff(thelist)[0] * np.diff(philist)[0])

if plot_initial:
    the, ph = np.meshgrid(thelist, philist)
    kx = np.cos(ph)*np.sin(the)*np.real(nstack[-1])
    ky = np.sin(ph)*np.sin(the)*np.real(nstack[-1])
    
    fig,ax=plt.subplots(figsize=(6,5))
    s=Enh_up.max()
    pcm = ax.pcolormesh(kx, ky, Enh_up, vmin=0, vmax=s, cmap='inferno', shading='gouraud')
    ax.set_title('Hexagonal Arrangement: Enhancement towards Air')
    ax.set_aspect('equal')
    ax.set_xlim([np.min(kx)-.05,np.max(kx)+0.05])
    ax.set_ylim([np.min(ky)-.05,np.max(ky)+0.05])
    ax.set_aspect('equal')
    
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_ticks([0, 0.5*s, s])
    cbar.set_ticklabels([f'{0:.1f}', f'{0.5*s:.2f}', f'{s:.2f}'])
    cbar.ax.tick_params(labelsize=16)

    ax.set_xlabel('k$_{x}$/k$_{0}$',fontsize = 16)
    ax.set_ylabel('k$_{y}$/k$_{0}$',fontsize = 16)
  
    xticks = [-1, 0, 1]
    yticks = [-1, 0, 1]

    ax = plt.gca()  # Get the current Axes object
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f'{tick:.1f}' for tick in xticks])
    ax.set_yticklabels([f'{tick:.1f}' for tick in yticks])
    

    ax.tick_params(axis='both', labelsize=14, direction='in')
    plt.tight_layout() 
    
    if Nscatterer == 60:        
        file = [folder,' Fig_9e_hexagonal_arrangement_emission_ref'+'.pdf']
        savefig(file[0], file[1])
    
    plt.show(block=False)
    plt.close()
    
    
#%%
## basically total_UP and cone towards air -> you use the maximum of two as reference
P_total_up_reference = max(P_total_up_square, P_total_up_hexagonal)
P_cone_up_reference = max(P_cone_up_square, P_cone_up_hexagonal)

    
#%%
if RUN_SAVED:
    print('------  READING SAVED OPTIMIZATION RUN and CALCULTE  ----------')

    '''#%% get the coordinates of the best population'''
    result = pop_hist[-1, max_idx, :]
    Np   = len(result)// 2 # Number of scatterers is len(x)/2, as each point x has both x,y coordinates.
    xy_coordinate_best = np.asarray(result).reshape((Np,2)) 


    '''#%% initial population'''
    print(initial_pop.shape)
    print(f'Number of particles in random arrangement: {int(initial_pop.shape[0]/2)}')
    initial_pop = initial_pop
    Np   = len(initial_pop)// 2
    xy_coordinate_initial = np.asarray(initial_pop).reshape((Np,2)) 
    
    
    plt.figure(dpi=500, figsize = (4,4))
    plt.plot(xy_coordinate_initial[:,0], xy_coordinate_initial[:,1], 'ko', markersize = 8)
    # ---- particles not within a box of 0.15 x 0.15 um centred around origin ----
    half = centre_square_L / 2
    ax = plt.gca()
    square = patches.Rectangle(
        (-half, -half),  # bottom-left corner
        centre_square_L, centre_square_L,
        linewidth=1.5, edgecolor='red',facecolor='none')
    ax.add_patch(square)

    # ---- dipole at origin ----
    plt.plot(0, 0, marker='x', color='blue', markersize=6, mew=2)
    # --------------------------
    plt.xlabel("$x$ [$\\mu$m]", fontsize = 14)
    plt.ylabel("$y$ [$\\mu$m]", fontsize = 14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.title('Initial random arrangement')
    plt.axis('equal')
    plt.xlim([-L/2, L/2])
    plt.ylim([-L/2, L/2])
 
    xticks = [-L/2, 0, L/2]
    yticks = [-L/2, 0, L/2]
 
    ax = plt.gca()  # Get the current Axes object
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f'{tick:.1f}' for tick in xticks])
    ax.set_yticklabels([f'{tick:.1f}' for tick in yticks])
    
    ax.tick_params(axis='both', labelsize=14)
    ax.tick_params(axis='both', which='major', direction='in', length=4, width=1.)  # Major ticks
    
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()
    
    
    plt.figure(dpi=500, figsize = (4,4))
    plt.plot(xy_coordinate_best[:,0], xy_coordinate_best[:,1], 'ro', markersize = 8)
    # ---- particles not within a box of 0.15 x 0.15 um centred around origin ----
    half = centre_square_L / 2
    ax = plt.gca()
    square = patches.Rectangle(
        (-half, -half),  # bottom-left corner
        centre_square_L, centre_square_L,
        linewidth=1.5, edgecolor='red',facecolor='none')
    ax.add_patch(square)

    # ---- dipole at origin ----
    plt.plot(0, 0, marker='x', color='blue', markersize=6, mew=2)
    # --------------------------
    plt.xlabel("$x$ [$\\mu$m]", fontsize = 14)
    plt.ylabel("$y$ [$\\mu$m]", fontsize = 14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.title('Optimized arrangement')
    plt.axis('equal')
    plt.xlim([-L/2, L/2])
    plt.ylim([-L/2, L/2])
 
    xticks = [-L/2, 0, L/2]
    yticks = [-L/2, 0, L/2]
 
    ax = plt.gca()  # Get the current Axes object
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f'{tick:.1f}' for tick in xticks])
    ax.set_yticklabels([f'{tick:.1f}' for tick in yticks])
    
    ax.tick_params(axis='both', labelsize=14, which='major', direction='in', length=4, width=1.)  # Major ticks
    plt.tight_layout() 
    
    if Nscatterer == 60:        
        file = [folder,' Fig_9f_optimized_arrangement'+'.pdf']
        savefig(file[0], file[1])
    
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()


    plt.figure(dpi=600, figsize = (4,4))
    plt.plot(xy_coordinate_best[:,0], xy_coordinate_best[:,1], 'ro', markersize = 8)
    plt.plot(xy_coordinate_initial[:,0], xy_coordinate_initial[:,1], 'ko', alpha=0.6, markersize = 8, rasterized = True)
    # ---- particles not within a box of 0.15 x 0.15 um centred around origin ----
    half = centre_square_L / 2
    ax = plt.gca()
    square = patches.Rectangle(
        (-half, -half),  # bottom-left corner
        centre_square_L, centre_square_L,
        linewidth=1.5,edgecolor='red',facecolor='none')
    ax.add_patch(square)

    # ---- dipole at origin ----
    plt.plot(0, 0, marker='x', color='blue', markersize=6, mew=2)
    # --------------------------
    plt.xlabel("$x$ [$\\mu$m]", fontsize = 14)
    plt.ylabel("$y$ [$\\mu$m]", fontsize = 14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.title('Arrangments overlap (black: initial, red: optimized)')
    plt.axis('equal')
    plt.xlim([-L/2, L/2])
    plt.ylim([-L/2, L/2])
 
    xticks = [-L/2, 0, L/2]
    yticks = [-L/2, 0, L/2]
 
    ax = plt.gca()  # Get the current Axes object
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f'{tick:.1f}' for tick in xticks])
    ax.set_yticklabels([f'{tick:.1f}' for tick in yticks])
    ax.tick_params(axis='both', labelsize=14, which='major', direction='in', length=4, width=1.)  # Major ticks
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()
    
    # Find the minimum distance
    from scipy.spatial import distance_matrix
    dist_matrix = distance_matrix(xy_coordinate_best, xy_coordinate_best)
    np.fill_diagonal(dist_matrix, np.inf)
    min_distance = np.min(dist_matrix)
    print(f"The minimum distance between any two points is: {min_distance}")

    if min_distance >= min_separation - resolution:
        print("Minimum separation constraint satisfied.")
    else:
        print("Warning: Minimum separation constraint violated!")
    
    
    rdip_opt = population_to_xy_particle(result, z)
    rdip_opt, Ndip, diplayer, M = compute_coupling_matrix(rdip_opt, nstack, dstack, n_particle, r_particle, k0)

    Enh_up, _, _, _, _, _ = incoherent_summation(Ndipoles, pnmsource, rsource, rdip_opt, M, thelist, philist, 
                                                           k0, nstack, dstack, num_cores, substrate = False)

    if plot_initial:    
        the, ph = np.meshgrid(thelist, philist)
        kx = np.cos(ph)*np.sin(the)*np.real(nstack[-1])
        ky = np.sin(ph)*np.sin(the)*np.real(nstack[-1])
        
        fig,ax=plt.subplots(figsize=(6,5))
        s=Enh_up.max()
        pcm = ax.pcolormesh(kx, ky, Enh_up, vmin=0, vmax=s, cmap='inferno', shading='gouraud')
        ax.set_title('Best Arrangement: Enhancement towards Air')
        ax.set_aspect('equal')
        ax.set_xlim([np.min(kx)-.05,np.max(kx)+0.05])
        ax.set_ylim([np.min(ky)-.05,np.max(ky)+0.05])
        ax.set_aspect('equal')
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.ax.tick_params(labelsize=16)        
        kx_circle = np.cos(ph) * np.sin(cone_angle) * np.real(nstack[-1])
        ky_circle = np.sin(ph) * np.sin(cone_angle) * np.real(nstack[-1])
        ax.plot(kx_circle, ky_circle, color='white', linestyle='--', linewidth=2, label=f'Theta = {cone_angle*180/np.pi}°')
    
        ax.set_xlabel('k$_{x}$/k$_{0}$',fontsize = 16)
        ax.set_ylabel('k$_{y}$/k$_{0}$',fontsize = 16)
        ax.tick_params(axis='both', labelsize=16, direction='in')
        plt.tight_layout() 
        
        if Nscatterer == 60:        
            file = [folder,' Fig_9g_optimized_arrangement_emission'+'.pdf']
            savefig(file[0], file[1])
        
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    
    P_total_up_opt = np.sum(Enh_up * np.sin(thelist)  * np.diff(thelist)[0] * np.diff(philist)[0])
    P_cone_up_opt = np.sum(Enh_up[:,idx] * np.sin(thelist[idx])  * np.diff(thelist)[0] * np.diff(philist)[0])
    
    print(f'% increase with desired cone compared to reference: {100*(P_cone_up_opt - P_cone_up_reference)/P_cone_up_reference}')

#%%
if RUN_OPT:    
    
    print('------  STARTED OPTIMIZATION  ----------')
    print('Doing Initial random Arrangement')
    
    #%%
    rdip = random_arrangement(total_scatterers, z, 0.5*L-r_particle, min_separation, sq_exclude_L= 3*centre_square_L, margin=r_particle)
    rdip, Ndip, diplayer, M = compute_coupling_matrix(rdip, nstack, dstack, n_particle, r_particle, k0)
    
    print(f'Number of particles in random arrangement: {rdip.shape[1]}')
    
    plt.figure(dpi=300, figsize = (4,4))
    plt.plot(rdip[:1], rdip[1:2], 'ko')
    # ---- particles not within a box of 0.15 x 0.15 um centred around origin ----
    half = centre_square_L / 2
    ax = plt.gca()
    square = patches.Rectangle(
        (-half, -half),  # bottom-left corner
        centre_square_L, centre_square_L,
        linewidth=1.5, edgecolor='red',facecolor='none')
    ax.add_patch(square)
    
    # ---- dipole at origin ----
    plt.plot(0, 0, marker='x', color='blue', markersize=6, mew=2)
    # --------------------------
    plt.xlabel("$x$ [$\\mu$m]", fontsize = 14)
    plt.ylabel("$y$ [$\\mu$m]", fontsize = 14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.title('Reference square arrangement')
    plt.axis('equal')
    plt.xlim([-L/2, L/2])
    plt.ylim([-L/2, L/2])
    
    xticks = [-L/2, 0, L/2]
    yticks = [-L/2, 0, L/2]
    
    ax = plt.gca()  # Get the current Axes object
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f'{tick:.1f}' for tick in xticks])
    ax.set_yticklabels([f'{tick:.1f}' for tick in yticks])
    
    ax.tick_params(axis='both', labelsize=14)
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()
    
    # Parallelize over dipole positions (j) for each orientation    
    Enh_up, _, _, _, _, _ = incoherent_summation(Ndipoles, pnmsource, rsource, rdip, M, thelist, philist, 
                                                          k0, nstack, dstack, num_cores, substrate = False)
    
    
    P_total_up_random = np.sum(Enh_up * np.sin(thelist)  * np.diff(thelist)[0] * np.diff(philist)[0])
    P_cone_up_random = np.sum(Enh_up[:,idx] * np.sin(thelist[idx])  * np.diff(thelist)[0] * np.diff(philist)[0])
    
    if plot_initial:
        the, ph = np.meshgrid(thelist, philist)
        kx = np.cos(ph)*np.sin(the)*np.real(nstack[-1])
        ky = np.sin(ph)*np.sin(the)*np.real(nstack[-1])
        
        fig,ax=plt.subplots(figsize=(6,5))
        s=Enh_up.max()
        pcm = ax.pcolormesh(kx, ky, Enh_up, vmin=0, vmax=s, cmap='inferno', shading='gouraud')
        ax.set_title('Random Arrangement: Enhancement towards Air')
        ax.set_aspect('equal')
        ax.set_xlim([np.min(kx)-.05,np.max(kx)+0.05])
        ax.set_ylim([np.min(ky)-.05,np.max(ky)+0.05])
        ax.set_aspect('equal')
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.ax.tick_params(labelsize=16)        
        ax.tick_params(axis='both', labelsize=16)   
        ax.set_xlabel('k$_{x}$/k$_{0}$',fontsize = 16)
        ax.set_ylabel('k$_{y}$/k$_{0}$',fontsize = 16)
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    
    #%%
    '''SIMPLE MERIT FUNCTION DEFINITION... just improve integrated power within directional cone '''
    Merit = (P_cone_up_random/P_cone_up_reference) 
    print(f'Given Initial Random Arrangement Merit: {Merit}')#%%
    
    #%%
    ## Initialize for optimization
    bound_length = L
    centre_square_length = centre_square_L
    z_position = z
    nstack = nstack
    dstack = dstack
    ref_particle = n_particle 
    rad_particle = r_particle
    k0 = k0
    num_dipoles = Ndipoles
    dip_orientations = pnmsource
    dip_coords = rsource
    polar_angle = thelist
    azimu_angle = philist
    cores = num_cores
    emission_cone = cone_angle
    seperation_constraint = min_separation
    P_target_initial = P_cone_up_reference    
    
    
    
    args = (bound_length,
        centre_square_length,    
        z_position,
        nstack,
        dstack,
        ref_particle,
        rad_particle,
        k0,
        num_dipoles,
        dip_orientations,
        dip_coords,
        polar_angle,
        azimu_angle,
        cores,
        emission_cone,
        seperation_constraint,
        P_target_initial)
    
    
    def violates_constraints(rdip, half_L, min_sep, sq_exclude_L, margin):
        # bounds (xy)
        if (np.any(rdip[0, :] < -half_L) or np.any(rdip[0, :] > half_L) or
            np.any(rdip[1, :] < -half_L) or np.any(rdip[1, :] > half_L)):
            return True
    
        # exclude central square
        half = 0.5*sq_exclude_L + margin
        
        if np.any((np.abs(rdip[0, :]) <= half) & (np.abs(rdip[1, :]) <= half)):
            return True
    
        # separation in xy
        d = pdist(rdip[:2, :].T)
        if np.any(d < min_sep):
            return True
    
        return False
    
        
    def FOM(x, *args):
        (bound_length,
         centre_square_length,    
         z_position,
         nstack,
         dstack,
         ref_particle,
         rad_particle,
         k0,
         num_dipoles,
         dip_orientations,
         dip_coords,
         polar_th,
         azimu_ph,
         cores,
         emission_cone,
         seperation_constraint,
         P_target_initial) = args
        
        rdip =  population_to_xy_particle(x, z_position)
        
        ## cheap checks first: constraint
        if violates_constraints(rdip, (0.5*bound_length - r_particle), seperation_constraint - resolution, 
                                                                        sq_exclude_L= centre_square_length, margin=rad_particle):
            return 1e2  # geo
        
        # expensive part after passing geometry
        rdip, Ndip, diplayer, M = compute_coupling_matrix(rdip, nstack, dstack, ref_particle, rad_particle, k0)
        
        Enh_up, _, _, _, _, _ = incoherent_summation( num_dipoles, dip_orientations, dip_coords, rdip, M,
                                                     polar_th, azimu_ph, k0, nstack, dstack, cores, substrate=False)
        
        
        
        if np.isnan(Enh_up).any() or np.isinf(Enh_up).any():
            return 1e6      # coupling
        
        idx = polar_th <= emission_cone
        dth = np.diff(polar_th)[0]
        dph = np.diff(azimu_ph)[0]
        
        P_total = np.sum(Enh_up * np.sin(polar_th) * dth * dph)
        P_cone  = np.sum(Enh_up[:, idx] * np.sin(polar_th[idx]) * dth * dph)
        
        if P_total <= 0 or P_cone <= 0:
            return 1e6      # naninf
        
        Merit = (P_cone / P_target_initial)
        return -Merit
    
    
    def FOM_no_output(x, *args):
        with io.capture_output():
            return FOM(x, *args)
    
    #%%
    ''' ---- optimization setup ---- '''
    bounds = [(-(0.5*bound_length - rad_particle), (0.5*bound_length - rad_particle))] * (2*total_scatterers)  # Both x and y bounds for each particle, thats why 2
    
    pop_size = int(0.1*2*total_scatterers)   ## so totaldimensionsility 2N... pop size is 10% of 2N.. small to be fair
    
    initial_population = []
    
    candidate_solution = rdip[:2, :].T.flatten()
    initial_population.append(candidate_solution)
    initial_population = np.array(initial_population)
    
    print(f"TOTAL No. of SCATTERERS: {total_scatterers}")
    #%%
    output_filename = file_path
    
    de = de.DifferentialEvolution(
        bounds=bounds,
        pop_size=pop_size,
        max_generations=max_gen,
        merit_func=FOM_no_output,
        args=args,
        outputFilename=output_filename,
        dipole_locations=dip_coords,
        num_cores=num_cores,
        F=0.5,
        CR=0.7,
        min_separation=seperation_constraint,
        initial_population= initial_population,
        step_size=resolution,
        exclude_square_side= centre_square_length,
        exclude_margin= rad_particle)
    
    print("DE OPTIMIZATION starts running")
    # Run the optimization
    de.optimize()