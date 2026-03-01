# PyRAMIDS

**A Python package for Radiation, Magnetoelectric Interactions, and Dipoles in Stratified Layers**
---

## Intended Use

PyRAMIDS is a simulation framework for nanophotonics and electromagnetic modeling in planar multilayer structures. It can serve both as a teaching tool and as a research platform.

It is designed for:

- Local density of states (LDOS) engineering  
- Dipolar emitter photophysics in layered media  
- Far-field radiation pattern analysis  
- Fourier microscopy calibration and benchmarking  
- Multiple scattering of magnetoelectric dipoles  
- Device design workflows (LEDs, photovoltaics, cavities, multilayer metamaterials)  
- Inverse-design and optimization studies  

---

## Scientific Scope

PyRAMIDS implements a Green-function based angular spectrum formulation for electromagnetic sources and dipole scatterers in arbitrary 1D multilayer stacks.

The framework addresses the following fundamental problem:
> Given an electric, magnetic, or magnetoelectric dipole embedded in a stratified system,  
> - What is the total emitted or scattered power?  
> - How is this power distributed across guided and radiative channels?  
> - What is the angular and polarimetric far-field signature?  
> - How do multiple dipolar scatterers interact self-consistently within the stack?

The implementation combines a stable S-matrix formalism with a rigorous 6×6 dyadic Green-function framework, as detailed in the accompanying manuscript and mathematical Supplement.
---

## Core Engines

### 1. S-Matrix Solver – Plane-wave response.

- Stable Redheffer star-product implementation
- Complex Fresnel coefficients for arbitrary stacks
- Reflectance, transmittance, absorptance
- Layer-resolved absorption and energy balance
- Local field distributions inside stacks
- Guided-mode and evanescent wave physics at large \( $\text{k}_{\parallel}$ \)

> Applications:
> Mirrors, LED stacks, photovoltaic layers, dielectric cavities.

---

### 2. Radiation Pattern - angular information.

- Angle-resolved far-field emission  
- s–p polarization basis and Cartesian basis  
- Integrated upward and downward radiated power  
- Radiative LDOS extraction  
- Back-focal-plane (Fourier plane) imaging simulations  
- Stokes polarimetry (S0–S3)

> Applications:
> Emitter calibration, high-NA objective benchmarking, COMSOL/FDTD validation, LED outcoupling analysis.


---

### 3. Local Density of States (LDOS) Framework.

LDOS is computed via the imaginary part of the dyadic Green function:

$\[
\rho \propto \mathbf{e}_d^T \cdot \mathrm{Im}\,G(\mathbf{r},\mathbf{r}) \cdot \mathbf{e}_d
\]$

Implemented features (aka Amos & Barnes):

- Electric LDOS  
- Magnetic LDOS  
- Magnetoelectric (chiral / bianisotropic) LDOS 
- Complex-contour integration over \( $\text{k}_{\parallel}$ \)  
- Total LDOS 
- \( $\text{k}_{\parallel}$ \)-resolved modal analysis  

### 4. Dyadic Green Function Engine

- Full 6x6 magnetoelectric Green tensor
- Angular spectrum representation
- Slab-centric internal coordinate formalism
- Consistent unit system (rationalized units)
- Explicit electric/magnetic cross-coupling blocks

> Applications:
> Drexhage experiments, Purcell engineering, materials quantum efficiency extraction, LEDs and photovoltaics.

---


---

### 5. Multiple Scattering of Dipolar Particles

- Coupled-dipole formalism in layered media  
- Radiation damping corrections (dynamic polarizability dressing)  
- Extinction via work $\( \mathrm{Re}[\mathbf{j}^* \cdot \mathbf{E}] \)$  
- Scattering cross sections via far-field integration  
- Optical theorem validation  
- Electric, magnetic, and magnetoelectric polarizabilities  

> Applications:
> Metasurface design, layered nanoantenna arrays, fast co-optimization with multilayers.
> (Orders of magnitude faster than full-wave FEM/FDTD in large footprints).

---

## Architecture

```
Library/
    Core/          # S-matrix, Green function, LDOS integrands, radiation kernels
    Utility/       # Argument checking, coordinate wrappers, visualization, DE optimizer
    Use/           # High-level user interfaces

Benchmarks/
    Literature/            # Reproduction of seminal LDOS and radiation papers
    Internal_Consistency/  # Cross-validation tests

Examples/          # Reproducible figures and workflows
UserSandbox/       # Custom simulation workspace
Manual/            # Folder structure PDF documentation
```

Internally, the code uses a slab-centric representation, while users interact with global stack coordinates.

---

## Installation

PyRAMIDS is implemented in Python.

Validated development environment:

- Python 3.12.7  
- NumPy 1.26.4  
- SciPy 1.15.1  
- Numba 0.60.0  
- Matplotlib  

Install dependencies:

```bash
pip install package==version
```

Other versions are typically compatible, but only the above have been formally validated.

---

## Benchmarks and Validation

The repository includes:

- Reproduction of canonical LDOS literature results  
- Electric, magnetic, and chiral dipole benchmarks  
- Guided-mode resolved LDOS verification  
- LDOS vs far-field consistency checks  
- Optical theorem tests in multiple scattering  

These tests ensure physical consistency and numerical robustness.


---

## Authors

**Debapriya Pal**  
**A. Femius Koenderink**  

Department of Physics of Information in Matter  
Center for Nanophotonics  
NWO-I Institute AMOLF  
Amsterdam 1098 XG, The Netherlands  

Contact: f.koenderink@amolf.nl  

---

## Citation

If you use PyRAMIDS in your research, please cite:

Pal, D. & Koenderink, A. F.  
*PyRAMIDS — A Python package for Radiation, Magnetoelectric Interactions, and Dipoles in Stratified Layers*
---

## License
