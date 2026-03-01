# PyRAMIDS

**A Python package for modeling electromagnetic sources and scatterers in complex stratified multilayer stacks**

---
## Intended Use

PyRAMIDS is a Python package for modeling radiation, local density of states (LDOS), magnetoelectric dipoles, and multiple scattering in planar stratified media. It is intended for:

- Optical system modeling in layered structures  
- Dipolar emitter engineering  
- Stratified metasurface and multilayer analysis  
- Multiple-scattering simulations  
- Inverse-design and parameter-optimization workflows  

---



## Overview

The framework provides:

- Plane-wave response of arbitrary multilayer stacks via S-matrix formalism  
- Electric, magnetic, and magnetoelectric LDOS (ImG definition aka Amos & Barnes)  
- Angle-resolved far-field radiation patterns  
- Dyadic Green functions in layered media  
- Multiple scattering of dipolar particles embedded in stratified systems  
- Internal benchmark and validation routines  

The package is designed to be:

- Transparent for experts with access to the mathematical framework and intends to extend it 
- Accessible for users to perform optical system design simulations  

A detailed PDF (Manual) describing the architecture and mathematical background is included in this repository.

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

## Installation

PyRAMIDS is scripted in Python.

The development and testing environment uses:

- Python 3.12.7  
- NumPy 1.26.4  
- SciPy 1.15.1  
- Numba 0.60.0  
- Matplotlib  

Install dependencies using:

```bash
pip install package==version
```

The listed versions correspond to the validated development environment.  
The package is generally compatible with earlier and newer versions of these libraries, although only the versions specified above have been formally tested.

---
## Repository Structure

```
Library/
    Core/          # Low-level slab-centric physics implementations - mathematical formulations
    Utility/       # Argument handling, verification, wrappers, visualization, custom DE optimization
    Use/           # User–facing high-level interfaces

Benchmarks/
    Literature/            # Reproduces published results
    Internal_Consistency/  # Cross-validation routines

Examples/          # Usage demonstrations and figure reproduction
UserSandbox/       # User workspace and batch execution scripts
Manual/            # PDF documentation
```

---

## Key Capabilities

### Plane-Wave Illumination

- Complex Fresnel coefficients via S-matrix formalism  
- Reflectance, transmittance, and absorptance  
- Layer-resolved absorption profiles  
- Spatial field distributions inside multilayer stacks  

### LDOS and Green-Function Analysis

- Electric and magnetic LDOS  
- Magnetoelectric cross terms  
- Arbitrary dipole orientations  
- Total LDOS via complex contour integration  
- Modal analysis through $k_{\parallel}$-resolved integrands  

### Radiation Patterns

- Far-field emission in s–p or Cartesian basis  
- Angular power distribution  
- Integrated upward and downward radiated power  
- Radiative LDOS extraction via angular integration  

### Multiple Scattering

- Coupled dipole interactions in layered media  
- Radiation damping and dressed polarizability corrections  
- Extinction and scattering cross sections  
- Optical theorem validation in stratified environments  

---

## Benchmarks and Validation

The repository includes:

- Reproduction of established literature results  
- Internal consistency checks:
  - LDOS vs radiation pattern integration  
  - Green-function comparisons  
  - Optical theorem validation  
  - Magnetoelectric dipole symmetry checks  

These tests ensure physical consistency and numerical robustness across the framework.

---
---

## Citation

If you use PyRAMIDS in your research, please cite:

****#########****
---

## License
