# PyRAMIDS

**A Python package for modeling electromagnetic sources and scatterers in complex stratified multilayer stacks**

---

## Overview

PyRAMIDS is a research-oriented Python package for modeling radiation, local density of states (LDOS), magnetoelectric dipoles, and multiple scattering in planar stratified media.

The framework provides:

- Plane-wave response of arbitrary multilayer stacks via S-matrix formalism  
- Electric, magnetic, and magnetoelectric LDOS  
- Angle-resolved far-field radiation patterns  
- Dyadic Green functions in layered media  
- Multiple scattering of dipolar particles embedded in stratified systems  
- Internal benchmark and validation routines  

The package is designed to be:

- Transparent for experts extending the mathematical framework  
- Accessible for users performing simulations or inverse design  

A detailed PDF manual describing the architecture and mathematical background is included in this repository.

---

## Authors

**Debapriya Pal**  
**A. Femius Koenderink**  

Department of Physics of Information in Matter  
Center for Nanophotonics  
NWO-I Institute AMOLF  
Amsterdam, The Netherlands  

Contact: f.koenderink@amolf.nl  

---

## Installation

PyRAMIDS is implemented in Python.

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

or via a `requirements.txt` file if provided.

The listed versions correspond to the validated development environment.  
The package is generally compatible with earlier and newer versions of these libraries, although only the versions specified above have been formally tested.

---

## Repository Structure

```
Library/
    Core/          # Low-level slab-centric physics implementations
    Utility/       # Argument handling, wrappers, visualization, optimization
    Use/           # User-facing high-level interfaces

Benchmarks/
    Literature/            # Reproduces published results
    Internal_Consistency/  # Cross-validation routines

Examples/          # Usage demonstrations and figure reproduction
UserSandbox/       # User workspace and batch execution scripts
Manual/            # PDF documentation
```

---

## Key Capabilities

### Plane-Wave Optics

- Reflectance, transmittance, absorptance  
- Layer-resolved absorption  
- Field distributions inside stacks  

### LDOS Calculations

- Electric and magnetic LDOS  
- Magnetoelectric cross terms  
- Arbitrary dipole orientations  
- Modal analysis via k∥ integrands  

### Radiation Patterns

- s–p basis far-field emission  
- Integrated radiated power  
- Radiative LDOS extraction  

### Multiple Scattering

- Coupled dipole interactions in layered media  
- Radiation damping corrections  
- Extinction and scattering cross sections  
- Optical theorem validation  

---

## Benchmarks and Validation

The repository includes:

- Reproduction of established literature results  
- Internal consistency checks:
  - LDOS vs radiation pattern integration  
  - Green function comparisons  
  - Optical theorem validation  
  - Magnetoelectric dipole symmetry checks  

These tests ensure physical consistency and numerical robustness.

---

## Intended Use

PyRAMIDS is intended for:

- Nanophotonics research  
- Stratified optical system modeling  
- Dipolar emitter engineering  
- Inverse design workflows  

It is a research code and assumes familiarity with electromagnetic theory in layered media.

---

## Citation

If you use PyRAMIDS in your research, please cite:

(Insert thesis / paper citation here)

---

## License

(Insert license information — e.g., MIT / GPL / academic license)