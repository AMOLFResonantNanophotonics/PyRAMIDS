# PyRAMIDS

**A Python package for Radiation, Magnetoelectric Interactions, and Dipoles in Stratified Layers**
---

## Intended Use

PyRAMIDS is a simulation framework for nanophotonics and electromagnetic modeling in planar multilayer structures.

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

PyRAMIDS implements a Green-function based angular spectrum formulation for electromagnetic sources and scatterers in arbitrary 1D multilayer stacks.

The framework solves the following core problem:

> Given an electric, magnetic, or magnetoelectric dipole embedded in a stratified system,  
> - What is the total emitted or scattered power?  
> - How is this power distributed across guided and radiative channels?  
> - What is the angular and polarimetric far-field signature?  
> - How do multiple dipolar scatterers interact self-consistently within the stack?

The implementation follows the S-matrix formalism combined with a rigorous dyadic Green-function approach as detailed in the main paper article and its mathematical Supplement.

---

## Core Capabilities

### 1. Plane-Wave Multilayer Optics (S-Matrix Formalism)

- S-matrix implementation (Redheffer star product)
- Complex Fresnel coefficients for arbitrary stacks
- Reflectance, transmittance, absorptance
- Layer-resolved absorption and energy balance
- Local field distributions inside stacks
- Guided-mode and evanescent wave physics at large \($\text{k}_{\parallel}$ \)

---

### 2. Local Density of States (LDOS)

LDOS is computed via the imaginary part of the dyadic Green function:

$\[
\rho \propto \mathbf{e}_d^T \cdot \mathrm{Im}\,G(\mathbf{r},\mathbf{r}) \cdot \mathbf{e}_d
\]$

Implemented features:

- Electric LDOS  
- Magnetic LDOS  
- Magnetoelectric (chiral / bianisotropic) LDOS  
- Total and radiative LDOS separation  
- Complex-contour integration over \($\text{k}_{\parallel}$ \)  
- \($\text{k}_{\parallel}$ \)-resolved modal analysis  
- Guided-mode and surface-polariton contributions  

This enables direct insight into Purcell enhancement, guided-mode coupling, and non-radiative channels.

---

### 3. Radiation Patterns and Fourier Microscopy

- Angle-resolved far-field emission  
- s–p polarization basis and Cartesian basis  
- Integrated upward and downward radiated power  
- Radiative LDOS extraction  
- Back-focal-plane (Fourier plane) imaging simulations  
- Stokes polarimetry (S0–S3)

The package provides exact multilayer radiation patterns suitable as ground truth for calibrating high-NA Fourier microscopes.

---

### 4. Dyadic Green Function Engine

- Full 6×6 magnetoelectric Green tensor
- Angular spectrum representation
- Slab-centric internal coordinate formalism
- Consistent unit system (rationalized units)
- Explicit electric/magnetic cross-coupling blocks

---

### 5. Multiple Scattering of Dipolar Particles

- Self-consistent coupled-dipole formalism in layered media  
- Radiation damping corrections (dynamic polarizability dressing)  
- Extinction via work $\( \mathrm{Re}[\mathbf{j}^* \cdot \mathbf{E}] \)$  
- Scattering cross sections via far-field integration  
- Optical theorem validation  
- Electric, magnetic, and magnetoelectric polarizabilities  

The formulation is fully Green-function based and avoids near-to-far field transformation approximations that fail near interfaces.

This enables fast and physically explainable device-scale modeling (orders of magnitude faster than full-wave FEM/FDTD in large footprints).

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
Manual/            # Full PDF documentation
```

The internal implementation uses a slab-centric representation, while the user interacts in global stack coordinates.

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

The listed versions correspond to the validated development environment.  
The package is typically compatible with earlier and newer versions, although only the versions above have been formally tested.

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

## Didactic and Research Applications

PyRAMIDS supports:

- Purcell effect exploration  
- Drexhage-type mirror experiments  
- Guided-mode physics in multilayers  
- Hyperbolic metamaterial LDOS  
- Kerker and Huygens dipoles  
- Chiral and pseudochiral emitters  
- Device-scale LED and photovoltaic modeling  
- Fourier microscopy calibration  

It can serve both as a teaching tool and as a research platform.

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

## Citation

If you use PyRAMIDS in your research, please cite:

Pal, D. & Koenderink, A. F.  
*PyRAMIDS — A Python package for Radiation, Magnetoelectric Interactions, and Dipoles in Stratified Layers*
---

## License

(Add license information here)