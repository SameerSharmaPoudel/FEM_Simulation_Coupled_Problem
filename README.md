# Simulation of Coupled Problems with FEM: Electromechanical Coupling

## Overview
This repository contains the numerical implementation and Finite Element Method (FEM) simulation scripts for modeling **electromechanical coupling** in smart materials (specifically barium titanate, BaTiO₃). The project evaluates coupled mechanical deformation and electrical potential fields using non-linear governing equations solved via iterative scheme methods.

---

## Formulation & Governing Equations

### 1. Energy Functions
The energy density function $\Psi$ and total potential energy function $H$ describe the electromechanical state based on the strain tensor $\boldsymbol{\varepsilon}$ and electric field vector $\mathbf{E}$:

$$\Psi(\boldsymbol{\varepsilon}, \mathbf{E}) = \frac{1}{2} \lambda (\text{tr}\,\boldsymbol{\varepsilon})^2 + \mu \, \text{tr}(\boldsymbol{\varepsilon}^2) - \frac{1}{2} \beta \, \mathbf{E} \cdot \mathbf{E} + \gamma \, (\mathbf{a} \cdot \mathbf{E}) \, \text{tr}(\boldsymbol{\varepsilon})$$

$$H(\boldsymbol{\varepsilon}, \mathbf{E}) = \Psi(\boldsymbol{\varepsilon}, \mathbf{E}) - \mathbf{b} \cdot \mathbf{u} - q_v \phi$$

Where:
* **$\boldsymbol{\varepsilon} = \frac{1}{2}(\nabla \mathbf{u} + \nabla \mathbf{u}^T)$**: Linearized strain tensor
* **$\mathbf{E} = -\nabla \phi$**: Electric field derived from electric potential $\phi$
* **$\mathbf{a}$**: Unit polarization vector
* **$\lambda, \mu$**: Lamé constants
* **$\beta$**: Dielectric permittivity coefficient
* **$\gamma$**: Electromechanical coupling coefficient

### 2. Constitutive Equations
* **Cauchy Stress Tensor ($ oldsym{\sigma}$)**:
  $$\boldsymbol{\sigma} = \frac{\partial \Psi}{\partial \boldsymbol{\varepsilon}} = \lambda (\text{tr}\,\boldsymbol{\varepsilon})\mathbf{I} + 2\mu\boldsymbol{\varepsilon} + \gamma (\mathbf{a} \cdot \mathbf{E})\mathbf{I}$$
* **Electric Displacement ($\mathbf{D}$)**:
  $$\mathbf{D} = -\frac{\partial \Psi}{\partial \mathbf{E}} = \beta \mathbf{E} - \gamma (\text{tr}\,\boldsymbol{\varepsilon})\mathbf{a}$$

---

## Discretization & Numerical Implementation

### Strong & Weak Forms
The balance equations for linear momentum and Gauss's law for electricity in weak form:
* **Mechanical Equilibrium**: $\int_{\Omega} \boldsymbol{\sigma} : \delta\boldsymbol{\varepsilon} \, d\Omega - \int_{\Omega} \mathbf{b} \cdot \delta\mathbf{u} \, d\Omega - \int_{\Gamma_t} \mathbf{t} \cdot \delta\mathbf{u} \, d\Gamma = 0$
* **Electrical Equilibrium**: $\int_{\Omega} \mathbf{D} \cdot \delta\mathbf{E} \, d\Omega - \int_{\Omega} q_v \delta\phi \, d\Omega - \int_{\Gamma_d} q_s \delta\phi \, d\Gamma = 0$

### Linearized System Matrix
The discretized system yields a coupled system of equations solved via the Newton-Raphson scheme:

$$\begin{bmatrix} \mathbf{K}_{uu} & \mathbf{K}_{u\phi} \\ \mathbf{K}_{\phi u} & \mathbf{K}_{\phi\phi} \end{bmatrix} \begin{bmatrix} \Delta\mathbf{u} \\ \Delta\boldsymbol{\phi} \end{bmatrix} = \begin{bmatrix} \mathbf{R}_u \\ \mathbf{R}_\phi \end{bmatrix}$$

Where:
* $\mathbf{K}_{uu}$: Mechanical stiffness matrix
* $\mathbf{K}_{u\phi}, \mathbf{K}_{\phi u}$: Electromechanical coupling stiffness matrices
* $\mathbf{K}_{\phi\phi}$: Dielectric stiffness matrix
* $\mathbf{R}_u, \mathbf{R}_\phi$: Mechanical and electrical residual vectors

---

## Standard Model Configuration & Setup

### Material Properties ($	ext{BaTiO}_3$)
| Property | Notation | Value |
| :--- | :--- | :--- |
| Lamé First Parameter | $\lambda$ | $76.6 \times 10^9 \, \text{N/m}^2$ |
| Shear Modulus | $\mu$ | $44.7 \times 10^9 \, \text{N/m}^2$ |
| Coupling Parameter | $\gamma$ | $-0.56 \times 10^6 \, \text{MN}/(\text{V}\cdot\text{m})$ |
| Permittivity Parameter | $\beta$ | $4.4 \, \text{C}/(\text{V}\cdot\text{m})$ |
| Polarization Vector | $\mathbf{a}$ | $[0, 0, 1]^T$ |

### Domain & Mesh Discretization
* **Geometry**: $L_x = 1.0 \, \text{m}$, $L_y = 0.8 \, \text{m}$, $L_z = 1.4 \, \text{m}$
* **Element Type**: 8 3D Solid Elements (8 Gauss points per element for numerical integration)
* **Nodal Degrees of Freedom**: 4 DOFs per node ($u_x, u_y, u_z, \phi$)
* **Interpolation**: Linear shape functions

---

## Key Test Cases & Results

1. **Pure Mechanical Test Case**:
   * Evaluates pure deformation under external force traction to verify structural response without electric field interaction.
2. **Pure Electrical Test Case**:
   * Evaluates potential distribution across domain boundaries to verify dielectric behavior.
3. **Coupled Linear Response Test**:
   * Verifies mutual electromechanical interaction under single-step loading.
4. **Non-linear Convergence Analysis**:
   * Measures residual decrease over Newton-Raphson iterations to confirm quadratic convergence rate.
5. **Mesh Convergence Study**:
   * Assesses stability and degree-of-freedom scaling across refined meshes.

---

## Getting Started

### Requirements
* Python 3.x 
* NumPy / SciPy / Matplotlib (if using Python)

### Running the Code
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```
2. Run the main simulation script:
   ```bash
   python main.py
