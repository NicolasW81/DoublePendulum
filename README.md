# Transient Stability Landscapes in the Double Pendulum System

**Author:** Nicolas Wallner  
**DOI / Citation:** 
**GitHub Repository:** [https://github.com/NicolasW81/DoublePendulum](https://github.com/NicolasW81/DoublePendulum)

---

## Overview
This repository contains Python code and simulation data used in the study:

**Transient Stability Landscapes in the Double Pendulum System: A Geometric Analysis of Finite-Time Dynamics**  
The study identifies previously unreported transient stability patterns in the nonlinear double pendulum system. By mapping initial conditions to a two-dimensional parameter space (u,v) and analyzing energy dissipation over finite time horizons, structured landscapes revealing predictable organization in the chaotic system were discovered.

---

## Contents

- `DoublePendulum.py` – Core simulations (Simulations 1-3)  
- `DoublePendulum3DSimulation.py` – 3D animated stability landscapes (Simulation 4)
- `DoublePendulum_Simulation4.gif` - 3D GIF animation (Simulation 4 Result)
- `rungekutta_test.py` – Verification using fourth-order Runge–Kutta (Simulation 5a)  
- `leapfrog_test.py` – Verification using symplectic Leapfrog integration (Simulation 5b) 
- `README.md` – This file

---

## Installation

1. **Python version**: ≥ 3.10 recommended  
2. **Install dependencies**:

```bash
pip install numpy matplotlib

