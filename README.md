# Project Overview

**Pure-SU(2)-Lattice-Gauge-Theory**  
This repository contains a small research project developed as part of the Research Scholarship Project at the University of Edinburgh. It implements a Metropolis–Hastings MCMC simulation of a 2D pure SU(2) Yang–Mills (gauge) field on a square lattice, computes local plaquettes and Wilson loops, and produces simple diagnostics (action convergence and Wilson-loop expectation values). The goal of the project is to give an introductory, reproducible implementation of lattice gauge methods and to explore how the coupling `β` affects sampling/observables.

---

## Files in this repository

- `Gluon Field Code.py`  
  The main Python script that implements the lattice, SU(2) link variables, Metropolis updates, plaquette/action computation, and Wilson-loop calculation. Produces plots (action & Wilson loop) and writes `action.txt`, `wilson_loop.txt`, and `su2_matrices.txt`.

- `Lattice_Gauge_Theory_Project_Report.pdf`  
  The full project report (theory, method, results, discussion and conclusions).

- `Poster_Lattice_Gauge_Simulation.pdf`  
  A poster summarising the project.

---

## Quick start

### Requirements
Tested with:
- Python 3.8+  
- `numpy`  
- `matplotlib`  
- `tqdm`

Install dependencies (example):
```bash
pip install numpy matplotlib tqdm
