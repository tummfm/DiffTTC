## Simulation of Free Energy for Solid Phases of Titanium

## 1) Prepare Phases

Scripts ``prepare_bcc.lmp`` / ``prepare_hcp.lmp``
run NPT simulations to determine the lattice parameters and atom mobilities

## 2)  Frenkel Ladd Absolute Free Energy

Scripts ``frenkel_ladd_bcc.lmp`` / ``frenkel_ladd_hcp.lmp`` 
run NVT simulations to determine the absolute free energy of solid phases
via NE-TI from einstein crystal reference

## 3) Reversible Scaling for Temperature Dependence

Scripts ``reversible_scaling_bcc.lmp`` / ``reversible_scaling_hcp.lmp``
run NPT simulations and scale the potential to obtain the temperature
change of the Gibbs free energy via NE TI

## 4) Analyze Results
Scripts ``notebooks/evaluate_crystal.ipynb`` contains evaluations
for all computations. The script ``eval_cst`` computes the crystal structures
using the Polyhedral Template Matching method.

## 5) Preparation of Training
Scripts ``create_bcc_box.lmp`` / ``create_hcp_box.lmp`` prepare simulation boxes
to match the free energy difference.
