# Differentiable Transition Temperature Correction (DiffTTC)

Code supporting the paper [Refining machine learning potentials through thermodynamic theory of phase transitions](https://doi.org/10.1038/s41524-026-02195-7).

## Overview

This repository contains the following files:

```text
examples/titanium/
| data/ -- Refererence data (properties)
  | ...
| model/ -- Trained and refined model
  | ...
| notebooks/ -- Evaluation of training and simulations
  | ...
| scripts/ -- Training scripts
  | eval_utils.py -- Evaluate free energy computations
  | export.py -- Export model to LAMMPS
  | predict.py -- Predict U, F, etc. on DFT samples
  | train_all_parallel.py -- Refine model on phase diagram/
  | train_utils.py -- Model definition, optimizer, etc.
  | train.py -- Pretrain model with Force Matching
  | visualize.py -- Visualize results from multiple simulations
| simulations/
  | lammps/ -- Simulation input scripts
    | ...
  | mace-mp/ -- MACE-MP-0b3 foundational model files
    | ..
  | scripts/ -- Run multiple bundled lammps simulations
    | ...
| external/ -- Dependencies to be installed
  | chemtrain/ -- Modified version of chemtrain
    | ...
  | chemutils/ -- MACE model and dataset loader
    | ...
  | lammps/ -- Used LAMMPS version (subrepo)
    | ...
| juwels/ -- Extra scripts to run on HPC cluster
  | ...

```

## Quickstart

First, install the external dependencies in the following order:

```bash
pip install -e "./external/chemtrain[all]"
pip install -e "./external/chemutils"
pip install "jax[cuda12]==0.4.37" --force-reinstall
```

To pretrain a model with the given hyperparameter on the CUDA device ``0``, run the following command:

```bash
cd examples/titanium/scripts
python train.py "0"
```

To refine the provided pretrained model via DiffTTC on the prepared setup, run the following command:

```bash
cd examples/titanium/scripts
python train_all_parallel.py "0"
```

To install the LAMMPS plugin, please refer to the [documentation](https://chemtrain.readthedocs.io/en/latest/chemtrain-deploy/installation.html).

## Full Cycle - Pretraining and Refining with DiffTTC

First, train an MLP using the command given in [Quickstart](#quickstart).
This command will create a model folder ``titanium__train*`` in the parent directory ``examples/titanium/scripts/output``.
The model can be copied into the models folder ``examples/titanium/models``.
Then, the LAMMPS simulations can be run by specifying the model in the ``run_*.sh`` scripts located in the folder ``examples/titanium/simulations`` by executing them outside their parent directory.
For example, to run the coexistence simulations, enter the command:

```bash
cd examples/titanium/simulations
bash scripts/run_coexistence.sh
```

Some simulations have to be run in a specific order, including postprocessing scripts.

For the HCP-BCC transition:

1. ``run_prepare.sh`` and compute ``hcp_constants.txt``and ``bcc_constants.txt``.
2. ``run_ti_bcc.sh`` and ``run_ti_hcp.sh``
3. ``run_rs_bcc-hcp.sh``
4. ``run_create_bcc_box.sh`` and ``run_create_hcp_box.sh`` and convert to pdb

For the BCC-Liquid transition:

1. ``run_coexistence.sh`` and compute ``temps.txt``
2. ``run_prepare_liquid.sh`` and compute ``bcc_constants.txt`` and ``liquid_constants.txt``
3. ``run_rs_bcc-liquid.sh``
4. ``run_create_bcc_box.sh`` and ``run_create_liquid_box.sh`` and convert to pdb

The thermal expansion, MSD and VACF have to be computed last:

1. ``run_thermal_expansion.sh``
2. ``run_compute_msd.sh`` and ``run_compute_vacf.sh``

The prepared free energy targets and simulation boxes have to be defined in the training script ``train_all_parallel.py``. Then, the script can be run via the instructions given in [Quickstart](#quickstart).


## Citation

If you use DiffTTC, please cite:

```bibtex

@article{fuchsDiffTTC2026,
  title = {Refining Machine Learning Potentials through Thermodynamic Theory of Phase Transitions},
  author = {Fuchs, Paul and Zavadlav, Julija},
  date = {2026-06-22},
  journaltitle = {npj Computational Materials},
  shortjournal = {npj Comput Mater},
  issn = {2057-3960},
  doi = {10.1038/s41524-026-02195-7},
  url = {https://doi.org/10.1038/s41524-026-02195-7},
}

```

