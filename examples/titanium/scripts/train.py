import os
import sys

import argparse

import tomli_w

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import numpy as onp

import jax
from jax import random

import jax.numpy as jnp

from jax_md_mod import custom_quantity
from jax_md import partition, space

from collections import OrderedDict

from chemtrain.data import preprocessing
from chemtrain import trainers

from chemtrain.deploy import exporter, graphs as export_graphs

import train_utils

from chemutils.datasets import titanium

def get_default_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, default="-1")
    parser.add_argument("--cutoff", type=float, default=0.55)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()

    print(f"Run on device {args.device}")
    return OrderedDict(
        seed=11,
        model=OrderedDict(
            edge_multiplier=1.1,
            r_cutoff=args.cutoff,
            type="MACE",
            model_kwargs=OrderedDict(
                hidden_irreps="64x0e + 32x1o",
                max_ell=3,
                num_interactions=2,
                correlation=3,
                n_radial_basis=8,
                envelope_p=6,
            ),
            prior_type="lennard_jones",
            prior_kwargs=OrderedDict(
                train=False,
                epsilon=[0.5,],
                sigma=[0.325 * onp.sqrt(3) / 2 / (2 ** (1 / 6)),], # Infer from the lattice constant
            )
        ),
        optimizer=OrderedDict(
            init_lr=args.lr,
            lr_decay=0.01,
            epochs=args.epochs,
            batch=args.batch,
            cache=8,
            weight_decay=1e-2,
            optimizer_kwargs=OrderedDict(
                b1=0.95,
                b2=0.995,
                eps=1e-8,
                eps_root=1e-16,
                nesterov=False,
            )
        ),
        gammas=OrderedDict(
            virial=4e-6,
            U=1e-5,
            F=1e-2,
        ),
    )

def main():
    config = get_default_config()
    key = random.PRNGKey(config["seed"])

    out_dir = train_utils.create_out_dir(config)

    dataset = titanium.download_dataset("/home/paul/Datasets", max_samples=config.get("max_samples"))
    for split in dataset.keys():
        dataset[split]["species"] = jnp.zeros(
            dataset[split]["R"].shape[:-1], dtype=jnp.int32)

    displacement_fn, shift_fn = space.periodic_general(1.0, fractional_coordinates=True)

    # We estimate the maximum number of edges and triplets and also initialize
    # a sufficiently big neighbor list.
    format = partition.Sparse
    nbrs_init, (max_neighbors, max_edges, avg_num_neighbors, max_triplets) = preprocessing.allocate_neighborlist(
        dataset["training"], displacement_fn, 0.0, config["model"]["r_cutoff"],
        box_key="box", format=format, count_triplets=True,
        default_list=True, disable_cell_list=True, fractional_coordinates=True,
    )

    max_neighbors = int(max_neighbors * config["model"]["edge_multiplier"])
    max_edges = int(max_edges * config["model"]["edge_multiplier"])
    max_triplets = int(max_triplets * config["model"]["edge_multiplier"] ** 2)

    print(f"Estimated: "
          f"\tMax. neighbors: {max_neighbors},"
          f"\tMax. edges: {max_edges},"
          f"\tMax. triplets: {max_triplets}"
    )

    energy_fn_template, init_params = train_utils.define_model(
        config, dataset, nbrs_init, max_edges, per_particle=False,
        avg_num_neighbors=avg_num_neighbors, positive_species=False,
        displacement_fn=displacement_fn
    )

    optimizer = train_utils.init_optimizer(config, dataset)

    trainer_fm = trainers.ForceMatching(
        init_params, optimizer, energy_fn_template, nbrs_init,
        batch_per_device=config["optimizer"]["batch"] // len(jax.devices()),
        batch_cache=config["optimizer"]["cache"],
        gammas=config["gammas"],
        additional_targets={
            'virial': custom_quantity.init_virial_stress_tensor(
                energy_fn_template, reference_box=None, include_kinetic=False)
        },
        weights_keys={
            'virial': 'virial_weights'
        }
    )

    trainer_fm.set_dataset(
        dataset['training'], stage='training')
    trainer_fm.set_dataset(
        dataset['validation'], stage='validation', include_all=True)
    trainer_fm.set_dataset(
        dataset['testing'], stage='testing', include_all=True)

    with open(out_dir / "config.toml", "wb") as f:
        tomli_w.dump(config, f)

    # Train and save the results to a new folder
    trainer_fm.train(config["optimizer"]["epochs"])

    train_utils.save_training_results(config, out_dir, trainer_fm)

    test_predictions = trainer_fm.predict(dataset['testing'], batch_size=config["optimizer"]["batch"])
    train_predictions = trainer_fm.predict(dataset['training'], batch_size=config["optimizer"]["batch"])

    train_utils.plot_predictions(test_predictions, dataset["testing"], out_dir, "test_predictions")
    train_utils.plot_predictions(train_predictions, dataset["training"], out_dir, "train_predictions")
    train_utils.plot_convergence(trainer_fm, out_dir)

    displacement_fn, _ = space.free()
    export_template, _ = train_utils.define_model(
        config, dataset, nbrs_init, max_edges, per_particle=True,
        avg_num_neighbors=avg_num_neighbors, positive_species=False,
        displacement_fn=displacement_fn
    )

    num_mpl_lookup = {
        "MACE": [
            config["model"]["model_kwargs"].get("num_interactions", 0),
            2 * config["model"]["model_kwargs"].get("num_interactions", 0)
        ],
    }

    class Model(exporter.Exporter):

        graph_type = export_graphs.SimpleDenseNeighborList if config["model"]["type"] == "DimeNetPP" else export_graphs.SimpleSparseNeighborList

        nbr_order: int = num_mpl_lookup[config["model"]["type"]]

        r_cutoff = config["model"]["r_cutoff"] * 10. # Cutoff in angstrom

        def __init__(self, export_template, params, *args, **kwargs):
            self.model = export_template(params)

            super().__init__(*args, **kwargs)

        def energy_fn(self, position, species, graph):
            neighbor = graph.to_neighborlist()
            # We trained the model with units nm and kJ/mol, so we need some scaling
            position /= 10.0
            energies = self.model(position, neighbor, species=species)
            energies /= 4.184

            return energies

    trained_model = Model(export_template, trainer_fm.best_params)
    trained_model.export()

    trained_model.save(out_dir / "model.ptb")

if __name__ == "__main__":
    main()

