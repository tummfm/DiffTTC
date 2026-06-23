import os
import sys

import argparse

from pathlib import Path

import tomli_w

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import tomli

from jax import tree_util

import jax.numpy as jnp

from jax_md_mod import custom_quantity
from jax_md import partition, space

import pickle as pkl

from collections import OrderedDict

from chemtrain.data import preprocessing
from chemtrain import trainers

import train_utils

from chemutils.datasets import titanium

def get_default_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, default="-1")
    parser.add_argument("dir", type=str)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print(f"Run on device {args.device}")
    return OrderedDict(
        seed=11,
        pretrained_model=args.dir,
        confs=[
            ("data/confs/titanium_bcc_6_6_9.pdb", 0.01805, 1727., 0.0),
        ],
    )

def main():
    config = get_default_config()

    reference_dir = Path(config["pretrained_model"])
    with open(reference_dir / "config.toml", "rb") as f:
        pretrained_settings = tomli.load(f)

    pretrained_dir = None
    if 'pretrained_model' in pretrained_settings:
        print(f"Loading pretrained model from {pretrained_settings['pretrained_model']}")

        pretrained_dir = Path(pretrained_settings['pretrained_model'])

        with open(pretrained_dir / "config.toml", "rb") as f:
            pretrained_settings = tomli.load(f)

    config["pretrained_settings"] = pretrained_settings

    out_dir = train_utils.create_out_dir(config["pretrained_settings"], "predict")

    dataset = titanium.download_dataset("/home/paul/Datasets")
    for split in dataset.keys():
        dataset[split]["species"] = jnp.zeros(
            dataset[split]["R"].shape[:-1], dtype=jnp.int32)

    _displacement_fn, shift_fn = space.periodic_general(dataset["training"]["box"][0], fractional_coordinates=True)

    def displacement_fn(*args, perturbation=None, **kwargs):
        return _displacement_fn(*args, perturbation=perturbation, **kwargs)

    # We estimate the maximum number of edges and triplets and also initialize
    # a sufficiently big neighbor list.
    format = partition.Sparse
    if config["pretrained_settings"]["model"]["type"] == "DimeNetPP":
        format = partition.Dense
    nbrs_init, (max_neighbors, max_edges, avg_num_neighbors, max_triplets) = preprocessing.allocate_neighborlist(
        dataset["training"], displacement_fn, dataset["training"]["box"][0],
        config["pretrained_settings"]["model"]["r_cutoff"],
        box_key="box", format=format, count_triplets=True,
        default_list=True, disable_cell_list=True, fractional_coordinates=True,
        capacity_multiplier=2.0
    )

    max_neighbors = int(max_neighbors * config["pretrained_settings"]["model"]["edge_multiplier"])
    max_edges = int(max_edges * config["pretrained_settings"]["model"]["edge_multiplier"])
    max_triplets = int(max_triplets * config["pretrained_settings"]["model"]["edge_multiplier"] ** 2)

    print(f"Estimated: "
          f"\tMax. neighbors: {max_neighbors},"
          f"\tMax. edges: {max_edges},"
          f"\tMax. triplets: {max_triplets}"
    )

    energy_fn_template, _ = train_utils.define_model(
        config["pretrained_settings"], dataset, nbrs_init, max_edges, per_particle=False,
        avg_num_neighbors=avg_num_neighbors, positive_species=False,
        displacement_fn=displacement_fn
    )

    if pretrained_dir is None:
        with open(reference_dir / "best_params.pkl", "rb") as f:
            ref_params = pkl.load(f)
    else:
        with open(pretrained_dir / "best_params.pkl", "rb") as f:
            ref_params = pkl.load(f)

        with open(reference_dir / "final_params.pkl", "rb") as f:
            ref_params = tree_util.tree_map(jnp.add, ref_params, pkl.load(f))


    trainer_fm = trainers.ForceMatching(
        ref_params, None, energy_fn_template, nbrs_init,
        batch_per_device=128,
        batch_cache=100,
        gammas=pretrained_settings["gammas"],
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
    test_predictions = trainer_fm.predict(dataset['testing'], batch_size=128)
    train_predictions = trainer_fm.predict(dataset['training'], batch_size=128)

    train_utils.save_predictions(out_dir, f"test_predictions", test_predictions)
    train_utils.save_predictions(out_dir, f"train_predictions", train_predictions)

    train_utils.plot_predictions(test_predictions, dataset["testing"], out_dir, "test_predictions")
    train_utils.plot_predictions(train_predictions, dataset["training"], out_dir, "train_predictions")

if __name__ == "__main__":
    main()

