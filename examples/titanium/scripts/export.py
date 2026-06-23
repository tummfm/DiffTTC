import os
import sys

import argparse

from pathlib import Path

import tomli

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

from jax import tree_util

import jax.numpy as jnp

import jax_md_mod
from jax_md import partition, space

from collections import OrderedDict

from chemtrain.data import preprocessing

from chemtrain.deploy import exporter, graphs as export_graphs

import pickle as pkl

import train_utils

from chemutils.datasets import titanium

def get_default_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, default="-1")
    parser.add_argument("dir", type=str)
    args = parser.parse_args()

    print(f"Run on device {args.device}")
    return OrderedDict(
        seed=11,
        pretrained_model=args.dir,
        confs=[
            ("../../../juwels/output/hcp-bcc/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/hcp_box_T_1154.7_P_0_a_2.958_ca_1.592.pdb", 1727.),
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

    dataset = titanium.download_dataset("/home/paul/Datasets")
    for split in dataset.keys():
        dataset[split]["species"] = jnp.zeros(
            dataset[split]["R"].shape[:-1], dtype=jnp.int32)

    confs = train_utils.load_confs(config, fractional=True)
    _displacement_fn, shift_fn = space.periodic_general(confs["box"][0], fractional_coordinates=True)

    def displacement_fn(*args, perturbation=None, **kwargs):
        return _displacement_fn(*args, perturbation=perturbation, **kwargs)

    # We estimate the maximum number of edges and triplets and also initialize
    # a sufficiently big neighbor list.
    format = partition.Sparse
    if config["pretrained_settings"]["model"]["type"] == "DimeNetPP":
        format = partition.Dense
    nbrs_init, (max_neighbors, max_edges, avg_num_neighbors, max_triplets) = preprocessing.allocate_neighborlist(
        confs, displacement_fn, confs["box"][0],
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
        config["pretrained_settings"], {"training": confs}, nbrs_init, max_edges, per_particle=False,
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

    displacement_fn, _ = space.free()

    num_mpl_lookup = {
        "MACE": [
            config["pretrained_settings"]["model"]["model_kwargs"].get("num_interactions", 0),
            2 * config["pretrained_settings"]["model"]["model_kwargs"].get("num_interactions", 0)
        ],
    }

    class Model(exporter.Exporter):

        graph_type = export_graphs.SimpleSparseNeighborList
        nbr_order = num_mpl_lookup[config["pretrained_settings"]["model"]["type"]]
        unit_style = "real"

        r_cutoff = config["pretrained_settings"]["model"]["r_cutoff"] * 10. # Cutoff in angstrom

        def __init__(self, params, *args, **kwargs):
            self.params = params

            super().__init__(*args, **kwargs)

        def energy_fn(self, position, species, graph):
            _max_edges = max_edges
            neighbor = graph.to_neighborlist()

            export_template = train_utils.define_model(
                config["pretrained_settings"], None, nbrs_init, _max_edges, per_particle=True,
                avg_num_neighbors=avg_num_neighbors, positive_species=False,
                displacement_fn=displacement_fn
            )
            model_fn = export_template(self.params)

            species = jnp.ones_like(species)
            # We trained the model with units nm and kJ/mol, so we need some scaling
            position /= 10.0
            energies = model_fn(position, neighbor, species=species)
            energies /= 4.184

            return energies

    trained_model = Model(ref_params)
    trained_model.export()

    trained_model.save(reference_dir / "model.ptb")

if __name__ == "__main__":
    main()

