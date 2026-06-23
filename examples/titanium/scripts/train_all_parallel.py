import os
import sys

import argparse

from pathlib import Path

import tomli_w

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.975"

import numpy as onp

import tomli

import pickle as pkl

import jax

jax.config.update("jax_debug_nans", True)

from jax import random

import jax.numpy as jnp

from jax_md_mod import custom_quantity
from jax_md import simulate, partition, space, quantity as snapshot_quantity

from collections import OrderedDict

from chemtrain.data import preprocessing
from chemtrain.ensemble import sampling
from chemtrain import quantity, trainers
from chemtrain.quantity import observables

import train_utils

from chemutils.datasets import titanium

def get_default_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, default="-1")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--test", type=bool, default=False)
    args = parser.parse_args()

    print(f"Run on device {args.device} in mode {'TEST' if args.test else 'PRODUCTION'}")
    return OrderedDict(
        seed=11,
        pretrained_model="../models/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5",
        confs=[
            # Statepoint for BCC-HCP coexistence: HCP
            ("../../../juwels/output/hcp-bcc/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/hcp_box_T_1154.7_P_0_a_2.958_ca_1.592.pdb", 1154.7),
            ("../../../juwels/output/hcp-bcc/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/hcp_box_T_1124.5_P_1_a_2.947_ca_1.592.pdb", 1124.5),
            ("../../../juwels/output/hcp-bcc/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/hcp_box_T_1094.4_P_2_a_2.936_ca_1.592.pdb", 1094.4),
            ("../../../juwels/output/hcp-bcc/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/hcp_box_T_1064.3_P_3_a_2.926_ca_1.592.pdb", 1064.3),
            ("../../../juwels/output/hcp-bcc/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/hcp_box_T_1034.2_P_4_a_2.916_ca_1.592.pdb", 1034.2),
            ("../../../juwels/output/hcp-bcc/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/hcp_box_T_1005.0_P_5_a_2.907_ca_1.593.pdb", 1005.0),
            # Statepoints for BCC-liquid coexistence: BCC
            ("../../../juwels/output/bcc-liquid/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/bcc_T_1940.52_P_0_a_3.315.pdb", 1940.52),
            ("../../../juwels/output/bcc-liquid/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/bcc_T_1979.41_P_1_a_3.303.pdb", 1979.41),
            ("../../../juwels/output/bcc-liquid/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/bcc_T_2011.3_P_2_a_3.291.pdb",  2011.3),
            ("../../../juwels/output/bcc-liquid/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/bcc_T_2041.43_P_3_a_3.281.pdb", 2041.43),
            ("../../../juwels/output/bcc-liquid/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/bcc_T_2071.56_P_4_a_3.271.pdb", 2071.56),
            ("../../../juwels/output/bcc-liquid/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/bcc_T_2092.93_P_5_a_3.261.pdb", 2092.93),
            # Statepoints for BCC-HCP coexistence: BCC
            ("../../../juwels/output/hcp-bcc/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/bcc_T_1940.52_P_0_a_3.286.pdb", 1154.7),
            ("../../../juwels/output/hcp-bcc/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/bcc_T_1979.41_P_1_a_3.274.pdb", 1124.5),
            ("../../../juwels/output/hcp-bcc/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/bcc_T_2011.3_P_2_a_3.263.pdb",  1094.4),
            ("../../../juwels/output/hcp-bcc/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/bcc_T_2041.43_P_3_a_3.252.pdb", 1064.3),
            ("../../../juwels/output/hcp-bcc/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/bcc_T_2071.56_P_4_a_3.242.pdb", 1034.2),
            ("../../../juwels/output/hcp-bcc/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/bcc_T_2092.93_P_5_a_3.232.pdb", 1005.0),
            # Statepoints for BCC-liquid coexistence: Liquid
            ("../../../juwels/output/bcc-liquid/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/liquid_T_1940.52_P_0_a_3.335.pdb", 1940.52),
            ("../../../juwels/output/bcc-liquid/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/liquid_T_1979.41_P_1_a_3.322.pdb", 1979.41),
            ("../../../juwels/output/bcc-liquid/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/liquid_T_2011.3_P_2_a_3.309.pdb",  2011.3),
            ("../../../juwels/output/bcc-liquid/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/liquid_T_2041.43_P_3_a_3.298.pdb", 2041.43),
            ("../../../juwels/output/bcc-liquid/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/liquid_T_2071.56_P_4_a_3.286.pdb", 2071.56),
            ("../../../juwels/output/bcc-liquid/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/boxes/liquid_T_2092.93_P_5_a_3.275.pdb", 2092.93),
        ],
        simulator_type="nose_hoover",
        simulator_settings=OrderedDict(
            kT=1942.0 * quantity.kb,  # For melting
            # kT=1035.0 * quantity.kb, # For BCC - HCP transition
            # pressure=4.0 * 602.2, # From GPa to kJ/mol/nm^3,
            thermostat_kwargs=OrderedDict(
                chain_steps=1, tau=2., chain_length=3,
            ),
        ),
        targets=OrderedDict(
            pressure = [i * 1e4 /  16.6054 for i in range(6)] * 2,
            free_energy_diff=[
                # Convert from per-atom to total free energy
                -0.35334453 * 576,
                -0.33564248 * 576,
                -0.31674853 * 576,
                -0.28214487 * 576,
                -0.23802428 * 576,
                -0.17726994 * 576,
                -1.08541916 * 576,
                -1.13525304 * 576,
                -1.09785235 * 576,
                -1.14602803 * 576,
                -1.15615396 * 576,
                -1.13893508 * 576
            ] #
        ),
        timings=OrderedDict(
            dt=4e-3,
            print_every=0.5 if not args.test else 0.1,
            t_equilib=20. if not args.test else 1.0,
            total_time=120. if not args.test else 2.0,
            # t_equilib=50.,  # 1000.0,
            # total_time=350.,  # 2000.0,
        ),
        optimizer=OrderedDict(
            init_lr=args.lr,
            lr_decay=1e-1,
            epochs=args.epochs,
            weight_decay=1e-2,
            batch=6, # Note: Batch size -1 possible -> All statepoints
            optimizer_kwargs=OrderedDict(
                b1=0.9,
                b2=0.99,
                eps=1e-8,
                eps_root=1e-16,
                nesterov=False,
            )
        ),
        gammas=OrderedDict(
            pressure=[1e-4] * 24,
            free_energy=[1e-5] * 24,
        ),
        reweighting_ratio=0.9,
    )

def main():
    config = get_default_config()

    key = random.PRNGKey(config["seed"])

    reference_dir = Path(config["pretrained_model"])
    with open(reference_dir / "config.toml", "rb") as f:
        config["pretrained_settings"] = tomli.load(f)

    out_dir = train_utils.create_out_dir(
        config["pretrained_settings"], "train_solid")

    dataset = titanium.download_dataset("/home/paul/Datasets")
    for split in dataset.keys():
        dataset[split]["species"] = jnp.zeros(
            dataset[split]["R"].shape[:-1], dtype=jnp.int32)

    confs = train_utils.load_confs(config, fractional=True)
    displacement_fn, shift_fn = space.periodic_general(confs["box"][0],
                                                        fractional_coordinates=True)

    # We estimate the maximum number of edges and triplets and also initialize
    # a sufficiently big neighbor list.
    format = partition.Sparse
    if config["pretrained_settings"]["model"]["type"] == "DimeNetPP":
        format = partition.Dense
    nbrs_init, (max_neighbors, max_edges, avg_num_neighbors,
                max_triplets) = preprocessing.allocate_neighborlist(
        confs, displacement_fn, confs["box"][0],
        config["pretrained_settings"]["model"]["r_cutoff"],
        box_key="box", format=format, count_triplets=True,
        default_list=True, disable_cell_list=True, fractional_coordinates=True,
        capacity_multiplier=2.0
    )

    max_neighbors = int(max_neighbors * config["pretrained_settings"]["model"][
        "edge_multiplier"])
    max_edges = int(
        max_edges * config["pretrained_settings"]["model"]["edge_multiplier"])
    max_triplets = int(max_triplets * config["pretrained_settings"]["model"][
        "edge_multiplier"] ** 2)

    print(f"Estimated: "
          f"\tMax. neighbors: {max_neighbors},"
          f"\tMax. edges: {max_edges},"
          f"\tMax. triplets: {max_triplets}"
          )

    energy_fn_template, _ = train_utils.define_model(
        config["pretrained_settings"], {"training": confs}, nbrs_init,
        max_edges, per_particle=False,
        avg_num_neighbors=avg_num_neighbors, positive_species=False,
        displacement_fn=displacement_fn
    )

    with open(reference_dir / "best_params.pkl", "rb") as f:
        ref_params = pkl.load(f)

    # ref_params["prior"]["sigma"] = 0.3 # For testing purposes

    # Learn only the difference to the prior parameters
    energy_fn_template, init_params = train_utils.init_difference_template(
        energy_fn_template, ref_params
    )

    num_pairs = len(config["confs"]) // 2
    print(f"Run for {num_pairs} pairs of statepoints.")

    if config["simulator_type"] == "nose_hoover":
        simulator_fn = simulate.nvt_nose_hoover
    else:
        raise ValueError(f"Unknown simulator type: {config['simulator_type']}")

    sim_template, timings = train_utils.init_simulator(
        config, shift_fn, simulator=simulator_fn,
    )

    @jax.vmap
    def init_sim_state(key, sample):
        assert "mass" in sample.keys(), "Masses are required for the simulation."

        pos = sample.pop("R")
        init_fn, _ = sim_template(energy_fn_template(init_params))
        nbrs = nbrs_init.update(pos, **sample)
        sim_state = init_fn(key, pos, neighbor=nbrs, **sample)
        return sampling.SimulatorState(sim_state=sim_state, nbrs=nbrs)

    key, split = random.split(key)
    init_states = init_sim_state(random.split(split, confs["R"].shape[0]), confs)

    rdf_discretization = custom_quantity.rdf_discretization(
        rdf_start=0.0, rdf_cut=0.9, nbins=150
    )
    rdf_params = custom_quantity.RDFParams(jnp.zeros(150), *rdf_discretization)

    quantities = {
        "pressure": custom_quantity.init_pressure(energy_fn_template),
        "rdf":  custom_quantity.init_rdf(displacement_fn, rdf_params)
    }

    def free_energy_fn(F, P, **state_dict):
        volume = snapshot_quantity.volume(3, state_dict['box'])
        return F + (P - state_dict['pressure']) * volume

    observables = {
        "pressure": quantity.observables.init_traj_mean_fn("pressure"),
        "free_energy": quantity.observables.init_identity_fn("free_energy"),
        "rdf": quantity.observables.init_traj_mean_fn("rdf") # Save for asserting the structure
    }

    optimizer = train_utils.init_difftre_optimizer(config)

    state_kwargs = {
            key: val for key, val in confs.items()
            if key not in ["R", "mass"]
    }

    state_kwargs["lambda_prior"] = jnp.ones(confs["R"].shape[0])
    state_kwargs["lambda_neural_network"] = jnp.ones(confs["R"].shape[0])

    key, split = random.split(key)
    trainer_difftre = trainers.DifftreParallel(
        split, init_params, optimizer,
        log_dir=out_dir / "training.log", checkpoint_path=out_dir / "checkpoints",
        sim_batch_size=config["optimizer"]["batch"],
        targets={
            "pressure": {
                "target": jnp.asarray(config["targets"]["pressure"] * 2),
                "gamma": jnp.asarray(config["gammas"]["pressure"])
            },
            "free_energy": {
                "target": jnp.concatenate([
                    jnp.asarray(config["targets"]["free_energy_diff"]),
                    -jnp.asarray(config["targets"]["free_energy_diff"]),
                   ], axis=0),
                "gamma": jnp.asarray(config["gammas"]["free_energy"])
            }
        },
        observables=observables,
        state_kwargs=state_kwargs,
        quantities=quantities,
        reference_states=init_states,
        neighbor_fn=nbrs_init,
        energy_fn_template=energy_fn_template,
        simulator_template=sim_template,
        timings=timings,
        reweight_ratio=config.get("reweighting_ratio", 0.9),
        allowed_reduction=None,
        num_runs_init=1,
        pair_batches=True, # Optimize corresponding statepoints together
    )

    def update_targets(trainer: trainers.DifftreParallel, batch, *args, **kwargs):

        # Get the predictions for the corresponding statepoints
        pred_batch = jnp.mod(batch + num_pairs, num_pairs * 2)
        predictions = trainer.predict(pred_batch)

        print(f"Predicted dFs:")
        for i in pred_batch:
            print(f"\t Statepoint {int(i)}:")
            for key, val in predictions.items():
                if not jnp.isscalar(val[i]): continue
                print(f"\t\t {key}: {val[i]:.3f}")

        targets = jnp.concatenate((onp.asarray(config["targets"]["free_energy_diff"]), -onp.asarray(config["targets"]["free_energy_diff"])), axis=0)
        for idx, key in enumerate(batch):
            print(f"Update target for statepoint {key}: {predictions['free_energy'][idx]} + {targets[key]}")

        trainer.targets["free_energy"]["target"] = trainer.targets["free_energy"]["target"].at[batch].set(predictions["free_energy"] + targets[batch])

        for key, vals in trainer.targets.items():
            if onp.any(onp.isnan(vals["target"])):
                raise ValueError(
                    f"Target {key} contains NaN values: {targets['target']}"
                )

        # trainer._converged = True

    trainer_difftre.add_task("pre_batch", update_targets)

    with open(out_dir / "config.toml", "wb") as f:
        tomli_w.dump(config, f)

    # Train and save the results to a new folder
    trainer_difftre.train(config["optimizer"]["epochs"], checkpoint_freq=10)

    trainer_difftre.save_trainer(
        out_dir / "trainer.pkl")
    trainer_difftre.save_energy_params(
        out_dir / "final_params.pkl", best=False, save_format=".pkl"
    )


if __name__ == "__main__":
    main()

