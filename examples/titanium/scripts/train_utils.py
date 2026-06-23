from pathlib import Path
import uuid
import datetime

import mdtraj
import tomli_w

import numpy as onp
from chemtrain.ensemble import sampling

import functools

import jax
from jax import tree_util, lax, random, nn

import jax.numpy as jnp


import matplotlib.pyplot as plt

from jax_md_mod.model import layers, neural_networks, prior
from jax_md_mod import custom_energy, io
from jax_md import simulate, partition, space, util, energy, quantity as snapshot_quantity, smap


import optax

import haiku as hk

from mdtraj import utils as md_utils

import e3nn_jax


from chemtrain.trainers import ForceMatching
from chemtrain import quantity
from chemtrain.learn import difftre

from chemutils.models.mace import mace_neighborlist_pp
from matplotlib.pyplot import legend
from cycler import cycler

def define_model(config,
                 dataset=None,
                 nbrs_init=None,
                 max_edges=None,
                 max_triplets=None,
                 per_particle=False,
                 avg_num_neighbors=1.0,
                 positive_species=False,
                 displacement_fn=None,
                 tether_config=None):
    """Initializes a concrete model for a system given path to model parameters."""

    # Requirement to capture all species in the dataset
    n_species = 1

    model_type = config["model"].get("type", "MACE")
    print(f"Run model {model_type}")
    if model_type == "MACE":
        _init_fn, gnn_energy_fn = mace_neighborlist_pp(
            displacement_fn, config["model"]["r_cutoff"], n_species,
            max_edges=max_edges, output_irreps="1x0e",
            per_particle=per_particle,
            avg_num_neighbors=avg_num_neighbors, mode="energy",
            positive_species=positive_species,
            **config["model"]["model_kwargs"]
        )
    elif model_type == "Spline":
        n_points = config["model"]["model_kwargs"].get("n_points", 100)
        def _init_fn(key, r, nbrs, **kwargs):
            return {
                "x": jnp.linspace(0.02, config["model"]["r_cutoff"], n_points),
                "y": 0.01 * random.normal(key, (n_points,)),
            }

        def gnn_energy_fn(params, pos, neighbor, **kwargs):
            energy_fn = custom_energy.tabulated_neighbor_list(
                displacement_fn, params["x"], params["y"], None,
                r_onset=0.8 * config["model"]["r_cutoff"],
                r_cutoff=config["model"]["r_cutoff"],
                initialize_neighbor_list=False,
            )
            return energy_fn(pos, neighbor, **kwargs)
    else:
        raise NotImplementedError(f"Model {model_type} not implemented.")

    prior_type = config["model"].get("prior_type")
    prior_kwargs = config["model"].get("prior_kwargs", {})

    if prior_type is None:
        print(f"No prior specified.")
        init_prior_fn = lambda *args, **kwargs: None
        apply_prior_fn = lambda *args, **kwargs: 0.0
    elif prior_type in ["repulsion", "lennard_jones", "truncated_lennard_jones"]:
        def apply_prior_fn(prior_params, pos, neighbor, **kwargs):
            if prior_kwargs.get("train", True):
                sigma = nn.softplus(prior_params["sigma"])
                epsilon = nn.softplus(prior_params["epsilon"])
            else:
                sigma = jnp.asarray(prior_kwargs["sigma"])
                epsilon = jnp.asarray(prior_kwargs["epsilon"])
                print(f"Initialized prior with sigma {prior_kwargs['sigma']} and epsilon {prior_kwargs['epsilon']}")

            if prior_type == "repulsion":
                _prior_fn = custom_energy.generic_repulsion_neighborlist(
                    displacement_fn, sigma=sigma, epsilon=epsilon,
                    species=kwargs["species"],
                    r_onset=0.9 * config["model"]["r_cutoff"],
                    r_cutoff=config["model"]["r_cutoff"],
                    initialize_neighbor_list=False,
                    per_particle=per_particle
                )
            elif prior_type == "truncated_lennard_jones":
                _prior_fn = custom_energy.truncated_lennard_jones_neighborlist(
                    displacement_fn, sigma=sigma, epsilon=epsilon,
                    species=kwargs["species"],
                    initialize_neighbor_list=False,
                    per_particle=per_particle
                )
            elif prior_type == "lennard_jones":
                _prior_fn = custom_energy.customn_lennard_jones_neighbor_list(
                    displacement_fn, sigma=sigma, epsilon=epsilon,
                    species=kwargs["species"],
                    r_onset=0.9 * config["model"]["r_cutoff"],
                    r_cutoff=config["model"]["r_cutoff"],
                    initialize_neighbor_list=False,
                    per_particle=per_particle, box_size=jnp.asarray(0.0)
                )

            else:
                raise NotImplementedError(f"Prior {prior_type} not implemented.")

            return _prior_fn(pos, neighbor, **kwargs)

        def init_prior_fn(key, r, nbrs, **kwargs):
            split1, split2 = random.split(key)
            if prior_kwargs.get("train", True):
                _params = {
                    "sigma": random.normal(split1, (n_species,)),
                    "epsilon": random.normal(split2, (n_species,))
                }
            else:
                _params = None
            return _params
    else:
        raise NotImplementedError(f"Prior {prior_type} not implemented.")

    def init_fn(key, r, nbrs, **kwargs):
        _params = {
            "prior": init_prior_fn(key, r, nbrs, **kwargs),
            "neural_network": _init_fn(key, r, nbrs, **kwargs)
        }
        return _params

    def energy_fn_template(energy_params):
        def energy_fn(pos, neighbor, mode=None, **dynamic_kwargs):
            assert 'species'  in dynamic_kwargs.keys()

            energies = {}

            energies["prior"] = apply_prior_fn(energy_params["prior"], pos, neighbor, **dynamic_kwargs)

            if model_type == "DimeNetPP":
                species = dynamic_kwargs.pop("species") + 1

                energies["neural_network"] = gnn_energy_fn(
                    energy_params["neural_network"], pos, neighbor,
                    species=species, **dynamic_kwargs
                )
            else:
                energies["neural_network"] = gnn_energy_fn(
                    energy_params["neural_network"], pos, neighbor, **dynamic_kwargs
                )

            # Restrict the magnitude of the potential by the prior
            # TODO

            for key in dynamic_kwargs:
                if key.startswith("lambda_"):
                    assert key.strip("lambda_") in energies.keys(), f"Unknown energy component {key.strip('lambda_')}"

            # Return the individual components to enable application of the
            # MBAR approach
            if mode == "components":
                return energies
            else:
                pot = 0.0
                for key, val in energies.items():
                    pot += dynamic_kwargs.get(f"lambda_{key}", 1.0) * val
                return pot

        return energy_fn

    if dataset is None:
        return energy_fn_template

    # Set up NN model
    init_kwargs = {
        "box": jnp.asarray(dataset['training']['box'][0]),
        "species": jnp.asarray(dataset['training']['species'][0]),
    }
    r_init = jnp.asarray(dataset['training']['R'][0])
    nbrs_init = nbrs_init.update(r_init, **init_kwargs)

    key = random.PRNGKey(config.get("seed", 0))

    # Load a pretrained model
    init_params = init_fn(
        key, r_init, nbrs_init, **init_kwargs
    )

    print(f"Initial energy is {jax.jit(energy_fn_template(init_params))(r_init, nbrs_init, **init_kwargs)}")

    if not per_particle:
        print(f"Initial forces are {jax.jit(jax.grad(energy_fn_template(init_params)))(r_init, nbrs_init, **init_kwargs)}")

    return energy_fn_template, init_params


def init_optimizer(config, dataset):

    transition_steps = int(
        config["optimizer"]["epochs"] * dataset['training']['U'].size
    ) // config["optimizer"]["batch"]

    lr_schedule_fm = optax.exponential_decay(
        config["optimizer"]["init_lr"], transition_steps, decay_rate=config["optimizer"]["lr_decay"])
    optimizer_fm = optax.chain(
        optax.scale_by_adam(**config["optimizer"]["optimizer_kwargs"]),
        optax.add_decayed_weights(config["optimizer"]["weight_decay"]),
        optax.scale_by_learning_rate(lr_schedule_fm, flip_sign=True),
    )

    return optimizer_fm


def init_difftre_optimizer(config):
    transition_steps = int(
        config["optimizer"]["epochs"] * len(config["confs"]) / config["optimizer"]["batch"]
    )

    lr_schedule_fm = optax.exponential_decay(
        config["optimizer"]["init_lr"], transition_steps, decay_rate=config["optimizer"]["lr_decay"])
    optimizer_fm = optax.chain(
        optax.scale_by_adam(**config["optimizer"]["optimizer_kwargs"]),
        optax.add_decayed_weights(config["optimizer"]["weight_decay"]),
        optax.scale_by_learning_rate(lr_schedule_fm, flip_sign=True),
    )

    return optimizer_fm


def load_confs(config, fractional=True):

    _data = []
    for conf in config["confs"]:
        extra_args = {}
        if isinstance(conf, str):
            (box, coords, mass, species) = io.load_box(conf)
        elif len(conf) == 2:
            conf, temp = conf

            (box, coords, mass, species) = io.load_box(conf)
            extra_args["kT"] = temp * quantity.kb
        else:
            conf, pvol, temp, press = conf
            (box, coords, mass, species) = io.load_box(conf)

            # Resize the box to the specified volume
            assert fractional, "Only fractional coordinates are supported"

            vol = snapshot_quantity.volume(coords.shape[-1], box)
            box *= (pvol * coords.shape[0] / vol) ** (1 / 3)

            extra_args["kT"] = temp * quantity.kb
            extra_args["pressure"] = 1e4 * press / 0.001661  # Convert from GPa

        if fractional:
            coords = jnp.einsum("ij,nj->ni", jnp.linalg.inv(box), coords)

        # We only have one type of species
        species = jnp.zeros_like(species, dtype=jnp.int32)

        extra_args.update({
            "box": box,
            "R": coords,
            "species": species,
            "mass": mass,
        })
        _data.append(extra_args)

    # Padd the data to have the same number of atoms
    max_atoms = max([d["R"].shape[0] for d in _data])
    n_confs = len(_data)
    data = {
        "box": onp.zeros((n_confs, 3, 3)),
        "R": onp.zeros((n_confs, max_atoms, 3)),
        "species": onp.zeros((n_confs, max_atoms), dtype=jnp.int32),
        "mass": onp.zeros((n_confs, max_atoms)),
        "mask": onp.zeros((n_confs, max_atoms), dtype=jnp.bool),
        "kT": onp.zeros((n_confs,)),
        "pressure": onp.zeros((n_confs,)),
    }
    for idx, d in enumerate(_data):
        data["box"][idx, :] = d["box"]
        data["R"][idx, :d["R"].shape[0], :] = d["R"]
        data["species"][idx, :d["species"].shape[0]] = d["species"]
        data["mass"][idx, :d["mass"].shape[0]] = d["mass"]
        data["mask"][idx, :d["mass"].shape[0]] = True
        if "kT" in d:
            data["kT"][idx] = d["kT"]
        else:
            data.pop("kT", None)
        if "pressure" in d:
            data["pressure"][idx] = d["pressure"]
        else:
            data.pop("pressure", None)

    return jax.tree.map(jnp.asarray, data)


def init_simulator(config, shift_fn, simulator=None):
    """Initializes simulator"""
    simulator_template = functools.partial(
        simulator, shift_fn=shift_fn,
        dt=config["timings"]["dt"], **config["simulator_settings"]
    )

    timings = sampling.process_printouts(
        config["timings"]["dt"], config["timings"]["total_time"],
        config["timings"]["t_equilib"], config["timings"]["print_every"]
    )

    return simulator_template, timings


def init_difference_template(energy_fn_template, ref_params):
    @functools.wraps(energy_fn_template)
    def wrapper(params):
        params = jax.tree.map(jnp.add, ref_params, params)
        return energy_fn_template(params)

    init_params = jax.tree.map(jnp.zeros_like, ref_params)
    return wrapper, init_params


def init_minimization(config, shift_fn, energy_fn_template, init_params):
    energy_fn = energy_fn_template(init_params)

    def init_apply_fn(neighbor, **kwargs):

        @jax.jit
        def apply_fn(pos):
            nbrs = neighbor.update(pos, **kwargs)
            return energy_fn(pos, neighbor=nbrs, **kwargs)

        return apply_fn

    max_iter = config["minimize"]["max_iter"]
    tol = config["minimize"]["tol"]
    max_disp = config["minimize"]["max_displacement"]

    def minimize_with_lbfs(init_pos, init_nbrs, **kwargs):
        lbfgs_opt = optax.lbfgs()
        value_fn = init_apply_fn(init_nbrs, **kwargs)

        def step(carry):
            pos, state = carry

            value, grad = optax.value_and_grad_from_state(value_fn)(pos, state=state)
            updates, state = lbfgs_opt.update(
                grad, state, pos, value=value, grad=grad, value_fn=value_fn
            )

            # jax.debug.print("Energy: {}, Max. force: {}", value, jnp.linalg.norm(grad).max())

            pos = shift_fn(pos, jnp.clip(updates, -max_disp, max_disp))
            return pos, state

        def continuing_criterion(carry):
            _, state = carry
            iter_num = optax.tree_utils.tree_get(state, 'count')
            grad = optax.tree_utils.tree_get(state, 'grad')
            err = optax.tree_utils.tree_l2_norm(grad)
            return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

        init_carry = (init_pos, lbfgs_opt.init(init_pos))
        min_pos, min_state = jax.lax.while_loop(
            continuing_criterion, step, init_carry
        )

        jax.debug.print('Minimized from {} energy to {}', value_fn(init_pos), value_fn(min_pos))

        return min_pos, min_state

    return minimize_with_lbfs


def create_out_dir(config, prefix=""):
    now = datetime.datetime.now()
    prefix += "_"

    model = config["model"].get("type", "DimeNet")
    name = f"titanium_{prefix}{model}_r_cutoff_{config['model']['r_cutoff']}_{now.year}_{now.month}_{now.day}_{uuid.uuid4()}"

    out_dir = Path("../output") / name
    out_dir.mkdir(exist_ok=False, parents=True)

    # Save the config values
    with open(out_dir / "config.toml", "wb") as f:
        tomli_w.dump(config, f)

    return out_dir


def plot_convergence(trainer, out_dir):
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5),
                                        layout="constrained")

    ax1.set_title("Loss")
    ax1.semilogy(trainer.train_losses, label="Training")
    ax1.semilogy(trainer.val_losses, label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    fig.savefig(out_dir / f"convergence.pdf", bbox_inches="tight")


def save_training_results(config, out_dir, trainer: ForceMatching, *rest):
    # Save the config values
    with open(out_dir / "config.toml", "wb") as f:
        tomli_w.dump(config, f)

    # Save all the outputs
    trainer.save_energy_params(out_dir / "best_params.pkl", ".pkl", best=True)
    trainer.save_energy_params(out_dir / "final_params.pkl", ".pkl", best=True)
    trainer.save_trainer(out_dir / "trainer.pkl", ".pkl")

    for idx, r in enumerate(rest):
        r.save_trainer(out_dir / f"trainer_{idx}.pkl")


def save_predictions(out_dir, name, predictions):
    predictions = tree_util.tree_map(
        onp.asarray, predictions
    )

    onp.savez(out_dir / f"{name}.npz", **predictions)

def plot_predictions(predictions, reference_data, out_dir, name):
    scale_energy = 96.4853722  # [eV] ->   [kJ/mol]
    scale_pos = 0.1  # [Å] -> [nm]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 5),
                                        layout="constrained")

    fig.suptitle("Predictions")

    mae = onp.mean(onp.abs(
        predictions['U'] - reference_data['U'])) / scale_energy / 256
    ax1.set_title(f"Energy (MAE: {mae * 1000:.1f} meV/atom)")
    ax1.plot(reference_data['U'] / scale_energy / 256,
             predictions['U'] / scale_energy / 256, "*")
    ax1.set_xlabel("Ref. U [eV/atom]")
    ax1.set_ylabel("Pred. U [eV/atom]")

    mae = onp.mean(onp.abs(predictions['F'] - reference_data[
        'F'])) / scale_energy * scale_pos
    ax2.set_title(f"Force (MAE: {mae * 1000:.1f} meV/A)")
    ax2.plot(reference_data['F'][::50].ravel() / scale_energy * scale_pos,
             predictions['F'][::50].ravel() / scale_energy * scale_pos,
             "*")
    ax2.set_xlabel("Ref. F [eV/A]")
    ax2.set_ylabel("Pred. F [eV/A]")

    mae = onp.mean(onp.abs(predictions['virial'] - reference_data[
        'virial'])) / scale_energy * (scale_pos ** 3)
    ax3.set_title(f"Virial (MAE: {mae * 1000:.1f} meV/A^3)")
    ax3.plot(reference_data['virial'][
                 reference_data['type'] == 0].ravel() / scale_energy * (
                     scale_pos ** 3), predictions['virial'][
                 reference_data['type'] == 0].ravel() / scale_energy * (
                     scale_pos ** 3), "*")
    ax3.set_xlabel("Ref. W [eV/A^3]")
    ax3.set_ylabel("Pred. W [eV/A^3]")

    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
