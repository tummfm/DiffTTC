"""Functions to evaluate single simulations."""

import re
from pathlib import Path

import mdtraj
import numpy as onp
import scipy

from chemtrain import quantity

def compute_fe(fE, fl, bE, bl, k, natoms, T, Va, P, m=47.87):
    fE *= 4.184 # Convert from kcal/mol to kJ/mol
    bE *= 4.184 # Convert from kcal/mol to kJ/mol
    k *= 4.184 * 100. # Convert from kcal/mol/Å² to kJ/mol/nm²
    Va /= 1000. # Convert from Å³ to nm³
    P *= 602.2  # Convert from GPa to kJ/mol/nm^3

    kT = quantity.kb * T
    hbar = 0.06351 # [kJ/mol·ps]

    fW = onp.trapezoid(fE, fl)
    bW = onp.trapezoid(bE, bl)

    W = 0.5 * (fW - bW)

    # Define harmonic reference system
    omega = onp.sqrt(k / m) # [1/ps
    F_harm = 3 * natoms * kT * onp.log(hbar * omega / kT) # [kJ/mol].

    # Fixed center of mass correction [Eq.(24) in the paper].
    V = Va * natoms
    F_CM = kT * onp.log(natoms / V * (2 * onp.pi * kT / (natoms * k)) ** (3 / 2)) # [kcal/mol].

    # Compute absolute free energy per atom [Eq.(16) in the paper] and save data.
    # Add the pressure contribution to obtain the gibbs free energy
    F = (F_harm + W + F_CM + P * V) / natoms # [eV/atom].

    return F

def compute_fe_rs(F0, fE, fl, bE, bl, T):
    # Convert to kJ/mol
    fE *= 4.184
    bE *= 4.184

    fE /= fl
    bE /= bl

    # Compute work done using cummulative integrals [Eq.(21) in the paper].
    I_f = scipy.integrate.cumulative_trapezoid(fE,fl,initial=0)
    I_b = scipy.integrate.cumulative_trapezoid(bE[::-1],bl[::-1],initial=0)
    W = (I_f + I_b) / (2 * fl)

    Tl = T / fl
    Fl = F0 / fl + 1.5 * quantity.kb * Tl * onp.log(fl) + W

    return Tl, Fl

def parse_dir(path, structure="hcp"):
    if structure == "hcp":
        _pattern = re.compile(r'.*_T_([0-9]+(?:\.[0-9]+)?)_P_([0-9]+(?:\.[0-9]+)?)_k_([0-9]+(?:\.[0-9]+)?)_a_([0-9]+(?:\.[0-9]+)?)_ca_([0-9]+(?:\.[0-9]+)?)')
    else:
        _pattern = re.compile(r'.*_T_([0-9]+(?:\.[0-9]+)?)_P_([0-9]+(?:\.[0-9]+)?)_k_([0-9]+(?:\.[0-9]+)?)_a_([0-9]+(?:\.[0-9]+)?)')

    hcp_paths = [p.parent for p in path.glob(f"frenkel*/{structure}*forward*.csv")]

    for test_path in hcp_paths:
        # Read out estimated temp and simulated pressure from filename
        m = _pattern.search(test_path.name)
        if not m:
            continue

        with open(next(test_path.glob(f"{structure}_forward*.lammpstrj")), "r") as f:
            for line in f:
                if line.strip() == "ITEM: NUMBER OF ATOMS":
                    natoms = int(next(f).strip())

        res = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
        if structure == "hcp":
            res = res + (float(m.group(5)),)

        yield test_path, *res, natoms

def parse_dir_rs(path, structure="hcp"):
    _pattern = re.compile(r'.*_T_([0-9]+(?:\.[0-9]+)?)_P_([0-9]+(?:\.[0-9]+)?)')

    matched_paths = [p.parent for p in path.glob(f"reversible*/{structure}*forward*.csv")]

    if len(matched_paths) == 0:
        matched_paths = [p.parent for p in path.glob(f"reversible*{structure}/forward*.csv")]

    if len(matched_paths) == 0:
        raise ValueError(f"No paths found in {path} for structure {structure}")

    for test_path in matched_paths:
        # Read out estimated temp and simulated pressure from filename
        m = _pattern.search(test_path.name)
        if not m:
            continue

        yield test_path, float(m.group(1)), float(m.group(2))


def batch_fe_hcp(path):
    runs = list(parse_dir(path))
    runs.sort(key=lambda r: r[2]) # Sort by pressure

    temps, press, fa = [], [], []
    for (p, T, P, k, a, ca, natoms) in runs:
        fE, fl = onp.loadtxt(next(p.glob("hcp_forward_*.csv")), skiprows=0, delimiter=" ", unpack=True)
        bE, bl = onp.loadtxt(next(p.glob("hcp_backward_*.dat")), skiprows=0, delimiter=" ", unpack=True)

        # Volume per atom for HCP cell
        Va = a ** 3 * onp.sqrt(3) * ca / 4 # Å³/atom

        Fa = compute_fe(fE, fl, bE, bl, k=k, natoms=natoms, T=T, Va=Va, P=P)

        temps.append(T)
        press.append(P)
        fa.append(Fa)

    return onp.asarray(temps), onp.asarray(press), onp.asarray(fa), onp.asarray(Va)

def batch_fe_bcc(path):
    runs = list(parse_dir(path, structure="bcc"))
    runs.sort(key=lambda r: r[2]) # Sort by pressure

    temps, press, fa = [], [], []
    for (p, T, P, k, a, natoms) in runs:
        fE, fl = onp.loadtxt(next(p.glob("bcc_forward_*.csv")), skiprows=0, delimiter=" ", unpack=True)
        bE, bl = onp.loadtxt(next(p.glob("bcc_backward_*.dat")), skiprows=0, delimiter=" ", unpack=True)

        # Volume per atom for BCC cell
        Va = a ** 3 / 2 # Å³/atom

        Fa = compute_fe(fE, fl, bE, bl, k=k, natoms=natoms, T=T, Va=Va, P=P)

        temps.append(T)
        press.append(P)
        fa.append(Fa)

    return onp.asarray(temps), onp.asarray(press), onp.asarray(fa), onp.asarray(Va)


def batch_rs(Fa, path, max=2, structure="hcp"):
    runs = list(parse_dir_rs(path, structure=structure))
    runs.sort(key=lambda r: r[2]) # Sort by pressure
    runs = runs[:max]

    temps, press, fa = [], [], []
    for idx, (p, T, P) in enumerate(runs):
        fE, fl = onp.loadtxt(next(p.glob(f"*forward_*.csv")), skiprows=0, delimiter=" ", unpack=True)
        bE, bl = onp.loadtxt(next(p.glob(f"*backward_*.csv")), skiprows=0, delimiter=" ", unpack=True)

        Tl, Fl = compute_fe_rs(Fa[idx], fE, fl, bE, bl, T=T)

        temps.append(Tl)
        press.append(P)
        fa.append(Fl)

    return temps, press, fa

def parse_dir_rs_melting(path, structure="hcp"):
    _pattern = re.compile(r'.*_T_([0-9]+(?:\.[0-9]+)?)_P_([0-9]+(?:\.[0-9]+)?).*')
    hcp_paths = [p.parent for p in path.glob(f"reversible*_{structure}*/forward*.csv") if not "cst" in p.name]

    for test_path in hcp_paths:
        # Read out estimated temp and simulated pressure from filename
        m = _pattern.search(test_path.name)
        if not m:
            continue

        yield test_path, float(m.group(1)), float(m.group(2))


def batch_rs_melting(Fa, path, max=2, structure="hcp"):
    runs = list(parse_dir_rs_melting(path, structure=structure))
    runs.sort(key=lambda r: r[2]) # Sort by pressure
    runs = runs[:max]

    temps, press, fa = [], [], []
    for idx, (p, T, P) in enumerate(runs):
        fE, fl = onp.loadtxt([p for p in p.glob(f"forward_*.csv") if "cst" not in p.name][0], skiprows=0, delimiter=" ", unpack=True)
        bE, bl = onp.loadtxt([p for p in p.glob(f"backward_*.csv") if "cst" not in p.name][0], skiprows=0, delimiter=" ", unpack=True)

        Tl, Fl = compute_fe_rs(Fa[idx], fE, fl, bE, bl, T=T)

        temps.append(Tl)
        press.append(P)
        fa.append(Fl)

    return tuple(onp.asarray(a) for a in [temps, press, fa])


def batch_rs_cst(path, structure="hcp"):
    runs = list(parse_dir_rs_melting(path, structure=structure))
    runs.sort(key=lambda r: r[2]) # Sort by pressure

    temps, press, fcst, bcst = [], [], [], []
    for idx, (p, T, P) in enumerate(runs):
        f = onp.loadtxt([p for p in p.glob(f"forward_*cst.csv")][0], skiprows=0, delimiter=" ", unpack=True)[1:, :]
        b = onp.loadtxt([p for p in p.glob(f"backward_*cst.csv")][0], skiprows=0, delimiter=" ", unpack=True)[1:, :]

        fE, fl = onp.loadtxt([p for p in p.glob(f"forward_*.csv") if "cst" not in p.name][0], skiprows=0, delimiter=" ", unpack=True)
        bE, bl = onp.loadtxt([p for p in p.glob(f"backward_*.csv") if "cst" not in p.name][0], skiprows=0, delimiter=" ", unpack=True)

        Tl,_ = compute_fe_rs(0.0, fE, fl, bE, bl, T=T)

        temps.append(Tl)
        press.append(P)
        fcst.append(f)
        bcst.append(b)

    return onp.asarray(temps), onp.asarray(press), onp.asarray(fcst), onp.asarray(bcst)


def parse_coexistence_dir(path, delta=0):
    _pattern = re.compile(r'coexistence_T_([0-9]+(?:\.[0-9]+)?)_P_([0-9]+(?:\.[0-9]+)?)_delta_(\+?-?[0-9]+(?:\.[0-9]+)?).*')
    coex_paths = [p.parent for p in path.glob(f"*/dump.coex")]

    for test_path in coex_paths:
        # Read out estimated temp and simulated pressure from filename
        m = _pattern.search(test_path.name)
        if not m:
            continue

        if int(m.group(3)) != delta:
            continue

        yield test_path, float(m.group(1)), float(m.group(2)), float(m.group(3))


def glob_coexistence_dirs(path, pattern="aniso_titanium__MACE"):
    _pattern = re.compile(fr'.*EstTemp_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)_{pattern}.*')
    coex_paths = [p.parent for p in path.glob(f"*/dump.coex")]

    for test_path in coex_paths:
        # Read out estimated temp and simulated pressure from filename
        m = _pattern.search(test_path.name)
        if not m:
            continue

        yield test_path, float(m.group(1)), float(m.group(2))


def lmpdat_to_pdb(file: Path):
    section = ""
    coords = None
    idx = 0
    box_length = onp.zeros(3)
    with open(file, "r") as f:
      for line in f:
          if line.strip() == "":
              continue

          if line.startswith("Atoms"):
              section = "atoms"
              continue
          if line.startswith("Velocities"):
              break

          if section == "atoms":
              coords[idx, :] = onp.asarray(line.split(" ")[2:5], dtype=float) / 10.
              idx += 1
              continue

          if re.compile(r"([0-9]+?)\satoms").match(line):
              natoms = int(line.split()[0])
              coords = onp.zeros((natoms, 3))
              continue

          if m := re.compile(r"([0-9]+(?:\.[0-9]+)?)\s([0-9]+(?:\.[0-9]+)?)\s(.)lo\s.?hi").match(line):
              size = (float(m.group(2)) - float(m.group(1))) / 10.
              if m.group(3) == "x":
                  box_length[0] = size
              if m.group(3) == "y":
                  box_length[1] = size
              if m.group(3) == "z":
                  box_length[2] = size
              continue

    top = mdtraj.Topology()
    for idx in range(576):
        c = top.add_chain(f"{idx + 1}")
        r = top.add_residue(f"{idx + 1}", c)
        a = top.add_atom(f"Ti{idx + 1}", mdtraj.element.get_by_symbol("Ti"), r)

    traj = mdtraj.Trajectory(xyz=coords, topology=top, unitcell_lengths=box_length, unitcell_angles=90.*onp.ones_like(box_length))
    traj.save(file.parent / f"{file.stem}.pdb")

def get_diffs(trainer, weight=None, key="free_energy"):
    statepoints = list(trainer["predictions"].keys())
    diffs = []
    for idx in range(len(statepoints) // 2):
        if weight is None:
            w1, w2 = 1.0, 1.0
        else:
            w1 = weight[idx]
            w2 = weight[idx + len(statepoints) // 2]
        diffs.append(
            w1 * onp.asarray([p[key] for p in trainer["predictions"][idx].values()])
            - w2 * onp.asarray([p[key] for p in trainer["predictions"][idx + len(statepoints) // 2].values()])
        )

    return onp.asarray(diffs)


def sorted_list(func):
    def wrapper(*args, **kwargs):
        data = list(func(*args, **kwargs))
        data.sort(key=lambda d: d[0])
        return tuple(onp.asarray(d) for d in zip(*data))
    return wrapper

@sorted_list
def batch_thermal_expansion(base_dir, structure="hcp"):
    _pattern = re.compile(fr'.*thermo_T_([0-9]+(?:\.[0-9]+)?).*\.csv')

    for test_path in base_dir.glob(f"{structure}_thermal*/*.csv"):
        # Read out estimated temp and simulated pressure from filename
        m = _pattern.search(test_path.name)
        if not m:
            continue

        data = onp.loadtxt(test_path, skiprows=1, delimiter=" ", unpack=True)

        if structure == "hcp":
            yield float(m.group(1)), data[0].mean(), data[1].mean(), data[2].mean(), data[3].mean() / 20, (data[4] / data[3] * 2).mean()
        elif structure == "bcc":
            yield float(m.group(1)), data[0].mean(), data[1].mean(), data[2].mean(), data[3].mean() / 16
        elif structure == "liquid":
            yield float(m.group(1)), data[0].mean(), data[1].mean(), data[2].mean(), data[3].mean() / 16