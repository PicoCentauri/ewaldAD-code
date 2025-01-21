#!/usr/bin/env python3

import time
import ase
import numpy as np
import torch
import vesin.torch
from torchpme import EwaldCalculator, CoulombPotential, PMECalculator, P3MCalculator
from torchpme.tuning import tune_ewald, tune_pme, tune_p3m
import pickle
import math

from helpers import define_crystal


def compute_error(
    atoms: ase.Atoms,
    accuracy: float,
    energy_ref: float,
    method: str,
    dtype: torch.dtype,
    device: str,
) -> tuple[float, float]:
    positions = torch.tensor(atoms.positions, dtype=dtype, device=device)
    charges = torch.tensor(
        atoms.get_initial_charges(), dtype=dtype, device=device
    ).unsqueeze(1)
    cell = torch.tensor(atoms.cell.array, dtype=dtype, device=device)
    cutoff = 4.4

    if method == "ewald":
        smearing, params = tune_ewald(
            charges=charges, cell=cell, positions=positions, cutoff=cutoff
        )
        calculator = EwaldCalculator(CoulombPotential(smearing), **params)
    elif method == "pme":
        smearing, params = tune_pme(
            charges=charges, cell=cell, positions=positions, cutoff=cutoff
        )
        calculator = PMECalculator(CoulombPotential(smearing), **params)
    elif method == "p3m":
        smearing, params = tune_p3m(
            charges=charges, cell=cell, positions=positions, cutoff=cutoff
        )
        calculator = P3MCalculator(CoulombPotential(smearing), **params)
    else:
        raise ValueError(f"Invalid method {method}. Choose ewald or pme.")

    # vesin requires double precision
    nl = vesin.torch.NeighborList(cutoff=cutoff, full_list=False)
    i, j, neighbor_distances = nl.compute(
        points=positions.to(dtype=torch.float64, device="cpu"),
        box=cell.to(dtype=torch.float64, device="cpu"),
        periodic=True,
        quantities="ijd",
    )

    neighbor_indices = torch.stack([i, j], dim=1).to(device=device)
    neighbor_distances = neighbor_distances.to(dtype=dtype, device=device)

    potentials = torch.zeros([len(positions), 1], dtype=dtype, device=device)

    # warm-up
    for _ in range(5):
        calculator.forward(
            charges=charges,
            positions=positions,
            cell=cell,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
        )

    n_samples = 50
    t0 = time.time()
    for _ in range(n_samples):
        potentials += calculator.forward(
            charges=charges,
            positions=positions,
            cell=cell,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
        )
    if device is torch.device("cuda"):
        torch.cuda.synchronize()

    execution_time = (time.time() - t0) / n_samples

    energy = torch.sum(charges * potentials / n_samples)
    accuracy = float(torch.abs(energy - energy_ref))
    accuracy = float(torch.abs((energy - energy_ref) / energy_ref))

    return accuracy, execution_time


def compute(
    target_accuracies: np.array,
    method: str,
    dtype: torch.dtype,
    device: str,
) -> dict:
    results = {}

    for crystal_name in crystal_symbols:
        print(f"Computing {crystal_name}..." + 10 * " ", end="\r")

        positions, charges, cell, madelung_ref, num_formula_units = define_crystal(
            crystal_name
        )

        atoms_unitcell = ase.Atoms(
            symbols=len(positions) * ["H"],
            positions=positions,
            charges=charges.flatten(),
            pbc=True,
            cell=cell,
        )

        rep = [8, 8, 8]
        atoms = atoms_unitcell.repeat(rep)

        results[crystal_name] = {}
        results[crystal_name]["acc_target"] = target_accuracies.copy()
        results[crystal_name]["acc_actual"] = np.zeros_like(target_accuracies)
        results[crystal_name]["time"] = np.zeros_like(target_accuracies)

        for i, accuracy in enumerate(target_accuracies):
            acc, exc_time = compute_error(
                atoms=atoms,
                accuracy=accuracy,
                energy_ref=-madelung_ref * num_formula_units * math.prod(rep),
                method=method,
                dtype=dtype,
                device=device,
            )
            results[crystal_name]["acc_actual"][i] = acc
            results[crystal_name]["time"][i] = exc_time

    return results


crystal_symbols = [
    "CsCl",
    "NaCl_primitive",
    "NaCl_cubic",
    "zincblende",
    "wurtzite",
    "cu2o",
    "fluorite",
]

target_accuracies = np.logspace(-1, -8, 8)

if torch.cuda.is_available():
    print("Using CUDA")
    cuda = True
    device = "cuda"
else:
    print("Using CPU")
    cuda = False
    device = "cpu"


for dtype in [torch.float64, torch.float32]:
    for method in ["ewald", "pme", "p3m"]:
        print(
            f"Running {method} with {str(dtype)[-2:]} precision...",
        )
        results = compute(
            target_accuracies=target_accuracies,
            method=method,
            dtype=dtype,
            device=device,
        )

        pickle.dump(results, open(f"results_{method}_{str(dtype)[-2:]}.pkl", "wb"))
