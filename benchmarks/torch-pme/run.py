import torch
from ase.io import read
import numpy as np
from vesin import NeighborList
import torchpme
from time import monotonic
import yaml

atomss = read("crystals.xyz", ":")

# we iterate over 10 samples at a time,
# repeating 5 times overall
repeats = 5
samples = 10

max_power = 6

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

hypers = {}

with open("p3m_parameters.yaml", "r") as f:
    hypers["P3M"] = yaml.safe_load(f)

with open("pme_parameters.yaml", "r") as f:
    hypers["PME"] = yaml.safe_load(f)

with open("ewald_parameters.yaml", "r") as f:
    hypers["Ewald"] = yaml.safe_load(f)


def compute_distances(positions, neighbor_indices, cell=None, neighbor_shifts=None):
    """Compute pairwise distances."""
    atom_is = neighbor_indices[:, 0]
    atom_js = neighbor_indices[:, 1]

    pos_is = positions[atom_is]
    pos_js = positions[atom_js]

    distance_vectors = pos_js - pos_is

    if cell is not None and neighbor_shifts is not None:
        shifts = neighbor_shifts.type(cell.dtype)
        distance_vectors += shifts @ cell
    elif cell is not None and neighbor_shifts is None:
        raise ValueError("Provided `cell` but no `neighbor_shifts`.")
    elif cell is None and neighbor_shifts is not None:
        raise ValueError("Provided `neighbor_shifts` but no `cell`.")

    return torch.linalg.norm(distance_vectors, dim=1)


def atoms_to_inputs(atoms, device, cutoff):
    charges = (
        torch.from_numpy(atoms.arrays["initial_charges"]).type(torch.float32).to(device)
    )
    charges = charges.reshape(-1, 1)
    positions = torch.from_numpy(atoms.positions).type(torch.float32).to(device)
    cell = torch.from_numpy(atoms.cell.array).type(torch.float32).to(device)

    nl = NeighborList(cutoff=cutoff, full_list=False)

    i, j, S = nl.compute(
        points=positions.cpu(), box=cell.cpu(), periodic=True, quantities="ijS"
    )

    i = torch.from_numpy(i.astype(int)).to(device)
    j = torch.from_numpy(j.astype(int)).to(device)
    neighbor_indices = torch.stack([i, j], dim=1).to(device)
    neighbor_shifts = torch.from_numpy(S.astype(int)).to(device)

    return charges, positions, cell, neighbor_indices, neighbor_shifts


def get_calculate_fn(potential):
    def calculate(charges, positions, cell, neighbor_indices, neighbor_shifts):
        positions.requires_grad = True
        cell.requires_grad = True
        distances = compute_distances(positions, neighbor_indices, cell, neighbor_shifts)
        potentials = potential(charges, cell, positions, neighbor_indices, distances)
        energy = (charges * potentials).sum()
        forces = -torch.autograd.grad(energy, positions, retain_graph=True)[0]
        cell_derivative = torch.autograd.grad(energy, cell)[0]

        stress = torch.einsum("ia,ib->ab", positions.detach(), -forces) + torch.einsum(
            "Aa,Ab->ab", cell.detach(), cell_derivative
        )

        return energy, forces, stress

    return calculate


def timed_run(calculate_fn, inputs, repeats, device, warmup=True):
    # we disturb input geometries slightly based on previous results
    # to make sure that there is data dependence betwen subsequent
    # evaluations -- this makes sure that there's no unfair optimisation
    # and mimics a kind of MD setting

    inputs = list(inputs)

    if warmup:
        e = torch.tensor(0.0, device=device)
        for _ in range(samples):
            inputs = [i.detach() for i in inputs]
            e = e.detach()
            inputs[1] += (e - 1) / e
            e, f, s = calculate_fn(*inputs)

        if device == "cuda":
            torch.cuda.synchronize()

    start = monotonic()
    for _ in range(repeats):
        for _ in range(samples):
            inputs = [i.detach() for i in inputs]
            e = e.detach()
            inputs[1] += (e - 1) / e
            e, f, s = calculate_fn(*inputs)

        if device == "cuda":
            torch.cuda.synchronize()
    end = monotonic()
    duration = end - start
    time_per_calculation = duration / (repeats * samples)
    return time_per_calculation


def execute(atoms, device, hypers, mode="P3M"):
    N = len(atoms)
    print(f"N = {N}")

    cutoff = hypers["cutoff"]
    inputs = atoms_to_inputs(atoms, device, cutoff)
    potential = torchpme.CoulombPotential(smearing=hypers["smearing"])

    if mode == "P3M":
        calculator = torchpme.P3MCalculator(
            potential,
            hypers["mesh_spacing"],
            interpolation_nodes=hypers["interpolation_nodes"],
            full_neighbor_list=False,
        )
    elif mode == "PME":
        calculator = torchpme.PMECalculator(
            potential,
            hypers["mesh_spacing"],
            interpolation_nodes=hypers["interpolation_nodes"],
            full_neighbor_list=False,
        )
    elif mode == "Ewald":
        calculator = torchpme.EwaldCalculator(
            potential,
            hypers["lr_wavelength"],
            full_neighbor_list=False,
        )

    calculator = torch.jit.script(calculator)
    calculator = calculator.to(device)
    calculate = get_calculate_fn(calculator)
    time_per_calculation = timed_run(calculate, inputs, repeats, device, warmup=True)
    print(f"... -> {1e3*time_per_calculation:.3f}ms")

    return time_per_calculation


for mode in ["P3M", "PME", "Ewald"]:
    print(f"device={device}, mode={mode}")

    results = {}

    for i, atoms in enumerate(atomss):
        N = len(atoms)

        h = hypers[mode][i]

        if mode == "Ewald" and N > 64000:
            # this causes OOM otherwise
            continue

        crystal_name = atoms.info["crystal_name"]
        try:
            time = execute(atoms, device, h, mode=mode)
        except torch.OutOfMemoryError:
            print("... encountered OOM, continue")
            continue
        except RuntimeError:
            print("... encountered OOM/Runtime error, continue")
            continue

        if crystal_name in results:
            results[crystal_name].append([N, time])

        else:
            results[crystal_name] = [[N, time]]

        r = {k: np.array(v) for k, v in results.items()}

        np.savez_compressed(f"{mode}_results.npz", **r)
