import jax
import jax.numpy as jnp
import numpy as np

from ase.io import read
from time import monotonic
import yaml

from jaxpme import PME, Ewald

atomss = read("crystals.xyz", ":")
# we generate 10 rattled samples over which we scan,
# and repeat the whole procedure 5 times
repeats = 5
samples = 10


def timed_run(calculator, inputs, repeats, warmup=True):
    # we make a function to lax.scan over the number of samples,
    # which makes a dummy change to positions (to avoid caching)
    # based on previous outputs (to enforce data dependences)
    # this should keep the JIT compiler from parallelising or
    # caching the whole calculation
    @jax.jit
    def calculate_fn(carry, ignored):
        inp, x = carry
        charges, cell, positions, *rest = inp
        positions += x

        energy, forces, stress = calculator.energy_forces_stress(
            charges, cell, positions, *rest
        )
        x = (energy - 1.0) / energy
        x += (forces[0, 0] - 1.0) / forces[0, 0]
        x += (stress[0, 0] - 1.0) / stress[0, 0]

        return (inp, x), (energy, forces, stress)

    # this does tracing and jitting
    if warmup:
        result = jax.lax.scan(calculate_fn, (inputs, jnp.array(0.0)), length=samples)
        result[-1][0].block_until_ready()
        result[-1][1].block_until_ready()
        result[-1][2].block_until_ready()

    start = monotonic()
    for _ in range(repeats):
        result = jax.lax.scan(calculate_fn, (inputs, jnp.array(0.0)), length=samples)
        result[-1][0].block_until_ready()
        result[-1][1].block_until_ready()
        result[-1][2].block_until_ready()

    end = monotonic()
    duration = end - start
    time_per_calculation = duration / (repeats * samples)
    return time_per_calculation


def execute(atoms, hypers, mode="PME"):
    N = len(atoms)
    print(f"N = {N}")

    charges = atoms.arrays["initial_charges"]

    if mode == "PME":
        calc = PME()
        inputs = calc.prepare(
            atoms, charges, hypers["cutoff"], hypers["mesh_spacing"], hypers["smearing"]
        )
    elif mode == "Ewald":
        calc = Ewald()
        inputs = calc.prepare(
            atoms, charges, hypers["cutoff"], hypers["lr_wavelength"], hypers["smearing"]
        )

    time_per_calculation = timed_run(calc, inputs, repeats, warmup=True)

    print(f"... -> {1e3*time_per_calculation:.3f}ms")

    return time_per_calculation


for mode in ["Ewald", "PME"]:
    print(f"mode={mode}")

    with open(f"{mode.lower()}_parameters.yaml", "r") as f:
        hyperss = yaml.safe_load(f)

    results = {}

    for i, atoms in enumerate(atomss):
        crystal_name = atoms.info["crystal_name"]
        N = len(atoms)

        # avoid OOM events
        if mode == "Ewald" and N > 40000:
            continue

        hypers = hyperss[i]
        time = execute(atoms, hypers, mode=mode)

        if crystal_name in results:
            results[crystal_name].append([len(atoms), time])

        else:
            results[crystal_name] = [[len(atoms), time]]

        r = {k: np.array(v) for k, v in results.items()}

        np.savez_compressed(f"{mode}_results.npz", **r)
