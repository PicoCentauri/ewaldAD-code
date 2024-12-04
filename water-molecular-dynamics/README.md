# Running Molecular Dynamics of Water

This folder contains the following files required to run an NpT simulation of 256 rigid
SPC/E molecules using [LAMMPS](https://www.lammps.org):

- **[model.pt](model.pt)**: The trained machine learning model used for the molecular
  dynamics simulation with `torch-pme` and Metatensor.
- **[water-mts.in](water-mts.in)**: The LAMMPS input file for running the simulation
  with the Metatensor-enabled version.
- **[water-lmp.in](water-lmp.in)**: The LAMMPS input file for running the simulation
  with the pure LAMMPS binary using the P3M algorithm.
- **[water.structure](structure.data)**: The initial structure file describing the
  atomic positions and types in the simulation.

## Using `torch-pme` and `Metatensor`

You can recreate the model using the tutorial in the [torch-pme
documentation](https://lab-cosmo.github.io/torch-pme/latest/examples/9-atomistic-model.html).
Replace the `add_charges` function in the tutorial with the following function to use
the correct charges for SPC/E, which are `-0.8476` for oxygen and `0.4238` for hydrogen:

```python
charge_dict = {8: -0.8476}
charge_dict[1] = -charge_dict[8] / 2

def add_charges(system: System) -> None:
    dtype = system.positions.dtype
    device = system.positions.device

    # Set charges
    charges = torch.zeros(len(system), 1, dtype=dtype, device=device)
    for atomic_type, atomic_charge in charge_dict.items():
        charges[atomic_type == system.types] = atomic_charge

    # Create metadata for the charges TensorBlock
    samples = Labels("atom", torch.arange(len(system), device=device).reshape(-1, 1))
    properties = Labels("charge", torch.zeros(1, 1, device=device, dtype=torch.int32))
    data = TensorBlock(
        values=charges, samples=samples, components=[], properties=properties
    )
    system.add_data(name="charges", data=data)
```

## Building the Metatensor-Enabled LAMMPS Version

To run MD, you need to build the [Metatensor-enabled
LAMMPS](https://github.com/metatensor/lammps/) version. For detailed building and usage
instructions, refer to the [Metatensor
documentation](https://docs.metatensor.org/latest/atomistic/engines/lammps.html#engine-lammps).
Here’s a quick overview:

1. Clone the repository:

   ```bash
   git clone https://github.com/metatensor/lammps lammps-metatensor
   cd lammps-metatensor

   # Patch a bug in PyTorch's MKL detection
   ./src/ML-METATENSOR/patch-torch.sh "$TORCH_PREFIX"
   ```

2. After cloning the repository and ensuring you have a [PyTorch
   installation](https://pytorch.org/get-started/locally/), build the
   Metatensor-enabled version:
   
   ```bash # Set the path to your C++ libtorch installation
   TORCH_PREFIX=<path/to/torch/installation>

   # If you installed PyTorch via Python, use:

   TORCH_PREFIX=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")

   mkdir build-mts && cd build-mts

   cmake -DPKG_BODY=yes \
    -DPKG_EXTRA-DUMP=yes \
    -DPKG_MOLECULE=yes \
    -DPKG_RIGID=yes \
    -DPKG_ML-METATENSOR=yes \
    -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" \
    ../cmake

   cmake --build . --parallel 4 # Use make -jX as an alternative
   ```

3. Run the simulation for 2 ns with one thread on a single GPU:

## Using LAMMPS’ P3M

For comparison, you can build a version that uses LAMMPS’ P3M code, with its GPU package
handling the long-range electrostatics instead of torch-pme.

1.	Build the LAMMPS version with P3M:

    ```bash
    mkdir build-lmp && cd build-lmp

    cmake -DPKG_BODY=yes \
        -DPKG_EXTRA-DUMP=yes \
        -DPKG_MOLECULE=yes \
        -DPKG_RIGID=yes \
        -DPKG_KSPACE=yes \
        -DPKG_GPU=yes \
        ../cmake

    cmake --build . --parallel 4 # Use make -jX as an alternative
    ```

2.	Run the simulation with the pure LAMMPS binary:

## Output

Both runs will generate a single XTC trajectory file that stores the positions after
every 200 steps. These files, along with the [structure file](water.structure), can be
analyzed with common libraries like [MDAnalysis](https://www.mdanalysis.org).

Happy simulating! 😊
