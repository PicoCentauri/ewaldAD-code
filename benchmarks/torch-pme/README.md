# `torch-pme` benchmark

This runs a basic benchmark that mimics an MD setting: Samples are processed sequentially with pre-computed neighborlists and settings. 

Files:

- `run.py` runs the benchmark
- `crystals.xyz` contains the used structures (needs to be unzipped and copied here from parent)
- `*_parameters.yaml` contains the settings used for each structure
- `*_results.npz` contains the results, grouped by crystal prototype

This benchmark was performed with `torch-pme` commit `fac5b1e7a11a50ebcbadeae6c658f2034c1edae3`, `torch==2.4.1`, and CUDA 12.4.1, on a H100.
