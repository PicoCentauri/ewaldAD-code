# `jax-pme` benchmark

This runs a basic benchmark that mimics an MD setting: Samples are processed
sequentially with pre-computed neighborlists and settings. 

Files:

- **[run.py](run.py)**: runs the benchmark
- `crystals.xyz` contains the used structures (needs to be unzipped and copied here from
  parent)
- `*_parameters.yaml` contains the settings used for each structure. For Ewald the
  settings are taken from LAMMPS with a relative force accuracy of $10^{-4}$. For PME
  from our tuning code with the same but absolute accuracy.
- `*_results.npz` contains the results, grouped by crystal prototype

Benchmarks were run with commit `db1e662bdd8e8a68f675b05a94693e39dafc0fdd` of `jax-pme`,
`jax==0.4.35` and `jaxlib==0.4.34`, and CUDA 12.4.1 on a H100. Note that with this
combination of versions, this benchmark script triggers various loss of precision
warnings in `gemm_fusion_autotuner.cc` -- we disregard this here.
