# README

To run the script, execute the `train.py` script while passing `parameters.yml` as a parameter.

Most of the parameters in `parameters.yml` are self-explanatory and standard for an ML pipeline. Below are explanations for parameters that might be ambiguous:

- **neighborlist_cutoff_radius**: Radius beyond which atoms are excluded from the neighbor list (required for SOAP as well as for torch-PME).
- **target_properties**: Supports either `['energies']` or `['energies', 'forces']`. Determines whether to train only on energies or on both energies and forces. This should align with the `predict_forces` flag. The model trains on both energies and forces if at least one flag specifies training on forces.
- **energies_weight, forces_weight**: Specify the weight with which the corresponding error contribution is added to the loss function. These weights are applicable only if training on both energies and forces.
- **long_range**: Indicates whether to include the long-range contribution in the model.
- **hidden_sizes_ps**: Corresponds to the number of linear layers and the number of neurons in the short-range part.
- **output_size**: Always set to `1`. Determines the output size of the neural network. Since forces are predicted via backpropagation, there is no scenario where this should differ from `1`.
- **max_radial, max_angular, radial_basis_type**: Parameters related to the SOAP method.
- **lode_charge_channels, lode_atomic_smearing, lode_lr_wavelength, lode_subtract_interior**: Parameters specific to torchPME. It is recommended to study torchPME documentation for further details.

### TorchPME Integration

All changes related to torchPME potentials are implemented manually in the `bpps_lode_model.py` script within the `LRModel` class. To better understand its workings, consult the [torchPME documentation](https://lab-cosmo.github.io/torch-pme/latest/index.html).
