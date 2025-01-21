from typing import List, Optional

import torch
from spex.spherical_expansion import SphericalExpansion
from atomistic_dataset import AtomicSystem


class PowerSpectrumFeatures(torch.nn.Module):
    def __init__(
        self,
        all_species: List[int],
        cutoff_radius: float,
        max_radial: int,
        max_angular: int,
        radial_basis_type: Optional[str] = "LaplacianEigenstates",
    ):
        self.all_species = all_species
        self.max_radial = max_radial
        self.max_angular = max_angular
        super().__init__()
        radial = {
            radial_basis_type: {
                "cutoff": cutoff_radius,
                "max_radial": max_radial,
                "max_angular": max_angular,
                "trim": False,
                "spliner_accuracy": 1e-8,
            },
        }
        self.spex_calculator = SphericalExpansion(
            radial,
            angular="SphericalHarmonics",
            species={"Orthogonal": {"species": all_species}},
            cutoff_function={"ShiftedCosine": {"width": 0.5}},
        )
        self.ps_calculator = PowerSpectrum(max_angular)

    def forward(
        self, x: List[AtomicSystem], distance_vectors: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        out = []
        for system, distance_vector in zip(x, distance_vectors):
            spex = self.spex_calculator(
                R_ij=distance_vector,
                i=system.neighbor_list["edge_indices"][0],
                j=system.neighbor_list["edge_indices"][1],
                species=system.numbers,
            )
            self.power_spectrum = self.ps_calculator(spex)
            out.append(self.power_spectrum)
        return out

    @property
    def num_features(self):
        return (
            (self.max_angular + 1)
            * (self.max_radial + 1) ** 2
            * len(self.all_species) ** 2
        )


class PowerSpectrum(torch.nn.Module):
    def __init__(self, l_max):
        super(PowerSpectrum, self).__init__()

        self.l_max = l_max

    def forward(self, spex: List[torch.Tensor]) -> torch.Tensor:
        ps_values_ai = []
        for l in range(self.l_max + 1):
            cg = (-1) ** l * (2 * l + 1) ** (-0.5)
            c_ai_l = spex[l]
            ps_ai_l = cg * torch.einsum("imaq, imbe -> iabqe", c_ai_l, c_ai_l)
            ps_values_ai.append(ps_ai_l)
        ps_values_ai_cat = torch.stack(ps_values_ai, dim=-1)
        return ps_values_ai_cat.flatten(start_dim=1)
