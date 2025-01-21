from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torchpme.calculators import EwaldCalculator
from torchpme import InversePowerLawPotential, CombinedPotential

from atomistic_dataset import AtomicSystem
from power_spectrum import PowerSpectrumFeatures


@dataclass
class LRModelConfig:
    charge_channels: int
    cutoff_radius: float
    hidden_sizes: List[int]
    atomic_smearing: float
    lr_wavelength: float
    subtract_interior: Optional[bool] = False
    exponent: Optional[torch.Tensor] = 1.0
    full_neighbor_list: Optional[bool] = True
    prefactor: Optional[float] = None


@dataclass
class SRModelConfig:
    charge_channels: int
    hidden_sizes: List[int]
    unique_elements: List[int]
    cutoff_radius: float
    max_radial: int
    max_angular: int
    radial_basis_type: str
    long_range: bool


class LRModel(torch.nn.Module):
    def __init__(self, config: LRModelConfig) -> None:
        super().__init__()
        exclusion_radius = config.cutoff_radius if config.subtract_interior else None
        potentials_list = CombinedPotential(
            potentials=[
                InversePowerLawPotential(
                    exponent=1,
                    smearing=config.atomic_smearing,
                    exclusion_radius=exclusion_radius,
                ),
                InversePowerLawPotential(
                    exponent=3,
                    smearing=config.atomic_smearing,
                    exclusion_radius=exclusion_radius,
                ),
                InversePowerLawPotential(
                    exponent=6,
                    smearing=config.atomic_smearing,
                    exclusion_radius=exclusion_radius,
                ),
            ],
            initial_weights=torch.tensor([1 / 3, 1 / 3, 1 / 3]),
            learnable_weights=True,
            smearing=config.atomic_smearing,
            exclusion_radius=exclusion_radius,
        )
        self.calculator = EwaldCalculator(
            potential=potentials_list,
            full_neighbor_list=config.full_neighbor_list,
            lr_wavelength=config.lr_wavelength,
            prefactor=config.prefactor,
        )

    def forward(
        self,
        systems: List[AtomicSystem],
        charges: List[torch.Tensor],
        distance_vectors: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        positions, cells, neighbor_indices = self.extract_atomic_properties(systems)
        potentials = []
        for position, charge, cell, neighbor_index, distance_vector in zip(
            positions, charges, cells, neighbor_indices, distance_vectors
        ):
            potential = self.calculator(
                positions=position,
                charges=charge,
                cell=cell,
                neighbor_indices=neighbor_index,
                neighbor_distances=distance_vector.norm(dim=-1),
            )
            potentials.append(potential * charge)

        return potentials

    @staticmethod
    def extract_atomic_properties(
        systems: List[AtomicSystem],
    ) -> Tuple[
        List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]
    ]:
        positions = [system.positions for system in systems]
        cells = [system.cell for system in systems]
        neighbor_indices = [
            system.neighbor_list["edge_indices"].T for system in systems
        ]
        return positions, cells, neighbor_indices


class SRModel(torch.nn.Module):
    def __init__(self, config: SRModelConfig) -> None:
        super().__init__()

        self.ps_features_layer = PowerSpectrumFeatures(
            all_species=config.unique_elements,
            cutoff_radius=config.cutoff_radius,
            max_radial=config.max_radial,
            max_angular=config.max_angular,
            radial_basis_type=config.radial_basis_type,
        )
        self.long_range = config.long_range
        self.layer_norm = torch.nn.LayerNorm(self.num_features)
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(self.num_features, config.hidden_sizes[0]),
            torch.nn.GELU(),
        )
        self.nn_map = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(config.hidden_sizes[i], config.hidden_sizes[i + 1]),
                    torch.nn.GELU(),
                )
                for i in range(len(config.hidden_sizes) - 1)
            ]
        )
        self.short_range_energy_map = torch.nn.Linear(config.hidden_sizes[-1], 1)
        if self.long_range:
            self.charges_map = torch.nn.Linear(
                config.hidden_sizes[-1], config.charge_channels
            )

    def forward(
        self, systems: List[AtomicSystem], distance_vectors: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        ps_features = self.ps_features_layer(systems, distance_vectors)
        nn_features = []
        for f in ps_features:
            f = self.layer_norm(f)
            f = self.projection(f)
            for layer in self.nn_map:
                f = layer(f)
            nn_features.append(f)
        energies = [self.short_range_energy_map(f) for f in nn_features]
        if self.long_range:
            charges = [self.charges_map(f) for f in nn_features]
            return energies, charges
        else:
            return energies, None

    @property
    def num_features(self) -> int:
        return self.ps_features_layer.num_features


class BPPSLodeModel(torch.nn.Module):
    def __init__(
        self,
        unique_elements: List[int],
        long_range: bool,
        hidden_sizes_ps: List[int],
        output_size: int,
        cutoff: float,
        max_radial: int,
        max_angular: int,
        radial_basis_type: str,
        hidden_sizes_mp: Optional[List[int]] = None,
        lode_prefactor: Optional[float] = None,
        lode_charge_channels: Optional[int] = None,
        lode_atomic_smearing: Optional[float] = None,
        lode_lr_wavelength: Optional[float] = None,
        lode_subtract_interior: Optional[bool] = False,
        lode_exponent: Optional[torch.Tensor] = 1.0,
    ):
        super().__init__()

        self.unique_elements = unique_elements
        self.long_range = long_range

        self.register_buffer(
            "compositions_weights", torch.zeros((output_size, len(unique_elements)))
        )

        # Initialize Short-Range Model
        self.sr_model = SRModel(
            SRModelConfig(
                charge_channels=lode_charge_channels,
                hidden_sizes=hidden_sizes_ps,
                unique_elements=unique_elements,
                cutoff_radius=cutoff,
                max_radial=max_radial,
                max_angular=max_angular,
                radial_basis_type=radial_basis_type,
                long_range=long_range,
            )
        )

        # Initialize Long-Range Model if applicable
        self.lr_model = (
            LRModel(
                LRModelConfig(
                    charge_channels=lode_charge_channels,
                    cutoff_radius=cutoff,
                    hidden_sizes=hidden_sizes_mp,
                    atomic_smearing=lode_atomic_smearing,
                    lr_wavelength=lode_lr_wavelength,
                    subtract_interior=lode_subtract_interior,
                    exponent=lode_exponent,
                    prefactor=lode_prefactor,
                )
            )
            if self.long_range
            else None
        )

    def forward(
        self, x: List[AtomicSystem]
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        distance_vectors = self._extract_distance_vectors(x)
        if self.long_range:
            features_ps, charges = self.sr_model(x, distance_vectors)
            features_mp = self.lr_model(x, charges, distance_vectors)
            features_energies = [ps + mp for ps, mp in zip(features_ps, features_mp)]
        else:
            features_energies, charges = self.sr_model(x, distance_vectors)
        return torch.stack([f.sum() for f in features_energies]).view(-1, 1), charges

    def _extract_distance_vectors(self, data: List[AtomicSystem]) -> List[torch.Tensor]:
        distance_vectors = []
        for system in data:
            atoms_is = system.neighbor_list["edge_indices"][0]
            atoms_js = system.neighbor_list["edge_indices"][1]
            pos_is = system.positions[atoms_is]
            pos_js = system.positions[atoms_js]
            shifts = system.neighbor_list["edge_shifts"].type(system.cell.dtype)
            distance_vector = pos_js - pos_is + shifts @ system.cell
            distance_vectors.append(distance_vector)
        return distance_vectors
