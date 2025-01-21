from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from ase import Atoms
import vesin.torch
from tqdm import tqdm

import lightning.pytorch as pl
from ase.io import read
from torch.utils.data import DataLoader

AVAILABLE_TARGET_PROPERTIES = {"energies", "forces"}


@dataclass
class AtomicSystem:
    numbers: torch.Tensor
    positions: torch.Tensor
    cell: torch.Tensor
    pbc: torch.Tensor
    target_properties: Dict[str, torch.Tensor]
    neighbor_list: Optional[Dict[str, torch.Tensor]] = None
    info: Optional[Dict] = None


class AtomisticDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frames: List[Atoms],
        target_properties: List[str],
        cutoff_radius: Optional[float] = None,
        verbose: bool = True,
    ):
        super().__init__()
        if not set(target_properties).issubset(AVAILABLE_TARGET_PROPERTIES):
            raise ValueError(
                f"Target properties must be a subset of {AVAILABLE_TARGET_PROPERTIES}"
            )

        self.frames = frames
        self.target_properties = target_properties
        self.cutoff_radius = cutoff_radius
        self.verbose = verbose
        self.dataset = self._process_frames()

    def _process_frames(self) -> List[AtomicSystem]:
        positions_requires_grad = "forces" in self.target_properties

        if self.verbose:
            frames_iter = tqdm(self.frames, desc="Processing data")
        else:
            frames_iter = self.frames

        dataset = []
        for frame in frames_iter:
            target_properties = self._get_target_properties(frame)
            data = AtomicSystem(
                numbers=torch.tensor(frame.numbers, dtype=torch.int32),
                positions=torch.tensor(
                    frame.positions,
                    requires_grad=positions_requires_grad,
                    dtype=torch.get_default_dtype(),
                ),
                cell=torch.tensor(frame.cell.array, dtype=torch.get_default_dtype()),
                pbc=frame.pbc[0],
                target_properties=target_properties,
                info=frame.info,
            )
            if self.cutoff_radius is not None:
                data.neighbor_list = self._get_neighbor_list(
                    points=data.positions,
                    box=data.cell,
                    periodic=data.pbc
                    )
            dataset.append(data)
        return dataset

    def _get_target_properties(self, frame: Atoms) -> Dict[str, torch.Tensor]:
        property_funcs = {
            "energies": frame.get_potential_energy,
            "forces": frame.get_forces,
        }

        return {
            prop: torch.tensor(property_funcs[prop](), dtype=torch.get_default_dtype())
            for prop in self.target_properties
        }

    def _get_neighbor_list(self, points, box, periodic) -> Dict[str, torch.Tensor]:
        nl = vesin.torch.NeighborList(cutoff=self.cutoff_radius, full_list=True)
        i, j, shifts = nl.compute(points=points, box=box, periodic=periodic, quantities="ijS")
        return {
            "edge_indices": torch.stack([i, j], dim=0),
            "edge_shifts": shifts,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __setitem__(self, idx, item):
        self.dataset[idx] = item

    def __str__(self):
        return self.__class__.__name__ + f"({len(self)})"

    def __repr__(self):
        return str(self)


class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_frames_path: str,
        val_frames_path: str,
        target_properties: List[str],
        neighborlist_cutoff_radius: float,
        test_frames_path: Optional[str] = None,
        batch_size: int = 16,
        shuffle: bool = True,
        verbose: bool = True,
        num_workers: int = 0,
    ):
        super().__init__()
        self.train_frames_path = train_frames_path
        self.val_frames_path = val_frames_path
        self.test_frames_path = test_frames_path
        self.batch_size = batch_size
        self.neighborlist_cutoff_radius = neighborlist_cutoff_radius
        self.target_properties = target_properties
        self.verbose = verbose
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = lambda x: x

        # Internal attributes
        self.train_frames = read(self.train_frames_path, ":")
        self.val_frames = read(self.val_frames_path, ":")
        self.test_frames = (
            read(self.test_frames_path, ":") if self.test_frames_path else []
        )

        # Collect unique atomic numbers
        self.unique_elements = sorted(
            set(
                number
                for frame in self.train_frames + self.val_frames + self.test_frames
                for number in frame.numbers.tolist()
            )
        )

        self.train_dataset = self._create_dataset(
            self.train_frames, self.neighborlist_cutoff_radius
        )
        self.val_dataset = self._create_dataset(
            self.val_frames, self.neighborlist_cutoff_radius
        )
        self.test_dataset = (
            self._create_dataset(self.test_frames, self.neighborlist_cutoff_radius)
            if self.test_frames_path
            else []
        )

    def _create_dataset(self, frames, cutoff_radius):
        return AtomisticDataset(
            frames,
            target_properties=self.target_properties,
            verbose=self.verbose,
            cutoff_radius=cutoff_radius,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def prepare_compositions_weights(self) -> torch.Tensor:
        compositions = torch.stack(
            get_compositions_from_numbers(
                self.train_dataset, self.unique_elements, torch.get_default_dtype()
            )
        ).to(torch.get_default_dtype())

        energies = torch.cat(
            [
                data.target_properties["energies"].view(1, -1)
                for data in self.train_dataset
            ],
            dim=0,
        )
        weights = torch.linalg.lstsq(compositions, energies).solution
        return weights.T

def get_compositions_from_numbers(
    systems: List[AtomisticDataset],
    unique_numbers: List[int],
    dtype: Optional[torch.dtype] = torch.get_default_dtype(),
) -> List[torch.Tensor]:
    compositions = []

    for system in systems:
        composition = torch.zeros(
            len(unique_numbers), dtype=dtype, device=system.numbers.device
        )

        elements, counts = torch.unique(system.numbers, return_counts=True)
        indices = torch.tensor(unique_numbers, device=system.numbers.device)
        mask = torch.isin(indices, elements).nonzero().squeeze()

        composition[mask] = counts.to(dtype)
        compositions.append(composition)

    return compositions