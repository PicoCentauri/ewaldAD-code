import os
from typing import Optional

import lightning.pytorch as pl
import torch
from torch.nn import MSELoss

from get_autograd_derivatives import get_autograd_forces
from atomistic_dataset import get_compositions_from_numbers


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model,
        predict_forces: bool,
        energies_weight: Optional[float] = 1.0,
        forces_weight: Optional[float] = 1.0,
        charges_weight: Optional[float] = 1.0,
        scheduler: Optional[bool] = False,
        lr: Optional[float] = 1e-4,
        weight_decay: Optional[float] = 1e-5,
        log_wandb_tables: Optional[bool] = True,
    ):
        super().__init__()
        self.model = model
        self.energies_weight = energies_weight
        self.forces_weight = forces_weight if predict_forces else None
        self.charges_weight = charges_weight if self.model.long_range else None
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_wandb_tables = log_wandb_tables
        self.predict_forces = predict_forces
        self.scheduler = scheduler

    def on_train_epoch_start(self):
        self.predicted_energies_train = []
        self.target_energies_train = []
        self.predicted_forces_train = [] if self.predict_forces else None
        self.target_forces_train = [] if self.predict_forces else None
        self.predicted_charges_train = [] if self.model.long_range else None
        self.target_charges_train = [] if self.model.long_range else None

    def forward(self, x):
        self.compositions = torch.stack(
            get_compositions_from_numbers(
                x,
                self.model.unique_elements,
                self.model.compositions_weights.dtype,
            )
        )
        predicted_energies, predicted_charges = self.model(x)
        target_energies = (
            torch.stack([system.target_properties["energies"] for system in x]).view(
                -1, 1
            )
            - self.compositions @ self.model.compositions_weights.T
        )
        target_charges = (
            torch.stack(
                [
                    torch.tensor(
                        sum(
                            system.info[charge_key]
                            for charge_key in system.info
                            if charge_key.startswith("charge")
                        ),
                        device=self.device,
                        dtype=self.dtype,
                    )
                    for system in x
                ]
            ).view(-1, 1)
            if self.model.long_range
            else None
        )
        predicted_forces = (
            torch.cat(
                [
                    get_autograd_forces(predicted_energies, system.positions)[0]
                    for system in x
                ]
            )
            if self.predict_forces
            else None
        )
        target_forces = (
            torch.cat([system.target_properties["forces"] for system in x])
            if self.predict_forces
            else None
        )

        return (
            predicted_energies,
            predicted_forces,
            predicted_charges,
            target_energies,
            target_forces,
            target_charges,
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.predict_forces:
            with torch.enable_grad():
                for system in batch:
                    system.positions.requires_grad_(requires_grad=True)
                return self.forward(batch)
        else:
            return self.forward(batch)

    def training_step(self, batch, batch_idx):
        (
            predicted_energies,
            predicted_forces,
            predicted_charges,
            target_energies,
            target_forces,
            target_charges,
        ) = self.forward(batch)

        loss_fn = MSELoss()
        # num_of_atoms = torch.tensor([[system.numbers.shape[0]] for system in batch], device=self.device, dtype=self.dtype)
        loss = self.energies_weight * loss_fn(predicted_energies, target_energies)

        if self.predict_forces:
            loss_force = loss_fn(predicted_forces, target_forces)
            loss += self.forces_weight * loss_force
        if self.model.long_range:
            loss += self.charges_weight * loss_fn(
                torch.stack([charge.sum(dim=0) for charge in predicted_charges]),
                target_charges,
            )
        self.log(
            "train_loss",
            loss.item(),
            on_step=True,
            prog_bar=True,
        )
        lr = (
            self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
            if self.scheduler
            else self.lr
        )
        self.log("learning_rate", lr, on_step=True, prog_bar=True)

        predicted_energies = predicted_energies.cpu().detach()
        self.predicted_energies_train.append(predicted_energies)
        target_energies = target_energies.cpu().detach()
        self.target_energies_train.append(target_energies)
        if self.predict_forces:
            predicted_forces = predicted_forces.cpu().detach()
            self.predicted_forces_train.append(predicted_forces)
            target_forces = target_forces.cpu().detach()
            self.target_forces_train.append(target_forces)
        if self.model.long_range:
            predicted_charges = (
                torch.stack([charge.sum(dim=0) for charge in predicted_charges])
                .cpu()
                .detach()
            )
            self.predicted_charges_train.append(predicted_charges)
            target_charges = target_charges.cpu().detach()
            self.target_charges_train.append(target_charges)
        return loss

    def on_train_epoch_end(self):
        print(self.model.lr_model.calculator.potential.weights)
        preds_energy_train = torch.cat(self.predicted_energies_train)
        targets_energy_train = torch.cat(self.target_energies_train)
        if self.predict_forces:
            preds_forces_train = torch.cat(self.predicted_forces_train)
            targets_forces_train = torch.cat(self.target_forces_train)
        if self.model.long_range:
            preds_charges_train = torch.cat(self.predicted_charges_train)
            targets_charges_train = torch.cat(self.target_charges_train)
        loss_fn = MSELoss()
        train_energies_rmse = torch.sqrt(
            loss_fn(preds_energy_train, targets_energy_train)
        ).item()
        print("\n")
        print(f"Energies RMSE: Train {train_energies_rmse:.4f}")
        self.log(
            "train_energies_rmse",
            train_energies_rmse,
            prog_bar=True,
        )

        if self.predict_forces:
            train_forces_rmse = torch.sqrt(
                loss_fn(preds_forces_train, targets_forces_train)
            ).item()
            print(f"Forces RMSE: Train {train_forces_rmse:.4f}")
            self.log(
                "train_forces_rmse",
                train_forces_rmse,
                prog_bar=True,
            )

        if self.model.long_range:
            train_charges_rmse = torch.sqrt(
                loss_fn(preds_charges_train, targets_charges_train)
            ).item()
            print(f"Charges RMSE: Train {train_charges_rmse:.4f}")
            self.log(
                "train_charges_rmse",
                train_charges_rmse,
                prog_bar=True,
            )

    def on_validation_epoch_start(self):
        if self.predict_forces:
            torch.set_grad_enabled(True)
        self.predicted_energies_val = []
        self.target_energies_val = []
        self.predicted_forces_val = [] if self.predict_forces else None
        self.target_forces_val = [] if self.predict_forces else None
        self.predicted_charges_val = [] if self.model.long_range else None
        self.target_charges_val = [] if self.model.long_range else None

    def validation_step(self, batch, batch_idx):
        (
            predicted_energies,
            predicted_forces,
            predicted_charges,
            target_energies,
            target_forces,
            target_charges,
        ) = self.forward(batch)

        self.predicted_energies_val.append(predicted_energies.cpu().detach())
        self.target_energies_val.append(target_energies.cpu().detach())
        if self.predict_forces:
            self.predicted_forces_val.append(predicted_forces.cpu().detach())
            self.target_forces_val.append(target_forces.cpu().detach())
        if self.model.long_range:
            self.predicted_charges_val.append(
                torch.stack([charge.sum(dim=0) for charge in predicted_charges])
                .cpu()
                .detach()
            )
            self.target_charges_val.append(target_charges.cpu().detach())

    def on_validation_epoch_end(self):
        preds_energy_val = torch.cat(self.predicted_energies_val)
        targets_energy_val = torch.cat(self.target_energies_val)
        if self.predict_forces:
            preds_forces_val = torch.cat(self.predicted_forces_val)
            targets_forces_val = torch.cat(self.target_forces_val)
        if self.model.long_range:
            preds_charges_val = torch.cat(self.predicted_charges_val)
            targets_charges_val = torch.cat(self.target_charges_val)
        loss_fn = MSELoss()
        val_energies_rmse = torch.sqrt(
            loss_fn(preds_energy_val, targets_energy_val)
        ).item()
        print("\n")
        print(f"Energies RMSE: Val {val_energies_rmse:.4f}")
        self.log(
            "val_energies_rmse",
            val_energies_rmse,
            prog_bar=True,
        )
        if self.predict_forces:
            val_forces_rmse = torch.sqrt(
                loss_fn(preds_forces_val, targets_forces_val)
            ).item()
            print(f"Forces RMSE: Val {val_forces_rmse:.4f}")
            self.log(
                "val_forces_rmse",
                val_forces_rmse,
                prog_bar=True,
            )
        if self.model.long_range:
            val_charges_rmse = torch.sqrt(
                loss_fn(preds_charges_val, targets_charges_val)
            ).item()
            print(f"Charges RMSE: Val {val_charges_rmse:.4f}")
            self.log(
                "val_charges_rmse",
                val_charges_rmse,
                prog_bar=True,
            )
        if isinstance(self.logger, pl.loggers.WandbLogger):
            torch.save(
                preds_energy_val,
                os.path.join(self.logger.experiment.dir, "val_predicted_energies.pt"),
            )
            torch.save(
                targets_energy_val,
                os.path.join(self.logger.experiment.dir, "val_target_energies.pt"),
            )
            if self.predict_forces:
                torch.save(
                    preds_forces_val,
                    os.path.join(self.logger.experiment.dir, "val_predicted_forces.pt"),
                )
                torch.save(
                    targets_forces_val,
                    os.path.join(self.logger.experiment.dir, "val_target_forces.pt"),
                )
            if self.model.long_range:
                torch.save(
                    preds_charges_val,
                    os.path.join(
                        self.logger.experiment.dir, "val_predicted_charges.pt"
                    ),
                )
                torch.save(
                    targets_charges_val,
                    os.path.join(self.logger.experiment.dir, "val_target_charges.pt"),
                )

    def on_test_epoch_start(self):
        if self.predict_forces:
            torch.set_grad_enabled(True)
        self.predicted_energies_test = []
        self.target_energies_test = []
        self.predicted_forces_test = [] if self.predict_forces else None
        self.target_forces_test = [] if self.predict_forces else None
        self.predicted_charges_test = [] if self.model.long_range else None
        self.target_charges_test = [] if self.model.long_range else None

    def test_step(self, batch, batch_idx):
        (
            predicted_energies,
            predicted_forces,
            predicted_charges,
            target_energies,
            target_forces,
            target_charges,
        ) = self.forward(batch)

        self.predicted_energies_test.append(predicted_energies.cpu().detach())
        self.target_energies_test.append(target_energies.cpu().detach())
        if self.predict_forces:
            self.predicted_forces_test.append(predicted_forces.cpu().detach())
            self.target_forces_test.append(target_forces.cpu().detach())
        if self.model.long_range:
            self.predicted_charges_test.append(
                torch.stack([charge.sum(dim=0) for charge in predicted_charges])
                .cpu()
                .detach()
            )
            self.target_charges_test.append(target_charges.cpu().detach())

    def on_test_epoch_end(self):
        preds_energy_test = torch.cat(self.predicted_energies_test)
        targets_energy_test = torch.cat(self.target_energies_test)
        if self.predict_forces:
            preds_forces_test = torch.cat(self.predicted_forces_test)
            targets_forces_test = torch.cat(self.target_forces_test)
        if self.model.long_range:
            preds_charges_test = torch.cat(self.predicted_charges_test)
            targets_charges_test = torch.cat(self.target_charges_test)
        loss_fn = MSELoss()
        test_energies_rmse = torch.sqrt(
            loss_fn(preds_energy_test, targets_energy_test)
        ).item()
        print("\n")
        print(f"Energies RMSE: Test {test_energies_rmse:.4f}")
        self.log(
            "test_energies_rmse",
            test_energies_rmse,
            prog_bar=True,
        )
        if self.predict_forces:
            test_forces_rmse = torch.sqrt(
                loss_fn(preds_forces_test, targets_forces_test)
            ).item()
            print(f"Forces RMSE: Test {test_forces_rmse:.4f}")
            self.log(
                "test_forces_rmse",
                test_forces_rmse,
                prog_bar=True,
            )
        if self.model.long_range:
            test_charges_rmse = torch.sqrt(
                loss_fn(preds_charges_test, targets_charges_test)
            ).item()
            print(f"Charges RMSE: Test {test_charges_rmse:.4f}")
            self.log(
                "test_charges_rmse",
                test_charges_rmse,
                prog_bar=True,
            )

        if isinstance(self.logger, pl.loggers.WandbLogger):
            torch.save(
                preds_energy_test,
                os.path.join(self.logger.experiment.dir, "test_predicted_energies.pt"),
            )
            torch.save(
                targets_energy_test,
                os.path.join(self.logger.experiment.dir, "test_target_energies.pt"),
            )
            if self.predict_forces:
                torch.save(
                    preds_forces_test,
                    os.path.join(
                        self.logger.experiment.dir, "test_predicted_forces.pt"
                    ),
                )
                torch.save(
                    targets_forces_test,
                    os.path.join(self.logger.experiment.dir, "test_target_forces.pt"),
                )
            if self.model.long_range:
                torch.save(
                    preds_charges_test,
                    os.path.join(
                        self.logger.experiment.dir, "test_predicted_charges.pt"
                    ),
                )
                torch.save(
                    targets_charges_test,
                    os.path.join(self.logger.experiment.dir, "test_target_charges.pt"),
                )

    def configure_optimizers(self):
        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.scheduler:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=5e-4,
                total_steps=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            return optimizer
