"""Utilities for fine-tuning an RF-DETR detector on football datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


@dataclass
class TrainingConfig:
    """Static configuration used during training.

    Attributes:
        num_epochs: Number of training epochs.
        gradient_accumulation: Number of gradient accumulation steps.
        clip_grad_norm: Optional gradient clipping value.
        checkpoint_dir: Directory where checkpoints will be stored.
        log_interval: Frequency (in batches) to emit training logs.
    """

    num_epochs: int
    gradient_accumulation: int = 1
    clip_grad_norm: Optional[float] = None
    checkpoint_dir: Optional[Path] = None
    log_interval: int = 50


class RFDetrFineTuner:
    """Encapsulates the fine-tuning loop for an RF-DETR detector.

    The class assumes the RF-DETR implementation follows the common Detectron2
    style where calling the model in training mode returns a dictionary of loss
    components. During evaluation the model is expected to return predictions
    that can be consumed by an evaluator callable provided by the user.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: Iterable[Dict[str, Any]],
        val_loader: Optional[Iterable[Dict[str, Any]]] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        evaluator: Optional[Callable[[Iterable[Any]], Dict[str, float]]] = None,
        device: Optional[torch.device] = None,
        config: Optional[TrainingConfig] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or TrainingConfig(num_epochs=10)

        self.model.to(self.device)

        if self.config.checkpoint_dir is not None:
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> Dict[str, Any]:
        """Executes the training loop and returns the history of metrics."""
        history: Dict[str, list] = {"loss": []}

        for epoch in range(1, self.config.num_epochs + 1):
            epoch_loss = self._train_one_epoch(epoch)
            history["loss"].append(epoch_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            if self.val_loader is not None and self.evaluator is not None:
                metrics = self.evaluate()
                history.setdefault("val", []).append(metrics)

            if self.config.checkpoint_dir is not None:
                self._save_checkpoint(epoch)

        return history

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        for step, batch in enumerate(self.train_loader, start=1):
            images = batch["images"].to(self.device)
            targets = batch["targets"]
            targets = [{k: v.to(self.device) if hasattr(v, "to") else v for k, v in t.items()} for t in targets]

            outputs = self.model(images, targets)
            loss = self._reduce_loss(outputs)
            loss = loss / self.config.gradient_accumulation
            loss.backward()

            if step % self.config.gradient_accumulation == 0:
                if self.config.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)

                if self.optimizer is not None:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            running_loss += loss.item()

            if step % self.config.log_interval == 0:
                avg_loss = running_loss / step
                print(f"Epoch {epoch} | Step {step} | Loss {avg_loss:.4f}")

        return running_loss / max(1, step)

    def evaluate(self) -> Dict[str, float]:
        if self.val_loader is None or self.evaluator is None:
            raise ValueError("Validation loader and evaluator must be provided for evaluation.")

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["images"].to(self.device)
                outputs = self.model(images)
                predictions.extend(outputs)

        metrics = self.evaluator(predictions)
        print(f"Validation metrics: {metrics}")
        return metrics

    def _save_checkpoint(self, epoch: int) -> None:
        if self.optimizer is None:
            raise ValueError("Optimizer must be provided to save checkpoints.")

        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state"] = self.scheduler.state_dict()

        path = self.config.checkpoint_dir / f"rfdetr_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    @staticmethod
    def _reduce_loss(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not outputs:
            raise ValueError("The model must return a dictionary of losses during training.")

        total_loss = sum(outputs.values())
        if not torch.is_tensor(total_loss):
            raise TypeError("The summed loss must be a torch.Tensor.")
        return total_loss


__all__ = ["TrainingConfig", "RFDetrFineTuner"]
