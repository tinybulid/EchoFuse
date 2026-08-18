from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import torch.nn.functional as F

from ..batches import move_batch, parse_batch
from ..checkpoints import save_fusion_checkpoint
from ..config import FusionTrainingConfig
from ..models.ensemble import TeacherPool, average_teacher_logits, blend_logits, weighted_teacher_logits
from ..models.fusion import Z1FusionNetwork, Z2PerClassFusion
from .engine import EarlyStopping, EpochStats, TrainingHistory, make_cosine_scheduler


def _accuracy(logits, labels):
    return int((logits.argmax(1) == labels).sum().item())


@torch.no_grad()
def validate_fusion(
    teacher_pool: TeacherPool,
    z1: Z1FusionNetwork,
    z2: Z2PerClassFusion,
    batches: Iterable,
    device: torch.device,
) -> EpochStats:
    teacher_pool.eval(); z1.eval(); z2.eval()
    loss_sum = 0.0
    correct = 0
    samples = 0
    pieces = {"a1": 0.0, "z1": 0.0, "z2": 0.0}
    for raw in batches:
        batch = move_batch(parse_batch(raw), device)
        teacher_logits = teacher_pool(batch.inputs)
        a1 = average_teacher_logits(teacher_logits)
        z1_logits = weighted_teacher_logits(teacher_logits, z1(batch.inputs))
        z2_logits = z2(teacher_logits)
        a1z1 = blend_logits((a1, 0.5), (z1_logits, 0.5))
        loss = F.cross_entropy(a1z1, batch.labels)
        n = batch.labels.numel()
        loss_sum += float(loss.cpu()) * n
        correct += _accuracy(a1z1, batch.labels)
        samples += n
        pieces["a1"] += _accuracy(a1, batch.labels)
        pieces["z1"] += _accuracy(z1_logits, batch.labels)
        pieces["z2"] += _accuracy(z2_logits, batch.labels)
    if samples == 0:
        raise ValueError("validation iterable produced no samples")
    return EpochStats(
        loss=loss_sum / samples,
        accuracy=correct / samples,
        samples=samples,
        pieces={key: value / samples for key, value in pieces.items()},
    )


def train_fusion_heads(
    teacher_pool: TeacherPool,
    z1: Z1FusionNetwork,
    z2: Z2PerClassFusion,
    train_batches: Iterable,
    validation_batches: Iterable,
    device: str | torch.device,
    config: FusionTrainingConfig = FusionTrainingConfig(),
    checkpoint_path: str | Path | None = None,
    augment: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> TrainingHistory:
    """Jointly optimize the learned z1/z2 ensemble heads.

    When train_teachers=True, gradients are also allowed to refine the teacher
    models. Set it False when the supplied teacher checkpoints must remain fixed.
    """
    device = torch.device(device)
    teacher_pool = teacher_pool.to(device)
    z1 = z1.to(device)
    z2 = z2.to(device)

    for parameter in teacher_pool.parameters():
        parameter.requires_grad = config.train_teachers

    params = list(z1.parameters()) + list(z2.parameters())
    if config.train_teachers:
        params += list(teacher_pool.parameters())
    optimizer = torch.optim.Adam(
        params,
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = make_cosine_scheduler(optimizer, config.epochs)
    checkpoint_path = checkpoint_path or config.checkpoint_name
    history = TrainingHistory()
    stopper = EarlyStopping(config.early_stopping_patience)

    for epoch in range(config.epochs):
        teacher_pool.train(config.train_teachers)
        z1.train(); z2.train()
        loss_sum = 0.0
        correct = 0
        samples = 0
        piece_sums = {"z1_loss": 0.0, "z2_loss": 0.0}

        for raw in train_batches:
            batch = move_batch(parse_batch(raw), device)
            x = augment(batch.inputs) if augment is not None else batch.inputs

            if config.train_teachers:
                teacher_logits = teacher_pool(x)
            else:
                with torch.no_grad():
                    teacher_logits = teacher_pool(x)

            z1_weights = z1(x)
            z1_logits = weighted_teacher_logits(teacher_logits, z1_weights)
            z2_logits = z2(teacher_logits)
            loss_z1 = F.cross_entropy(z1_logits, batch.labels)
            loss_z2 = F.cross_entropy(z2_logits, batch.labels)
            loss = config.z1_loss_weight * loss_z1 + config.z2_loss_weight * loss_z2

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            n = batch.labels.numel()
            loss_sum += float(loss.detach().cpu()) * n
            correct += _accuracy(z1_logits, batch.labels)
            samples += n
            piece_sums["z1_loss"] += float(loss_z1.detach().cpu()) * n
            piece_sums["z2_loss"] += float(loss_z2.detach().cpu()) * n

        scheduler.step()
        if samples == 0:
            raise ValueError("training iterable produced no samples")
        train_stats = EpochStats(
            loss=loss_sum / samples,
            accuracy=correct / samples,
            samples=samples,
            pieces={key: value / samples for key, value in piece_sums.items()},
        )
        val_stats = validate_fusion(teacher_pool, z1, z2, validation_batches, device)
        history.train.append(train_stats)
        history.validation.append(val_stats)

        improved = val_stats.accuracy > history.best_accuracy
        if improved:
            history.best_accuracy = val_stats.accuracy
            history.best_epoch = epoch
            save_fusion_checkpoint(
                checkpoint_path,
                z1,
                z2,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=val_stats.accuracy,
                config=config,
                teacher_pool=teacher_pool if config.train_teachers else None,
            )
            stopper.best = val_stats.accuracy
            stopper.bad_epochs = 0
        elif stopper.update(val_stats.accuracy):
            break

    return history
