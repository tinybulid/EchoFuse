#!/usr/bin/env python3
"""Train z1 and z2 using externally supplied TAU batches and teacher models."""

from __future__ import annotations

import argparse
from pathlib import Path

from echofuse.config import FusionTrainingConfig, StudentConfig
from echofuse.external import resolve_provider
from echofuse.models import TeacherPool, Z1FusionNetwork, Z2PerClassFusion
from echofuse.training.engine import choose_device, set_seed
from echofuse.training.fusion import train_fusion_heads


def main():
    parser = argparse.ArgumentParser(description="Train learned teacher-fusion heads.")
    parser.add_argument("--data-provider", required=True)
    parser.add_argument("--teacher-provider", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", default=None)
    parser.add_argument("--freeze-teachers", action="store_true")
    parser.add_argument("--output", default="checkpoints/tau_fusion_best.pth")
    args = parser.parse_args()

    set_seed(42)
    device = choose_device(args.device)
    data = resolve_provider(args.data_provider)()
    teachers = list(resolve_provider(args.teacher_provider)())
    pool = TeacherPool(teachers)
    student_config = StudentConfig()
    z1 = Z1FusionNetwork(len(teachers), student_config)
    z2 = Z2PerClassFusion(len(teachers), student_config.n_classes)
    config = FusionTrainingConfig(epochs=args.epochs, train_teachers=not args.freeze_teachers)
    history = train_fusion_heads(
        pool,
        z1,
        z2,
        data["train"],
        data["validation"],
        device,
        config=config,
        checkpoint_path=Path(args.output),
    )
    print({"best_epoch": history.best_epoch, "best_accuracy": history.best_accuracy})


if __name__ == "__main__":
    main()
