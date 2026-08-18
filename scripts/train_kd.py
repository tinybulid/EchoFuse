#!/usr/bin/env python3
"""Minimal end-user KD entry point.

The package intentionally does not define a dataset. Supply two callables:

  data provider    -> {"train": iterable, "validation": iterable}
  teacher provider -> sequence[nn.Module]

The z1 checkpoint must come from the learned fusion stage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from echofuse.checkpoints import load_fusion_checkpoint
from echofuse.config import KDTrainingConfig, StudentConfig
from echofuse.external import resolve_provider
from echofuse.models import CPMobileStudent, TeacherPool, Z1FusionNetwork, Z2PerClassFusion
from echofuse.training.engine import choose_device, set_seed
from echofuse.training.kd import train_student_kd


def main():
    parser = argparse.ArgumentParser(description="Train the compact CP-Mobile student with a1+z1 knowledge distillation.")
    parser.add_argument("--data-provider", required=True, help="module.path:function returning train/validation iterables")
    parser.add_argument("--teacher-provider", required=True, help="module.path:function returning the teacher model sequence")
    parser.add_argument("--z1-checkpoint", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="checkpoints/tau_student_kd_best.pth")
    args = parser.parse_args()

    set_seed(42)
    device = choose_device(args.device)
    data = resolve_provider(args.data_provider)()
    teachers = list(resolve_provider(args.teacher_provider)())
    pool = TeacherPool(teachers)

    student_config = StudentConfig()
    student = CPMobileStudent(student_config)
    z1 = Z1FusionNetwork(num_teachers=len(teachers), student_config=student_config)
    # z2 is instantiated only so the shared fusion checkpoint format can be loaded.
    z2 = Z2PerClassFusion(num_teachers=len(teachers), num_classes=student_config.n_classes)
    load_fusion_checkpoint(z1, z2, args.z1_checkpoint, device=device, strict=False)

    config = KDTrainingConfig(
        epochs=args.epochs,
        alpha=args.alpha,
        temperature=args.temperature,
    )
    history = train_student_kd(
        student,
        pool,
        z1,
        data["train"],
        data["validation"],
        device,
        config,
        checkpoint_path=Path(args.output),
    )
    print({"best_epoch": history.best_epoch, "best_accuracy": history.best_accuracy})


if __name__ == "__main__":
    main()
