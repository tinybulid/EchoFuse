#!/usr/bin/env python3
"""Evaluate a compact student on externally supplied batches."""

from __future__ import annotations

import argparse

from echofuse.checkpoints import load_model_checkpoint
from echofuse.config import StudentConfig
from echofuse.evaluation import evaluate_model, model_summary
from echofuse.external import resolve_provider
from echofuse.models import CPMobileStudent
from echofuse.training.engine import choose_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-provider", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = choose_device(args.device)
    data = resolve_provider(args.data_provider)()
    model = CPMobileStudent(StudentConfig()).to(device)
    load_model_checkpoint(model, args.checkpoint, device=device, strict=False)
    report = evaluate_model(model, data.get("test", data["validation"]), device, num_classes=10)
    print({
        "accuracy": report.accuracy,
        "macro_recall": report.macro_recall,
        "device_accuracy": report.device_accuracy,
        "model": model_summary(model, input_shape=(1, 1, 256, 64)),
    })


if __name__ == "__main__":
    main()
