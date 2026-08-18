# Training Design

The package separates the training-time ensemble from the deployed student.

## Fusion stage

The teacher pool emits a tensor of shape `[batch, teachers, classes]`.

- `z1` reuses the compact CP-Mobile backbone and outputs one mixture score per teacher. Softmax converts those scores into sample-adaptive teacher weights.
- `z2` performs per-class nonlinear fusion with a small one-hidden-layer MLP for every target class.
- The fusion trainer supervises both branches with cross-entropy and can optionally refine the teacher models jointly.

## Final KD stage

The final student target uses the reported `a1 + z1` combination:

- `a1`: arithmetic mean of teacher logits.
- `z1`: sample-adaptive weighted teacher logits.
- The package combines the two streams with configurable weights; the default is an equal blend because no different coefficient is specified in the supplied method description.

The compact student is optimized with hard-label cross-entropy plus temperature-scaled KL divergence. The KD coefficient `alpha` and temperature `T` are explicit inputs because the supplied method description defines the formula but does not provide fixed numeric values for them.

## Optimization defaults

The project configuration uses Adam, learning rate `1e-4`, batch-size reference 64, and 80 epochs.

## Device robustness

`EnergyAdaptiveIRAugment` provides a dataset-agnostic implementation of energy-modulated impulse-response augmentation. The caller supplies the IR bank. Higher-RMS inputs receive a stronger blend of the convolved signal.
