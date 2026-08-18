# Model Components

## Compact student

`CPMobileStudent` is adapted from the supplied CP-Mobile implementation. The original block type, GRN behavior, quantization stubs, and fusion method are retained, while the default configuration is reduced to approximately the 60K-parameter / 30M-MAC regime for a 256 x 64 Mel input.

Default compact configuration:

| Setting | Value |
|---|---:|
| Base channels | 24 |
| Channel multiplier | 2.0 |
| Expansion rate | 2.5 |
| Blocks by stage | 2 / 3 / 1 |
| Downsampling block | b4: 2 x 2 |
| Classes | 10 |

## Teacher code

The previous teacher architecture source files are copied unchanged into `echofuse/models/teachers/`:

- `Lenv1.py`
- `LenV2.py`
- `LenV2Prime.py`
- `cp_mobile.py`
- `cp_resnet.py`
- `repconv.py`

Only the package registry/import layer is standardized. The ensemble container accepts any additional external PyTorch model, so transformer or pretrained audio teachers can be supplied without changing the fusion or KD code.

## z1

`Z1FusionNetwork` uses the same compact student backbone with the final output dimension changed to the number of teachers. Its softmax scores are used as sample-specific mixture coefficients.

## z2

`Z2PerClassFusion` consumes the teacher-logit stack and applies one small MLP per class. This makes its output independent of convex teacher weighting.
