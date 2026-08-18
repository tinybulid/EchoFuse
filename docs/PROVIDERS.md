# External Provider Contract

The project intentionally does not embed dataset paths, private checkpoints, or notebook-specific globals.

The command-line scripts use `module.path:function` provider specifications.

A data provider should return a mapping containing reusable iterables under `train` and `validation`; an optional `test` iterable may also be supplied.

Accepted batch formats are inherited from `echofuse.batches.parse_batch`:

- `(inputs, labels)`
- `(inputs, device_ids, labels)`
- mappings containing `inputs`/`x`, `labels`/`y`, and optional `device_ids`/`devices`

A teacher provider should return a sequence of initialized PyTorch teacher modules. This keeps teacher loading, external pretrained model dependencies, and checkpoint locations outside the reusable core package.
