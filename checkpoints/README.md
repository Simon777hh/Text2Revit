# Model checkpoints

Model weights are not distributed in this source repository. The Windows
release includes the inference weights required by the Revit add-in. The online
installer retrieves them automatically; users do not download a separate checkpoint.

For source inference or release builds, supply a compatible trained checkpoint
at `checkpoints/flow_matching_best.pth`, or pass its path with `--checkpoint`.
Local weight files are ignored by Git. Training from scratch does not require
the author's checkpoint.

The reference full training checkpoint is 770,009,261 bytes; its metadata is
retained in `manifest.json` for provenance.

SHA-256:
`a2e67e9c96069847d1a0c64b61fb65536faa54c0245e56364e4aea5d98b57bdd`

The release builder exports only inference tensors/architecture arguments into
the installer, reducing the bundled checkpoint to approximately 193 MB.
The installer checkpoint omits optimizer state and is not a complete training
resume checkpoint. Author-provided weights are covered by LICENSE.
