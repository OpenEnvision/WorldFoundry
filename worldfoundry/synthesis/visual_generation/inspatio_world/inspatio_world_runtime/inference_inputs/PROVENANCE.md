# InSpatio-World inference input reader

- Source: https://github.com/inspatio/inspatio-world/blob/fef970664e33f519a31f0ee19d58689e41752c0e/datasets/test_dataset.py
- Revision: `fef970664e33f519a31f0ee19d58689e41752c0e`
- License: Apache-2.0; see root `THIRD-PARTY-NOTICES`.

`scene_inputs.py` restores the upstream inference input reader as `SceneInputs`.
Its original `TestDataset` name refers to inference-time videos, depth, masks and
camera trajectories; this is not a test suite or a training data loader.
Local changes: relative utility imports, single-frame bounce handling, reliable
video filename stems, and removal of unused imports. `video_dataset.py` reports
invalid inference inputs immediately instead of randomly retrying forever.
