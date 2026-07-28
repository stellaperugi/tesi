# Changelog

## 2.2.0

- Measurement-noise datasets generated in the same run now share a common
  standard-normal noise realization.
- Different measurement standard deviations rescale the same noise samples
  instead of generating independent realizations.
- Added `measurement_noise_seed` and `common_noise_across_std` metadata.

## 2.1.1

- Added BSD 3-Clause licensing information.
- Updated the README with the project license.
- Removed internal release instructions from the public repository.

## 2.1.0

- Added within-trajectory switching among OU, positive CT and negative CT.
- Added a new OU cruise velocity sample at the start of every OU segment.
- Added per-step segment identifiers and switching flags.
- Updated sample plots to identify active model segments.
- Kept the main `.npz` fields compatible with the OU-only and single-model three-class generators.

## 2.0.0

- Added an OU-only generator with trajectory-specific cruise velocities sampled from user-defined x/y ranges.
- Added a balanced three-model generator with OU, positive coordinated turn, and negative coordinated turn trajectories.
- Standardized measurement-noise levels as user-provided standard deviations in metres.
- Preserved a common `.npz` schema across both generators.
- Added sample 2D plots for each generated noise level.
