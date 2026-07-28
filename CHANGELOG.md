# Changelog

## 2.0.0

- Added an OU-only generator with trajectory-specific cruise velocities sampled from user-defined x/y ranges.
- Added a balanced three-model generator with OU, positive coordinated turn, and negative coordinated turn trajectories.
- Standardized measurement-noise levels as user-provided standard deviations in metres.
- Preserved a common `.npz` schema across both generators.
- Added sample 2D plots for each generated noise level.
