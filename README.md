# State-Space Trajectory Dataset Generator

Compact Python generator for deterministic target trajectories from two linear state-space motion models: Ornstein-Uhlenbeck velocity dynamics and coordinated turn dynamics.

The generated state is

```text
x[k] = [px[k], py[k], vx[k], vy[k]]
```

and the measurement is

```text
y[k] = [px[k], py[k]] + v[k]
```

where `v[k]` is zero-mean isotropic Gaussian noise. Measurement uncertainty is specified directly through its standard deviation in meters.

## Files

```text
trajectory_dataset_core.py
    Common state-space models, dataset saving and plotting utilities.

generate_ou_range_dataset.py
    OU-only trajectories. Each trajectory receives one cruise velocity sampled
    independently from user-defined x/y ranges.

generate_three_model_dataset.py
    Balanced dataset containing three trajectory classes:
      1. OU with variable cruise velocity;
      2. CT with a fixed positive turn rate;
      3. CT with the corresponding fixed negative turn rate.
```

Both scripts save the same main arrays and can therefore be loaded by the same downstream code.

## Installation

```bat
conda create -n ssm_traj python=3.11
conda activate ssm_traj
pip install -r requirements.txt
```

## 1. OU-only dataset with variable cruise velocity

Windows Command Prompt, one line:

```bat
python generate_ou_range_dataset.py --length-mode variable --min-len 200 --max-len 300 --batch-size 1000 --sampling-time 1 --ou-gamma 0.05 --ou-vx-min 4 --ou-vx-max 12 --ou-vy-min -4 --ou-vy-max 4 --measurement-std 0 1 2 3 5
```

The cruise velocity is constant within each trajectory but varies across trajectories:

```text
dx ~ Uniform(ou-vx-min, ou-vx-max)
dy ~ Uniform(ou-vy-min, ou-vy-max)
```

## 2. Three-model dataset

Windows Command Prompt, one line:

```bat
python generate_three_model_dataset.py --length-mode variable --min-len 200 --max-len 300 --batch-size 1200 --sampling-time 1 --ou-gamma 0.05 --ou-vx-min 4 --ou-vx-max 12 --ou-vy-min -4 --ou-vy-max 4 --ct-turn-rate-deg 3 --measurement-std 0 1 2 3 5
```

The dataset is balanced as closely as possible across:

```text
ou
ct_positive   (+3 deg/time-unit in the example)
ct_negative   (-3 deg/time-unit in the example)
```

For exact balance, choose a batch size divisible by 3, such as `1200`.

## Saved data

Each `.npz` file contains at least:

```text
states                    (N, T, 4)
observations              (N, T, 2)
mask                      (N, T)
lengths                   (N,)
transition_matrices       (N, T, 4, 4)
input_matrices            (N, T, 4, 2)
control_inputs            (N, T, 2)
turn_rates_deg            (N, T)
active_model_ids          (N, T)
active_model_codes        (N, T)
trajectory_model_ids      (N,)
trajectory_model_codes    (N,)
cruise_velocities         (N, 2)
measurement_noise_std     scalar
```

Model codes are

```text
ou           = 0
ct_positive  = 1
ct_negative  = 2
pad          = -1
```

For a variable-length trajectory:

```python
import numpy as np

data = np.load("data/example.npz")

i = 0
Ti = int(data["lengths"][i])
x_i = data["states"][i, :Ti]
y_i = data["observations"][i, :Ti]
model_i = data["trajectory_model_ids"][i]
```

## Output files

For each value supplied to `--measurement-std`, the scripts save:

- one compressed `.npz` dataset;
- one `.png` plot of the trajectory selected through `--plot-sample-index`.

The default measurement-noise levels are:

```text
0, 1, 2, 3, 5 m
```

## Modelling assumptions

- State and observation models are linear.
- Process noise is zero.
- The OU cruise velocity is fixed within one trajectory and sampled again for the next trajectory.
- Each three-model trajectory follows one model for its full duration; there is no within-trajectory switching in this release.
- CT positive and CT negative use equal turn-rate magnitude and opposite sign.
