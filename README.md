# State-Space Trajectory Dataset Generator

Compact Python generator for deterministic target trajectories from two linear state-space motion models: Ornstein--Uhlenbeck velocity dynamics and coordinated turn dynamics.

The generated state is

```text
x[k] = [px[k], py[k], vx[k], vy[k]]
```

and the measurement is

```text
y[k] = [px[k], py[k]] + measurement noise
```

Measurement noise is Gaussian, zero mean, isotropic, and controlled only through its standard deviation in meters.

## Installation

```bash
conda create -n ssm_traj python=3.11
conda activate ssm_traj
pip install -r requirements.txt
```

## Examples

### Switching OU/CT dataset

A 2D PNG plot of a sample trajectory is saved automatically in the output folder.


```bash
python generate_trajectory_dataset.py \
  --model-mode switching \
  --length-mode variable \
  --min-len 200 \
  --max-len 300 \
  --batch-size 1000 \
  --sampling-time 1 \
  --measurement-std 0 5 10 50 \
  --switch-min-duration 20 \
  --switch-max-duration 50
```

### OU-only dataset with assigned cruise velocity

```bash
python generate_trajectory_dataset.py \
  --model-mode ou \
  --length-mode fixed \
  --data-len 250 \
  --batch-size 1000 \
  --sampling-time 1 \
  --ou-gamma 0.05 \
  --ou-cruise-vx 8 \
  --ou-cruise-vy 0 \
  --measurement-std 0 5 10
```

### CT-only dataset with assigned turn rate

```bash
python generate_trajectory_dataset.py \
  --model-mode ct \
  --length-mode fixed \
  --data-len 250 \
  --batch-size 1000 \
  --sampling-time 1 \
  --ct-turn-rate-deg 3 \
  --measurement-std 0 5 10
```

## Saved arrays

Each `.npz` file contains:

```text
states                  shape (N, T, 4)
observations            shape (N, T, 2)
mask                    shape (N, T)
lengths                 shape (N,)
transition_matrices     shape (N, T, 4, 4)
input_matrices          shape (N, T, 4, 2)
control_inputs          shape (N, T, 2)
turn_rates_deg          shape (N, T)
active_model_ids        shape (N, T)
active_model_codes      shape (N, T)
trajectory_model_ids    shape (N,)
measurement_noise_std   scalar
```

The valid portion of trajectory `i` is

```python
Ti = lengths[i]
x_i = states[i, :Ti]
y_i = observations[i, :Ti]
models_i = active_model_ids[i, :Ti]
```

The model codes are

```text
ct  = 1
ou  = 2
pad = -1
```

For trajectory `i` and time step `k`, the ground-truth dynamics are

```python
F = transition_matrices[i, k]
B = input_matrices[i, k]
d = control_inputs[i, k]

x_next = F @ x + B @ d
```

For CT, `B` and `d` are zero. For OU, `B` and `d` encode the cruise velocity contribution.

## Notes

- Process noise is not added in this generator.
- A measurement standard deviation of `0` produces noiseless position measurements.
- Multiple measurement-noise levels can be generated in a single run.

## Sample plot

For each requested measurement-noise standard deviation, the script saves a PNG file in the output folder.
By default it plots trajectory index `0`. You can choose a different one with:

```bash
--plot-sample-index 5
```

The PNG shows the true 2D trajectory and the corresponding noisy observations. In switching mode, the active OU/CT segments are also highlighted.


## Expected output

A typical command is:

```bash
python generate_trajectory_dataset.py \
  --model-mode switching \
  --length-mode variable \
  --min-len 200 \
  --max-len 300 \
  --batch-size 1000 \
  --sampling-time 1 \
  --measurement-std 0 5 10 50 \
  --switch-min-duration 20 \
  --switch-max-duration 50 \
  --plot-sample-index 0
```

The output folder will contain one dataset file and one sample plot for each requested measurement-noise level:

```text
data/
├── trajectory_dataset_switching_ou_ct_variable_std_0m_dt_1.npz
├── trajectory_dataset_switching_ou_ct_variable_std_5m_dt_1.npz
├── trajectory_dataset_switching_ou_ct_variable_std_10m_dt_1.npz
├── trajectory_dataset_switching_ou_ct_variable_std_50m_dt_1.npz
├── sample_trajectory_switching_ou_ct_variable_std_0m_dt_1_idx_0.png
├── sample_trajectory_switching_ou_ct_variable_std_5m_dt_1_idx_0.png
├── sample_trajectory_switching_ou_ct_variable_std_10m_dt_1_idx_0.png
└── sample_trajectory_switching_ou_ct_variable_std_50m_dt_1_idx_0.png
```

The `.npz` files contain the simulated states, noisy or noiseless observations, trajectory lengths, masks, active-model ground truth, and the per-step state-space matrices. The `.png` files provide a quick 2D check of one generated trajectory.

## Suggested repository layout

```text
ssm-trajectory-generator/
├── generate_trajectory_dataset.py
├── README.md
├── requirements.txt
└── data/                  # generated locally, usually not committed
```

Large generated datasets should usually be kept outside Git, or added to `.gitignore`, especially when they become large.
