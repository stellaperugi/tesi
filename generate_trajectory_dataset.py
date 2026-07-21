#!/usr/bin/env python3
"""
State-space trajectory dataset generator.

The script generates Cartesian target trajectories from deterministic linear
state-space models:

    OU: x[k+1] = F_ou x[k] + B_ou d
    CT: x[k+1] = F_ct x[k]

where x = [px, py, vx, vy]. Measurements are Cartesian positions corrupted by
Gaussian noise with user-specified standard deviation.

Supported features:
    - fixed-length and variable-length trajectories;
    - single-model datasets, OU or CT;
    - within-trajectory switching between OU and CT;
    - multiple measurement-noise standard deviations in one run;
    - active-model ground truth and model parameters saved at each step.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_CODE = {"pad": -1, "ct": 1, "ou": 2}
MODEL_NAMES_BY_CODE = np.array(["ct", "ou"], dtype="U8")


@dataclass(frozen=True)
class Config:
    output_dir: Path
    batch_size: int
    length_mode: str
    data_len: int
    min_len: int
    max_len: int
    sampling_time: float
    model_mode: str
    switch_min_duration: int
    switch_max_duration: int
    measurement_std: tuple[float, ...]
    seed: int

    position_min: float
    position_max: float
    speed_min: float
    speed_max: float

    ou_gamma: float
    ou_cruise_vx: float
    ou_cruise_vy: float

    ct_turn_rate_deg: float
    plot_sample_index: int

    @property
    def padded_len(self) -> int:
        return self.data_len if self.length_mode == "fixed" else self.max_len


def ncv_transition(dt: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def ct_transition(dt: float, turn_rate_deg: float) -> np.ndarray:
    if abs(turn_rate_deg) < 1e-12:
        return ncv_transition(dt)

    omega = np.deg2rad(turn_rate_deg)
    s = np.sin(omega * dt)
    c = np.cos(omega * dt)

    return np.array(
        [
            [1.0, 0.0, s / omega, (c - 1.0) / omega],
            [0.0, 1.0, -(c - 1.0) / omega, s / omega],
            [0.0, 0.0, c, -s],
            [0.0, 0.0, s, c],
        ],
        dtype=np.float64,
    )


def ou_transition_and_input(dt: float, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    if gamma <= 0.0:
        raise ValueError("OU gamma must be strictly positive.")

    e = float(np.exp(-gamma * dt))
    a = (1.0 - e) / gamma
    b = dt - a
    c = 1.0 - e

    F = np.array(
        [
            [1.0, 0.0, a, 0.0],
            [0.0, 1.0, 0.0, a],
            [0.0, 0.0, e, 0.0],
            [0.0, 0.0, 0.0, e],
        ],
        dtype=np.float64,
    )

    B = np.array(
        [
            [b, 0.0],
            [0.0, b],
            [c, 0.0],
            [0.0, c],
        ],
        dtype=np.float64,
    )
    return F, B


def parse_measurement_std(values: list[str]) -> tuple[float, ...]:
    std_values = tuple(float(v) for v in values)
    if len(std_values) == 0:
        raise ValueError("At least one measurement standard deviation is required.")
    if any(v < 0.0 for v in std_values):
        raise ValueError("Measurement standard deviations must be nonnegative.")
    return std_values


def validate_config(cfg: Config) -> None:
    if cfg.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if cfg.sampling_time <= 0.0:
        raise ValueError("--sampling-time must be positive.")
    if cfg.length_mode not in {"fixed", "variable"}:
        raise ValueError("--length-mode must be fixed or variable.")
    if cfg.length_mode == "fixed" and cfg.data_len <= 1:
        raise ValueError("--data-len must be larger than 1.")
    if cfg.length_mode == "variable":
        if cfg.min_len <= 1 or cfg.max_len <= 1:
            raise ValueError("--min-len and --max-len must be larger than 1.")
        if cfg.min_len > cfg.max_len:
            raise ValueError("--min-len must be <= --max-len.")
    if cfg.model_mode not in {"ou", "ct", "switching"}:
        raise ValueError("--model-mode must be ou, ct, or switching.")
    if cfg.switch_min_duration <= 0 or cfg.switch_max_duration <= 0:
        raise ValueError("Switching durations must be positive.")
    if cfg.switch_min_duration > cfg.switch_max_duration:
        raise ValueError("--switch-min-duration must be <= --switch-max-duration.")
    if cfg.position_min < 0.0 or cfg.position_max < cfg.position_min:
        raise ValueError("Require 0 <= --position-min <= --position-max.")
    if cfg.speed_max < cfg.speed_min:
        raise ValueError("Require --speed-min <= --speed-max.")
    if cfg.ou_gamma <= 0.0:
        raise ValueError("--ou-gamma must be strictly positive.")
    if cfg.plot_sample_index < 0 or cfg.plot_sample_index >= cfg.batch_size:
        raise ValueError("--plot-sample-index must be between 0 and batch-size-1.")


def sample_lengths(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    if cfg.length_mode == "fixed":
        return np.full(cfg.batch_size, cfg.data_len, dtype=np.int64)
    return rng.integers(cfg.min_len, cfg.max_len + 1, size=cfg.batch_size, dtype=np.int64)


def sample_initial_state(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    radius = rng.uniform(cfg.position_min, cfg.position_max)
    position_angle = rng.uniform(-np.pi, np.pi)
    speed = rng.uniform(cfg.speed_min, cfg.speed_max)
    velocity_angle = rng.uniform(-np.pi, np.pi)

    return np.array(
        [
            radius * np.cos(position_angle),
            radius * np.sin(position_angle),
            speed * np.cos(velocity_angle),
            speed * np.sin(velocity_angle),
        ],
        dtype=np.float64,
    )


def model_parameters(model_id: str, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    B = np.zeros((4, 2), dtype=np.float64)
    d = np.zeros(2, dtype=np.float64)
    turn_rate = 0.0

    if model_id == "ou":
        F, B = ou_transition_and_input(cfg.sampling_time, cfg.ou_gamma)
        d = np.array([cfg.ou_cruise_vx, cfg.ou_cruise_vy], dtype=np.float64)
    elif model_id == "ct":
        F = ct_transition(cfg.sampling_time, cfg.ct_turn_rate_deg)
        turn_rate = cfg.ct_turn_rate_deg
    else:
        raise ValueError(f"Unknown model id: {model_id}")

    return F, B, d, turn_rate


def active_model_sequence(cfg: Config, rng: np.random.Generator, length: int) -> np.ndarray:
    sequence = np.empty(length, dtype="U8")

    if cfg.model_mode in {"ou", "ct"}:
        sequence[:] = cfg.model_mode
        return sequence

    k = 0
    current = str(rng.choice(["ou", "ct"]))
    while k < length:
        duration = int(rng.integers(cfg.switch_min_duration, cfg.switch_max_duration + 1))
        end = min(length, k + duration)
        sequence[k:end] = current
        current = "ct" if current == "ou" else "ou"
        k = end

    return sequence


def generate_states(cfg: Config) -> dict[str, np.ndarray]:
    validate_config(cfg)
    rng = np.random.default_rng(cfg.seed)

    lengths = sample_lengths(cfg, rng)
    padded_len = cfg.padded_len

    states = np.zeros((cfg.batch_size, padded_len, 4), dtype=np.float64)
    mask = np.zeros((cfg.batch_size, padded_len), dtype=bool)
    transition_matrices = np.zeros((cfg.batch_size, padded_len, 4, 4), dtype=np.float64)
    input_matrices = np.zeros((cfg.batch_size, padded_len, 4, 2), dtype=np.float64)
    control_inputs = np.zeros((cfg.batch_size, padded_len, 2), dtype=np.float64)
    turn_rates = np.zeros((cfg.batch_size, padded_len), dtype=np.float64)
    active_model_ids = np.full((cfg.batch_size, padded_len), "pad", dtype="U8")
    active_model_codes = np.full((cfg.batch_size, padded_len), MODEL_CODE["pad"], dtype=np.int8)
    trajectory_model_ids = np.empty(cfg.batch_size, dtype="U16")

    for i in range(cfg.batch_size):
        length_i = int(lengths[i])
        model_sequence = active_model_sequence(cfg, rng, length_i)
        unique_models = set(model_sequence.tolist())
        trajectory_model_ids[i] = model_sequence[0] if len(unique_models) == 1 else "switching"

        x = sample_initial_state(cfg, rng)
        current_model = None
        F = np.eye(4, dtype=np.float64)
        B = np.zeros((4, 2), dtype=np.float64)
        d = np.zeros(2, dtype=np.float64)
        turn_rate = 0.0

        for k in range(length_i):
            model_k = str(model_sequence[k])

            if model_k != current_model:
                F, B, d, turn_rate = model_parameters(model_k, cfg)
                current_model = model_k

            states[i, k] = x
            mask[i, k] = True
            transition_matrices[i, k] = F
            input_matrices[i, k] = B
            control_inputs[i, k] = d
            turn_rates[i, k] = turn_rate
            active_model_ids[i, k] = model_k
            active_model_codes[i, k] = MODEL_CODE[model_k]

            x = F @ x + B @ d

    return {
        "states": states,
        "mask": mask,
        "lengths": lengths,
        "transition_matrices": transition_matrices,
        "input_matrices": input_matrices,
        "control_inputs": control_inputs,
        "turn_rates_deg": turn_rates,
        "active_model_ids": active_model_ids,
        "active_model_codes": active_model_codes,
        "trajectory_model_ids": trajectory_model_ids,
    }


def add_measurement_noise(
    states: np.ndarray,
    mask: np.ndarray,
    measurement_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    positions = states[:, :, 0:2]
    observations = np.zeros_like(positions)
    observations[mask] = positions[mask]

    if measurement_std > 0.0:
        noise = rng.normal(0.0, measurement_std, size=positions[mask].shape)
        observations[mask] = positions[mask] + noise

    return observations


def save_sample_plot(
    cfg: Config,
    clean_data: dict[str, np.ndarray],
    observations: np.ndarray,
    measurement_std: float,
) -> Path:
    sample_idx = cfg.plot_sample_index
    length_i = int(clean_data["lengths"][sample_idx])

    true_positions = clean_data["states"][sample_idx, :length_i, 0:2]
    obs_positions = observations[sample_idx, :length_i, :]
    model_ids = clean_data["active_model_ids"][sample_idx, :length_i]

    std_tag = f"std_{measurement_std:g}m".replace(".", "p")
    file_name = (
        f"sample_trajectory_{model_tag(cfg)}_"
        f"{cfg.length_mode}_{std_tag}_dt_{cfg.sampling_time:g}_idx_{sample_idx}.png"
    ).replace(" ", "")
    path = cfg.output_dir / file_name

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(true_positions[:, 0], true_positions[:, 1], linewidth=2, label="true trajectory")
    ax.scatter(obs_positions[:, 0], obs_positions[:, 1], s=12, alpha=0.55, label="observations")
    ax.scatter(true_positions[0, 0], true_positions[0, 1], s=50, marker="o", label="start")
    ax.scatter(true_positions[-1, 0], true_positions[-1, 1], s=60, marker="x", label="end")

    if cfg.model_mode == "switching":
        change_points = [0]
        for k in range(1, length_i):
            if model_ids[k] != model_ids[k - 1]:
                change_points.append(k)
        change_points.append(length_i)

        for start, end in zip(change_points[:-1], change_points[1:]):
            label = f"segment: {model_ids[start]}"
            ax.plot(
                true_positions[start:end, 0],
                true_positions[start:end, 1],
                linewidth=3,
                alpha=0.8,
                label=label,
            )

    ax.set_xlabel("p_x")
    ax.set_ylabel("p_y")
    ax.set_title(
        f"Sample trajectory {sample_idx} | mode={cfg.model_mode} | std={measurement_std:g} m"
    )
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_handles = []
    unique_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    ax.legend(unique_handles, unique_labels, loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def model_tag(cfg: Config) -> str:
    if cfg.model_mode == "switching":
        return "switching_ou_ct"
    if cfg.model_mode == "ou":
        tag = f"ou_vx_{cfg.ou_cruise_vx:g}_vy_{cfg.ou_cruise_vy:g}"
    else:
        tag = f"ct_turn_{cfg.ct_turn_rate_deg:g}deg"
    return tag.replace(".", "p").replace("-", "m")


def save_datasets(cfg: Config, clean_data: dict[str, np.ndarray]) -> list[Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for idx, std in enumerate(cfg.measurement_std):
        rng = np.random.default_rng(cfg.seed + 1000 + idx)
        observations = add_measurement_noise(clean_data["states"], clean_data["mask"], std, rng)

        std_tag = f"std_{std:g}m".replace(".", "p")
        file_name = (
            f"trajectory_dataset_{model_tag(cfg)}_"
            f"{cfg.length_mode}_{std_tag}_dt_{cfg.sampling_time:g}.npz"
        ).replace(" ", "")
        path = cfg.output_dir / file_name

        np.savez_compressed(
            path,
            observations=observations,
            measurement_noise_std=np.array(std, dtype=np.float64),
            sampling_time=np.array(cfg.sampling_time, dtype=np.float64),
            model_mode=np.array(cfg.model_mode),
            length_mode=np.array(cfg.length_mode),
            padded_len=np.array(cfg.padded_len, dtype=np.int64),
            data_len=np.array(cfg.data_len, dtype=np.int64),
            min_len=np.array(cfg.min_len, dtype=np.int64),
            max_len=np.array(cfg.max_len, dtype=np.int64),
            batch_size=np.array(cfg.batch_size, dtype=np.int64),
            ou_gamma=np.array(cfg.ou_gamma, dtype=np.float64),
            ou_cruise_velocity=np.array([cfg.ou_cruise_vx, cfg.ou_cruise_vy], dtype=np.float64),
            ct_turn_rate_deg=np.array(cfg.ct_turn_rate_deg, dtype=np.float64),
            switch_min_duration=np.array(cfg.switch_min_duration, dtype=np.int64),
            switch_max_duration=np.array(cfg.switch_max_duration, dtype=np.int64),
            model_code_names=MODEL_NAMES_BY_CODE,
            model_code_pad=np.array(MODEL_CODE["pad"], dtype=np.int8),
            seed=np.array(cfg.seed, dtype=np.int64),
            **clean_data,
        )
        plot_path = save_sample_plot(cfg, clean_data, observations, std)
        saved_paths.append(path)
        saved_paths.append(plot_path)

    return saved_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate OU/CT state-space trajectory datasets with std-based measurement noise."
    )

    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--length-mode", choices=["fixed", "variable"], default="fixed")
    parser.add_argument("--data-len", type=int, default=250)
    parser.add_argument("--min-len", type=int, default=200)
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--sampling-time", type=float, default=1.0)

    parser.add_argument(
        "--model-mode",
        choices=["ou", "ct", "switching"],
        default="switching",
        help="ou: all trajectories use OU; ct: all trajectories use CT; switching: within-trajectory OU/CT switching.",
    )
    parser.add_argument("--switch-min-duration", type=int, default=20)
    parser.add_argument("--switch-max-duration", type=int, default=50)

    parser.add_argument(
        "--measurement-std",
        nargs="+",
        default=["0", "10", "50"],
        help="One or more Cartesian measurement-noise standard deviations in meters.",
    )

    parser.add_argument("--position-min", type=float, default=0.0)
    parser.add_argument("--position-max", type=float, default=1000.0)
    parser.add_argument("--speed-min", type=float, default=4.0)
    parser.add_argument("--speed-max", type=float, default=12.0)

    parser.add_argument("--ou-gamma", type=float, default=0.05)
    parser.add_argument("--ou-cruise-vx", type=float, default=8.0)
    parser.add_argument("--ou-cruise-vy", type=float, default=0.0)

    parser.add_argument("--ct-turn-rate-deg", type=float, default=3.0)
    parser.add_argument("--plot-sample-index", type=int, default=0, help="Trajectory index used for the saved 2D sample plot.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    cfg = Config(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        length_mode=args.length_mode,
        data_len=args.data_len,
        min_len=args.min_len,
        max_len=args.max_len,
        sampling_time=args.sampling_time,
        model_mode=args.model_mode,
        switch_min_duration=args.switch_min_duration,
        switch_max_duration=args.switch_max_duration,
        measurement_std=parse_measurement_std(args.measurement_std),
        seed=args.seed,
        position_min=args.position_min,
        position_max=args.position_max,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        ou_gamma=args.ou_gamma,
        ou_cruise_vx=args.ou_cruise_vx,
        ou_cruise_vy=args.ou_cruise_vy,
        ct_turn_rate_deg=args.ct_turn_rate_deg,
        plot_sample_index=args.plot_sample_index,
    )

    clean_data = generate_states(cfg)
    saved_items = save_datasets(cfg, clean_data)

    valid_codes = clean_data["active_model_codes"][clean_data["mask"]]
    ct_count = int(np.sum(valid_codes == MODEL_CODE["ct"]))
    ou_count = int(np.sum(valid_codes == MODEL_CODE["ou"]))
    lengths = clean_data["lengths"]

    print("Dataset generation completed.")
    print(f"model_mode: {cfg.model_mode}")
    print(f"length_mode: {cfg.length_mode}")
    print(f"states: {clean_data['states'].shape}")
    print("observations: same first two dimensions, 2")
    print(f"lengths: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.2f}")
    print(f"active time-step counts: ct={ct_count}, ou={ou_count}")

    data_paths = [p for p in saved_items if p.suffix == ".npz"]
    plot_paths = [p for p in saved_items if p.suffix == ".png"]

    for path, std in zip(data_paths, cfg.measurement_std):
        print(f"saved dataset std={std:g} m: {path}")
    for path, std in zip(plot_paths, cfg.measurement_std):
        print(f"saved sample plot std={std:g} m: {path}")


if __name__ == "__main__":
    main()
