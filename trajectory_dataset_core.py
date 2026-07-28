#!/usr/bin/env python3
"""Common functions for deterministic OU and coordinated-turn datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


MODEL_CODE = {
    "pad": -1,
    "ou": 0,
    "ct_positive": 1,
    "ct_negative": 2,
}
MODEL_NAMES_BY_CODE = np.array(
    ["ou", "ct_positive", "ct_negative"], dtype="U16"
)


@dataclass(frozen=True)
class CommonConfig:
    output_dir: Path
    batch_size: int
    length_mode: str
    data_len: int
    min_len: int
    max_len: int
    sampling_time: float
    measurement_std: tuple[float, ...]
    seed: int
    position_min: float
    position_max: float
    initial_speed_min: float
    initial_speed_max: float
    plot_sample_index: int

    @property
    def padded_len(self) -> int:
        return self.data_len if self.length_mode == "fixed" else self.max_len


def parse_float_list(values: Iterable[str]) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if not parsed:
        raise ValueError("At least one value is required.")
    return parsed


def validate_common_config(cfg: CommonConfig) -> None:
    if cfg.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if cfg.sampling_time <= 0.0:
        raise ValueError("--sampling-time must be positive.")
    if cfg.length_mode == "fixed":
        if cfg.data_len <= 1:
            raise ValueError("--data-len must be larger than 1.")
    elif cfg.length_mode == "variable":
        if cfg.min_len <= 1 or cfg.max_len <= 1:
            raise ValueError("--min-len and --max-len must be larger than 1.")
        if cfg.min_len > cfg.max_len:
            raise ValueError("--min-len must be <= --max-len.")
    else:
        raise ValueError("--length-mode must be fixed or variable.")
    if cfg.position_min < 0.0 or cfg.position_max < cfg.position_min:
        raise ValueError("Require 0 <= --position-min <= --position-max.")
    if cfg.initial_speed_min < 0.0 or cfg.initial_speed_max < cfg.initial_speed_min:
        raise ValueError(
            "Require 0 <= --initial-speed-min <= --initial-speed-max."
        )
    if any(std < 0.0 for std in cfg.measurement_std):
        raise ValueError("Measurement standard deviations must be nonnegative.")
    if cfg.plot_sample_index < 0 or cfg.plot_sample_index >= cfg.batch_size:
        raise ValueError(
            "--plot-sample-index must be between 0 and batch-size - 1."
        )


def sample_lengths(cfg: CommonConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.length_mode == "fixed":
        return np.full(cfg.batch_size, cfg.data_len, dtype=np.int64)
    return rng.integers(
        cfg.min_len,
        cfg.max_len + 1,
        size=cfg.batch_size,
        dtype=np.int64,
    )


def sample_initial_state(
    cfg: CommonConfig, rng: np.random.Generator
) -> np.ndarray:
    radius = rng.uniform(cfg.position_min, cfg.position_max)
    position_heading = rng.uniform(-np.pi, np.pi)
    speed = rng.uniform(cfg.initial_speed_min, cfg.initial_speed_max)
    velocity_heading = rng.uniform(-np.pi, np.pi)

    return np.array(
        [
            radius * np.cos(position_heading),
            radius * np.sin(position_heading),
            speed * np.cos(velocity_heading),
            speed * np.sin(velocity_heading),
        ],
        dtype=np.float64,
    )


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
    """Coordinated-turn transition for state [px, py, vx, vy]."""
    if abs(turn_rate_deg) < 1e-12:
        return ncv_transition(dt)

    omega = np.deg2rad(turn_rate_deg)
    sine = np.sin(omega * dt)
    cosine = np.cos(omega * dt)

    return np.array(
        [
            [1.0, 0.0, sine / omega, (cosine - 1.0) / omega],
            [0.0, 1.0, -(cosine - 1.0) / omega, sine / omega],
            [0.0, 0.0, cosine, -sine],
            [0.0, 0.0, sine, cosine],
        ],
        dtype=np.float64,
    )


def ou_transition_and_input(
    dt: float, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exact first-moment discretization of integrated OU velocity dynamics."""
    if gamma <= 0.0:
        raise ValueError("OU gamma must be strictly positive.")

    decay = float(np.exp(-gamma * dt))
    velocity_gain = (1.0 - decay) / gamma
    position_input_gain = dt - velocity_gain
    velocity_input_gain = 1.0 - decay

    transition = np.array(
        [
            [1.0, 0.0, velocity_gain, 0.0],
            [0.0, 1.0, 0.0, velocity_gain],
            [0.0, 0.0, decay, 0.0],
            [0.0, 0.0, 0.0, decay],
        ],
        dtype=np.float64,
    )

    input_matrix = np.array(
        [
            [position_input_gain, 0.0],
            [0.0, position_input_gain],
            [velocity_input_gain, 0.0],
            [0.0, velocity_input_gain],
        ],
        dtype=np.float64,
    )
    return transition, input_matrix


def add_measurement_noise(
    states: np.ndarray,
    mask: np.ndarray,
    measurement_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    positions = states[:, :, :2]
    observations = np.zeros_like(positions)
    observations[mask] = positions[mask]

    if measurement_std > 0.0:
        noise = rng.normal(
            loc=0.0,
            scale=measurement_std,
            size=positions[mask].shape,
        )
        observations[mask] = positions[mask] + noise

    return observations


def safe_number_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def range_tag(low: float, high: float) -> str:
    return f"{safe_number_tag(low)}_{safe_number_tag(high)}"


def save_npz_variants(
    cfg: CommonConfig,
    clean_data: dict[str, np.ndarray],
    file_stem: str,
    metadata: dict[str, np.ndarray],
) -> tuple[list[Path], list[Path]]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_paths: list[Path] = []
    plot_paths: list[Path] = []

    for index, std in enumerate(cfg.measurement_std):
        noise_rng = np.random.default_rng(cfg.seed + 10_000 + index)
        observations = add_measurement_noise(
            clean_data["states"], clean_data["mask"], std, noise_rng
        )

        std_tag = safe_number_tag(std)
        dt_tag = safe_number_tag(cfg.sampling_time)
        dataset_path = cfg.output_dir / (
            f"trajectory_dataset_{file_stem}_{cfg.length_mode}_"
            f"std_{std_tag}m_dt_{dt_tag}.npz"
        )

        np.savez_compressed(
            dataset_path,
            observations=observations,
            measurement_noise_std=np.array(std, dtype=np.float64),
            sampling_time=np.array(cfg.sampling_time, dtype=np.float64),
            length_mode=np.array(cfg.length_mode),
            batch_size=np.array(cfg.batch_size, dtype=np.int64),
            data_len=np.array(cfg.data_len, dtype=np.int64),
            min_len=np.array(cfg.min_len, dtype=np.int64),
            max_len=np.array(cfg.max_len, dtype=np.int64),
            padded_len=np.array(cfg.padded_len, dtype=np.int64),
            model_code_names=MODEL_NAMES_BY_CODE,
            model_code_pad=np.array(MODEL_CODE["pad"], dtype=np.int8),
            seed=np.array(cfg.seed, dtype=np.int64),
            **metadata,
            **clean_data,
        )
        dataset_paths.append(dataset_path)

        plot_path = save_sample_plot(
            cfg=cfg,
            clean_data=clean_data,
            observations=observations,
            measurement_std=std,
            file_stem=file_stem,
        )
        plot_paths.append(plot_path)

    return dataset_paths, plot_paths


def save_sample_plot(
    cfg: CommonConfig,
    clean_data: dict[str, np.ndarray],
    observations: np.ndarray,
    measurement_std: float,
    file_stem: str,
) -> Path:
    sample_index = cfg.plot_sample_index
    length_i = int(clean_data["lengths"][sample_index])
    positions = clean_data["states"][sample_index, :length_i, :2]
    measured = observations[sample_index, :length_i]
    model_id = str(clean_data["trajectory_model_ids"][sample_index])

    std_tag = safe_number_tag(measurement_std)
    dt_tag = safe_number_tag(cfg.sampling_time)
    plot_path = cfg.output_dir / (
        f"sample_trajectory_{file_stem}_{cfg.length_mode}_"
        f"std_{std_tag}m_dt_{dt_tag}_idx_{sample_index}.png"
    )

    fig, axis = plt.subplots(figsize=(7, 6))
    axis.plot(
        positions[:, 0],
        positions[:, 1],
        linewidth=2.0,
        label="true trajectory",
    )
    axis.scatter(
        measured[:, 0],
        measured[:, 1],
        s=12,
        alpha=0.55,
        label="observations",
    )
    axis.scatter(
        positions[0, 0], positions[0, 1], marker="o", s=55, label="start"
    )
    axis.scatter(
        positions[-1, 0], positions[-1, 1], marker="x", s=65, label="end"
    )
    axis.set_xlabel("p_x [m]")
    axis.set_ylabel("p_y [m]")
    axis.set_title(
        f"Sample {sample_index}: {model_id}, measurement std = {measurement_std:g} m"
    )
    axis.axis("equal")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def print_summary(
    cfg: CommonConfig,
    clean_data: dict[str, np.ndarray],
    dataset_paths: list[Path],
    plot_paths: list[Path],
) -> None:
    lengths = clean_data["lengths"]
    model_ids, counts = np.unique(
        clean_data["trajectory_model_ids"], return_counts=True
    )
    counts_text = ", ".join(
        f"{model_id}={count}" for model_id, count in zip(model_ids, counts)
    )

    print("Dataset generation completed.")
    print(f"states shape: {clean_data['states'].shape}")
    print(f"mask shape: {clean_data['mask'].shape}")
    print(
        f"lengths: min={lengths.min()}, max={lengths.max()}, "
        f"mean={lengths.mean():.2f}"
    )
    print(f"trajectory model counts: {counts_text}")

    for std, dataset_path, plot_path in zip(
        cfg.measurement_std, dataset_paths, plot_paths
    ):
        print(f"saved std={std:g} m: {dataset_path}")
        print(f"saved plot: {plot_path}")
