#!/usr/bin/env python3
"""Generate deterministic OU-only datasets with trajectory-specific cruise velocities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trajectory_dataset_core import (
    CommonConfig,
    MODEL_CODE,
    ou_transition_and_input,
    parse_float_list,
    print_summary,
    range_tag,
    sample_initial_state,
    sample_lengths,
    save_npz_variants,
    validate_common_config,
)


@dataclass(frozen=True)
class OUConfig:
    common: CommonConfig
    gamma: float
    cruise_vx_min: float
    cruise_vx_max: float
    cruise_vy_min: float
    cruise_vy_max: float


def validate_config(cfg: OUConfig) -> None:
    validate_common_config(cfg.common)
    if cfg.gamma <= 0.0:
        raise ValueError("--ou-gamma must be strictly positive.")
    if cfg.cruise_vx_min > cfg.cruise_vx_max:
        raise ValueError("--ou-vx-min must be <= --ou-vx-max.")
    if cfg.cruise_vy_min > cfg.cruise_vy_max:
        raise ValueError("--ou-vy-min must be <= --ou-vy-max.")


def generate_clean_data(cfg: OUConfig) -> dict[str, np.ndarray]:
    validate_config(cfg)
    rng = np.random.default_rng(cfg.common.seed)
    lengths = sample_lengths(cfg.common, rng)
    padded_len = cfg.common.padded_len

    states = np.zeros((cfg.common.batch_size, padded_len, 4), dtype=np.float64)
    mask = np.zeros((cfg.common.batch_size, padded_len), dtype=bool)
    transition_matrices = np.zeros(
        (cfg.common.batch_size, padded_len, 4, 4), dtype=np.float64
    )
    input_matrices = np.zeros(
        (cfg.common.batch_size, padded_len, 4, 2), dtype=np.float64
    )
    control_inputs = np.zeros(
        (cfg.common.batch_size, padded_len, 2), dtype=np.float64
    )
    turn_rates_deg = np.zeros(
        (cfg.common.batch_size, padded_len), dtype=np.float64
    )
    active_model_ids = np.full(
        (cfg.common.batch_size, padded_len), "pad", dtype="U16"
    )
    active_model_codes = np.full(
        (cfg.common.batch_size, padded_len), MODEL_CODE["pad"], dtype=np.int8
    )
    trajectory_model_ids = np.full(
        cfg.common.batch_size, "ou", dtype="U16"
    )
    trajectory_model_codes = np.full(
        cfg.common.batch_size, MODEL_CODE["ou"], dtype=np.int8
    )
    cruise_velocities = np.zeros((cfg.common.batch_size, 2), dtype=np.float64)

    transition, input_matrix = ou_transition_and_input(
        cfg.common.sampling_time, cfg.gamma
    )

    for i in range(cfg.common.batch_size):
        length_i = int(lengths[i])
        cruise_velocity = np.array(
            [
                rng.uniform(cfg.cruise_vx_min, cfg.cruise_vx_max),
                rng.uniform(cfg.cruise_vy_min, cfg.cruise_vy_max),
            ],
            dtype=np.float64,
        )
        cruise_velocities[i] = cruise_velocity
        state = sample_initial_state(cfg.common, rng)

        for k in range(length_i):
            states[i, k] = state
            mask[i, k] = True
            transition_matrices[i, k] = transition
            input_matrices[i, k] = input_matrix
            control_inputs[i, k] = cruise_velocity
            active_model_ids[i, k] = "ou"
            active_model_codes[i, k] = MODEL_CODE["ou"]
            state = transition @ state + input_matrix @ cruise_velocity

    return {
        "states": states,
        "mask": mask,
        "lengths": lengths,
        "transition_matrices": transition_matrices,
        "input_matrices": input_matrices,
        "control_inputs": control_inputs,
        "turn_rates_deg": turn_rates_deg,
        "active_model_ids": active_model_ids,
        "active_model_codes": active_model_codes,
        "trajectory_model_ids": trajectory_model_ids,
        "trajectory_model_codes": trajectory_model_codes,
        "cruise_velocities": cruise_velocities,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate OU-only trajectories with one cruise velocity sampled "
            "for each trajectory."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--length-mode", choices=["fixed", "variable"], default="fixed"
    )
    parser.add_argument("--data-len", type=int, default=250)
    parser.add_argument("--min-len", type=int, default=200)
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--sampling-time", type=float, default=1.0)
    parser.add_argument(
        "--measurement-std",
        nargs="+",
        default=["0", "1", "2", "3", "5"],
        help="One or more measurement-noise standard deviations in metres.",
    )
    parser.add_argument("--position-min", type=float, default=0.0)
    parser.add_argument("--position-max", type=float, default=1000.0)
    parser.add_argument("--initial-speed-min", type=float, default=4.0)
    parser.add_argument("--initial-speed-max", type=float, default=12.0)
    parser.add_argument("--plot-sample-index", type=int, default=0)
    parser.add_argument("--ou-gamma", type=float, default=0.05)
    parser.add_argument("--ou-vx-min", type=float, default=4.0)
    parser.add_argument("--ou-vx-max", type=float, default=12.0)
    parser.add_argument("--ou-vy-min", type=float, default=-4.0)
    parser.add_argument("--ou-vy-max", type=float, default=4.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    common = CommonConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        length_mode=args.length_mode,
        data_len=args.data_len,
        min_len=args.min_len,
        max_len=args.max_len,
        sampling_time=args.sampling_time,
        measurement_std=parse_float_list(args.measurement_std),
        seed=args.seed,
        position_min=args.position_min,
        position_max=args.position_max,
        initial_speed_min=args.initial_speed_min,
        initial_speed_max=args.initial_speed_max,
        plot_sample_index=args.plot_sample_index,
    )
    cfg = OUConfig(
        common=common,
        gamma=args.ou_gamma,
        cruise_vx_min=args.ou_vx_min,
        cruise_vx_max=args.ou_vx_max,
        cruise_vy_min=args.ou_vy_min,
        cruise_vy_max=args.ou_vy_max,
    )

    clean_data = generate_clean_data(cfg)
    file_stem = (
        "ou_variable_cruise_"
        f"vx_{range_tag(cfg.cruise_vx_min, cfg.cruise_vx_max)}_"
        f"vy_{range_tag(cfg.cruise_vy_min, cfg.cruise_vy_max)}"
    )
    metadata = {
        "dataset_type": np.array("ou_variable_cruise"),
        "ou_gamma": np.array(cfg.gamma, dtype=np.float64),
        "ou_cruise_vx_range": np.array(
            [cfg.cruise_vx_min, cfg.cruise_vx_max], dtype=np.float64
        ),
        "ou_cruise_vy_range": np.array(
            [cfg.cruise_vy_min, cfg.cruise_vy_max], dtype=np.float64
        ),
        "process_noise_std": np.array(0.0, dtype=np.float64),
    }
    dataset_paths, plot_paths = save_npz_variants(
        cfg.common, clean_data, file_stem, metadata
    )
    print_summary(cfg.common, clean_data, dataset_paths, plot_paths)


if __name__ == "__main__":
    main()
