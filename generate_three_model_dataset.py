#!/usr/bin/env python3
"""Generate datasets containing OU, positive-CT and negative-CT trajectories."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trajectory_dataset_core import (
    CommonConfig,
    MODEL_CODE,
    ct_transition,
    ou_transition_and_input,
    parse_float_list,
    print_summary,
    range_tag,
    safe_number_tag,
    sample_initial_state,
    sample_lengths,
    save_npz_variants,
    validate_common_config,
)


@dataclass(frozen=True)
class ThreeModelConfig:
    common: CommonConfig
    ou_gamma: float
    cruise_vx_min: float
    cruise_vx_max: float
    cruise_vy_min: float
    cruise_vy_max: float
    ct_turn_rate_magnitude_deg: float


def validate_config(cfg: ThreeModelConfig) -> None:
    validate_common_config(cfg.common)
    if cfg.ou_gamma <= 0.0:
        raise ValueError("--ou-gamma must be strictly positive.")
    if cfg.cruise_vx_min > cfg.cruise_vx_max:
        raise ValueError("--ou-vx-min must be <= --ou-vx-max.")
    if cfg.cruise_vy_min > cfg.cruise_vy_max:
        raise ValueError("--ou-vy-min must be <= --ou-vy-max.")
    if cfg.ct_turn_rate_magnitude_deg <= 0.0:
        raise ValueError("--ct-turn-rate-deg must be strictly positive.")


def balanced_model_assignment(
    batch_size: int, rng: np.random.Generator
) -> np.ndarray:
    base = np.array(["ou", "ct_positive", "ct_negative"], dtype="U16")
    assignment = np.resize(base, batch_size).copy()
    rng.shuffle(assignment)
    return assignment


def generate_clean_data(cfg: ThreeModelConfig) -> dict[str, np.ndarray]:
    validate_config(cfg)
    rng = np.random.default_rng(cfg.common.seed)
    lengths = sample_lengths(cfg.common, rng)
    model_assignment = balanced_model_assignment(cfg.common.batch_size, rng)
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
    trajectory_model_ids = model_assignment.copy()
    trajectory_model_codes = np.array(
        [MODEL_CODE[model_id] for model_id in model_assignment], dtype=np.int8
    )
    cruise_velocities = np.zeros((cfg.common.batch_size, 2), dtype=np.float64)

    ou_transition, ou_input = ou_transition_and_input(
        cfg.common.sampling_time, cfg.ou_gamma
    )
    positive_rate = abs(cfg.ct_turn_rate_magnitude_deg)
    negative_rate = -positive_rate
    ct_positive_transition = ct_transition(
        cfg.common.sampling_time, positive_rate
    )
    ct_negative_transition = ct_transition(
        cfg.common.sampling_time, negative_rate
    )

    for i in range(cfg.common.batch_size):
        length_i = int(lengths[i])
        model_id = str(model_assignment[i])
        state = sample_initial_state(cfg.common, rng)
        input_matrix = np.zeros((4, 2), dtype=np.float64)
        cruise_velocity = np.zeros(2, dtype=np.float64)
        turn_rate = 0.0

        if model_id == "ou":
            transition = ou_transition
            input_matrix = ou_input
            cruise_velocity = np.array(
                [
                    rng.uniform(cfg.cruise_vx_min, cfg.cruise_vx_max),
                    rng.uniform(cfg.cruise_vy_min, cfg.cruise_vy_max),
                ],
                dtype=np.float64,
            )
            cruise_velocities[i] = cruise_velocity
        elif model_id == "ct_positive":
            transition = ct_positive_transition
            turn_rate = positive_rate
        else:
            transition = ct_negative_transition
            turn_rate = negative_rate

        for k in range(length_i):
            states[i, k] = state
            mask[i, k] = True
            transition_matrices[i, k] = transition
            input_matrices[i, k] = input_matrix
            control_inputs[i, k] = cruise_velocity
            turn_rates_deg[i, k] = turn_rate
            active_model_ids[i, k] = model_id
            active_model_codes[i, k] = MODEL_CODE[model_id]
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
            "Generate a balanced dataset with OU, positive coordinated-turn "
            "and negative coordinated-turn trajectories."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=999)
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
    parser.add_argument(
        "--ct-turn-rate-deg",
        type=float,
        default=3.0,
        help=(
            "Positive turn-rate magnitude in deg/time-unit. The third model "
            "uses the corresponding negative value."
        ),
    )
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
    cfg = ThreeModelConfig(
        common=common,
        ou_gamma=args.ou_gamma,
        cruise_vx_min=args.ou_vx_min,
        cruise_vx_max=args.ou_vx_max,
        cruise_vy_min=args.ou_vy_min,
        cruise_vy_max=args.ou_vy_max,
        ct_turn_rate_magnitude_deg=args.ct_turn_rate_deg,
    )

    clean_data = generate_clean_data(cfg)
    rate_tag = safe_number_tag(abs(cfg.ct_turn_rate_magnitude_deg))
    file_stem = (
        "three_models_ou_ctpos_ctneg_"
        f"ou_vx_{range_tag(cfg.cruise_vx_min, cfg.cruise_vx_max)}_"
        f"vy_{range_tag(cfg.cruise_vy_min, cfg.cruise_vy_max)}_"
        f"ct_{rate_tag}deg"
    )
    metadata = {
        "dataset_type": np.array("ou_ct_positive_ct_negative"),
        "model_sampling_policy": np.array("balanced_per_trajectory"),
        "ou_gamma": np.array(cfg.ou_gamma, dtype=np.float64),
        "ou_cruise_vx_range": np.array(
            [cfg.cruise_vx_min, cfg.cruise_vx_max], dtype=np.float64
        ),
        "ou_cruise_vy_range": np.array(
            [cfg.cruise_vy_min, cfg.cruise_vy_max], dtype=np.float64
        ),
        "ct_positive_turn_rate_deg": np.array(
            abs(cfg.ct_turn_rate_magnitude_deg), dtype=np.float64
        ),
        "ct_negative_turn_rate_deg": np.array(
            -abs(cfg.ct_turn_rate_magnitude_deg), dtype=np.float64
        ),
        "process_noise_std": np.array(0.0, dtype=np.float64),
    }
    dataset_paths, plot_paths = save_npz_variants(
        cfg.common, clean_data, file_stem, metadata
    )
    print_summary(cfg.common, clean_data, dataset_paths, plot_paths)


if __name__ == "__main__":
    main()
