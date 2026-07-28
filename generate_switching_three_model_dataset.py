#!/usr/bin/env python3
"""Generate trajectories that switch among OU, positive-CT and negative-CT models."""

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


SWITCHING_CODE = -2
MODEL_IDS = ("ou", "ct_positive", "ct_negative")


@dataclass(frozen=True)
class SwitchingThreeModelConfig:
    common: CommonConfig
    ou_gamma: float
    cruise_vx_min: float
    cruise_vx_max: float
    cruise_vy_min: float
    cruise_vy_max: float
    ct_turn_rate_magnitude_deg: float
    switch_min_duration: int
    switch_max_duration: int
    ensure_all_models: bool


def validate_config(cfg: SwitchingThreeModelConfig) -> None:
    validate_common_config(cfg.common)
    if cfg.ou_gamma <= 0.0:
        raise ValueError("--ou-gamma must be strictly positive.")
    if cfg.cruise_vx_min > cfg.cruise_vx_max:
        raise ValueError("--ou-vx-min must be <= --ou-vx-max.")
    if cfg.cruise_vy_min > cfg.cruise_vy_max:
        raise ValueError("--ou-vy-min must be <= --ou-vy-max.")
    if cfg.ct_turn_rate_magnitude_deg <= 0.0:
        raise ValueError("--ct-turn-rate-deg must be strictly positive.")
    if cfg.switch_min_duration <= 0 or cfg.switch_max_duration <= 0:
        raise ValueError("Switching durations must be positive.")
    if cfg.switch_min_duration > cfg.switch_max_duration:
        raise ValueError("--switch-min-duration must be <= --switch-max-duration.")

    minimum_length = (
        cfg.common.data_len
        if cfg.common.length_mode == "fixed"
        else cfg.common.min_len
    )
    if cfg.ensure_all_models and minimum_length < 3 * cfg.switch_min_duration:
        raise ValueError(
            "To guarantee all three models in every trajectory, the minimum "
            "trajectory length must be at least 3 * switch-min-duration."
        )


def choose_next_model(
    rng: np.random.Generator,
    previous: str | None,
) -> str:
    candidates = [model_id for model_id in MODEL_IDS if model_id != previous]
    return str(rng.choice(candidates))


def sample_model_segments(
    length: int,
    cfg: SwitchingThreeModelConfig,
    rng: np.random.Generator,
) -> list[tuple[int, int, str]]:
    """Return half-open segments (start, end, model_id)."""
    segments: list[tuple[int, int, str]] = []
    position = 0
    previous: str | None = None

    required_order: list[str] = []
    if cfg.ensure_all_models:
        required_order = [str(value) for value in rng.permutation(MODEL_IDS)]

    while position < length:
        remaining = length - position

        if required_order:
            model_id = required_order.pop(0)
            required_after = len(required_order)
            maximum_allowed = remaining - required_after * cfg.switch_min_duration
            maximum_duration = min(cfg.switch_max_duration, maximum_allowed)
            duration = int(
                rng.integers(cfg.switch_min_duration, maximum_duration + 1)
            )
        else:
            if remaining < cfg.switch_min_duration and segments:
                start, _, model_id = segments[-1]
                segments[-1] = (start, length, model_id)
                break

            model_id = choose_next_model(rng, previous)
            maximum_duration = min(cfg.switch_max_duration, remaining)
            duration = int(
                rng.integers(cfg.switch_min_duration, maximum_duration + 1)
            )

        end = min(length, position + duration)
        segments.append((position, end, model_id))
        previous = model_id
        position = end

    return segments


def sample_ou_cruise_velocity(
    cfg: SwitchingThreeModelConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    return np.array(
        [
            rng.uniform(cfg.cruise_vx_min, cfg.cruise_vx_max),
            rng.uniform(cfg.cruise_vy_min, cfg.cruise_vy_max),
        ],
        dtype=np.float64,
    )


def generate_clean_data(
    cfg: SwitchingThreeModelConfig,
) -> dict[str, np.ndarray]:
    validate_config(cfg)
    rng = np.random.default_rng(cfg.common.seed)
    lengths = sample_lengths(cfg.common, rng)
    padded_len = cfg.common.padded_len

    states = np.zeros(
        (cfg.common.batch_size, padded_len, 4), dtype=np.float64
    )
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
        cfg.common.batch_size, "switching", dtype="U16"
    )
    trajectory_model_codes = np.full(
        cfg.common.batch_size, SWITCHING_CODE, dtype=np.int8
    )
    segment_ids = np.full(
        (cfg.common.batch_size, padded_len), -1, dtype=np.int16
    )
    switch_flags = np.zeros(
        (cfg.common.batch_size, padded_len), dtype=bool
    )

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
        segments = sample_model_segments(length_i, cfg, rng)
        state = sample_initial_state(cfg.common, rng)

        for segment_index, (start, end, model_id) in enumerate(segments):
            input_matrix = np.zeros((4, 2), dtype=np.float64)
            control_input = np.zeros(2, dtype=np.float64)
            turn_rate = 0.0

            if model_id == "ou":
                transition = ou_transition
                input_matrix = ou_input
                control_input = sample_ou_cruise_velocity(cfg, rng)
            elif model_id == "ct_positive":
                transition = ct_positive_transition
                turn_rate = positive_rate
            else:
                transition = ct_negative_transition
                turn_rate = negative_rate

            switch_flags[i, start] = segment_index > 0

            for k in range(start, end):
                states[i, k] = state
                mask[i, k] = True
                transition_matrices[i, k] = transition
                input_matrices[i, k] = input_matrix
                control_inputs[i, k] = control_input
                turn_rates_deg[i, k] = turn_rate
                active_model_ids[i, k] = model_id
                active_model_codes[i, k] = MODEL_CODE[model_id]
                segment_ids[i, k] = segment_index
                state = transition @ state + input_matrix @ control_input

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
        "segment_ids": segment_ids,
        "switch_flags": switch_flags,
        "ou_cruise_velocities_per_step": control_inputs.copy(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate trajectories with within-trajectory switching among "
            "OU, positive coordinated turn, and negative coordinated turn."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--length-mode", choices=["fixed", "variable"], default="variable"
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
            "Positive turn-rate magnitude in deg/time-unit. CT negative uses "
            "the opposite sign."
        ),
    )
    parser.add_argument("--switch-min-duration", type=int, default=20)
    parser.add_argument("--switch-max-duration", type=int, default=50)
    parser.add_argument(
        "--allow-missing-models",
        action="store_true",
        help=(
            "Do not force all three models to appear in every trajectory. "
            "By default, each trajectory contains OU, CT positive, and CT negative."
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
    cfg = SwitchingThreeModelConfig(
        common=common,
        ou_gamma=args.ou_gamma,
        cruise_vx_min=args.ou_vx_min,
        cruise_vx_max=args.ou_vx_max,
        cruise_vy_min=args.ou_vy_min,
        cruise_vy_max=args.ou_vy_max,
        ct_turn_rate_magnitude_deg=args.ct_turn_rate_deg,
        switch_min_duration=args.switch_min_duration,
        switch_max_duration=args.switch_max_duration,
        ensure_all_models=not args.allow_missing_models,
    )

    clean_data = generate_clean_data(cfg)
    rate_tag = safe_number_tag(abs(cfg.ct_turn_rate_magnitude_deg))
    file_stem = (
        "switching_three_models_ou_ctpos_ctneg_"
        f"ou_vx_{range_tag(cfg.cruise_vx_min, cfg.cruise_vx_max)}_"
        f"vy_{range_tag(cfg.cruise_vy_min, cfg.cruise_vy_max)}_"
        f"ct_{rate_tag}deg"
    )
    positive_rate = abs(cfg.ct_turn_rate_magnitude_deg)
    metadata = {
        "dataset_type": np.array(
            "within_trajectory_switching_ou_ct_positive_ct_negative"
        ),
        "model_sampling_policy": np.array("piecewise_constant_segments"),
        "ou_cruise_resampling_policy": np.array("new_value_per_ou_segment"),
        "ou_gamma": np.array(cfg.ou_gamma, dtype=np.float64),
        "ou_cruise_vx_range": np.array(
            [cfg.cruise_vx_min, cfg.cruise_vx_max], dtype=np.float64
        ),
        "ou_cruise_vy_range": np.array(
            [cfg.cruise_vy_min, cfg.cruise_vy_max], dtype=np.float64
        ),
        "ct_positive_turn_rate_deg": np.array(
            positive_rate, dtype=np.float64
        ),
        "ct_negative_turn_rate_deg": np.array(
            -positive_rate, dtype=np.float64
        ),
        "switch_min_duration": np.array(
            cfg.switch_min_duration, dtype=np.int64
        ),
        "switch_max_duration": np.array(
            cfg.switch_max_duration, dtype=np.int64
        ),
        "ensure_all_models_per_trajectory": np.array(
            cfg.ensure_all_models, dtype=bool
        ),
        "trajectory_model_code_switching": np.array(
            SWITCHING_CODE, dtype=np.int8
        ),
        "process_noise_std": np.array(0.0, dtype=np.float64),
    }
    dataset_paths, plot_paths = save_npz_variants(
        cfg.common, clean_data, file_stem, metadata
    )
    print_summary(cfg.common, clean_data, dataset_paths, plot_paths)

    valid_codes = clean_data["active_model_codes"][clean_data["mask"]]
    print(
        "active time-step counts: "
        f"ou={int(np.sum(valid_codes == MODEL_CODE['ou']))}, "
        f"ct_positive={int(np.sum(valid_codes == MODEL_CODE['ct_positive']))}, "
        f"ct_negative={int(np.sum(valid_codes == MODEL_CODE['ct_negative']))}"
    )


if __name__ == "__main__":
    main()
