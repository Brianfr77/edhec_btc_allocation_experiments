#!/usr/bin/env python3
"""Part 14 experiment runner: pairwise bootstrap inference for Part 10 benchmarks.

The bootstrap conditions on the frozen Part 10 scenario definitions. It does not
re-estimate PCA/HMM states or re-select allocation rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import pickle
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_STATE_START = "2018-02-09"
EXPECTED_STATE_END = "2026-03-27"
EXPECTED_STATE_ROWS = 425
TRADING_WEEKS_PER_YEAR = 52
TAIL_ALPHA = 0.05
FLOAT_TOL = 1e-10

ASSETS = ["ret_btc", "ret_spy", "ret_tlt", "ret_ief", "ret_gld", "ret_dbc"]
PORTFOLIO_FAMILIES = ["all_weather", "erc"]
BOOTSTRAP_REPS = 2000
BLOCK_LENGTH = 13

CORE_METRICS = [
    "annualized_mean_arithmetic",
    "annualized_volatility",
    "cvar_95_weekly",
    "max_drawdown",
    "btc_share_vol",
    "btc_share_cvar",
    "average_btc_weight",
]

REQUIRED_RESULTS = [
    "input_validation_summary.json",
    "output_validation_summary.json",
    "data_lineage.csv",
    "model_assumption_audit.json",
    "methodology_audit.md",
    "bootstrap_configuration.json",
    "bootstrap_draw_index_audit.csv",
    "part14_pairwise_bootstrap_ci.csv",
    "part14_cap_exceedance_under_pairwise_bootstrap.csv",
    "part14_inference_decision_matrix.csv",
]

REQUIRED_FIGURES = [
    "part14_pairwise_ci_core_metrics.png",
    "part14_pairwise_ci_risk_contribution.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 14 pairwise bootstrap inference.")
    parser.add_argument("--input-dir", default="data_2026/cleaned", type=Path)
    parser.add_argument(
        "--part10-run-dir",
        default="outputs/part10_benchmark_cap_sensitivity_outputs/part10_benchmark_cap_sensitivity/colab_part10_seed42",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/part14_pairwise_inference_outputs/part14_pairwise_inference",
        type=Path,
    )
    parser.add_argument("--run-id", default="colab_part14_seed42")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--bootstrap-reps", default=BOOTSTRAP_REPS, type=int)
    parser.add_argument("--block-length", default=BLOCK_LENGTH, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def ensure_dirs(run_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "results": run_dir / "results",
        "figures": run_dir / "figures",
        "models": run_dir / "models",
        "logs": run_dir / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_logging(log_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "run.log", mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_pickle(path: Path, payload: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_or_run(dirs: dict[str, Path], name: str, resume: bool, compute_fn) -> Any:
    path = dirs["checkpoints"] / f"{name}.pkl"
    if resume and path.exists():
        logging.info("Loading checkpoint: %s", path)
        return load_pickle(path)
    payload = compute_fn()
    save_pickle(path, payload)
    logging.info("Saved checkpoint: %s", path)
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.replace("\n", " "), "platform": platform.platform()}
    for package in ["numpy", "pandas", "matplotlib"]:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def normalize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [normalize_for_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if not isinstance(value, (str, bytes, dict, list, tuple)) and pd.isna(value):
        return None
    return value


def date_string(series: pd.Series, fn: str) -> str:
    value = series.min() if fn == "min" else series.max()
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def read_inputs(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "state_model_panel_weekly": args.input_dir / "state_model_panel_weekly.csv",
        "part10_manifest": args.part10_run_dir / "run_manifest.json",
        "part10_input_validation": args.part10_run_dir / "results" / "input_validation_summary.json",
        "part10_output_validation": args.part10_run_dir / "results" / "output_validation_summary.json",
        "part10_scenario_dictionary": args.part10_run_dir / "results" / "part10_scenario_dictionary.csv",
        "part10_return_series": args.part10_run_dir / "results" / "part10_return_series.csv",
        "part10_weekly_weights": args.part10_run_dir / "results" / "part10_weekly_weights.csv",
        "part10_pairwise_comparison": args.part10_run_dir / "results" / "part10_pairwise_benchmark_comparison.csv",
        "part10_risk_contribution_summary": args.part10_run_dir / "results" / "part10_risk_contribution_summary.csv",
    }
    for name, path in paths.items():
        require(path.exists(), f"Missing required input file: {name} at {path}")

    return {
        "paths": paths,
        "state": pd.read_csv(paths["state_model_panel_weekly"], parse_dates=["date"]),
        "part10_manifest": read_json(paths["part10_manifest"]),
        "part10_input_validation": read_json(paths["part10_input_validation"]),
        "part10_output_validation": read_json(paths["part10_output_validation"]),
        "scenario_dictionary": pd.read_csv(paths["part10_scenario_dictionary"]),
        "returns": pd.read_csv(paths["part10_return_series"], parse_dates=["date"]),
        "weights": pd.read_csv(paths["part10_weekly_weights"], parse_dates=["date"]),
        "part10_pairwise": pd.read_csv(paths["part10_pairwise_comparison"]),
        "part10_risk": pd.read_csv(paths["part10_risk_contribution_summary"]),
        "input_hashes": {name: file_sha256(path) for name, path in paths.items()},
    }


def enforce_resume_input_hashes(args: argparse.Namespace, dirs: dict[str, Path], input_hashes: dict[str, str]) -> None:
    if not args.resume:
        return
    manifest_path = dirs["root"] / "run_manifest.json"
    if not manifest_path.exists():
        return
    old_manifest = read_json(manifest_path)
    require(old_manifest.get("input_hashes", {}) == input_hashes, "Input hashes changed since previous run")
    logging.info("Resume input hash check passed")


def validate_inputs(inputs: dict[str, Any], args: argparse.Namespace, dirs: dict[str, Path]) -> dict[str, Any]:
    state = inputs["state"]
    scenarios = inputs["scenario_dictionary"]
    returns = inputs["returns"]
    weights = inputs["weights"]
    part10_output = inputs["part10_output_validation"]

    require(part10_output.get("status") == "passed", "Part 10 output validation did not pass")
    require(inputs["part10_input_validation"].get("status") == "passed", "Part 10 input validation did not pass")
    require(inputs["part10_manifest"].get("status") == "passed", "Part 10 manifest is not passed")
    require(len(state) == EXPECTED_STATE_ROWS, f"Unexpected state panel rows: {len(state)}")
    require(date_string(state["date"], "min") == EXPECTED_STATE_START, "Unexpected state panel start")
    require(date_string(state["date"], "max") == EXPECTED_STATE_END, "Unexpected state panel end")
    require(all(col in state.columns for col in ["date"] + ASSETS), "Missing required asset return columns")

    require(scenarios["scenario_id"].is_unique, "Part 10 scenario ids are not unique")
    require(set(PORTFOLIO_FAMILIES).issubset(set(scenarios["portfolio_family"])), "Missing portfolio families")
    require(len(returns) == len(scenarios) * EXPECTED_STATE_ROWS, "Unexpected Part 10 return rows")
    require(weights.groupby(["scenario_id", "date"])["weight"].sum().sub(1.0).abs().max() < 1e-8, "Part 10 weights do not sum to one")

    required_pair_ids = core_pair_specs()
    missing_pairs = sorted(set(required_pair_ids) - set(inputs["part10_pairwise"]["comparison_id"]))
    require(not missing_pairs, f"Missing Part 10 pairwise comparisons: {missing_pairs}")

    summary = {
        "status": "passed",
        "sample_frozen": True,
        "sample_domain": "target_weight_full_state_sample",
        "sample_rows": EXPECTED_STATE_ROWS,
        "sample_start": EXPECTED_STATE_START,
        "sample_end": EXPECTED_STATE_END,
        "bootstrap_reps": args.bootstrap_reps,
        "block_length": args.block_length,
        "part10_scenario_count": int(len(scenarios)),
        "part10_return_rows": int(len(returns)),
        "part10_weight_rows": int(len(weights)),
        "input_hashes": inputs["input_hashes"],
        "part10_run_id": inputs["part10_manifest"].get("run_id"),
    }
    write_json(dirs["results"] / "input_validation_summary.json", normalize_for_json(summary))
    logging.info("Input validation passed")
    return summary


def core_pair_specs() -> dict[str, tuple[str, str]]:
    specs: dict[str, tuple[str, str]] = {}
    for family in PORTFOLIO_FAMILIES:
        conditional = f"part10__conditional_cap__{family}__cap_10pct"
        matched = f"part10__matched_fixed_btc__{family}__0109bp"
        cap_only = f"part10__cap_only__{family}__cap_10pct"
        no_btc = f"part10__no_btc_baseline__{family}"
        specs[f"{family}__conditional_cap_10pct_vs_matched_fixed_btc"] = (conditional, matched)
        specs[f"{family}__conditional_cap_10pct_vs_cap_only_10pct"] = (conditional, cap_only)
        specs[f"{family}__conditional_cap_10pct_vs_no_btc_baseline"] = (conditional, no_btc)
        specs[f"{family}__cap_only_10pct_vs_matched_fixed_btc"] = (cap_only, matched)
    return specs


def var_cvar(returns: pd.Series, alpha: float = TAIL_ALPHA) -> tuple[float, float, int]:
    clean = pd.Series(returns).dropna()
    if clean.empty:
        return float("nan"), float("nan"), 0
    var_value = float(clean.quantile(alpha))
    tail = clean[clean <= var_value]
    return var_value, float(tail.mean()) if len(tail) else var_value, int(len(tail))


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + pd.Series(returns).fillna(0.0)).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1.0


def performance_metrics(returns: pd.Series) -> dict[str, Any]:
    clean = pd.Series(returns).dropna()
    var_value, cvar_value, tail_count = var_cvar(clean)
    mean = float(clean.mean()) if len(clean) else float("nan")
    vol = float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")
    return {
        "annualized_mean_arithmetic": mean * TRADING_WEEKS_PER_YEAR if math.isfinite(mean) else float("nan"),
        "annualized_volatility": vol * math.sqrt(TRADING_WEEKS_PER_YEAR) if math.isfinite(vol) else float("nan"),
        "cvar_95_weekly": cvar_value,
        "max_drawdown": float(drawdown_series(clean).min()) if len(clean) else float("nan"),
        "tail_scenario_count": tail_count,
    }


def risk_shares_from_matrix(component_matrix: np.ndarray, asset_index: int = 0) -> dict[str, Any]:
    portfolio = component_matrix.sum(axis=1)
    vol = float(np.std(portfolio, ddof=1))
    if vol <= FLOAT_TOL:
        vol_share = float("nan")
    else:
        cov = float(np.cov(component_matrix[:, asset_index], portfolio, ddof=1)[0, 1])
        vol_share = cov / (vol * vol)
    var_value = float(np.quantile(portfolio, TAIL_ALPHA))
    tail_mask = portfolio <= var_value
    tail_count = int(tail_mask.sum())
    cvar_loss = float((-portfolio[tail_mask]).mean()) if tail_count else float("nan")
    if not math.isfinite(cvar_loss) or abs(cvar_loss) <= FLOAT_TOL:
        cvar_share = float("nan")
    else:
        cvar_share = float((-component_matrix[tail_mask, asset_index]).mean() / cvar_loss)
    return {
        "btc_share_vol": vol_share,
        "btc_share_cvar": cvar_share,
        "tail_scenario_count": tail_count,
    }


def circular_block_indices(n: int, block_length: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(math.ceil(n / block_length))
    draws = np.empty((reps, n), dtype=np.int32)
    for rep in range(reps):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(start, start + block_length) % n) for start in starts])[:n]
        draws[rep] = idx
    return draws


def build_bootstrap_draws(args: argparse.Namespace, dirs: dict[str, Path]) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    draws = circular_block_indices(EXPECTED_STATE_ROWS, args.block_length, args.bootstrap_reps, rng)
    audit = pd.DataFrame(
        [
            {
                "sample_domain": "target_weight_full_state_sample",
                "bootstrap_reps": int(draws.shape[0]),
                "sample_length": int(draws.shape[1]),
                "block_length": args.block_length,
                "min_index": int(draws.min()),
                "max_index": int(draws.max()),
                "first_rep_unique_index_count": int(len(np.unique(draws[0]))),
                "same_draw_used_within_domain": True,
            }
        ]
    )
    audit.to_csv(dirs["results"] / "bootstrap_draw_index_audit.csv", index=False)
    config = {
        "method": "circular_moving_block_bootstrap",
        "bootstrap_reps": args.bootstrap_reps,
        "block_length": args.block_length,
        "ci_levels": [0.90, 0.95],
        "seed": args.seed,
        "sample_domains": {"target_weight_full_state_sample": EXPECTED_STATE_ROWS},
        "conditional_on_fixed_rule": True,
        "reestimates_pca_hmm": False,
        "reselects_allocation_rule": False,
    }
    write_json(dirs["results"] / "bootstrap_configuration.json", normalize_for_json(config))
    return draws


def build_scenario_arrays(inputs: dict[str, Any], dirs: dict[str, Path]) -> dict[str, dict[str, Any]]:
    state = inputs["state"][["date"] + ASSETS].sort_values("date").reset_index(drop=True)
    returns = inputs["returns"].sort_values(["scenario_id", "date"]).copy()
    weights = inputs["weights"].sort_values(["scenario_id", "date", "asset"]).copy()
    scenarios = inputs["scenario_dictionary"].copy()

    scenario_arrays: dict[str, dict[str, Any]] = {}
    for scenario_id in sorted(scenarios["scenario_id"].unique()):
        ret_frame = returns[returns["scenario_id"].eq(scenario_id)].sort_values("date").reset_index(drop=True)
        require(len(ret_frame) == EXPECTED_STATE_ROWS, f"Unexpected return rows for {scenario_id}")
        require(ret_frame["date"].equals(state["date"]), f"Date alignment failed for returns: {scenario_id}")
        weight_frame = weights[weights["scenario_id"].eq(scenario_id)]
        weight_matrix = (
            weight_frame.pivot_table(index="date", columns="asset", values="weight", aggfunc="first")
            .reindex(index=state["date"], columns=ASSETS)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        return_matrix = state[ASSETS].to_numpy(dtype=float)
        component_matrix = weight_matrix * return_matrix
        portfolio_return = ret_frame["portfolio_return"].to_numpy(dtype=float)
        require(np.allclose(component_matrix.sum(axis=1), portfolio_return, atol=1e-12), f"Component return mismatch: {scenario_id}")
        metadata = scenarios[scenarios["scenario_id"].eq(scenario_id)].iloc[0].to_dict()
        original_perf = performance_metrics(pd.Series(portfolio_return))
        original_risk = risk_shares_from_matrix(component_matrix)
        original = {
            **original_perf,
            **original_risk,
            "average_btc_weight": float(weight_matrix[:, 0].mean()),
        }
        scenario_arrays[scenario_id] = {
            "scenario_id": scenario_id,
            "metadata": metadata,
            "portfolio_return": portfolio_return,
            "component_matrix": component_matrix,
            "btc_weight": weight_matrix[:, 0],
            "original": original,
        }
    logging.info("Prepared scenario arrays for %d scenarios", len(scenario_arrays))
    return scenario_arrays


def summarize_ci(values: list[float], original: float) -> dict[str, Any]:
    series = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan)
    clean = series.dropna()
    valid = int(len(clean))
    invalid = int(len(series) - valid)
    if valid == 0:
        return {
            "bootstrap_mean_difference": float("nan"),
            "bootstrap_std_difference": float("nan"),
            "ci90_lower": float("nan"),
            "ci90_upper": float("nan"),
            "ci95_lower": float("nan"),
            "ci95_upper": float("nan"),
            "valid_reps": 0,
            "invalid_reps": invalid,
        }
    q = clean.quantile([0.025, 0.05, 0.95, 0.975])
    return {
        "bootstrap_mean_difference": float(clean.mean()),
        "bootstrap_std_difference": float(clean.std(ddof=1)) if valid > 1 else 0.0,
        "ci90_lower": float(q.loc[0.05]),
        "ci90_upper": float(q.loc[0.95]),
        "ci95_lower": float(q.loc[0.025]),
        "ci95_upper": float(q.loc[0.975]),
        "valid_reps": valid,
        "invalid_reps": invalid,
    }


def scenario_metric(arrays: dict[str, Any], metric: str, idx: np.ndarray | None = None) -> float:
    if idx is None:
        return float(arrays["original"][metric])
    if metric in {"annualized_mean_arithmetic", "annualized_volatility", "cvar_95_weekly", "max_drawdown"}:
        return float(performance_metrics(pd.Series(arrays["portfolio_return"][idx]))[metric])
    if metric in {"btc_share_vol", "btc_share_cvar"}:
        return float(risk_shares_from_matrix(arrays["component_matrix"][idx, :])[metric])
    if metric == "average_btc_weight":
        return float(arrays["btc_weight"][idx].mean())
    raise ValueError(f"Unsupported metric: {metric}")


def metric_interpretation(metric: str, diff: float, ci_low: float, ci_high: float, comparison_id: str) -> str:
    ci_includes_zero = ci_low <= 0.0 <= ci_high
    if ci_includes_zero:
        return "CI includes zero; treat as statistically uncertain."
    if metric == "annualized_mean_arithmetic":
        return "Left scenario has higher mean return in the bootstrap interval." if diff > 0 else "Left scenario has lower mean return in the bootstrap interval."
    if metric in {"annualized_volatility", "btc_share_vol", "btc_share_cvar", "average_btc_weight"}:
        return "Left scenario is lower on this risk/exposure metric." if diff < 0 else "Left scenario is higher on this risk/exposure metric."
    if metric in {"cvar_95_weekly", "max_drawdown"}:
        return "Left scenario has a less negative tail/drawdown estimate." if diff > 0 else "Left scenario has a more negative tail/drawdown estimate."
    return "Context-dependent result."


def bootstrap_pairwise(inputs: dict[str, Any], scenario_arrays: dict[str, dict[str, Any]], draws: np.ndarray, dirs: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for comparison_id, (left_id, right_id) in core_pair_specs().items():
        left = scenario_arrays[left_id]
        right = scenario_arrays[right_id]
        family = str(left["metadata"]["portfolio_family"])
        require(family == str(right["metadata"]["portfolio_family"]), f"Portfolio family mismatch for {comparison_id}")
        for metric in CORE_METRICS:
            original_left = scenario_metric(left, metric)
            original_right = scenario_metric(right, metric)
            original_diff = original_left - original_right
            diffs = []
            for idx in draws:
                diffs.append(scenario_metric(left, metric, idx) - scenario_metric(right, metric, idx))
            ci = summarize_ci(diffs, original_diff)
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "left_scenario_id": left_id,
                    "right_scenario_id": right_id,
                    "portfolio_family": family,
                    "implementation_scope": "target_weight",
                    "metric": metric,
                    "original_left": original_left,
                    "original_right": original_right,
                    "original_difference_left_minus_right": original_diff,
                    **ci,
                    "interpretation": metric_interpretation(metric, original_diff, ci["ci95_lower"], ci["ci95_upper"], comparison_id),
                }
            )
    out = pd.DataFrame(rows).sort_values(["portfolio_family", "comparison_id", "metric"]).reset_index(drop=True)
    out.to_csv(dirs["results"] / "part14_pairwise_bootstrap_ci.csv", index=False)
    logging.info("Pairwise bootstrap CI completed with %d rows", len(out))
    return out


def bootstrap_cap_exceedance(
    scenario_arrays: dict[str, dict[str, Any]],
    draws: np.ndarray,
    dirs: dict[str, Path],
) -> pd.DataFrame:
    scenario_ids = sorted({sid for pair in core_pair_specs().values() for sid in pair})
    rows = []
    for scenario_id in scenario_ids:
        arrays = scenario_arrays[scenario_id]
        cap = arrays["metadata"].get("risk_budget_cap")
        if pd.isna(cap):
            continue
        cap = float(cap)
        vol_exceeded = []
        cvar_exceeded = []
        for idx in draws:
            risk = risk_shares_from_matrix(arrays["component_matrix"][idx, :])
            vol_exceeded.append(bool(risk["btc_share_vol"] > cap + FLOAT_TOL))
            cvar_exceeded.append(bool(risk["btc_share_cvar"] > cap + FLOAT_TOL))
        rows.append(
            {
                "scenario_id": scenario_id,
                "portfolio_family": arrays["metadata"]["portfolio_family"],
                "implementation_scope": "target_weight",
                "risk_budget_cap": cap,
                "bootstrap_reps": int(len(draws)),
                "vol_cap_exceedance_probability": float(np.mean(vol_exceeded)),
                "cvar_cap_exceedance_probability": float(np.mean(cvar_exceeded)),
                "any_cap_exceedance_probability": float(np.mean(np.logical_or(vol_exceeded, cvar_exceeded))),
                "notes": "Conditional on fixed Part 10 scenario weights and HMM labels.",
            }
        )
    out = pd.DataFrame(rows).sort_values(["portfolio_family", "scenario_id"]).reset_index(drop=True)
    out.to_csv(dirs["results"] / "part14_cap_exceedance_under_pairwise_bootstrap.csv", index=False)
    return out


def build_decision_matrix(pairwise_ci: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for _, row in pairwise_ci.iterrows():
        metric = row["metric"]
        ci_includes_zero = bool(row["ci95_lower"] <= 0.0 <= row["ci95_upper"])
        comparison_id = row["comparison_id"]
        diff = float(row["original_difference_left_minus_right"])
        supports_macro = False
        supports_risk_control = False
        if "conditional_cap_10pct_vs_matched_fixed_btc" in comparison_id:
            if metric == "annualized_mean_arithmetic" and diff > 0 and not ci_includes_zero:
                supports_macro = True
            if metric in {"annualized_volatility", "btc_share_vol", "btc_share_cvar"} and diff < 0 and not ci_includes_zero:
                supports_risk_control = True
        if "conditional_cap_10pct_vs_cap_only_10pct" in comparison_id:
            if metric == "annualized_mean_arithmetic" and diff > 0 and not ci_includes_zero:
                supports_macro = True
            if metric in {"annualized_volatility", "btc_share_vol", "btc_share_cvar"} and diff < 0 and not ci_includes_zero:
                supports_risk_control = True

        if ci_includes_zero:
            strength = "descriptive_only"
        elif metric == "annualized_mean_arithmetic":
            strength = "moderate"
        elif metric in {"btc_share_vol", "btc_share_cvar", "annualized_volatility"}:
            strength = "moderate"
        else:
            strength = "weak"

        rows.append(
            {
                "comparison_id": comparison_id,
                "portfolio_family": row["portfolio_family"],
                "metric_family": metric,
                "evidence_strength": strength,
                "ci_includes_zero": ci_includes_zero,
                "supports_macro_conditioning_incremental_value": supports_macro,
                "supports_risk_control_only": supports_risk_control,
                "main_text_use": bool(metric in {"annualized_mean_arithmetic", "annualized_volatility", "btc_share_vol", "btc_share_cvar"}),
                "recommended_sentence": recommended_sentence(row, ci_includes_zero),
            }
        )
    out = pd.DataFrame(rows).sort_values(["portfolio_family", "comparison_id", "metric_family"]).reset_index(drop=True)
    out.to_csv(dirs["results"] / "part14_inference_decision_matrix.csv", index=False)
    return out


def recommended_sentence(row: pd.Series, ci_includes_zero: bool) -> str:
    comparison = str(row["comparison_id"])
    metric = str(row["metric"])
    diff_pp = float(row["original_difference_left_minus_right"]) * 100
    low_pp = float(row["ci95_lower"]) * 100
    high_pp = float(row["ci95_upper"]) * 100
    if ci_includes_zero:
        return (
            f"For {metric}, {comparison} has an original difference of {diff_pp:.2f} percentage points, "
            f"but the 95% bootstrap interval [{low_pp:.2f}, {high_pp:.2f}] includes zero."
        )
    return (
        f"For {metric}, {comparison} has an original difference of {diff_pp:.2f} percentage points, "
        f"with a 95% bootstrap interval [{low_pp:.2f}, {high_pp:.2f}]."
    )


def make_figures(pairwise_ci: pd.DataFrame, dirs: dict[str, Path]) -> None:
    core = pairwise_ci[
        pairwise_ci["metric"].isin(["annualized_mean_arithmetic", "annualized_volatility", "cvar_95_weekly", "max_drawdown"])
        & pairwise_ci["comparison_id"].str.contains("conditional_cap_10pct_vs_(?:matched_fixed_btc|cap_only_10pct)", regex=True)
    ].copy()
    core["label"] = core["portfolio_family"] + " | " + core["comparison_id"].str.replace(r"^[^_]+__", "", regex=True) + " | " + core["metric"]
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(core))
    ax.errorbar(
        core["original_difference_left_minus_right"] * 100,
        y,
        xerr=[
            (core["original_difference_left_minus_right"] - core["ci95_lower"]) * 100,
            (core["ci95_upper"] - core["original_difference_left_minus_right"]) * 100,
        ],
        fmt="o",
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(core["label"], fontsize=7)
    ax.set_xlabel("Difference, left minus right (percentage points)")
    ax.set_title("Part 14 pairwise bootstrap CI: core performance metrics")
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "part14_pairwise_ci_core_metrics.png", dpi=170)
    plt.close(fig)

    risk = pairwise_ci[
        pairwise_ci["metric"].isin(["btc_share_vol", "btc_share_cvar"])
        & pairwise_ci["comparison_id"].str.contains("conditional_cap_10pct_vs_(?:matched_fixed_btc|cap_only_10pct)", regex=True)
    ].copy()
    risk["label"] = risk["portfolio_family"] + " | " + risk["comparison_id"].str.replace(r"^[^_]+__", "", regex=True) + " | " + risk["metric"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    y = np.arange(len(risk))
    ax.errorbar(
        risk["original_difference_left_minus_right"] * 100,
        y,
        xerr=[
            (risk["original_difference_left_minus_right"] - risk["ci95_lower"]) * 100,
            (risk["ci95_upper"] - risk["original_difference_left_minus_right"]) * 100,
        ],
        fmt="o",
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(risk["label"], fontsize=8)
    ax.set_xlabel("Difference, left minus right (percentage points)")
    ax.set_title("Part 14 pairwise bootstrap CI: BTC risk contribution")
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "part14_pairwise_ci_risk_contribution.png", dpi=170)
    plt.close(fig)


def write_audits(args: argparse.Namespace, inputs: dict[str, Any], dirs: dict[str, Path]) -> None:
    lineage_rows = []
    for name, path in inputs["paths"].items():
        lineage_rows.append(
            {
                "source_name": name,
                "path": str(path),
                "sha256": inputs["input_hashes"][name],
                "role": "frozen_input",
            }
        )
    pd.DataFrame(lineage_rows).to_csv(dirs["results"] / "data_lineage.csv", index=False)

    assumption_audit = {
        "status": "passed",
        "sample_frozen": True,
        "bootstrap_method": "circular_moving_block_bootstrap",
        "bootstrap_reps": args.bootstrap_reps,
        "block_length": args.block_length,
        "conditional_on_fixed_rule": True,
        "reestimates_pca_hmm": False,
        "reselects_allocation_rule": False,
        "implementation_scope": "target_weight",
        "interpretation_boundary": "Pairwise CIs quantify return-path uncertainty under fixed Part 10 scenarios only.",
    }
    write_json(dirs["results"] / "model_assumption_audit.json", normalize_for_json(assumption_audit))

    methodology = f"""# Part 14 Methodology Audit

Part 14 applies circular moving-block bootstrap inference to the Part 10 benchmark comparisons.
The sample is the frozen state-aligned target-weight sample from {EXPECTED_STATE_START} to {EXPECTED_STATE_END}
with {EXPECTED_STATE_ROWS} weekly observations.

Bootstrap design:
- Replications: {args.bootstrap_reps}
- Block length: {args.block_length} weeks
- Same draw index is used for both scenarios in each pair, preserving common shocks.
- The procedure is conditional on fixed Part 10 scenario weights and fixed Part 1 HMM-4 labels.
- It does not re-estimate PCA/HMM models and does not re-select allocation rules.

Pairs:
- conditional_cap_10pct vs matched_fixed_btc
- conditional_cap_10pct vs cap_only_10pct
- conditional_cap_10pct vs no_btc_baseline
- cap_only_10pct vs matched_fixed_btc

Interpretation boundary:
These confidence intervals support statistical caution around target-weight comparisons. They are not full
model-selection uncertainty intervals and should not be described as nested bootstrap evidence.
"""
    (dirs["results"] / "methodology_audit.md").write_text(methodology, encoding="utf-8")


def validate_outputs(
    pairwise_ci: pd.DataFrame,
    cap_exceedance: pd.DataFrame,
    decision: pd.DataFrame,
    dirs: dict[str, Path],
) -> dict[str, Any]:
    required_paths = [dirs["results"] / name for name in REQUIRED_RESULTS if name != "output_validation_summary.json"] + [
        dirs["figures"] / name for name in REQUIRED_FIGURES
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    require(not missing, f"Missing required output files: {missing}")
    require(len(pairwise_ci) == len(core_pair_specs()) * len(CORE_METRICS), "Unexpected pairwise CI row count")
    require(not pairwise_ci.duplicated(["comparison_id", "metric", "implementation_scope"]).any(), "Duplicate pairwise CI rows")
    require((pairwise_ci["valid_reps"] > 0).all(), "Pairwise CI has zero valid reps")
    require(set(pairwise_ci["implementation_scope"]) == {"target_weight"}, "Unexpected implementation scope")
    require(set(decision["comparison_id"]) == set(pairwise_ci["comparison_id"]), "Decision matrix comparisons mismatch")
    require(not cap_exceedance.empty, "Cap exceedance output is empty")

    summary = {
        "status": "passed",
        "pairwise_rows": int(len(pairwise_ci)),
        "cap_exceedance_rows": int(len(cap_exceedance)),
        "decision_rows": int(len(decision)),
        "bootstrap_reps_min": int(pairwise_ci["valid_reps"].min()),
        "bootstrap_reps_max": int(pairwise_ci["valid_reps"].max()),
        "implementation_scope": "target_weight",
        "required_files_present": True,
    }
    write_json(dirs["results"] / "output_validation_summary.json", normalize_for_json(summary))
    logging.info("Output validation passed")
    return summary


def write_manifest(args: argparse.Namespace, inputs: dict[str, Any], output_validation: dict[str, Any], dirs: dict[str, Path]) -> None:
    manifest = {
        "part_id": "part14_pairwise_inference",
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_frozen": True,
        "sample_end": EXPECTED_STATE_END,
        "seed": args.seed,
        "inputs": {name: str(path) for name, path in inputs["paths"].items()},
        "input_hashes": inputs["input_hashes"],
        "parameters": {
            "bootstrap_reps": args.bootstrap_reps,
            "block_length": args.block_length,
            "core_metrics": CORE_METRICS,
            "pair_specs": core_pair_specs(),
            "implementation_scope": "target_weight",
        },
        "outputs": {
            "results": REQUIRED_RESULTS,
            "figures": REQUIRED_FIGURES,
            "output_validation": output_validation,
        },
        "package_versions": package_versions(),
        "status": output_validation["status"],
    }
    write_json(dirs["root"] / "run_manifest.json", normalize_for_json(manifest))


def main() -> None:
    args = parse_args()
    run_dir = args.output_dir / args.run_id
    dirs = ensure_dirs(run_dir)
    setup_logging(dirs["logs"])
    np.random.seed(args.seed)

    logging.info("Starting Part 14 run in %s", run_dir)
    inputs = read_inputs(args)
    enforce_resume_input_hashes(args, dirs, inputs["input_hashes"])
    validation = load_or_run(dirs, "01_input_validation", args.resume, lambda: validate_inputs(inputs, args, dirs))
    scenario_arrays = load_or_run(dirs, "02_scenario_arrays", args.resume, lambda: build_scenario_arrays(inputs, dirs))
    draws = load_or_run(dirs, "03_bootstrap_draws", args.resume, lambda: build_bootstrap_draws(args, dirs))
    pairwise_ci = load_or_run(dirs, "04_pairwise_bootstrap", args.resume, lambda: bootstrap_pairwise(inputs, scenario_arrays, draws, dirs))
    cap_exceedance = load_or_run(dirs, "05_cap_exceedance", args.resume, lambda: bootstrap_cap_exceedance(scenario_arrays, draws, dirs))
    decision = build_decision_matrix(pairwise_ci, dirs)
    make_figures(pairwise_ci, dirs)
    write_audits(args, inputs, dirs)
    output_validation = validate_outputs(pairwise_ci, cap_exceedance, decision, dirs)
    write_manifest(args, inputs, output_validation, dirs)
    logging.info("Part 14 completed successfully")


if __name__ == "__main__":
    main()
