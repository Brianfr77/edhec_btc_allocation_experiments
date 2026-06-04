#!/usr/bin/env python3
"""Part 3 experiment runner: BTC state dependence and conditional beta diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import pickle
import platform
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


EXPECTED_ASSET_START = "2018-01-12"
EXPECTED_ASSET_END = "2026-03-27"
EXPECTED_ASSET_ROWS = 429
EXPECTED_STATE_START = "2018-02-09"
EXPECTED_STATE_END = "2026-03-27"
EXPECTED_STATE_ROWS = 425
EXPECTED_STATE_COUNTS = {"state_0": 149, "state_1": 167, "state_2": 30, "state_3": 79}
TRADING_WEEKS_PER_YEAR = 52
TAIL_ALPHA = 0.05
SMALL_STATE_THRESHOLD_WEEKS = 52
HAC_LAG_WEEKS = 4
CONSISTENCY_TOLERANCE = 1e-10

ASSET_RETURN_COLS = ["ret_btc", "ret_spy", "ret_tlt", "ret_ief", "ret_gld", "ret_dbc", "ret_bil"]
CORRELATION_ASSETS = ["ret_spy", "ret_tlt", "ret_ief", "ret_gld", "ret_dbc", "ret_bil"]
BETA_PREDICTORS = [
    {
        "predictor": "ret_spy",
        "label": "SPY weekly return",
        "source_column": "ret_spy",
        "predictor_type": "asset_return",
        "timing": "same-week asset return",
        "unit": "weekly simple return",
        "interpretation": "BTC weekly return sensitivity per 1 unit SPY weekly return.",
    },
    {
        "predictor": "macro_vix_z",
        "label": "Lagged VIX z-score",
        "source_column": "macro_vix_z",
        "predictor_type": "lagged_macro_zscore",
        "timing": "one-week lagged macro predictor",
        "unit": "full-sample z-score",
        "interpretation": "BTC weekly return sensitivity per 1 standard deviation lagged VIX predictor.",
    },
    {
        "predictor": "macro_real_yield_10y_z",
        "label": "Lagged 10Y real yield z-score",
        "source_column": "macro_real_yield_10y_z",
        "predictor_type": "lagged_macro_zscore",
        "timing": "one-week lagged macro predictor",
        "unit": "full-sample z-score",
        "interpretation": "BTC weekly return sensitivity per 1 standard deviation lagged 10Y real-yield predictor.",
    },
    {
        "predictor": "macro_dollar_chg_4w_z",
        "label": "Lagged dollar 4-week change z-score",
        "source_column": "macro_dollar_chg_4w_z",
        "predictor_type": "lagged_macro_zscore",
        "timing": "one-week lagged macro predictor",
        "unit": "full-sample z-score",
        "interpretation": "BTC weekly return sensitivity per 1 standard deviation lagged dollar-change predictor.",
    },
    {
        "predictor": "macro_credit_spread_baa10y_z",
        "label": "Lagged BAA10Y credit spread z-score",
        "source_column": "macro_credit_spread_baa10y_z",
        "predictor_type": "lagged_macro_zscore",
        "timing": "one-week lagged macro predictor",
        "unit": "full-sample z-score",
        "interpretation": "BTC weekly return sensitivity per 1 standard deviation lagged credit-spread predictor.",
    },
]

REQUIRED_RESULTS = [
    "input_validation_summary.json",
    "state_conditioned_btc_performance.csv",
    "state_conditioned_asset_correlations.csv",
    "full_sample_beta_diagnostics.csv",
    "state_conditioned_beta_diagnostics.csv",
    "state_beta_contrast_summary.csv",
    "part1_consistency_checks.csv",
    "part2_context_summary.csv",
    "state_sample_warnings.csv",
    "beta_predictor_dictionary.csv",
    "beta_methodology.json",
    "methodology_audit.md",
    "data_lineage.csv",
    "model_assumption_audit.json",
    "output_validation_summary.json",
]
REQUIRED_FIGURES = [
    "btc_state_performance_summary.png",
    "state_conditioned_correlation_heatmap.png",
    "state_conditioned_beta_heatmap.png",
    "conditional_beta_confidence_intervals.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BTC state dependence and conditional beta diagnostics."
    )
    parser.add_argument("--input-dir", default="data_2026/cleaned", type=Path)
    parser.add_argument(
        "--part1-run-dir",
        default="outputs/part1_btc_macro_state/colab_part1_seed42",
        type=Path,
    )
    parser.add_argument(
        "--part2-run-dir",
        default="outputs/part2_portfolio_risk_budget/colab_part2_seed42",
        type=Path,
    )
    parser.add_argument("--output-dir", default="outputs/part3_btc_state_dependence", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
    for package in ["numpy", "pandas", "matplotlib", "scipy"]:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def date_string(series: pd.Series, fn: str) -> str:
    value = series.min() if fn == "min" else series.max()
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def warning_for_state(n_weeks: int) -> str:
    return "small_state_sample" if n_weeks < SMALL_STATE_THRESHOLD_WEEKS else ""


def drawdown_from_returns(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    return float(drawdown.min())


def var_cvar(returns: pd.Series, alpha: float = TAIL_ALPHA) -> tuple[float, float, int]:
    clean = returns.dropna()
    var = float(clean.quantile(alpha))
    tail = clean[clean <= var]
    cvar = float(tail.mean())
    return var, cvar, int(len(tail))


def contiguous_state_episode_summary(panel: pd.DataFrame, state_name: str) -> dict[str, Any]:
    episodes: list[pd.DataFrame] = []
    current_rows: list[pd.Series] = []
    for _, row in panel.sort_values("date").iterrows():
        if row["hmm4_state"] == state_name:
            current_rows.append(row)
        elif current_rows:
            episodes.append(pd.DataFrame(current_rows))
            current_rows = []
    if current_rows:
        episodes.append(pd.DataFrame(current_rows))

    if not episodes:
        return {
            "btc_n_contiguous_state_episodes": 0,
            "btc_average_episode_weeks": float("nan"),
            "btc_worst_contiguous_episode_drawdown": float("nan"),
            "btc_worst_contiguous_episode_start": "",
            "btc_worst_contiguous_episode_end": "",
            "btc_worst_contiguous_episode_weeks": 0,
        }

    episode_rows = []
    for episode in episodes:
        episode_rows.append(
            {
                "drawdown": drawdown_from_returns(episode["ret_btc"]),
                "start": pd.Timestamp(episode["date"].min()).strftime("%Y-%m-%d"),
                "end": pd.Timestamp(episode["date"].max()).strftime("%Y-%m-%d"),
                "weeks": int(len(episode)),
            }
        )
    worst = min(episode_rows, key=lambda item: item["drawdown"])
    return {
        "btc_n_contiguous_state_episodes": int(len(episode_rows)),
        "btc_average_episode_weeks": float(np.mean([item["weeks"] for item in episode_rows])),
        "btc_worst_contiguous_episode_drawdown": float(worst["drawdown"]),
        "btc_worst_contiguous_episode_start": worst["start"],
        "btc_worst_contiguous_episode_end": worst["end"],
        "btc_worst_contiguous_episode_weeks": int(worst["weeks"]),
    }


def read_inputs(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "asset_returns_main_weekly": args.input_dir / "asset_returns_main_weekly.csv",
        "state_model_panel_weekly": args.input_dir / "state_model_panel_weekly.csv",
        "cleaning_report": args.input_dir / "cleaning_report.json",
        "part1_manifest": args.part1_run_dir / "run_manifest.json",
        "part1_validation_summary": args.part1_run_dir / "results" / "validation_summary.json",
        "hmm4_state_labels": args.part1_run_dir / "results" / "hmm4_state_labels.csv",
        "hmm4_state_profiles": args.part1_run_dir / "results" / "hmm4_state_profiles.csv",
        "part1_state_conditioned_btc_summary": args.part1_run_dir
        / "results"
        / "state_conditioned_btc_summary.csv",
        "part1_state_conditioned_correlations": args.part1_run_dir
        / "results"
        / "state_conditioned_correlations.csv",
        "part2_manifest": args.part2_run_dir / "run_manifest.json",
        "part2_input_validation_summary": args.part2_run_dir
        / "results"
        / "input_validation_summary.json",
        "part2_output_validation_summary": args.part2_run_dir
        / "results"
        / "output_validation_summary.json",
        "part2_btc_risk_budget_summary": args.part2_run_dir
        / "results"
        / "btc_risk_budget_summary.csv",
    }
    for name, path in paths.items():
        require(path.exists(), f"Missing required input file: {name} at {path}")

    return {
        "paths": paths,
        "asset": pd.read_csv(paths["asset_returns_main_weekly"], parse_dates=["date"]),
        "state_panel": pd.read_csv(paths["state_model_panel_weekly"], parse_dates=["date"]),
        "cleaning_report": json.loads(paths["cleaning_report"].read_text(encoding="utf-8")),
        "part1_manifest": json.loads(paths["part1_manifest"].read_text(encoding="utf-8")),
        "part1_validation": json.loads(paths["part1_validation_summary"].read_text(encoding="utf-8")),
        "labels": pd.read_csv(paths["hmm4_state_labels"], parse_dates=["date"]),
        "profiles": pd.read_csv(paths["hmm4_state_profiles"]),
        "part1_btc_summary": pd.read_csv(paths["part1_state_conditioned_btc_summary"]),
        "part1_correlations": pd.read_csv(paths["part1_state_conditioned_correlations"]),
        "part2_manifest": json.loads(paths["part2_manifest"].read_text(encoding="utf-8")),
        "part2_input_validation": json.loads(
            paths["part2_input_validation_summary"].read_text(encoding="utf-8")
        ),
        "part2_output_validation": json.loads(
            paths["part2_output_validation_summary"].read_text(encoding="utf-8")
        ),
        "part2_btc_risk_budget": pd.read_csv(paths["part2_btc_risk_budget_summary"]),
        "input_hashes": {name: file_sha256(path) for name, path in paths.items()},
    }


def enforce_resume_input_hashes(args: argparse.Namespace, dirs: dict[str, Path], input_hashes: dict[str, str]) -> None:
    if not args.resume:
        return
    manifest_path = dirs["root"] / "run_manifest.json"
    if not manifest_path.exists():
        return
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(old_manifest.get("input_hashes", {}) == input_hashes, "Input hashes changed since the previous run manifest")
    logging.info("Resume input hash check passed")


def validate_inputs(inputs: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    asset = inputs["asset"].copy()
    state = inputs["state_panel"].copy()
    labels = inputs["labels"].copy()
    profiles = inputs["profiles"].copy()
    report = inputs["cleaning_report"]
    part1_manifest = inputs["part1_manifest"]
    part1_validation = inputs["part1_validation"]
    part2_manifest = inputs["part2_manifest"]
    part2_input_validation = inputs["part2_input_validation"]
    part2_output_validation = inputs["part2_output_validation"]
    input_hashes = inputs["input_hashes"]

    raw_predictor_cols = list(report["column_mapping"]["state_predictors"].values())
    z_predictor_cols = [f"{col}_z" for col in raw_predictor_cols]
    beta_predictor_cols = [entry["predictor"] for entry in BETA_PREDICTORS]

    require(len(asset) == EXPECTED_ASSET_ROWS, f"Unexpected asset row count: {len(asset)}")
    require(date_string(asset["date"], "min") == EXPECTED_ASSET_START, "Unexpected asset start date")
    require(date_string(asset["date"], "max") == EXPECTED_ASSET_END, "Unexpected asset end date")
    require(asset["date"].dt.dayofweek.eq(4).all(), "Asset dates are not all Fridays")
    require(all(col in asset.columns for col in ASSET_RETURN_COLS), "Missing core asset return columns")
    require(asset[ASSET_RETURN_COLS].isna().sum().sum() == 0, "Missing values in main asset returns")

    require(len(state) == EXPECTED_STATE_ROWS, f"Unexpected state panel row count: {len(state)}")
    require(date_string(state["date"], "min") == EXPECTED_STATE_START, "Unexpected state panel start date")
    require(date_string(state["date"], "max") == EXPECTED_STATE_END, "Unexpected state panel end date")
    require(state["date"].dt.dayofweek.eq(4).all(), "State panel dates are not all Fridays")
    require(all(col in state.columns for col in ASSET_RETURN_COLS), "Missing asset returns in state panel")
    require(all(col in state.columns for col in raw_predictor_cols), "Missing raw macro predictor columns")
    require(all(col in state.columns for col in z_predictor_cols), "Missing z-score macro predictor columns")
    require(all(col in state.columns for col in beta_predictor_cols), "Missing conditional beta predictor columns")
    require(state[ASSET_RETURN_COLS + beta_predictor_cols].isna().sum().sum() == 0, "Missing values in Part 3 core columns")

    require(len(labels) == EXPECTED_STATE_ROWS, f"Unexpected HMM-4 label rows: {len(labels)}")
    require(date_string(labels["date"], "min") == EXPECTED_STATE_START, "Unexpected HMM-4 label start date")
    require(date_string(labels["date"], "max") == EXPECTED_STATE_END, "Unexpected HMM-4 label end date")
    require(labels["date"].dt.dayofweek.eq(4).all(), "HMM-4 label dates are not all Fridays")
    required_label_cols = ["date", "hmm4_state", "hmm4_state_id", "hmm4_state_posterior_probability"]
    require(all(col in labels.columns for col in required_label_cols), "Missing HMM-4 label columns")
    state_counts = labels["hmm4_state"].value_counts().sort_index().to_dict()
    require(state_counts == EXPECTED_STATE_COUNTS, f"Unexpected HMM-4 state counts: {state_counts}")
    require(len(profiles) == 4, "HMM-4 profiles must contain four states")

    require(part1_validation.get("status") == "passed", "Part 1 validation summary is not passed")
    require(part1_manifest.get("model_diagnostics", {}).get("hmm4_converged") is True, "Part 1 HMM-4 did not converge")
    require(part2_input_validation.get("status") == "passed", "Part 2 input validation is not passed")
    require(part2_output_validation.get("status") == "passed", "Part 2 output validation is not passed")
    require(part2_manifest.get("sample", {}).get("rows") == EXPECTED_STATE_ROWS, "Part 2 sample row count is not 425")

    require(
        input_hashes["asset_returns_main_weekly"]
        == part1_manifest["input_hashes"]["asset_returns_main_weekly"],
        "Asset hash does not match Part 1 manifest",
    )
    require(
        input_hashes["state_model_panel_weekly"]
        == part1_manifest["input_hashes"]["state_model_panel_weekly"],
        "State panel hash does not match Part 1 manifest",
    )
    require(
        input_hashes["cleaning_report"] == part1_manifest["input_hashes"]["cleaning_report"],
        "Cleaning report hash does not match Part 1 manifest",
    )
    require(
        input_hashes["asset_returns_main_weekly"]
        == part2_manifest["input_hashes"]["asset_returns_main_weekly"],
        "Asset hash does not match Part 2 manifest",
    )
    require(
        input_hashes["cleaning_report"] == part2_manifest["input_hashes"]["cleaning_report"],
        "Cleaning report hash does not match Part 2 manifest",
    )
    require(
        input_hashes["part1_manifest"] == part2_manifest["input_hashes"]["part1_manifest"],
        "Part 1 manifest hash does not match Part 2 lineage",
    )
    require(
        input_hashes["hmm4_state_labels"] == part2_manifest["input_hashes"]["hmm4_state_labels"],
        "HMM-4 label hash does not match Part 2 lineage",
    )
    require(
        input_hashes["hmm4_state_profiles"] == part2_manifest["input_hashes"]["hmm4_state_profiles"],
        "HMM-4 profile hash does not match Part 2 lineage",
    )
    require(
        input_hashes["part1_validation_summary"] == part2_manifest["input_hashes"]["part1_validation_summary"],
        "Part 1 validation hash does not match Part 2 lineage",
    )

    merged = state.merge(labels[required_label_cols], on="date", how="inner", validate="one_to_one")
    require(len(merged) == EXPECTED_STATE_ROWS, f"State panel and HMM labels inner join produced {len(merged)} rows")
    asset_state = asset[["date"] + ASSET_RETURN_COLS].merge(
        state[["date"] + ASSET_RETURN_COLS], on="date", how="inner", suffixes=("_asset", "_state")
    )
    require(len(asset_state) == EXPECTED_STATE_ROWS, "Asset/state panel return alignment failed")
    max_return_diff = 0.0
    for col in ASSET_RETURN_COLS:
        diff = (asset_state[f"{col}_asset"] - asset_state[f"{col}_state"]).abs().max()
        max_return_diff = max(max_return_diff, float(diff))
    require(max_return_diff < 1e-14, f"Asset returns differ between main and state panels: {max_return_diff}")

    state_sample_warnings = []
    for state_name, n_weeks in state_counts.items():
        var, cvar, tail_count = var_cvar(merged.loc[merged["hmm4_state"] == state_name, "ret_btc"])
        state_sample_warnings.append(
            {
                "state": state_name,
                "state_id": int(merged.loc[merged["hmm4_state"] == state_name, "hmm4_state_id"].iloc[0]),
                "n_weeks": int(n_weeks),
                "sample_share": float(n_weeks / len(merged)),
                "btc_var_95_weekly": var,
                "btc_cvar_95_weekly": cvar,
                "cvar_tail_scenario_count": int(tail_count),
                "state_sample_warning": warning_for_state(int(n_weeks)),
                "inference_warning": (
                    "HAC confidence intervals are reported, but state-level inference is unstable below 52 weeks."
                    if n_weeks < SMALL_STATE_THRESHOLD_WEEKS
                    else ""
                ),
            }
        )
    state_warnings = pd.DataFrame(state_sample_warnings)
    state_warnings.to_csv(dirs["results"] / "state_sample_warnings.csv", index=False)

    validation = {
        "status": "passed",
        "asset_sample": {
            "rows": int(len(asset)),
            "start": date_string(asset["date"], "min"),
            "end": date_string(asset["date"], "max"),
        },
        "state_sample": {
            "rows": int(len(state)),
            "start": date_string(state["date"], "min"),
            "end": date_string(state["date"], "max"),
        },
        "part3_sample": {
            "rows": int(len(merged)),
            "start": date_string(merged["date"], "min"),
            "end": date_string(merged["date"], "max"),
        },
        "hmm4_state_counts": {k: int(v) for k, v in state_counts.items()},
        "hmm4_min_state_weeks": int(min(state_counts.values())),
        "asset_state_max_abs_return_diff": max_return_diff,
        "core_asset_columns": ASSET_RETURN_COLS,
        "beta_predictors": beta_predictor_cols,
        "raw_macro_predictor_columns": raw_predictor_cols,
        "zscore_macro_predictor_columns": z_predictor_cols,
        "input_hashes": input_hashes,
        "part1_model_diagnostics": part1_manifest.get("model_diagnostics", {}),
        "part2_output_status": part2_output_validation.get("status"),
    }
    write_json(dirs["results"] / "input_validation_summary.json", validation)
    logging.info("Input validation passed")
    return {"validation": validation, "analysis_panel": merged, "state_sample_warnings": state_warnings}


def compute_state_diagnostics(inputs: dict[str, Any], validation_payload: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    panel = validation_payload["analysis_panel"].copy()
    part1_btc = inputs["part1_btc_summary"].copy()
    part1_corr = inputs["part1_correlations"].copy()

    rows = []
    for state_name, frame in panel.groupby("hmm4_state", sort=True):
        returns = frame["ret_btc"]
        var, cvar, tail_count = var_cvar(returns)
        episode_summary = contiguous_state_episode_summary(panel, state_name)
        rows.append(
            {
                "state": state_name,
                "state_id": int(frame["hmm4_state_id"].iloc[0]),
                "n_weeks": int(len(frame)),
                "sample_share": float(len(frame) / len(panel)),
                "state_sample_warning": warning_for_state(len(frame)),
                "btc_mean_weekly": float(returns.mean()),
                "btc_median_weekly": float(returns.median()),
                "btc_volatility_weekly": float(returns.std(ddof=1)),
                "btc_annualized_mean_arithmetic": float(returns.mean() * TRADING_WEEKS_PER_YEAR),
                "btc_annualized_volatility": float(returns.std(ddof=1) * math.sqrt(TRADING_WEEKS_PER_YEAR)),
                "btc_min_weekly": float(returns.min()),
                "btc_max_weekly": float(returns.max()),
                "btc_var_95_weekly": var,
                "btc_cvar_95_weekly": cvar,
                "btc_cvar_tail_scenario_count": int(tail_count),
                "btc_max_drawdown": drawdown_from_returns(returns),
                "btc_max_drawdown_interpretation": "Noncontiguous state-ordered drawdown; see contiguous episode columns for calendar-adjacent state runs.",
                **episode_summary,
                "btc_positive_week_share": float((returns > 0).mean()),
            }
        )
    state_perf = pd.DataFrame(rows).sort_values("state_id").reset_index(drop=True)
    state_perf.to_csv(dirs["results"] / "state_conditioned_btc_performance.csv", index=False)

    full_corrs = {
        asset: float(panel["ret_btc"].corr(panel[asset]))
        for asset in CORRELATION_ASSETS
    }
    corr_rows = []
    for state_name, frame in panel.groupby("hmm4_state", sort=True):
        for asset in CORRELATION_ASSETS:
            state_corr = float(frame["ret_btc"].corr(frame[asset]))
            corr_rows.append(
                {
                    "state": state_name,
                    "state_id": int(frame["hmm4_state_id"].iloc[0]),
                    "asset": asset,
                    "n_weeks": int(len(frame)),
                    "state_sample_warning": warning_for_state(len(frame)),
                    "correlation_with_btc": state_corr,
                    "full_sample_correlation_with_btc": full_corrs[asset],
                    "state_minus_full_sample_correlation": state_corr - full_corrs[asset],
                }
            )
    correlations = pd.DataFrame(corr_rows).sort_values(["state_id", "asset"]).reset_index(drop=True)
    correlations.to_csv(dirs["results"] / "state_conditioned_asset_correlations.csv", index=False)

    consistency_rows = []
    perf_common = [
        "n_weeks",
        "sample_share",
        "btc_mean_weekly",
        "btc_median_weekly",
        "btc_volatility_weekly",
        "btc_annualized_mean_arithmetic",
        "btc_annualized_volatility",
        "btc_min_weekly",
        "btc_max_weekly",
        "btc_var_95_weekly",
        "btc_cvar_95_weekly",
        "btc_positive_week_share",
    ]
    left_perf = state_perf.set_index(["state", "state_id"])
    right_perf = part1_btc.set_index(["state", "state_id"])
    for col in perf_common:
        diffs = (left_perf[col] - right_perf[col]).abs()
        consistency_rows.append(
            {
                "check_name": f"part1_btc_summary_{col}",
                "source_file": "state_conditioned_btc_summary.csv",
                "max_abs_difference": float(diffs.max()),
                "tolerance": CONSISTENCY_TOLERANCE,
                "status": "passed" if float(diffs.max()) <= CONSISTENCY_TOLERANCE else "failed",
            }
        )

    left_corr = correlations.set_index(["state", "state_id", "asset"])["correlation_with_btc"]
    right_corr = part1_corr.set_index(["state", "state_id", "asset"])["correlation_with_btc"]
    corr_diffs = (left_corr - right_corr).abs()
    consistency_rows.append(
        {
            "check_name": "part1_state_conditioned_correlations",
            "source_file": "state_conditioned_correlations.csv",
            "max_abs_difference": float(corr_diffs.max()),
            "tolerance": CONSISTENCY_TOLERANCE,
            "status": "passed" if float(corr_diffs.max()) <= CONSISTENCY_TOLERANCE else "failed",
        }
    )

    consistency = pd.DataFrame(consistency_rows)
    consistency.to_csv(dirs["results"] / "part1_consistency_checks.csv", index=False)
    require((consistency["status"] == "passed").all(), "Part 1 consistency checks failed")

    plot_btc_state_performance(state_perf, dirs["figures"] / "btc_state_performance_summary.png")
    plot_correlation_heatmap(correlations, dirs["figures"] / "state_conditioned_correlation_heatmap.png")
    logging.info("State dependence diagnostics completed")
    return {
        "state_performance": state_perf,
        "correlations": correlations,
        "part1_consistency_checks": consistency,
    }


def ols_hac_single_predictor(y: pd.Series, x: pd.Series, predictor_name: str, hac_lag: int = HAC_LAG_WEEKS) -> dict[str, Any]:
    frame = pd.DataFrame({"y": y, "x": x}).dropna()
    n_obs = int(len(frame))
    require(n_obs >= 8, f"Too few observations for beta regression on {predictor_name}: {n_obs}")

    y_arr = frame["y"].to_numpy(dtype=float)
    x_arr = frame["x"].to_numpy(dtype=float)
    require(float(np.std(x_arr, ddof=1)) > 0.0, f"Predictor has zero variance: {predictor_name}")

    x_matrix = np.column_stack([np.ones(n_obs), x_arr])
    xtx = x_matrix.T @ x_matrix
    xtx_inv = np.linalg.pinv(xtx)
    coef = xtx_inv @ x_matrix.T @ y_arr
    residuals = y_arr - x_matrix @ coef
    df_resid = n_obs - x_matrix.shape[1]
    require(df_resid > 0, f"Regression has non-positive residual degrees of freedom: {predictor_name}")

    effective_lag = min(hac_lag, n_obs - 1)
    meat = np.zeros((2, 2), dtype=float)
    for t in range(n_obs):
        xt = x_matrix[t : t + 1].T
        meat += residuals[t] ** 2 * (xt @ xt.T)
    for lag in range(1, effective_lag + 1):
        weight = 1.0 - lag / (effective_lag + 1.0)
        gamma = np.zeros((2, 2), dtype=float)
        for t in range(lag, n_obs):
            xt = x_matrix[t : t + 1].T
            xlag = x_matrix[t - lag : t - lag + 1].T
            gamma += residuals[t] * residuals[t - lag] * (xt @ xlag.T)
        meat += weight * (gamma + gamma.T)
    meat *= n_obs / df_resid
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    require(np.isfinite(se).all(), f"Non-finite HAC standard error for {predictor_name}")

    t_values = coef / se
    p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(t_values), df=df_resid))
    ci_mult = float(stats.t.ppf(0.975, df=df_resid))
    ci_lower = coef - ci_mult * se
    ci_upper = coef + ci_mult * se

    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r_squared = float(1.0 - sse / sst) if sst > 0 else float("nan")
    residual_volatility = float(np.sqrt(sse / df_resid))
    condition_number = float(np.linalg.cond(xtx))

    result = {
        "predictor": predictor_name,
        "n_obs": n_obs,
        "alpha": float(coef[0]),
        "beta": float(coef[1]),
        "hac_se_alpha": float(se[0]),
        "hac_se_beta": float(se[1]),
        "t_alpha": float(t_values[0]),
        "t_beta": float(t_values[1]),
        "p_alpha": float(p_values[0]),
        "p_beta": float(p_values[1]),
        "alpha_ci95_lower": float(ci_lower[0]),
        "alpha_ci95_upper": float(ci_upper[0]),
        "beta_ci95_lower": float(ci_lower[1]),
        "beta_ci95_upper": float(ci_upper[1]),
        "r_squared": r_squared,
        "df_resid": int(df_resid),
        "hac_lag_weeks": int(hac_lag),
        "effective_hac_lag_weeks": int(effective_lag),
        "predictor_mean": float(np.mean(x_arr)),
        "predictor_std": float(np.std(x_arr, ddof=1)),
        "btc_mean_weekly": float(np.mean(y_arr)),
        "residual_volatility_weekly": residual_volatility,
        "condition_number_xtx": condition_number,
    }
    require(all(np.isfinite(result[key]) for key in ["alpha", "beta", "hac_se_alpha", "hac_se_beta", "t_alpha", "t_beta", "p_alpha", "p_beta", "beta_ci95_lower", "beta_ci95_upper"]), f"Non-finite beta output for {predictor_name}")
    return result


def predictor_dictionary() -> pd.DataFrame:
    rows = []
    for entry in BETA_PREDICTORS:
        row = dict(entry)
        row["dependent_variable"] = "ret_btc"
        row["model_form"] = "ret_btc_t = alpha + beta * predictor_t + error_t"
        rows.append(row)
    return pd.DataFrame(rows)


def compute_beta_diagnostics(validation_payload: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    panel = validation_payload["analysis_panel"].copy()
    pred_dict = predictor_dictionary()
    pred_meta = pred_dict.set_index("predictor").to_dict(orient="index")

    full_rows = []
    for predictor in pred_dict["predictor"]:
        result = ols_hac_single_predictor(panel["ret_btc"], panel[predictor], predictor)
        result.update(
            {
                "scope": "full_sample",
                "state": "full_sample",
                "state_id": -1,
                "state_sample_warning": "",
                "predictor_label": pred_meta[predictor]["label"],
                "predictor_type": pred_meta[predictor]["predictor_type"],
                "predictor_unit": pred_meta[predictor]["unit"],
                "beta_interpretation": pred_meta[predictor]["interpretation"],
            }
        )
        full_rows.append(result)
    full_beta = pd.DataFrame(full_rows)

    state_rows = []
    for state_name, frame in panel.groupby("hmm4_state", sort=True):
        state_id = int(frame["hmm4_state_id"].iloc[0])
        sample_warning = warning_for_state(len(frame))
        for predictor in pred_dict["predictor"]:
            result = ols_hac_single_predictor(frame["ret_btc"], frame[predictor], predictor)
            result.update(
                {
                    "scope": "state_conditioned",
                    "state": state_name,
                    "state_id": state_id,
                    "state_sample_warning": sample_warning,
                    "predictor_label": pred_meta[predictor]["label"],
                    "predictor_type": pred_meta[predictor]["predictor_type"],
                    "predictor_unit": pred_meta[predictor]["unit"],
                    "beta_interpretation": pred_meta[predictor]["interpretation"],
                }
            )
            state_rows.append(result)
    state_beta = pd.DataFrame(state_rows).sort_values(["state_id", "predictor"]).reset_index(drop=True)

    contrast_rows = []
    full_lookup = full_beta.set_index("predictor")
    for _, row in state_beta.iterrows():
        full = full_lookup.loc[row["predictor"]]
        contrast_rows.append(
            {
                "predictor": row["predictor"],
                "predictor_label": row["predictor_label"],
                "state": row["state"],
                "state_id": int(row["state_id"]),
                "state_n_obs": int(row["n_obs"]),
                "state_sample_warning": row["state_sample_warning"],
                "full_sample_beta": float(full["beta"]),
                "state_beta": float(row["beta"]),
                "beta_difference_vs_full_sample": float(row["beta"] - full["beta"]),
                "full_sample_beta_ci95_lower": float(full["beta_ci95_lower"]),
                "full_sample_beta_ci95_upper": float(full["beta_ci95_upper"]),
                "state_beta_ci95_lower": float(row["beta_ci95_lower"]),
                "state_beta_ci95_upper": float(row["beta_ci95_upper"]),
                "full_sample_p_beta": float(full["p_beta"]),
                "state_p_beta": float(row["p_beta"]),
                "contrast_interpretation_note": "Diagnostic beta contrast only; no allocation rule is inferred.",
            }
        )
    contrast = pd.DataFrame(contrast_rows).sort_values(["state_id", "predictor"]).reset_index(drop=True)

    full_beta.to_csv(dirs["results"] / "full_sample_beta_diagnostics.csv", index=False)
    state_beta.to_csv(dirs["results"] / "state_conditioned_beta_diagnostics.csv", index=False)
    contrast.to_csv(dirs["results"] / "state_beta_contrast_summary.csv", index=False)
    pred_dict.to_csv(dirs["results"] / "beta_predictor_dictionary.csv", index=False)

    methodology = {
        "dependent_variable": "ret_btc",
        "model_form": "ret_btc_t = alpha + beta * predictor_t + error_t",
        "predictors": BETA_PREDICTORS,
        "hac": {
            "method": "Newey-West HAC covariance implemented with numpy/scipy",
            "lag_weeks": HAC_LAG_WEEKS,
            "finite_sample_adjustment": "meat matrix multiplied by n / (n - k)",
            "confidence_interval": "two-sided 95 percent t interval with df = n - k",
        },
        "state_usage": "HMM-4 labels are full-sample ex-post descriptive groups from Part 1.",
        "scope_boundaries": [
            "No multivariate beta regression.",
            "No prediction model.",
            "No portfolio weights or allocation rules.",
            "No transaction costs or backtest.",
        ],
    }
    write_json(dirs["results"] / "beta_methodology.json", methodology)
    save_pickle(dirs["models"] / "conditional_beta_specification.pkl", methodology)

    plot_beta_heatmap(state_beta, dirs["figures"] / "state_conditioned_beta_heatmap.png")
    plot_beta_confidence_intervals(full_beta, state_beta, dirs["figures"] / "conditional_beta_confidence_intervals.png")
    logging.info("Conditional beta diagnostics completed")
    return {
        "full_beta": full_beta,
        "state_beta": state_beta,
        "contrast": contrast,
        "predictor_dictionary": pred_dict,
        "beta_methodology": methodology,
    }


def write_explainability_artifacts(
    inputs: dict[str, Any],
    validation_payload: dict[str, Any],
    beta_payload: dict[str, Any],
    dirs: dict[str, Path],
) -> dict[str, Any]:
    paths = inputs["paths"]
    input_hashes = inputs["input_hashes"]
    panel = validation_payload["analysis_panel"]
    state_warnings = validation_payload["state_sample_warnings"]
    part2_btc = inputs["part2_btc_risk_budget"].copy()

    lineage_rows = []
    for name, path in paths.items():
        rows = None
        start = None
        end = None
        if path.suffix == ".csv":
            frame = pd.read_csv(path)
            rows = int(len(frame))
            if "date" in frame.columns:
                start = str(frame["date"].min())
                end = str(frame["date"].max())
        lineage_rows.append(
            {
                "source_name": name,
                "path": str(path),
                "sha256": input_hashes[name],
                "rows": rows,
                "start_date": start,
                "end_date": end,
                "part3_usage": lineage_usage(name),
            }
        )
    lineage = pd.DataFrame(lineage_rows)
    lineage.to_csv(dirs["results"] / "data_lineage.csv", index=False)

    part2_context = part2_btc.copy()
    part2_context["part2_run_id"] = inputs["part2_manifest"].get("run_id")
    part2_context["part2_output_validation_status"] = inputs["part2_output_validation"].get("status")
    part2_context["part3_usage_note"] = (
        "Context only. Part 3 beta diagnostics are computed from cleaned returns, "
        "lagged macro predictors, and Part 1 HMM-4 labels."
    )
    part2_context.to_csv(dirs["results"] / "part2_context_summary.csv", index=False)

    audit_md = f"""# Part 3 Methodology Audit

## Purpose
Part 3 tests BTC state dependence and conditional beta diagnostics. It is diagnostic and descriptive. It does not construct portfolio weights, risk-budget thresholds, allocation rules, trading strategies, transaction costs, or thesis conclusions.

## Inputs
- Cleaned asset returns: `{paths["asset_returns_main_weekly"]}`.
- Cleaned state panel with lagged macro predictors: `{paths["state_model_panel_weekly"]}`.
- Part 1 HMM-4 state labels: `{paths["hmm4_state_labels"]}`.
- Part 2 risk-budget outputs: `{paths["part2_btc_risk_budget_summary"]}`.

The effective Part 3 sample is {len(panel)} weekly observations from {date_string(panel["date"], "min")} to {date_string(panel["date"], "max")}.

## State Dependence Diagnostics
The runner recomputes BTC performance by HMM-4 state and recomputes BTC correlations with SPY, TLT, IEF, GLD, DBC, and BIL by state. It verifies these overlapping quantities against the Part 1 light diagnostics before continuing. State-conditioned drawdown is reported in two forms: noncontiguous state-ordered drawdown and worst contiguous same-state episode drawdown.

## Conditional Beta Diagnostics
The dependent variable is `ret_btc`. Each beta is a separate single-predictor OLS diagnostic. The SPY predictor uses same-week `ret_spy`; macro predictors use one-week-lagged full-sample z-score variables from the cleaned state panel. HAC/Newey-West standard errors use a fixed four-week lag.

## Small Sample Handling
States with fewer than {SMALL_STATE_THRESHOLD_WEEKS} weeks are explicitly flagged. Coefficients and confidence intervals are still reported, but state-level inference is treated as unstable.

## Discussion Boundaries
- HMM-4 labels are full-sample ex-post descriptive labels, not real-time signals.
- Macro beta coefficients are z-score sensitivities, not conventional asset-return betas.
- Part 2 outputs are used only as lineage and context, not as inputs to beta estimation.
- Coin Metrics, SHY, IBIT/FBTC, HY OAS, transaction costs, and implementability checks remain outside Part 3.
"""
    (dirs["results"] / "methodology_audit.md").write_text(audit_md, encoding="utf-8")

    assumption_audit = {
        "status": "documented",
        "state_labels": {
            "source": "Part 1 HMM-4 state labels",
            "estimation_type": "full-sample ex-post descriptive regime identification",
            "not_a_real_time_signal": True,
        },
        "sample": {
            "rows": int(len(panel)),
            "start": date_string(panel["date"], "min"),
            "end": date_string(panel["date"], "max"),
            "state_counts": validation_payload["validation"]["hmm4_state_counts"],
        },
        "small_sample_policy": {
            "threshold_weeks": SMALL_STATE_THRESHOLD_WEEKS,
            "flagged_states": state_warnings.loc[
                state_warnings["state_sample_warning"] != "", "state"
            ].tolist(),
            "policy": "Report diagnostics, but do not hide small-state estimates.",
        },
        "regression_policy": {
            "main_specification": "single-predictor OLS with HAC/Newey-West inference",
            "hac_lag_weeks": HAC_LAG_WEEKS,
            "multivariate_regression": "excluded to avoid overfitting small state samples",
        },
        "excluded_from_part3": [
            "conditional allocation rules",
            "portfolio construction",
            "risk-budget thresholds",
            "transaction costs",
            "rolling ERC",
            "Coin Metrics and ETF robustness",
        ],
    }
    write_json(dirs["results"] / "model_assumption_audit.json", assumption_audit)

    logging.info("Explainability artifacts completed")
    return {
        "data_lineage": lineage,
        "part2_context_summary": part2_context,
        "model_assumption_audit": assumption_audit,
        "methodology_audit": audit_md,
        "beta_methodology": beta_payload["beta_methodology"],
    }


def lineage_usage(name: str) -> str:
    usage = {
        "asset_returns_main_weekly": "Validation and cross-check of state-panel returns.",
        "state_model_panel_weekly": "Primary Part 3 analysis panel.",
        "cleaning_report": "Column mapping and lagged predictor audit.",
        "part1_manifest": "Input hash and Part 1 model lineage.",
        "part1_validation_summary": "Part 1 validation status.",
        "hmm4_state_labels": "Required HMM-4 state labels.",
        "hmm4_state_profiles": "State profile context.",
        "part1_state_conditioned_btc_summary": "Consistency check only.",
        "part1_state_conditioned_correlations": "Consistency check only.",
        "part2_manifest": "Part 2 lineage and context.",
        "part2_input_validation_summary": "Part 2 input validation context.",
        "part2_output_validation_summary": "Part 2 output validation context.",
        "part2_btc_risk_budget_summary": "Risk-budget context only.",
    }
    return usage.get(name, "Supporting audit input.")


def validate_outputs(dirs: dict[str, Path]) -> dict[str, Any]:
    result_checks = []
    for name in REQUIRED_RESULTS:
        if name == "output_validation_summary.json":
            continue
        path = dirs["results"] / name
        result_checks.append(
            {"file": name, "exists": path.exists(), "nonempty": path.exists() and path.stat().st_size > 0}
        )
    figure_checks = []
    for name in REQUIRED_FIGURES:
        path = dirs["figures"] / name
        figure_checks.append(
            {"file": name, "exists": path.exists(), "nonempty": path.exists() and path.stat().st_size > 0}
        )

    part1_checks = pd.read_csv(dirs["results"] / "part1_consistency_checks.csv")
    full_beta = pd.read_csv(dirs["results"] / "full_sample_beta_diagnostics.csv")
    state_beta = pd.read_csv(dirs["results"] / "state_conditioned_beta_diagnostics.csv")
    state_warnings = pd.read_csv(dirs["results"] / "state_sample_warnings.csv")

    numeric_beta_cols = [
        "alpha",
        "beta",
        "hac_se_alpha",
        "hac_se_beta",
        "t_alpha",
        "t_beta",
        "p_alpha",
        "p_beta",
        "beta_ci95_lower",
        "beta_ci95_upper",
        "r_squared",
    ]
    beta_finite = bool(
        np.isfinite(full_beta[numeric_beta_cols].to_numpy(dtype=float)).all()
        and np.isfinite(state_beta[numeric_beta_cols].to_numpy(dtype=float)).all()
    )
    full_n_ok = bool(full_beta["n_obs"].eq(EXPECTED_STATE_ROWS).all())
    state_n_counts = state_beta.groupby("state")["n_obs"].first().sort_index().to_dict()
    state_n_ok = state_n_counts == EXPECTED_STATE_COUNTS
    part1_consistency_ok = bool((part1_checks["status"] == "passed").all())
    required_files_ok = all(item["exists"] and item["nonempty"] for item in result_checks + figure_checks)
    flagged_states = state_warnings.loc[state_warnings["state_sample_warning"].fillna("") != "", "state"].tolist()
    state2_flagged = flagged_states == ["state_2"]

    status = "passed" if all([required_files_ok, beta_finite, full_n_ok, state_n_ok, part1_consistency_ok, state2_flagged]) else "failed"
    summary = {
        "status": status,
        "required_result_checks": result_checks,
        "required_figure_checks": figure_checks,
        "part1_consistency_ok": part1_consistency_ok,
        "part1_max_abs_difference": float(part1_checks["max_abs_difference"].max()),
        "beta_outputs_finite": beta_finite,
        "full_sample_beta_n_obs_ok": full_n_ok,
        "state_conditioned_beta_n_obs": {k: int(v) for k, v in state_n_counts.items()},
        "state_conditioned_beta_n_obs_ok": state_n_ok,
        "small_sample_flagged_states": flagged_states,
        "small_sample_flags_ok": state2_flagged,
        "hac_lag_weeks": HAC_LAG_WEEKS,
    }
    require(status == "passed", f"Output validation failed: {summary}")
    write_json(dirs["results"] / "output_validation_summary.json", summary)
    logging.info("Output validation completed")
    return summary


def write_manifest(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    inputs: dict[str, Any],
    validation_payload: dict[str, Any],
    beta_payload: dict[str, Any],
    output_validation: dict[str, Any],
) -> dict[str, Any]:
    panel = validation_payload["analysis_panel"]
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "run_id": dirs["root"].name,
        "objective": "BTC state dependence and conditional beta diagnostics",
        "input_dir": str(args.input_dir),
        "part1_run_dir": str(args.part1_run_dir),
        "part2_run_dir": str(args.part2_run_dir),
        "output_dir": str(args.output_dir),
        "run_dir": str(dirs["root"]),
        "random_seed": int(args.seed),
        "sample": {
            "rows": int(len(panel)),
            "start": date_string(panel["date"], "min"),
            "end": date_string(panel["date"], "max"),
            "state_counts": validation_payload["validation"]["hmm4_state_counts"],
        },
        "input_hashes": inputs["input_hashes"],
        "package_versions": package_versions(),
        "parameters": {
            "tail_alpha": TAIL_ALPHA,
            "trading_weeks_per_year": TRADING_WEEKS_PER_YEAR,
            "small_state_threshold_weeks": SMALL_STATE_THRESHOLD_WEEKS,
            "hac_lag_weeks": HAC_LAG_WEEKS,
            "dependent_variable": "ret_btc",
            "beta_predictors": [entry["predictor"] for entry in BETA_PREDICTORS],
            "beta_specification": "single-predictor OLS, lagged macro z-score main specification",
            "state_usage": "Part 1 full-sample ex-post HMM-4 state labels; diagnostic grouping only.",
        },
        "lineage": {
            "part1_run_id": inputs["part1_manifest"].get("run_id"),
            "part1_hmm4_converged": inputs["part1_manifest"].get("model_diagnostics", {}).get("hmm4_converged"),
            "part2_run_id": inputs["part2_manifest"].get("run_id"),
            "part2_output_validation_status": inputs["part2_output_validation"].get("status"),
        },
        "beta_methodology": beta_payload["beta_methodology"],
        "output_validation": output_validation,
        "outputs": {
            "checkpoints": str(dirs["checkpoints"]),
            "results": str(dirs["results"]),
            "figures": str(dirs["figures"]),
            "models": str(dirs["models"]),
            "logs": str(dirs["logs"]),
        },
        "scope_notes": [
            "No conditional allocation rule in Part 3.",
            "No portfolio weights, transaction costs, turnover, or trading strategy in Part 3.",
            "Part 2 outputs are context only and do not enter beta estimation.",
            "Coin Metrics, SHY, IBIT/FBTC, and HY OAS robustness are reserved for later experiments.",
        ],
    }
    write_json(dirs["root"] / "run_manifest.json", manifest)
    return manifest


def plot_btc_state_performance(state_perf: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = state_perf["state"].tolist()
    colors = ["#3b7ddd", "#6f9ceb", "#f0a43a", "#c7524a"]
    axes[0, 0].bar(labels, state_perf["btc_mean_weekly"], color=colors)
    axes[0, 0].set_title("BTC Mean Weekly Return by HMM-4 State")
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].bar(labels, state_perf["btc_volatility_weekly"], color=colors)
    axes[0, 1].set_title("BTC Weekly Volatility by HMM-4 State")
    axes[1, 0].bar(labels, state_perf["btc_cvar_95_weekly"], color=colors)
    axes[1, 0].set_title("BTC 5% CVaR by HMM-4 State")
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].bar(labels, state_perf["btc_max_drawdown"], color=colors)
    axes[1, 1].set_title("BTC Max Drawdown by HMM-4 State")
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_correlation_heatmap(correlations: pd.DataFrame, output_path: Path) -> None:
    pivot = correlations.pivot(index="state", columns="asset", values="correlation_with_btc").sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_title("BTC State-Conditioned Correlations")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_beta_heatmap(state_beta: pd.DataFrame, output_path: Path) -> None:
    display = state_beta.copy()
    display["predictor_short"] = display["predictor"].map(
        {
            "ret_spy": "SPY",
            "macro_vix_z": "VIX z",
            "macro_real_yield_10y_z": "Real yield z",
            "macro_dollar_chg_4w_z": "Dollar z",
            "macro_credit_spread_baa10y_z": "BAA10Y z",
        }
    )
    pivot = display.pivot(index="state", columns="predictor_short", values="beta").sort_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    max_abs = float(np.nanmax(np.abs(pivot.to_numpy())))
    bound = max(max_abs, 0.01)
    im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_title("State-Conditioned BTC Beta Diagnostics")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.3f}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_beta_confidence_intervals(full_beta: pd.DataFrame, state_beta: pd.DataFrame, output_path: Path) -> None:
    predictors = full_beta["predictor"].tolist()
    short_names = {
        "ret_spy": "SPY",
        "macro_vix_z": "VIX z",
        "macro_real_yield_10y_z": "Real yield z",
        "macro_dollar_chg_4w_z": "Dollar z",
        "macro_credit_spread_baa10y_z": "BAA10Y z",
    }
    fig, axes = plt.subplots(len(predictors), 1, figsize=(11, 13), sharex=False)
    if len(predictors) == 1:
        axes = [axes]
    for ax, predictor in zip(axes, predictors):
        sub = state_beta[state_beta["predictor"] == predictor].sort_values("state_id")
        full = full_beta[full_beta["predictor"] == predictor].iloc[0]
        labels = ["full"] + sub["state"].tolist()
        betas = [float(full["beta"])] + sub["beta"].astype(float).tolist()
        lows = [float(full["beta_ci95_lower"])] + sub["beta_ci95_lower"].astype(float).tolist()
        highs = [float(full["beta_ci95_upper"])] + sub["beta_ci95_upper"].astype(float).tolist()
        x_pos = np.arange(len(labels))
        lower_err = np.array(betas) - np.array(lows)
        upper_err = np.array(highs) - np.array(betas)
        colors = ["#2f4858"] + ["#3b7ddd", "#6f9ceb", "#f0a43a", "#c7524a"]
        ax.errorbar(x_pos, betas, yerr=[lower_err, upper_err], fmt="none", ecolor="#333333", capsize=4, linewidth=1)
        ax.scatter(x_pos, betas, color=colors, s=45, zorder=3)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x_pos, labels)
        ax.set_title(f"{short_names[predictor]} beta with HAC 95% confidence intervals")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    run_id = args.run_id or now_run_id()
    run_dir = args.output_dir / run_id
    dirs = ensure_dirs(run_dir)
    setup_logging(dirs["logs"])
    logging.info("Starting Part 3 run: %s", run_id)

    inputs = read_inputs(args)
    enforce_resume_input_hashes(args, dirs, inputs["input_hashes"])
    validation_payload = load_or_run(
        dirs,
        "01_input_validation",
        args.resume,
        lambda: validate_inputs(inputs, dirs),
    )
    state_payload = load_or_run(
        dirs,
        "02_state_dependence_diagnostics",
        args.resume,
        lambda: compute_state_diagnostics(inputs, validation_payload, dirs),
    )
    beta_payload = load_or_run(
        dirs,
        "03_conditional_beta_diagnostics",
        args.resume,
        lambda: compute_beta_diagnostics(validation_payload, dirs),
    )
    explainability = load_or_run(
        dirs,
        "04_explainability_artifacts",
        args.resume,
        lambda: write_explainability_artifacts(inputs, validation_payload, beta_payload, dirs),
    )
    output_validation = load_or_run(
        dirs,
        "05_output_validation",
        args.resume,
        lambda: validate_outputs(dirs),
    )
    manifest = write_manifest(args, dirs, inputs, validation_payload, beta_payload, output_validation)
    _ = state_payload, explainability, manifest
    logging.info("Completed Part 3 run: %s", run_id)
    logging.info("Results directory: %s", dirs["results"])


if __name__ == "__main__":
    main()
