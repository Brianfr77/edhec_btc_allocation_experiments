#!/usr/bin/env python3
"""Part 8 runner: bootstrap uncertainty and HMM ensemble drift audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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


EXPECTED_ASSET_START = "2018-01-12"
EXPECTED_ASSET_END = "2026-03-27"
EXPECTED_ASSET_ROWS = 429
EXPECTED_STATE_START = "2018-02-09"
EXPECTED_STATE_END = "2026-03-27"
EXPECTED_STATE_ROWS = 425
EXPECTED_PART5_START = "2018-02-16"
EXPECTED_PART5_ROWS = 424
EXPECTED_PART7_PROB_START = "2021-02-05"
EXPECTED_PART7_RETURN_START = "2021-02-12"
EXPECTED_PART7_RETURN_ROWS = 268
EXPECTED_STATE_COUNTS = {"state_0": 149, "state_1": 167, "state_2": 30, "state_3": 79}

TAIL_ALPHA = 0.05
TRADING_WEEKS_PER_YEAR = 52
RISK_BUDGET_CAP = 0.10
FLOAT_TOL = 1e-10
MIN_STATE_N = 10
MIN_CVAR_TAIL_COUNT = 2

BASE_ASSETS = ["ret_spy", "ret_tlt", "ret_ief", "ret_gld", "ret_dbc"]
ASSETS = ["ret_btc"] + BASE_ASSETS + ["ret_bil"]
IMPLEMENTED_RULE_IDS = ["main_executed", "sensitivity_state2_low_executed"]
PORTFOLIO_FAMILIES = ["all_weather", "erc"]
FUNDING_CONVENTIONS = ["pro_rata_base", "bil_sleeve"]
REBALANCE_FREQUENCIES = ["monthly", "quarterly"]
MAIN_COST_SCENARIO = "moderate_cost"
MAIN_SIGNAL_TIMING = "lagged_one_week"
PART7_MAIN_RULE_TYPE = "realtime_probability_weighted_overlay"

STATE_BETA_PREDICTORS = [
    "ret_spy",
    "macro_vix_z",
    "macro_real_yield_10y_z",
    "macro_dollar_chg_4w_z",
    "macro_credit_spread_baa10y_z",
]

REQUIRED_RESULTS = [
    "input_validation_summary.json",
    "bootstrap_configuration.json",
    "bootstrap_draw_index_audit.csv",
    "state_conditioned_btc_bootstrap_ci.csv",
    "state_conditioned_beta_bootstrap_ci.csv",
    "conditional_rule_bootstrap_ci.csv",
    "risk_contribution_bootstrap_ci.csv",
    "implementability_bootstrap_ci.csv",
    "realtime_rule_bootstrap_ci.csv",
    "realtime_vs_expost_bootstrap_ci.csv",
    "risk_budget_cap_exceedance_summary.csv",
    "small_sample_state_uncertainty_audit.csv",
    "hmm_ensemble_variant_dictionary.csv",
    "hmm_ensemble_state_agreement.csv",
    "hmm_ensemble_rule_signal_sensitivity.csv",
    "hmm_ensemble_rule_performance_sensitivity.csv",
    "uncertainty_decision_matrix.csv",
    "methodology_audit.md",
    "data_lineage.csv",
    "model_assumption_audit.json",
    "output_validation_summary.json",
]
REQUIRED_FIGURES = [
    "state_btc_bootstrap_ci.png",
    "rule_performance_bootstrap_ci.png",
    "risk_contribution_bootstrap_ci.png",
    "realtime_vs_expost_bootstrap_ci.png",
    "hmm_ensemble_agreement.png",
    "small_sample_state_uncertainty.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 8 bootstrap uncertainty diagnostics.")
    parser.add_argument("--input-dir", default="data_2026/cleaned", type=Path)
    parser.add_argument("--part1-run-dir", default="outputs/part1_btc_macro_state/colab_part1_seed42", type=Path)
    parser.add_argument("--part2-run-dir", default="outputs/part2_portfolio_risk_budget/colab_part2_seed42", type=Path)
    parser.add_argument("--part3-run-dir", default="outputs/part3_btc_state_dependence/colab_part3_seed42", type=Path)
    parser.add_argument("--part4-run-dir", default="outputs/part4_conditional_btc_allocation/colab_part4_seed42", type=Path)
    parser.add_argument("--part5-run-dir", default="outputs/part5_implementability_rebalancing/colab_part5_seed42", type=Path)
    parser.add_argument("--part6-run-dir", default="outputs/part6_robustness_analysis/colab_part6_seed42", type=Path)
    parser.add_argument("--part7-run-dir", default="outputs/part7_realtime_probabilistic_regime_robustness/colab_part7_seed42", type=Path)
    parser.add_argument("--output-dir", default="outputs/part8_uncertainty_quantification", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--bootstrap-reps", default=2000, type=int)
    parser.add_argument("--block-length", default=13, type=int)
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
    for package in ["numpy", "pandas", "matplotlib", "scipy"]:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def date_string(series: pd.Series, fn: str) -> str:
    value = series.min() if fn == "min" else series.max()
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def validation_status(payload: dict[str, Any]) -> str:
    if "status" in payload:
        return str(payload["status"])
    if payload.get("validation", {}).get("status"):
        return str(payload["validation"]["status"])
    return ""


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
    sharpe = mean / vol * math.sqrt(TRADING_WEEKS_PER_YEAR) if vol and vol > 0 else float("nan")
    return {
        "mean_weekly": mean,
        "volatility_weekly": vol,
        "annualized_mean_arithmetic": mean * TRADING_WEEKS_PER_YEAR if math.isfinite(mean) else float("nan"),
        "annualized_volatility": vol * math.sqrt(TRADING_WEEKS_PER_YEAR) if math.isfinite(vol) else float("nan"),
        "var_95_weekly": var_value,
        "cvar_95_weekly": cvar_value,
        "tail_scenario_count": tail_count,
        "max_drawdown": float(drawdown_series(clean).min()) if len(clean) else float("nan"),
        "positive_week_share": float((clean > 0).mean()) if len(clean) else float("nan"),
        "sharpe_annualized_zero_rf": sharpe,
    }


def beta_r2(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    frame = pd.concat([pd.Series(y), pd.Series(x)], axis=1).dropna()
    if len(frame) < MIN_STATE_N:
        return float("nan"), float("nan")
    yy = frame.iloc[:, 0].astype(float)
    xx = frame.iloc[:, 1].astype(float)
    var_x = float(xx.var(ddof=1))
    if var_x <= FLOAT_TOL:
        return float("nan"), float("nan")
    beta = float(yy.cov(xx) / var_x)
    alpha = float(yy.mean() - beta * xx.mean())
    fitted = alpha + beta * xx
    ss_res = float(((yy - fitted) ** 2).sum())
    ss_tot = float(((yy - yy.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > FLOAT_TOL else float("nan")
    return beta, r2


def component_risk_shares(component: pd.DataFrame, asset: str = "ret_btc") -> dict[str, Any]:
    pivot = component.pivot_table(index="sample_order", columns="asset", values="component_return", aggfunc="sum")
    pivot = pivot.reindex(columns=ASSETS, fill_value=0.0)
    portfolio = pivot.sum(axis=1)
    vol = float(portfolio.std(ddof=1))
    if vol <= FLOAT_TOL:
        vol_share = float("nan")
    else:
        vol_share = float(np.cov(pivot[asset], portfolio, ddof=1)[0, 1] / (vol * vol))
    var_value = float(portfolio.quantile(TAIL_ALPHA))
    tail_mask = portfolio <= var_value
    tail_count = int(tail_mask.sum())
    cvar_loss = float((-portfolio.loc[tail_mask]).mean()) if tail_count else float("nan")
    if not math.isfinite(cvar_loss) or abs(cvar_loss) <= FLOAT_TOL:
        cvar_share = float("nan")
    else:
        cvar_share = float((-pivot.loc[tail_mask, asset]).mean() / cvar_loss)
    return {
        "btc_component_share_vol": vol_share,
        "btc_component_share_cvar": cvar_share,
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
        "btc_component_share_vol": vol_share,
        "btc_component_share_cvar": cvar_share,
        "tail_scenario_count": tail_count,
    }


def ci_summary(rows: list[dict[str, Any]], group_cols: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    out_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        info = dict(zip(group_cols, keys))
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        original = float(group["original_estimate"].dropna().iloc[0]) if group["original_estimate"].notna().any() else float("nan")
        valid = int(len(values))
        invalid = int(len(group) - valid)
        if valid:
            q = values.quantile([0.025, 0.05, 0.95, 0.975])
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if valid > 1 else 0.0
            ci90_lower = float(q.loc[0.05])
            ci90_upper = float(q.loc[0.95])
            ci95_lower = float(q.loc[0.025])
            ci95_upper = float(q.loc[0.975])
        else:
            mean = std = ci90_lower = ci90_upper = ci95_lower = ci95_upper = float("nan")
        out_rows.append(
            {
                **info,
                "original_estimate": original,
                "bootstrap_mean": mean,
                "bootstrap_std": std,
                "ci90_lower": ci90_lower,
                "ci90_upper": ci90_upper,
                "ci95_lower": ci95_lower,
                "ci95_upper": ci95_upper,
                "valid_reps": valid,
                "invalid_reps": invalid,
                "estimate_below_ci95": bool(math.isfinite(original) and math.isfinite(ci95_lower) and original < ci95_lower),
                "estimate_above_ci95": bool(math.isfinite(original) and math.isfinite(ci95_upper) and original > ci95_upper),
            }
        )
    return pd.DataFrame(out_rows)


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "asset": args.input_dir / "asset_returns_main_weekly.csv",
        "state": args.input_dir / "state_model_panel_weekly.csv",
        "cleaning_report": args.input_dir / "cleaning_report.json",
        "part1_manifest": args.part1_run_dir / "run_manifest.json",
        "part1_validation": args.part1_run_dir / "results" / "validation_summary.json",
        "part1_labels": args.part1_run_dir / "results" / "hmm4_state_labels.csv",
        "part2_manifest": args.part2_run_dir / "run_manifest.json",
        "part2_input_validation": args.part2_run_dir / "results" / "input_validation_summary.json",
        "part2_output_validation": args.part2_run_dir / "results" / "output_validation_summary.json",
        "part2_baseline_weights": args.part2_run_dir / "results" / "baseline_portfolio_weights.csv",
        "part3_manifest": args.part3_run_dir / "run_manifest.json",
        "part3_output_validation": args.part3_run_dir / "results" / "output_validation_summary.json",
        "part3_btc_performance": args.part3_run_dir / "results" / "state_conditioned_btc_performance.csv",
        "part3_beta": args.part3_run_dir / "results" / "state_conditioned_beta_diagnostics.csv",
        "part4_manifest": args.part4_run_dir / "run_manifest.json",
        "part4_input_validation": args.part4_run_dir / "results" / "input_validation_summary.json",
        "part4_output_validation": args.part4_run_dir / "results" / "output_validation_summary.json",
        "part4_rule_definition": args.part4_run_dir / "results" / "allocation_rule_definition.csv",
        "part4_weekly_weights": args.part4_run_dir / "results" / "weekly_conditional_weights.csv",
        "part4_returns": args.part4_run_dir / "results" / "conditional_portfolio_return_series.csv",
        "part5_manifest": args.part5_run_dir / "run_manifest.json",
        "part5_input_validation": args.part5_run_dir / "results" / "input_validation_summary.json",
        "part5_output_validation": args.part5_run_dir / "results" / "output_validation_summary.json",
        "part5_returns": args.part5_run_dir / "results" / "rebalanced_portfolio_return_series.csv",
        "part6_manifest": args.part6_run_dir / "run_manifest.json",
        "part6_input_validation": args.part6_run_dir / "results" / "input_validation_summary.json",
        "part6_output_validation": args.part6_run_dir / "results" / "output_validation_summary.json",
        "part7_manifest": args.part7_run_dir / "run_manifest.json",
        "part7_input_validation": args.part7_run_dir / "results" / "input_validation_summary.json",
        "part7_output_validation": args.part7_run_dir / "results" / "output_validation_summary.json",
        "part7_probabilities": args.part7_run_dir / "results" / "realtime_state_probabilities.csv",
        "part7_rule_signals": args.part7_run_dir / "results" / "realtime_rule_signal_series.csv",
        "part7_target_returns": args.part7_run_dir / "results" / "realtime_target_weight_return_series.csv",
        "part7_returns": args.part7_run_dir / "results" / "realtime_rebalanced_return_series.csv",
        "part7_overlay": args.part7_run_dir / "results" / "risk_budget_overlay_audit.csv",
    }


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    paths = resolve_paths(args)
    missing = [str(path) for path in paths.values() if not path.exists()]
    require(not missing, "Missing required input files: " + "; ".join(missing))
    return {
        "paths": paths,
        "asset": pd.read_csv(paths["asset"], parse_dates=["date"]),
        "state": pd.read_csv(paths["state"], parse_dates=["date"]),
        "cleaning_report": read_json(paths["cleaning_report"]),
        "part1_manifest": read_json(paths["part1_manifest"]),
        "part1_validation": read_json(paths["part1_validation"]),
        "part1_labels": pd.read_csv(paths["part1_labels"], parse_dates=["date"]),
        "part2_manifest": read_json(paths["part2_manifest"]),
        "part2_input_validation": read_json(paths["part2_input_validation"]),
        "part2_output_validation": read_json(paths["part2_output_validation"]),
        "part2_baseline_weights": pd.read_csv(paths["part2_baseline_weights"]),
        "part3_manifest": read_json(paths["part3_manifest"]),
        "part3_output_validation": read_json(paths["part3_output_validation"]),
        "part3_btc_performance": pd.read_csv(paths["part3_btc_performance"]),
        "part3_beta": pd.read_csv(paths["part3_beta"]),
        "part4_manifest": read_json(paths["part4_manifest"]),
        "part4_input_validation": read_json(paths["part4_input_validation"]),
        "part4_output_validation": read_json(paths["part4_output_validation"]),
        "part4_rule_definition": pd.read_csv(paths["part4_rule_definition"]),
        "part4_weekly_weights": pd.read_csv(paths["part4_weekly_weights"], parse_dates=["date"]),
        "part4_returns": pd.read_csv(paths["part4_returns"], parse_dates=["date"]),
        "part5_manifest": read_json(paths["part5_manifest"]),
        "part5_input_validation": read_json(paths["part5_input_validation"]),
        "part5_output_validation": read_json(paths["part5_output_validation"]),
        "part5_returns": pd.read_csv(paths["part5_returns"], parse_dates=["date"]),
        "part6_manifest": read_json(paths["part6_manifest"]),
        "part6_input_validation": read_json(paths["part6_input_validation"]),
        "part6_output_validation": read_json(paths["part6_output_validation"]),
        "part7_manifest": read_json(paths["part7_manifest"]),
        "part7_input_validation": read_json(paths["part7_input_validation"]),
        "part7_output_validation": read_json(paths["part7_output_validation"]),
        "part7_probabilities": pd.read_csv(paths["part7_probabilities"], parse_dates=["date", "training_end_date"]),
        "part7_rule_signals": pd.read_csv(paths["part7_rule_signals"], parse_dates=["decision_date", "return_date"]),
        "part7_target_returns": pd.read_csv(paths["part7_target_returns"], parse_dates=["decision_date", "return_date"]),
        "part7_returns": pd.read_csv(paths["part7_returns"], parse_dates=["decision_date", "return_date"]),
        "part7_overlay": pd.read_csv(paths["part7_overlay"], parse_dates=["decision_date", "return_date"]),
    }


def validate_inputs(inputs: dict[str, Any], args: argparse.Namespace, dirs: dict[str, Path]) -> dict[str, Any]:
    asset = inputs["asset"]
    state = inputs["state"]
    labels = inputs["part1_labels"]
    require(len(asset) == EXPECTED_ASSET_ROWS, "Unexpected asset rows")
    require(date_string(asset["date"], "min") == EXPECTED_ASSET_START, "Unexpected asset start")
    require(date_string(asset["date"], "max") == EXPECTED_ASSET_END, "Unexpected asset end")
    require(len(state) == EXPECTED_STATE_ROWS, "Unexpected state rows")
    require(date_string(state["date"], "min") == EXPECTED_STATE_START, "Unexpected state start")
    require(date_string(state["date"], "max") == EXPECTED_STATE_END, "Unexpected state end")
    require(len(labels) == EXPECTED_STATE_ROWS, "Unexpected Part 1 labels rows")
    require(labels["hmm4_state"].value_counts().sort_index().to_dict() == EXPECTED_STATE_COUNTS, "Unexpected Part 1 state counts")

    validation_payloads = {
        "part1_validation": inputs["part1_validation"],
        "part2_input_validation": inputs["part2_input_validation"],
        "part2_output_validation": inputs["part2_output_validation"],
        "part3_output_validation": inputs["part3_output_validation"],
        "part4_input_validation": inputs["part4_input_validation"],
        "part4_output_validation": inputs["part4_output_validation"],
        "part5_input_validation": inputs["part5_input_validation"],
        "part5_output_validation": inputs["part5_output_validation"],
        "part6_input_validation": inputs["part6_input_validation"],
        "part6_output_validation": inputs["part6_output_validation"],
        "part7_input_validation": inputs["part7_input_validation"],
        "part7_output_validation": inputs["part7_output_validation"],
    }
    for name, payload in validation_payloads.items():
        require(validation_status(payload) == "passed", f"Upstream validation did not pass: {name}")

    cleaned_hashes = {
        "asset_returns_main_weekly": file_sha256(inputs["paths"]["asset"]),
        "state_model_panel_weekly": file_sha256(inputs["paths"]["state"]),
        "cleaning_report": file_sha256(inputs["paths"]["cleaning_report"]),
    }
    part1_hashes = inputs["part1_manifest"].get("input_hashes", {})
    for key, value in cleaned_hashes.items():
        require(part1_hashes.get(key) == value, f"Cleaned hash mismatch against Part 1: {key}")
    require(file_sha256(inputs["paths"]["part1_labels"]) == inputs["part2_manifest"]["input_hashes"]["hmm4_state_labels"], "HMM label hash mismatch against Part 2")
    require(file_sha256(inputs["paths"]["part4_rule_definition"]) == inputs["part5_manifest"]["input_hashes"]["part4_allocation_rule_definition"], "Part 4 rule hash mismatch against Part 5")
    require(file_sha256(inputs["paths"]["part5_returns"]) == inputs["part7_manifest"]["input_hashes"]["part5_rebalanced_returns"], "Part 5 returns hash mismatch against Part 7")

    panel = state.merge(labels[["date", "hmm4_state", "hmm4_state_id"]], on="date", how="inner", validate="one_to_one")
    require(len(panel) == EXPECTED_STATE_ROWS, "State/label panel join lost rows")
    part5_main = inputs["part5_returns"][
        (inputs["part5_returns"]["signal_timing"] == MAIN_SIGNAL_TIMING)
        & (inputs["part5_returns"]["cost_scenario"] == MAIN_COST_SCENARIO)
        & (inputs["part5_returns"]["rule_id"].isin(IMPLEMENTED_RULE_IDS))
    ].copy()
    require(part5_main.groupby("scenario_id").size().nunique() == 1, "Part 5 scenario lengths differ")
    require(int(part5_main.groupby("scenario_id").size().iloc[0]) == EXPECTED_PART5_ROWS, "Unexpected Part 5 main rows")

    part7_main = inputs["part7_returns"][
        (inputs["part7_returns"]["rule_type"] == PART7_MAIN_RULE_TYPE)
        & (inputs["part7_returns"]["cost_scenario"] == MAIN_COST_SCENARIO)
        & (inputs["part7_returns"]["rule_id"].isin(IMPLEMENTED_RULE_IDS))
    ].copy()
    require(part7_main.groupby("scenario_id").size().nunique() == 1, "Part 7 scenario lengths differ")
    require(int(part7_main.groupby("scenario_id").size().iloc[0]) == EXPECTED_PART7_RETURN_ROWS, "Unexpected Part 7 main rows")
    require(date_string(part7_main["return_date"], "min") == EXPECTED_PART7_RETURN_START, "Unexpected Part 7 return start")

    input_hashes = {name: file_sha256(path) for name, path in inputs["paths"].items()}
    summary = {
        "status": "passed",
        "bootstrap_reps": args.bootstrap_reps,
        "block_length": args.block_length,
        "samples": {
            "full_state_sample": {"rows": EXPECTED_STATE_ROWS, "start": EXPECTED_STATE_START, "end": EXPECTED_STATE_END},
            "part5_implementability_sample": {"rows": EXPECTED_PART5_ROWS, "start": EXPECTED_PART5_START, "end": EXPECTED_STATE_END},
            "part7_realtime_sample": {"rows": EXPECTED_PART7_RETURN_ROWS, "start": EXPECTED_PART7_RETURN_START, "end": EXPECTED_STATE_END},
        },
        "state_counts": labels["hmm4_state"].value_counts().sort_index().to_dict(),
        "cleaned_hashes": cleaned_hashes,
        "upstream_validation_status": {name: validation_status(payload) for name, payload in validation_payloads.items()},
        "input_hashes": input_hashes,
    }
    write_json(dirs["results"] / "input_validation_summary.json", summary)
    return {"panel": panel, "part5_main": part5_main, "part7_main": part7_main, "input_hashes": input_hashes, "summary": summary}


def enforce_resume_input_hashes(args: argparse.Namespace, dirs: dict[str, Path], input_hashes: dict[str, str]) -> None:
    if not args.resume:
        return
    manifest_path = dirs["root"] / "run_manifest.json"
    if not manifest_path.exists():
        return
    old_manifest = read_json(manifest_path)
    require(old_manifest.get("input_hashes", {}) == input_hashes, "Input hashes changed since previous run")


def circular_block_indices(n: int, block_length: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(math.ceil(n / block_length))
    draws = np.empty((reps, n), dtype=np.int32)
    for rep in range(reps):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(start, start + block_length) % n) for start in starts])[:n]
        draws[rep] = idx
    return draws


def build_bootstrap_draws(validation: dict[str, Any], args: argparse.Namespace, dirs: dict[str, Path]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    domains = {
        "full_state_sample": EXPECTED_STATE_ROWS,
        "part5_implementability_sample": EXPECTED_PART5_ROWS,
        "part7_realtime_sample": EXPECTED_PART7_RETURN_ROWS,
    }
    draws = {domain: circular_block_indices(n, args.block_length, args.bootstrap_reps, rng) for domain, n in domains.items()}
    audit_rows = []
    for domain, matrix in draws.items():
        audit_rows.append(
            {
                "sample_domain": domain,
                "bootstrap_reps": int(matrix.shape[0]),
                "sample_length": int(matrix.shape[1]),
                "block_length": args.block_length,
                "min_index": int(matrix.min()),
                "max_index": int(matrix.max()),
                "first_rep_unique_index_count": int(len(np.unique(matrix[0]))),
                "same_draw_used_within_domain": True,
            }
        )
    pd.DataFrame(audit_rows).to_csv(dirs["results"] / "bootstrap_draw_index_audit.csv", index=False)
    config = {
        "method": "circular_moving_block_bootstrap",
        "bootstrap_reps": args.bootstrap_reps,
        "block_length": args.block_length,
        "ci_levels": [0.90, 0.95],
        "seed": args.seed,
        "sample_domains": domains,
        "state_metric_min_n": MIN_STATE_N,
        "cvar_min_tail_count": MIN_CVAR_TAIL_COUNT,
    }
    write_json(dirs["results"] / "bootstrap_configuration.json", config)
    return draws


def bootstrap_state_btc(validation: dict[str, Any], draws: dict[str, np.ndarray], dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    panel = validation["panel"].reset_index(drop=True)
    original_rows = []
    for state, frame in panel.groupby("hmm4_state", sort=True):
        metrics = performance_metrics(frame["ret_btc"])
        for metric, value in metrics.items():
            if metric in {"mean_weekly", "volatility_weekly", "var_95_weekly", "cvar_95_weekly", "max_drawdown", "positive_week_share"}:
                original_rows.append({"state": state, "metric": metric, "original_estimate": value})
    original = pd.DataFrame(original_rows)
    rows: list[dict[str, Any]] = []
    small_rows: list[dict[str, Any]] = []
    for rep, idx in enumerate(draws["full_state_sample"]):
        sample = panel.iloc[idx].copy().reset_index(drop=True)
        for state in EXPECTED_STATE_COUNTS:
            frame = sample[sample["hmm4_state"] == state]
            n = int(len(frame))
            metrics = performance_metrics(frame["ret_btc"]) if n >= MIN_STATE_N else {}
            _, _, tail_count = var_cvar(frame["ret_btc"]) if n >= MIN_STATE_N else (float("nan"), float("nan"), 0)
            small_rows.append({"rep": rep, "state": state, "state_n": n, "tail_scenario_count": tail_count, "valid_state_n": n >= MIN_STATE_N})
            for metric in ["mean_weekly", "volatility_weekly", "var_95_weekly", "cvar_95_weekly", "max_drawdown", "positive_week_share"]:
                original_value = float(original[(original["state"] == state) & (original["metric"] == metric)]["original_estimate"].iloc[0])
                valid = n >= MIN_STATE_N and (metric != "cvar_95_weekly" or tail_count >= MIN_CVAR_TAIL_COUNT)
                rows.append(
                    {
                        "state": state,
                        "metric": metric,
                        "rep": rep,
                        "value": metrics.get(metric, float("nan")) if valid else float("nan"),
                        "original_estimate": original_value,
                    }
                )
    ci = ci_summary(rows, ["state", "metric"])
    small = pd.DataFrame(small_rows)
    audit = small.groupby("state").agg(
        reps=("rep", "count"),
        state_n_mean=("state_n", "mean"),
        state_n_min=("state_n", "min"),
        state_n_p05=("state_n", lambda x: float(np.quantile(x, 0.05))),
        state_n_p95=("state_n", lambda x: float(np.quantile(x, 0.95))),
        invalid_state_n_reps=("valid_state_n", lambda x: int((~x).sum())),
        cvar_tail_count_min=("tail_scenario_count", "min"),
        cvar_tail_count_p05=("tail_scenario_count", lambda x: float(np.quantile(x, 0.05))),
    ).reset_index()
    ci.to_csv(dirs["results"] / "state_conditioned_btc_bootstrap_ci.csv", index=False)
    audit.to_csv(dirs["results"] / "small_sample_state_uncertainty_audit.csv", index=False)
    return {"ci": ci, "small_sample": audit}


def bootstrap_state_beta(validation: dict[str, Any], draws: dict[str, np.ndarray], dirs: dict[str, Path]) -> pd.DataFrame:
    panel = validation["panel"].reset_index(drop=True)
    original_rows = []
    for state, frame in panel.groupby("hmm4_state", sort=True):
        for predictor in STATE_BETA_PREDICTORS:
            beta, r2 = beta_r2(frame["ret_btc"], frame[predictor])
            original_rows.append({"state": state, "predictor": predictor, "metric": "beta", "original_estimate": beta})
            original_rows.append({"state": state, "predictor": predictor, "metric": "r_squared", "original_estimate": r2})
    original = pd.DataFrame(original_rows)
    rows: list[dict[str, Any]] = []
    for rep, idx in enumerate(draws["full_state_sample"]):
        sample = panel.iloc[idx].copy().reset_index(drop=True)
        for state in EXPECTED_STATE_COUNTS:
            frame = sample[sample["hmm4_state"] == state]
            for predictor in STATE_BETA_PREDICTORS:
                beta, r2 = beta_r2(frame["ret_btc"], frame[predictor])
                for metric, value in [("beta", beta), ("r_squared", r2)]:
                    original_value = float(
                        original[
                            (original["state"] == state) & (original["predictor"] == predictor) & (original["metric"] == metric)
                        ]["original_estimate"].iloc[0]
                    )
                    rows.append(
                        {
                            "state": state,
                            "predictor": predictor,
                            "metric": metric,
                            "rep": rep,
                            "value": value,
                            "original_estimate": original_value,
                        }
                    )
    ci = ci_summary(rows, ["state", "predictor", "metric"])
    ci.to_csv(dirs["results"] / "state_conditioned_beta_bootstrap_ci.csv", index=False)
    return ci


def part4_component_frame(inputs: dict[str, Any]) -> pd.DataFrame:
    weights = inputs["part4_weekly_weights"].copy()
    state_returns = inputs["state"][["date"] + ASSETS].copy()
    frame = weights.merge(state_returns, on="date", how="left", validate="many_to_one")
    frame["component_return"] = frame["weight"] * frame.lookup(frame.index, frame["asset"]) if hasattr(frame, "lookup") else np.nan
    if frame["component_return"].isna().all():
        frame["component_return"] = [float(row["weight"]) * float(row[row["asset"]]) for _, row in frame.iterrows()]
    return frame


def bootstrap_part4(inputs: dict[str, Any], draws: dict[str, np.ndarray], dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    returns = inputs["part4_returns"].sort_values(["rule_id", "portfolio_family", "date"]).copy()
    weights = inputs["part4_weekly_weights"].copy()
    state_returns = inputs["state"].set_index("date").sort_index()
    scenarios = returns[["rule_id", "portfolio_family"]].drop_duplicates().sort_values(["rule_id", "portfolio_family"])
    original_perf = {}
    scenario_arrays: dict[tuple[str, str], dict[str, Any]] = {}
    for _, scen in scenarios.iterrows():
        key = (str(scen["rule_id"]), str(scen["portfolio_family"]))
        frame = returns[(returns["rule_id"] == key[0]) & (returns["portfolio_family"] == key[1])].sort_values("date").reset_index(drop=True)
        dates = pd.to_datetime(frame["date"])
        w = weights[(weights["rule_id"] == key[0]) & (weights["portfolio_family"] == key[1])]
        weight_matrix = (
            w.pivot_table(index="date", columns="asset", values="weight", aggfunc="first")
            .reindex(index=dates, columns=ASSETS)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        return_matrix = state_returns.reindex(dates)[ASSETS].to_numpy(dtype=float)
        component_matrix = weight_matrix * return_matrix
        original_perf[key] = performance_metrics(frame["portfolio_return"])
        scenario_arrays[key] = {
            "portfolio_returns": frame["portfolio_return"].to_numpy(dtype=float),
            "component_matrix": component_matrix,
            "original_risk": risk_shares_from_matrix(component_matrix),
        }
    perf_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    exceed_rows: list[dict[str, Any]] = []
    for rep, idx in enumerate(draws["full_state_sample"]):
        for _, scen in scenarios.iterrows():
            key = (str(scen["rule_id"]), str(scen["portfolio_family"]))
            arrays = scenario_arrays[key]
            metrics = performance_metrics(pd.Series(arrays["portfolio_returns"][idx]))
            for metric in ["annualized_mean_arithmetic", "annualized_volatility", "cvar_95_weekly", "max_drawdown", "positive_week_share"]:
                perf_rows.append(
                    {
                        "rule_id": key[0],
                        "portfolio_family": key[1],
                        "metric": metric,
                        "rep": rep,
                        "value": metrics[metric],
                        "original_estimate": original_perf[key][metric],
                    }
                )
            risk = risk_shares_from_matrix(arrays["component_matrix"][idx, :])
            orig_risk = arrays["original_risk"]
            for metric in ["btc_component_share_vol", "btc_component_share_cvar"]:
                risk_rows.append(
                    {
                        "rule_id": key[0],
                        "portfolio_family": key[1],
                        "metric": metric,
                        "rep": rep,
                        "value": risk[metric],
                        "original_estimate": orig_risk[metric],
                    }
                )
            exceed_rows.append(
                {
                    "rule_id": key[0],
                    "portfolio_family": key[1],
                    "rep": rep,
                    "vol_cap_exceeded": bool(risk["btc_component_share_vol"] > RISK_BUDGET_CAP),
                    "cvar_cap_exceeded": bool(risk["btc_component_share_cvar"] > RISK_BUDGET_CAP),
                    "any_cap_exceeded": bool(
                        risk["btc_component_share_vol"] > RISK_BUDGET_CAP or risk["btc_component_share_cvar"] > RISK_BUDGET_CAP
                    ),
                }
            )
    perf_ci = ci_summary(perf_rows, ["rule_id", "portfolio_family", "metric"])
    risk_ci = ci_summary(risk_rows, ["rule_id", "portfolio_family", "metric"])
    exceed = pd.DataFrame(exceed_rows).groupby(["rule_id", "portfolio_family"]).agg(
        bootstrap_reps=("rep", "count"),
        vol_cap_exceedance_probability=("vol_cap_exceeded", "mean"),
        cvar_cap_exceedance_probability=("cvar_cap_exceeded", "mean"),
        any_cap_exceedance_probability=("any_cap_exceeded", "mean"),
    ).reset_index()
    perf_ci.to_csv(dirs["results"] / "conditional_rule_bootstrap_ci.csv", index=False)
    risk_ci.to_csv(dirs["results"] / "risk_contribution_bootstrap_ci.csv", index=False)
    exceed.to_csv(dirs["results"] / "risk_budget_cap_exceedance_summary.csv", index=False)
    return {"performance_ci": perf_ci, "risk_ci": risk_ci, "exceedance": exceed}


def bootstrap_part5(inputs: dict[str, Any], validation: dict[str, Any], draws: dict[str, np.ndarray], dirs: dict[str, Path]) -> pd.DataFrame:
    frame = validation["part5_main"].sort_values(["scenario_id", "date"]).copy()
    scenarios = frame[["scenario_id", "rule_id", "portfolio_family", "funding_convention", "rebalance_frequency"]].drop_duplicates().sort_values("scenario_id")
    rows: list[dict[str, Any]] = []
    for _, scen in scenarios.iterrows():
        scen_frame = frame[frame["scenario_id"] == scen["scenario_id"]].sort_values("date").reset_index(drop=True)
        original = performance_metrics(scen_frame["net_return"])
        original["total_turnover"] = float(scen_frame["turnover"].sum())
        original["total_transaction_cost"] = float(scen_frame["transaction_cost"].sum())
        for rep, idx in enumerate(draws["part5_implementability_sample"]):
            sample = scen_frame.iloc[idx].reset_index(drop=True)
            metrics = performance_metrics(sample["net_return"])
            metrics["total_turnover"] = float(sample["turnover"].sum())
            metrics["total_transaction_cost"] = float(sample["transaction_cost"].sum())
            for metric in ["annualized_mean_arithmetic", "annualized_volatility", "cvar_95_weekly", "max_drawdown", "total_turnover", "total_transaction_cost"]:
                rows.append(
                    {
                        "scenario_id": scen["scenario_id"],
                        "rule_id": scen["rule_id"],
                        "portfolio_family": scen["portfolio_family"],
                        "funding_convention": scen["funding_convention"],
                        "rebalance_frequency": scen["rebalance_frequency"],
                        "metric": metric,
                        "rep": rep,
                        "value": metrics[metric],
                        "original_estimate": original[metric],
                    }
                )
    ci = ci_summary(rows, ["scenario_id", "rule_id", "portfolio_family", "funding_convention", "rebalance_frequency", "metric"])
    ci.to_csv(dirs["results"] / "implementability_bootstrap_ci.csv", index=False)
    return ci


def bootstrap_part7(validation: dict[str, Any], inputs: dict[str, Any], draws: dict[str, np.ndarray], dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    part7 = validation["part7_main"].sort_values(["scenario_id", "return_date"]).copy()
    scenarios = part7[["scenario_id", "rule_id", "portfolio_family", "funding_convention", "rebalance_frequency"]].drop_duplicates().sort_values("scenario_id")
    p7_rows: list[dict[str, Any]] = []
    diff_rows: list[dict[str, Any]] = []
    part5 = inputs["part5_returns"][
        (inputs["part5_returns"]["signal_timing"] == MAIN_SIGNAL_TIMING)
        & (inputs["part5_returns"]["cost_scenario"] == MAIN_COST_SCENARIO)
        & (inputs["part5_returns"]["date"] >= pd.Timestamp(EXPECTED_PART7_RETURN_START))
    ].copy()
    for _, scen in scenarios.iterrows():
        p7_frame = part7[part7["scenario_id"] == scen["scenario_id"]].sort_values("return_date").reset_index(drop=True)
        p5_frame = part5[
            (part5["rule_id"] == scen["rule_id"])
            & (part5["portfolio_family"] == scen["portfolio_family"])
            & (part5["funding_convention"] == scen["funding_convention"])
            & (part5["rebalance_frequency"] == scen["rebalance_frequency"])
        ].sort_values("date").reset_index(drop=True)
        require(len(p5_frame) == len(p7_frame), f"Part5/Part7 shared sample mismatch for {scen['scenario_id']}")
        original = performance_metrics(p7_frame["net_return"])
        original["average_btc_beginning_weight"] = float(p7_frame["btc_beginning_weight"].mean())
        original["max_btc_beginning_weight"] = float(p7_frame["btc_beginning_weight"].max())
        p5_original = performance_metrics(p5_frame["net_return"])
        for rep, idx in enumerate(draws["part7_realtime_sample"]):
            s7 = p7_frame.iloc[idx].reset_index(drop=True)
            s5 = p5_frame.iloc[idx].reset_index(drop=True)
            metrics7 = performance_metrics(s7["net_return"])
            metrics7["average_btc_beginning_weight"] = float(s7["btc_beginning_weight"].mean())
            metrics7["max_btc_beginning_weight"] = float(s7["btc_beginning_weight"].max())
            metrics5 = performance_metrics(s5["net_return"])
            for metric in [
                "annualized_mean_arithmetic",
                "annualized_volatility",
                "cvar_95_weekly",
                "max_drawdown",
                "average_btc_beginning_weight",
                "max_btc_beginning_weight",
            ]:
                p7_rows.append(
                    {
                        "scenario_id": scen["scenario_id"],
                        "rule_id": scen["rule_id"],
                        "portfolio_family": scen["portfolio_family"],
                        "funding_convention": scen["funding_convention"],
                        "rebalance_frequency": scen["rebalance_frequency"],
                        "metric": metric,
                        "rep": rep,
                        "value": metrics7[metric],
                        "original_estimate": original[metric],
                    }
                )
            for metric in ["annualized_mean_arithmetic", "annualized_volatility", "max_drawdown", "sharpe_annualized_zero_rf"]:
                diff_rows.append(
                    {
                        "rule_id": scen["rule_id"],
                        "portfolio_family": scen["portfolio_family"],
                        "funding_convention": scen["funding_convention"],
                        "rebalance_frequency": scen["rebalance_frequency"],
                        "metric": f"part7_minus_part5_{metric}",
                        "rep": rep,
                        "value": metrics7[metric] - metrics5[metric],
                        "original_estimate": original[metric] - p5_original[metric],
                    }
                )
    p7_ci = ci_summary(p7_rows, ["scenario_id", "rule_id", "portfolio_family", "funding_convention", "rebalance_frequency", "metric"])
    diff_ci = ci_summary(diff_rows, ["rule_id", "portfolio_family", "funding_convention", "rebalance_frequency", "metric"])
    p7_ci.to_csv(dirs["results"] / "realtime_rule_bootstrap_ci.csv", index=False)
    diff_ci.to_csv(dirs["results"] / "realtime_vs_expost_bootstrap_ci.csv", index=False)
    return {"realtime_ci": p7_ci, "difference_ci": diff_ci}


def load_part7_module():
    module_path = Path("experiments/part7_realtime_probabilistic_regime_robustness/run_part7_realtime_probabilistic_regime_robustness.py")
    spec = importlib.util.spec_from_file_location("part7_runner_for_part8", module_path)
    require(spec is not None and spec.loader is not None, "Could not load Part 7 runner module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_refit_dates(panel: pd.DataFrame, initial_rows: int, frequency: str) -> list[pd.Timestamp]:
    initial_end = pd.Timestamp(panel.iloc[initial_rows - 1]["date"])
    if frequency == "monthly":
        endpoints = panel.groupby(panel["date"].dt.to_period("M")).tail(1)["date"]
    elif frequency == "quarterly":
        endpoints = panel.groupby(panel["date"].dt.to_period("Q")).tail(1)["date"]
    else:
        raise ValueError(f"Unknown refit frequency: {frequency}")
    return [initial_end] + [pd.Timestamp(date) for date in endpoints if pd.Timestamp(date) > initial_end]


def run_realtime_variant(part7mod, panel: pd.DataFrame, raw_cols: list[str], z_cols: list[str], initial_rows: int, frequency: str, seed: int) -> pd.DataFrame:
    refit_dates = build_refit_dates(panel, initial_rows, frequency)
    rows: list[dict[str, Any]] = []
    for idx, train_end in enumerate(refit_dates):
        next_end = refit_dates[idx + 1] if idx + 1 < len(refit_dates) else pd.Timestamp(panel["date"].max())
        train = panel[panel["date"] <= train_end].reset_index(drop=True)
        future = panel[(panel["date"] > train_end) & (panel["date"] <= next_end)].reset_index(drop=True)
        if future.empty:
            continue
        train_raw = train[raw_cols].to_numpy(dtype=float)
        train_z, _, scaler_mean, scaler_std = part7mod.realtime_standardize(train_raw)
        train_z_frame = pd.DataFrame(train_z, columns=z_cols)
        stress = part7mod.stress_composite_from_z(train_z_frame)
        pca = part7mod.fit_pca(train_z, 5)
        hmm_raw = part7mod.fit_diag_gaussian_hmm(pca["scores"], 4, seed=seed + idx * 104729)
        hmm, _ = part7mod.reorder_hmm_by_stress(hmm_raw, stress)
        posterior = hmm.gamma[-1].copy()
        for _, obs in future.iterrows():
            current_raw = obs[raw_cols].to_numpy(dtype=float).reshape(1, -1)
            current_z = (current_raw - scaler_mean) / scaler_std
            score = part7mod.pca_transform(current_z, pca).reshape(-1)
            posterior = part7mod.filtered_update(posterior, score, hmm)
            map_id = int(posterior.argmax())
            max_prob = float(posterior.max())
            entropy = -float(np.sum(np.where(posterior > 0, posterior * np.log(posterior), 0.0))) / math.log(4)
            row = {
                "date": pd.Timestamp(obs["date"]),
                "training_end_date": train_end,
                "realtime_map_state": f"state_{map_id}",
                "realtime_map_state_id": map_id,
                "realtime_max_probability": max_prob,
                "realtime_normalized_entropy": entropy,
                "part1_hmm4_state": obs["hmm4_state"],
                "part1_hmm4_state_id": int(obs["hmm4_state_id"]),
            }
            for state_id in range(4):
                row[f"realtime_prob_state_{state_id}"] = float(posterior[state_id])
            rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def base_weights(inputs: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for family, frame in inputs["part2_baseline_weights"].groupby("portfolio_family"):
        out[family] = {row["asset"]: float(row["weight"]) for _, row in frame.iterrows() if row["asset"] in BASE_ASSETS}
    return out


def rule_state_weights(inputs: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for rule_id in IMPLEMENTED_RULE_IDS:
        frame = inputs["part4_rule_definition"][(inputs["part4_rule_definition"]["rule_id"] == rule_id) & (inputs["part4_rule_definition"]["portfolio_family"] == "all_weather")]
        out[rule_id] = {row["hmm4_state"]: float(row["selected_btc_weight"]) for _, row in frame.iterrows()}
    return out


def target_return_for_candidate(row: pd.Series, btc_weight: float, family_weights: dict[str, float]) -> float:
    weights = {"ret_btc": btc_weight, "ret_bil": 0.0}
    for asset, base_weight in family_weights.items():
        weights[asset] = base_weight * (1.0 - btc_weight)
    return float(sum(weights[asset] * float(row[asset]) for asset in ASSETS))


def hmm_ensemble(inputs: dict[str, Any], validation: dict[str, Any], dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    part7mod = load_part7_module()
    report = inputs["cleaning_report"]
    raw_cols = list(report["column_mapping"]["state_predictors"].values())
    z_cols = [f"{col}_z" for col in raw_cols]
    panel = validation["panel"].copy()
    variants = [
        {"variant_id": "canonical_part7", "initial_rows": 156, "refit_frequency": "monthly", "seed": 42, "source": "part7_output"},
        {"variant_id": "seed_7", "initial_rows": 156, "refit_frequency": "monthly", "seed": 7, "source": "recomputed"},
        {"variant_id": "seed_21", "initial_rows": 156, "refit_frequency": "monthly", "seed": 21, "source": "recomputed"},
        {"variant_id": "seed_101", "initial_rows": 156, "refit_frequency": "monthly", "seed": 101, "source": "recomputed"},
        {"variant_id": "seed_202", "initial_rows": 156, "refit_frequency": "monthly", "seed": 202, "source": "recomputed"},
        {"variant_id": "window_130", "initial_rows": 130, "refit_frequency": "monthly", "seed": 42, "source": "recomputed"},
        {"variant_id": "window_208", "initial_rows": 208, "refit_frequency": "monthly", "seed": 42, "source": "recomputed"},
        {"variant_id": "quarterly_156", "initial_rows": 156, "refit_frequency": "quarterly", "seed": 42, "source": "recomputed"},
    ]
    variant_rows = []
    agreement_rows = []
    signal_rows = []
    perf_rows = []
    weights_by_rule = rule_state_weights(inputs)
    base = base_weights(inputs)
    for spec in variants:
        if spec["source"] == "part7_output":
            probs = inputs["part7_probabilities"].copy()
        else:
            probs = run_realtime_variant(part7mod, panel, raw_cols, z_cols, spec["initial_rows"], spec["refit_frequency"], spec["seed"])
        variant_rows.append({**spec, "probability_rows": int(len(probs)), "start": date_string(probs["date"], "min"), "end": date_string(probs["date"], "max")})
        match = probs["realtime_map_state"].eq(probs["part1_hmm4_state"])
        counts = probs["realtime_map_state"].value_counts().to_dict()
        agreement_rows.append(
            {
                "variant_id": spec["variant_id"],
                "rows": int(len(probs)),
                "overall_agreement_with_part1": float(match.mean()),
                "average_max_probability": float(probs["realtime_max_probability"].mean()),
                "average_normalized_entropy": float(probs["realtime_normalized_entropy"].mean()),
                "state_0_count": int(counts.get("state_0", 0)),
                "state_1_count": int(counts.get("state_1", 0)),
                "state_2_count": int(counts.get("state_2", 0)),
                "state_3_count": int(counts.get("state_3", 0)),
            }
        )
        next_date = {date: panel["date"].iloc[i + 1] for i, date in enumerate(panel["date"].iloc[:-1])}
        panel_by_date = panel.set_index("date")
        for rule_id in IMPLEMENTED_RULE_IDS:
            candidate = probs.apply(
                lambda row: sum(float(row[f"realtime_prob_state_{i}"]) * weights_by_rule[rule_id][f"state_{i}"] for i in range(4)),
                axis=1,
            )
            signal_rows.append(
                {
                    "variant_id": spec["variant_id"],
                    "rule_id": rule_id,
                    "rows": int(len(candidate)),
                    "average_candidate_btc_weight": float(candidate.mean()),
                    "max_candidate_btc_weight": float(candidate.max()),
                    "active_candidate_share": float((candidate > 1e-10).mean()),
                }
            )
            for family in PORTFOLIO_FAMILIES:
                returns = []
                for (_, row), btc_weight in zip(probs.iterrows(), candidate):
                    date = pd.Timestamp(row["date"])
                    if date not in next_date:
                        continue
                    rrow = panel_by_date.loc[next_date[date]]
                    returns.append(target_return_for_candidate(rrow, float(btc_weight), base[family]))
                metrics = performance_metrics(pd.Series(returns))
                perf_rows.append(
                    {
                        "variant_id": spec["variant_id"],
                        "rule_id": rule_id,
                        "portfolio_family": family,
                        "rows": int(len(returns)),
                        "average_candidate_btc_weight": float(candidate.mean()),
                        "annualized_mean_arithmetic": metrics["annualized_mean_arithmetic"],
                        "annualized_volatility": metrics["annualized_volatility"],
                        "max_drawdown": metrics["max_drawdown"],
                        "sharpe_annualized_zero_rf": metrics["sharpe_annualized_zero_rf"],
                    }
                )
    variants_df = pd.DataFrame(variant_rows)
    agreement = pd.DataFrame(agreement_rows)
    signal = pd.DataFrame(signal_rows)
    perf = pd.DataFrame(perf_rows)
    variants_df.to_csv(dirs["results"] / "hmm_ensemble_variant_dictionary.csv", index=False)
    agreement.to_csv(dirs["results"] / "hmm_ensemble_state_agreement.csv", index=False)
    signal.to_csv(dirs["results"] / "hmm_ensemble_rule_signal_sensitivity.csv", index=False)
    perf.to_csv(dirs["results"] / "hmm_ensemble_rule_performance_sensitivity.csv", index=False)
    return {"variants": variants_df, "agreement": agreement, "signals": signal, "performance": perf}


def uncertainty_decision_matrix(outputs: dict[str, Any], dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    state_ci = outputs["state_btc"]["ci"]
    state2 = state_ci[(state_ci["state"] == "state_2") & (state_ci["metric"] == "mean_weekly")]
    rows.append(
        {
            "evidence_area": "state_2_btc_performance",
            "diagnostic": "small_sample_bootstrap_uncertainty",
            "status": "caution",
            "note": "State 2 remains small and should not drive the main allocation rule.",
            "key_value": float(state2["bootstrap_std"].iloc[0]) if len(state2) else float("nan"),
        }
    )
    exceed = outputs["part4"]["exceedance"]
    rows.append(
        {
            "evidence_area": "risk_budget_cap",
            "diagnostic": "bootstrap_exceedance_probability",
            "status": "pass" if float(exceed["any_cap_exceedance_probability"].max()) < 0.50 else "caution",
            "note": "Risk cap exceedance probability is reported rather than hidden.",
            "key_value": float(exceed["any_cap_exceedance_probability"].max()),
        }
    )
    ensemble = outputs["ensemble"]["agreement"]
    rows.append(
        {
            "evidence_area": "part7_state_drift",
            "diagnostic": "hmm_ensemble_agreement_range",
            "status": "caution",
            "note": "Pseudo real-time state labels are treated as uncertain and not as stable forecasts.",
            "key_value": float(ensemble["overall_agreement_with_part1"].min()),
        }
    )
    matrix = pd.DataFrame(rows)
    matrix.to_csv(dirs["results"] / "uncertainty_decision_matrix.csv", index=False)
    return matrix


def write_audits(args: argparse.Namespace, inputs: dict[str, Any], validation: dict[str, Any], dirs: dict[str, Path]) -> None:
    methodology = f"""# Part 8 Methodology Audit

Part 8 quantifies uncertainty around the Part 3, Part 4/5, and Part 7 evidence chain. It does not search for a new BTC allocation rule and does not mutate upstream outputs.

The main uncertainty method is circular moving block bootstrap with `{args.bootstrap_reps}` replications and block length `{args.block_length}` weeks. The same draw index is used within each sample domain so cross-asset and cross-strategy shocks remain aligned.

The HMM ensemble is a sensitivity audit for Part 7 state drift. It changes seeds, initial windows, and refit frequency, then reports state agreement and rule-signal sensitivity. It does not replace the canonical Part 7 result.
"""
    (dirs["results"] / "methodology_audit.md").write_text(methodology, encoding="utf-8")
    model_assumptions = {
        "experiment_role": "uncertainty_quantification",
        "bootstrap_method": "circular_moving_block_bootstrap",
        "bootstrap_reps": args.bootstrap_reps,
        "block_length_weeks": args.block_length,
        "ci_levels": [0.90, 0.95],
        "state_metric_min_n": MIN_STATE_N,
        "cvar_min_tail_count": MIN_CVAR_TAIL_COUNT,
        "hmm_ensemble_role": "sensitivity_audit_not_strategy_search",
        "upstream_outputs_modified": False,
    }
    write_json(dirs["results"] / "model_assumption_audit.json", model_assumptions)
    lineage = [{"input_name": name, "path": str(path), "sha256": file_sha256(path), "usage": lineage_usage(name)} for name, path in inputs["paths"].items()]
    pd.DataFrame(lineage).to_csv(dirs["results"] / "data_lineage.csv", index=False)


def lineage_usage(name: str) -> str:
    if name in {"asset", "state", "cleaning_report"}:
        return "cleaned_input"
    if "part7" in name:
        return "pseudo_real_time_uncertainty_input"
    if "part5" in name:
        return "implementability_uncertainty_input"
    if "part4" in name:
        return "conditional_rule_uncertainty_input"
    if "part3" in name:
        return "state_dependence_uncertainty_reference"
    return "lineage_context"


def plot_outputs(outputs: dict[str, Any], dirs: dict[str, Path]) -> None:
    state = outputs["state_btc"]["ci"]
    fig, ax = plt.subplots(figsize=(9, 5))
    subset = state[state["metric"] == "mean_weekly"].sort_values("state")
    x = np.arange(len(subset))
    ax.errorbar(x, subset["original_estimate"], yerr=[subset["original_estimate"] - subset["ci95_lower"], subset["ci95_upper"] - subset["original_estimate"]], fmt="o")
    ax.set_xticks(x, subset["state"])
    ax.set_title("BTC State Mean Return Bootstrap 95% CI")
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "state_btc_bootstrap_ci.png", dpi=160)
    plt.close(fig)

    rule = outputs["part4"]["performance_ci"]
    subset = rule[(rule["rule_id"] == "main_executed") & (rule["metric"] == "annualized_mean_arithmetic")].sort_values("portfolio_family")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(subset))
    ax.errorbar(x, subset["original_estimate"], yerr=[subset["original_estimate"] - subset["ci95_lower"], subset["ci95_upper"] - subset["original_estimate"]], fmt="o")
    ax.set_xticks(x, subset["portfolio_family"])
    ax.set_title("Conditional Rule Annualized Mean Bootstrap 95% CI")
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "rule_performance_bootstrap_ci.png", dpi=160)
    plt.close(fig)

    risk = outputs["part4"]["risk_ci"]
    subset = risk[(risk["rule_id"] == "main_executed") & (risk["metric"] == "btc_component_share_vol")].sort_values("portfolio_family")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(subset))
    ax.errorbar(x, subset["original_estimate"], yerr=[subset["original_estimate"] - subset["ci95_lower"], subset["ci95_upper"] - subset["original_estimate"]], fmt="o")
    ax.axhline(RISK_BUDGET_CAP, color="red", linestyle="--", linewidth=1)
    ax.set_xticks(x, subset["portfolio_family"])
    ax.set_title("BTC Volatility Risk Contribution Bootstrap 95% CI")
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "risk_contribution_bootstrap_ci.png", dpi=160)
    plt.close(fig)

    diff = outputs["part7"]["difference_ci"]
    subset = diff[(diff["rule_id"] == "main_executed") & (diff["metric"] == "part7_minus_part5_annualized_mean_arithmetic")].sort_values(["portfolio_family", "funding_convention", "rebalance_frequency"])
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = subset["portfolio_family"] + "\n" + subset["funding_convention"] + "\n" + subset["rebalance_frequency"]
    x = np.arange(len(subset))
    ax.errorbar(x, subset["original_estimate"], yerr=[subset["original_estimate"] - subset["ci95_lower"], subset["ci95_upper"] - subset["original_estimate"]], fmt="o")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
    ax.set_title("Part 7 minus Part 5 Annualized Mean Bootstrap 95% CI")
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "realtime_vs_expost_bootstrap_ci.png", dpi=160)
    plt.close(fig)

    ensemble = outputs["ensemble"]["agreement"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(ensemble["variant_id"], ensemble["overall_agreement_with_part1"])
    ax.set_title("HMM Ensemble Agreement with Part 1 Labels")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "hmm_ensemble_agreement.png", dpi=160)
    plt.close(fig)

    small = outputs["state_btc"]["small_sample"].sort_values("state")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(small["state"], small["state_n_p05"], label="5th percentile state n")
    ax.plot(small["state"], small["state_n_mean"], marker="o", color="black", label="mean state n")
    ax.axhline(MIN_STATE_N, color="red", linestyle="--", linewidth=1)
    ax.set_title("Bootstrap State Sample Size Audit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "small_sample_state_uncertainty.png", dpi=160)
    plt.close(fig)


def validate_outputs(dirs: dict[str, Path], outputs: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result_checks = []
    for filename in REQUIRED_RESULTS:
        path = dirs["results"] / filename
        if filename == "output_validation_summary.json":
            result_checks.append({"file": filename, "exists": True, "nonempty": True})
        else:
            result_checks.append({"file": filename, "exists": path.exists(), "nonempty": path.exists() and path.stat().st_size > 0})
    figure_checks = []
    for filename in REQUIRED_FIGURES:
        path = dirs["figures"] / filename
        readable = False
        if path.exists() and path.stat().st_size > 0:
            try:
                plt.imread(path)
                readable = True
            except Exception:
                readable = False
        figure_checks.append({"file": filename, "exists": path.exists(), "nonempty": path.exists() and path.stat().st_size > 0, "readable": readable})
    checks = {
        "required_results_ok": all(row["exists"] and row["nonempty"] for row in result_checks if row["file"] != "output_validation_summary.json"),
        "required_figures_ok": all(row["exists"] and row["nonempty"] and row["readable"] for row in figure_checks),
        "bootstrap_reps_ok": bool(args.bootstrap_reps == 2000),
        "block_length_ok": bool(args.block_length == 13),
        "hmm_ensemble_variants_ok": bool(len(outputs["ensemble"]["variants"]) == 8),
        "state2_small_sample_recorded": bool("state_2" in set(outputs["state_btc"]["small_sample"]["state"])),
        "risk_cap_exceedance_recorded": bool(len(outputs["part4"]["exceedance"]) > 0),
    }
    status = "passed" if all(checks.values()) else "failed"
    payload = {"status": status, "checks": checks, "required_result_checks": result_checks, "required_figure_checks": figure_checks}
    write_json(dirs["results"] / "output_validation_summary.json", payload)
    require(status == "passed", "Output validation failed")
    return payload


def write_manifest(args: argparse.Namespace, dirs: dict[str, Path], validation: dict[str, Any], outputs: dict[str, Any], output_validation: dict[str, Any]) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": args.run_id or dirs["root"].name,
        "objective": "Part 8 bootstrap uncertainty quantification and HMM ensemble drift audit",
        "input_dir": str(args.input_dir),
        "part1_run_dir": str(args.part1_run_dir),
        "part2_run_dir": str(args.part2_run_dir),
        "part3_run_dir": str(args.part3_run_dir),
        "part4_run_dir": str(args.part4_run_dir),
        "part5_run_dir": str(args.part5_run_dir),
        "part6_run_dir": str(args.part6_run_dir),
        "part7_run_dir": str(args.part7_run_dir),
        "output_dir": str(args.output_dir),
        "run_dir": str(dirs["root"]),
        "random_seed": args.seed,
        "input_hashes": validation["input_hashes"],
        "package_versions": package_versions(),
        "parameters": {
            "bootstrap_method": "circular_moving_block_bootstrap",
            "bootstrap_reps": args.bootstrap_reps,
            "block_length": args.block_length,
            "ci_levels": [0.90, 0.95],
            "hmm_ensemble_variants": int(len(outputs["ensemble"]["variants"])),
        },
        "samples": validation["summary"]["samples"],
        "output_validation": output_validation,
        "outputs": {"results": REQUIRED_RESULTS, "figures": REQUIRED_FIGURES},
        "scope_notes": [
            "Uncertainty quantification only; no new allocation rule.",
            "HMM ensemble is a drift audit and does not replace Part 7 canonical results.",
            "Bootstrap uses historical resampling and does not establish causality.",
        ],
    }
    write_json(dirs["root"] / "run_manifest.json", manifest)


def main() -> None:
    args = parse_args()
    run_id = args.run_id or now_run_id()
    dirs = ensure_dirs(args.output_dir / run_id)
    setup_logging(dirs["logs"])
    logging.info("Starting Part 8 run_id=%s", run_id)

    inputs = load_inputs(args)
    validation = load_or_run(dirs, "01_input_validation", args.resume, lambda: validate_inputs(inputs, args, dirs))
    enforce_resume_input_hashes(args, dirs, validation["input_hashes"])
    draws = load_or_run(dirs, "02_bootstrap_draws", args.resume, lambda: build_bootstrap_draws(validation, args, dirs))
    state_btc = load_or_run(dirs, "03_state_btc_bootstrap", args.resume, lambda: bootstrap_state_btc(validation, draws, dirs))
    state_beta = load_or_run(dirs, "04_state_beta_bootstrap", args.resume, lambda: bootstrap_state_beta(validation, draws, dirs))
    part4 = load_or_run(dirs, "05_part4_bootstrap", args.resume, lambda: bootstrap_part4(inputs, draws, dirs))
    part5 = load_or_run(dirs, "06_part5_bootstrap", args.resume, lambda: bootstrap_part5(inputs, validation, draws, dirs))
    part7 = load_or_run(dirs, "07_part7_bootstrap", args.resume, lambda: bootstrap_part7(validation, inputs, draws, dirs))
    ensemble = load_or_run(dirs, "08_hmm_ensemble", args.resume, lambda: hmm_ensemble(inputs, validation, dirs))
    outputs = {"state_btc": state_btc, "state_beta": state_beta, "part4": part4, "part5": part5, "part7": part7, "ensemble": ensemble}
    outputs["decision_matrix"] = uncertainty_decision_matrix(outputs, dirs)
    write_audits(args, inputs, validation, dirs)
    plot_outputs(outputs, dirs)
    output_validation = validate_outputs(dirs, outputs, args)
    write_manifest(args, dirs, validation, outputs, output_validation)
    logging.info("Part 8 completed successfully: %s", dirs["root"])


if __name__ == "__main__":
    main()
