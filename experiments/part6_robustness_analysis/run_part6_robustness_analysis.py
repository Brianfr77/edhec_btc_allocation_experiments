#!/usr/bin/env python3
"""Part 6 experiment runner: full robustness analysis."""

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
EXPECTED_ROBUSTNESS_ROWS = 430
EXPECTED_STATE_START = "2018-02-09"
EXPECTED_STATE_END = "2026-03-27"
EXPECTED_STATE_ROWS = 425
EXPECTED_LAGGED_START = "2018-02-16"
EXPECTED_LAGGED_ROWS = 424
EXPECTED_STATE_COUNTS = {"state_0": 149, "state_1": 167, "state_2": 30, "state_3": 79}
ETF_RETURN_START = "2024-01-19"
ETF_RETURN_ROWS = 115
TAIL_ALPHA = 0.05
TRADING_WEEKS_PER_YEAR = 52
TRADING_MONTHS_PER_YEAR = 12
FLOAT_TOL = 1e-10
HAC_LAG_WEEKS = 4

BASE_ASSETS = ["ret_spy", "ret_tlt", "ret_ief", "ret_gld", "ret_dbc"]
CANONICAL_ASSETS = ["ret_btc"] + BASE_ASSETS + ["ret_bil"]
BTC_WEIGHT_GRID = [0.0, 0.01, 0.02, 0.03, 0.05]
IMPLEMENTED_RULE_IDS = ["main_executed", "sensitivity_state2_low_executed"]
PORTFOLIO_FAMILIES = ["all_weather", "erc"]
FUNDING_CONVENTIONS = ["pro_rata_base", "bil_sleeve"]
REBALANCE_FREQUENCIES = ["monthly", "quarterly"]
MAIN_SIGNAL_TIMING = "lagged_one_week"
MAIN_COST_SCENARIO = "moderate_cost"
COST_RATES = {"ret_btc": 0.0025, "etf_and_cash": 0.0005}

REQUIRED_RESULTS = [
    "input_validation_summary.json",
    "data_availability_and_missingness.csv",
    "robustness_variant_dictionary.csv",
    "btc_source_tracking_summary.csv",
    "btc_source_replacement_diagnostics.csv",
    "cash_proxy_bil_shy_summary.csv",
    "fixed_btc_risk_budget_robustness.csv",
    "conditional_rule_robustness_summary.csv",
    "implementability_robustness_summary.csv",
    "monthly_frequency_robustness_summary.csv",
    "monthly_state_conditioned_btc_summary.csv",
    "etf_post_tracking_summary.csv",
    "etf_post_implementation_observation.csv",
    "credit_proxy_overlap_summary.csv",
    "credit_proxy_beta_robustness.csv",
    "robustness_decision_matrix.csv",
    "methodology_audit.md",
    "data_lineage.csv",
    "model_assumption_audit.json",
    "output_validation_summary.json",
]
REQUIRED_FIGURES = [
    "btc_source_tracking.png",
    "cash_proxy_bil_shy.png",
    "fixed_btc_robustness_summary.png",
    "conditional_rule_robustness_summary.png",
    "monthly_vs_weekly_robustness.png",
    "etf_post_tracking.png",
    "credit_proxy_overlap.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 6 full robustness diagnostics.")
    parser.add_argument("--input-dir", default="data_2026/cleaned", type=Path)
    parser.add_argument("--part1-run-dir", default="outputs/part1_btc_macro_state/colab_part1_seed42", type=Path)
    parser.add_argument("--part2-run-dir", default="outputs/part2_portfolio_risk_budget/colab_part2_seed42", type=Path)
    parser.add_argument("--part3-run-dir", default="outputs/part3_btc_state_dependence/colab_part3_seed42", type=Path)
    parser.add_argument("--part4-run-dir", default="outputs/part4_conditional_btc_allocation/colab_part4_seed42", type=Path)
    parser.add_argument("--part5-run-dir", default="outputs/part5_implementability_rebalancing/colab_part5_seed42", type=Path)
    parser.add_argument("--output-dir", default="outputs/part6_robustness_analysis", type=Path)
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


def var_cvar(returns: pd.Series, alpha: float = TAIL_ALPHA) -> tuple[float, float, int]:
    clean = returns.dropna()
    if clean.empty:
        return float("nan"), float("nan"), 0
    var = float(clean.quantile(alpha))
    tail = clean[clean <= var]
    return var, float(tail.mean()), int(len(tail))


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    if returns.dropna().empty:
        return float("nan")
    return float(drawdown_series(returns).min())


def performance_metrics(returns: pd.Series, prefix: str = "", periods_per_year: int = TRADING_WEEKS_PER_YEAR) -> dict[str, Any]:
    clean = returns.dropna()
    var, cvar, tail_count = var_cvar(clean)
    vol = float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")
    mean = float(clean.mean()) if len(clean) else float("nan")
    sharpe = mean / vol * math.sqrt(periods_per_year) if vol and vol > 0 else float("nan")
    return {
        f"{prefix}count": int(len(clean)),
        f"{prefix}mean": mean,
        f"{prefix}median": float(clean.median()) if len(clean) else float("nan"),
        f"{prefix}volatility": vol,
        f"{prefix}annualized_mean_arithmetic": mean * periods_per_year if len(clean) else float("nan"),
        f"{prefix}annualized_volatility": vol * math.sqrt(periods_per_year) if np.isfinite(vol) else float("nan"),
        f"{prefix}var_95": var,
        f"{prefix}cvar_95": cvar,
        f"{prefix}tail_scenario_count": tail_count,
        f"{prefix}max_drawdown": max_drawdown(clean),
        f"{prefix}positive_period_share": float((clean > 0).mean()) if len(clean) else float("nan"),
        f"{prefix}sharpe_annualized_zero_rf": float(sharpe),
    }


def correlation(a: pd.Series, b: pd.Series) -> float:
    frame = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(frame) < 2:
        return float("nan")
    return float(frame["a"].corr(frame["b"]))


def risk_contributions_vol(returns: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    assets = list(weights.keys())
    w = np.array([weights[asset] for asset in assets], dtype=float)
    sigma = returns[assets].cov().to_numpy(dtype=float)
    portfolio_variance = float(w @ sigma @ w)
    portfolio_vol = math.sqrt(max(portfolio_variance, 0.0))
    if portfolio_vol <= 0:
        component = np.zeros_like(w)
        share = np.zeros_like(w)
    else:
        marginal = sigma @ w / portfolio_vol
        component = w * marginal
        share = component / portfolio_vol
    return pd.DataFrame(
        {
            "asset": assets,
            "weight": w,
            "portfolio_volatility": portfolio_vol,
            "component_contribution_vol": component,
            "component_share_vol": share,
            "share_sum_check": float(share.sum()),
        }
    )


def risk_contributions_cvar(returns: pd.DataFrame, weights: dict[str, float], alpha: float = TAIL_ALPHA) -> pd.DataFrame:
    assets = list(weights.keys())
    w = np.array([weights[asset] for asset in assets], dtype=float)
    asset_returns = returns[assets]
    portfolio_returns = asset_returns.to_numpy(dtype=float) @ w
    loss = -portfolio_returns
    var_loss = float(np.quantile(loss, 1.0 - alpha))
    tail_mask = loss >= var_loss - 1e-15
    tail_count = int(tail_mask.sum())
    require(tail_count > 0, "CVaR tail scenario count is zero")
    cvar_loss = float(loss[tail_mask].mean())
    component = np.array([-w[i] * float(asset_returns.iloc[tail_mask, i].mean()) for i in range(len(assets))])
    share = component / cvar_loss if abs(cvar_loss) > 1e-15 else np.zeros_like(component)
    return pd.DataFrame(
        {
            "asset": assets,
            "weight": w,
            "tail_alpha": alpha,
            "tail_scenario_count": tail_count,
            "portfolio_cvar_loss": cvar_loss,
            "component_contribution_cvar_loss": component,
            "component_share_cvar": share,
            "share_sum_check": float(share.sum()),
        }
    )


def ols_hac_single_predictor(y: pd.Series, x: pd.Series, predictor_name: str, hac_lag: int = HAC_LAG_WEEKS) -> dict[str, Any]:
    frame = pd.DataFrame({"y": y, "x": x}).dropna()
    n_obs = int(len(frame))
    require(n_obs >= 12, f"Too few observations for beta regression on {predictor_name}: {n_obs}")
    y_arr = frame["y"].to_numpy(dtype=float)
    x_arr = frame["x"].to_numpy(dtype=float)
    require(float(np.std(x_arr, ddof=1)) > 0.0, f"Predictor has zero variance: {predictor_name}")
    x_matrix = np.column_stack([np.ones(n_obs), x_arr])
    xtx_inv = np.linalg.pinv(x_matrix.T @ x_matrix)
    coef = xtx_inv @ x_matrix.T @ y_arr
    residuals = y_arr - x_matrix @ coef
    df_resid = n_obs - 2
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
            xl = x_matrix[t - lag : t - lag + 1].T
            gamma += residuals[t] * residuals[t - lag] * (xt @ xl.T)
        meat += weight * (gamma + gamma.T)
    meat *= n_obs / df_resid
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    t_values = coef / se
    p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(t_values), df=df_resid))
    ci_mult = float(stats.t.ppf(0.975, df=df_resid))
    ci_lower = coef - ci_mult * se
    ci_upper = coef + ci_mult * se
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y_arr - y_arr.mean()) ** 2))
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
        "r_squared": float(1.0 - sse / sst) if sst > 0 else float("nan"),
        "df_resid": int(df_resid),
        "hac_lag_weeks": int(hac_lag),
        "effective_hac_lag_weeks": int(effective_lag),
    }
    require(all(np.isfinite(result[key]) for key in ["alpha", "beta", "hac_se_beta", "t_beta", "p_beta"]), f"Non-finite beta output for {predictor_name}")
    return result


def required_input_paths(args: argparse.Namespace) -> dict[str, Path]:
    input_dir = args.input_dir
    return {
        "asset_returns_main_weekly": input_dir / "asset_returns_main_weekly.csv",
        "robustness_weekly_panel": input_dir / "robustness_weekly_panel.csv",
        "state_model_panel_weekly": input_dir / "state_model_panel_weekly.csv",
        "cleaning_report": input_dir / "cleaning_report.json",
        "part1_manifest": args.part1_run_dir / "run_manifest.json",
        "part1_validation_summary": args.part1_run_dir / "results" / "validation_summary.json",
        "hmm4_state_labels": args.part1_run_dir / "results" / "hmm4_state_labels.csv",
        "part2_manifest": args.part2_run_dir / "run_manifest.json",
        "part2_input_validation_summary": args.part2_run_dir / "results" / "input_validation_summary.json",
        "part2_output_validation_summary": args.part2_run_dir / "results" / "output_validation_summary.json",
        "part2_baseline_portfolio_weights": args.part2_run_dir / "results" / "baseline_portfolio_weights.csv",
        "part3_manifest": args.part3_run_dir / "run_manifest.json",
        "part3_input_validation_summary": args.part3_run_dir / "results" / "input_validation_summary.json",
        "part3_output_validation_summary": args.part3_run_dir / "results" / "output_validation_summary.json",
        "part3_state_conditioned_btc_performance": args.part3_run_dir / "results" / "state_conditioned_btc_performance.csv",
        "part4_manifest": args.part4_run_dir / "run_manifest.json",
        "part4_input_validation_summary": args.part4_run_dir / "results" / "input_validation_summary.json",
        "part4_output_validation_summary": args.part4_run_dir / "results" / "output_validation_summary.json",
        "part4_allocation_rule_definition": args.part4_run_dir / "results" / "allocation_rule_definition.csv",
        "part4_risk_budget_cap_audit": args.part4_run_dir / "results" / "risk_budget_cap_audit.csv",
        "part5_manifest": args.part5_run_dir / "run_manifest.json",
        "part5_input_validation_summary": args.part5_run_dir / "results" / "input_validation_summary.json",
        "part5_output_validation_summary": args.part5_run_dir / "results" / "output_validation_summary.json",
        "part5_rebalanced_performance_summary": args.part5_run_dir / "results" / "rebalanced_performance_summary.csv",
    }


def guard_resume_input_hashes(args: argparse.Namespace, dirs: dict[str, Path]) -> None:
    checkpoint = dirs["checkpoints"] / "01_load_inputs.pkl"
    if not args.resume or not checkpoint.exists():
        return
    previous = load_pickle(checkpoint)
    required_paths = required_input_paths(args)
    missing = [name for name, path in required_paths.items() if not path.exists()]
    require(not missing, f"Missing required input files: {missing}")
    hashes = {name: file_sha256(path) for name, path in required_paths.items()}
    mismatches = [
        name
        for name, digest in hashes.items()
        if previous.get("hashes", {}).get(name) != digest
    ]
    require(not mismatches, f"Resume aborted because input hashes changed: {mismatches}")
    logging.info("Resume input hash guard passed for %d files", len(hashes))


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    required_paths = required_input_paths(args)
    missing = [name for name, path in required_paths.items() if not path.exists()]
    require(not missing, f"Missing required input files: {missing}")
    hashes = {name: file_sha256(path) for name, path in required_paths.items()}

    return {
        "paths": required_paths,
        "hashes": hashes,
        "asset": pd.read_csv(required_paths["asset_returns_main_weekly"], parse_dates=["date"]),
        "robustness": pd.read_csv(required_paths["robustness_weekly_panel"], parse_dates=["date"]),
        "state_model": pd.read_csv(required_paths["state_model_panel_weekly"], parse_dates=["date"]),
        "cleaning_report": read_json(required_paths["cleaning_report"]),
        "labels": pd.read_csv(required_paths["hmm4_state_labels"], parse_dates=["date"]),
        "part1_manifest": read_json(required_paths["part1_manifest"]),
        "part1_validation": read_json(required_paths["part1_validation_summary"]),
        "part2_manifest": read_json(required_paths["part2_manifest"]),
        "part2_input_validation": read_json(required_paths["part2_input_validation_summary"]),
        "part2_output_validation": read_json(required_paths["part2_output_validation_summary"]),
        "part2_baseline_weights": pd.read_csv(required_paths["part2_baseline_portfolio_weights"]),
        "part3_manifest": read_json(required_paths["part3_manifest"]),
        "part3_input_validation": read_json(required_paths["part3_input_validation_summary"]),
        "part3_output_validation": read_json(required_paths["part3_output_validation_summary"]),
        "part3_state_btc_performance": pd.read_csv(required_paths["part3_state_conditioned_btc_performance"]),
        "part4_manifest": read_json(required_paths["part4_manifest"]),
        "part4_input_validation": read_json(required_paths["part4_input_validation_summary"]),
        "part4_output_validation": read_json(required_paths["part4_output_validation_summary"]),
        "part4_rule_definition": pd.read_csv(required_paths["part4_allocation_rule_definition"]),
        "part4_cap_audit": pd.read_csv(required_paths["part4_risk_budget_cap_audit"]),
        "part5_manifest": read_json(required_paths["part5_manifest"]),
        "part5_input_validation": read_json(required_paths["part5_input_validation_summary"]),
        "part5_output_validation": read_json(required_paths["part5_output_validation_summary"]),
        "part5_performance": pd.read_csv(required_paths["part5_rebalanced_performance_summary"]),
    }


def validate_inputs(inputs: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    asset = inputs["asset"].copy()
    robust = inputs["robustness"].copy()
    state_model = inputs["state_model"].copy()
    labels = inputs["labels"].copy()
    hashes = inputs["hashes"]

    require(inputs["part1_validation"].get("status") == "passed", "Part 1 validation did not pass")
    require(inputs["part1_manifest"].get("model_diagnostics", {}).get("hmm4_converged") is True, "Part 1 HMM-4 did not converge")
    for part in ["part2", "part3", "part4", "part5"]:
        require(inputs[f"{part}_input_validation"].get("status") == "passed", f"{part} input validation did not pass")
        require(inputs[f"{part}_output_validation"].get("status") == "passed", f"{part} output validation did not pass")

    require(len(asset) == EXPECTED_ASSET_ROWS, f"Unexpected asset row count: {len(asset)}")
    require(date_string(asset["date"], "min") == EXPECTED_ASSET_START, "Unexpected asset start date")
    require(date_string(asset["date"], "max") == EXPECTED_ASSET_END, "Unexpected asset end date")
    require(asset["date"].dt.dayofweek.eq(4).all(), "Asset dates are not all Fridays")
    require(asset[CANONICAL_ASSETS].isna().sum().sum() == 0, "Missing required main asset returns")

    require(len(robust) == EXPECTED_ROBUSTNESS_ROWS, f"Unexpected robustness row count: {len(robust)}")
    require(robust["date"].dt.dayofweek.eq(4).all(), "Robustness dates are not all Fridays")

    require(len(state_model) == EXPECTED_STATE_ROWS, f"Unexpected state model rows: {len(state_model)}")
    require(date_string(state_model["date"], "min") == EXPECTED_STATE_START, "Unexpected state model start date")
    require(date_string(state_model["date"], "max") == EXPECTED_STATE_END, "Unexpected state model end date")

    require(len(labels) == EXPECTED_STATE_ROWS, f"Unexpected HMM label rows: {len(labels)}")
    require(date_string(labels["date"], "min") == EXPECTED_STATE_START, "Unexpected HMM label start date")
    require(date_string(labels["date"], "max") == EXPECTED_STATE_END, "Unexpected HMM label end date")
    state_counts = labels["hmm4_state"].value_counts().sort_index().to_dict()
    require(state_counts == EXPECTED_STATE_COUNTS, f"Unexpected HMM-4 state counts: {state_counts}")

    part1_hashes = inputs["part1_manifest"].get("input_hashes", {})
    part2_hashes = inputs["part2_manifest"].get("input_hashes", {})
    part3_hashes = inputs["part3_manifest"].get("input_hashes", {})
    part4_hashes = inputs["part4_manifest"].get("input_hashes", {})
    part5_hashes = inputs["part5_manifest"].get("input_hashes", {})
    require(hashes["asset_returns_main_weekly"] == part1_hashes.get("asset_returns_main_weekly"), "Asset hash does not match Part 1")
    require(hashes["robustness_weekly_panel"] == part1_hashes.get("robustness_weekly_panel"), "Robustness hash does not match Part 1")
    require(hashes["state_model_panel_weekly"] == part1_hashes.get("state_model_panel_weekly"), "State panel hash does not match Part 1")
    require(hashes["cleaning_report"] == part1_hashes.get("cleaning_report"), "Cleaning report hash does not match Part 1")
    require(hashes["hmm4_state_labels"] == part2_hashes.get("hmm4_state_labels"), "HMM labels hash does not match Part 2")
    require(hashes["hmm4_state_labels"] == part3_hashes.get("hmm4_state_labels"), "HMM labels hash does not match Part 3")
    require(hashes["hmm4_state_labels"] == part4_hashes.get("hmm4_state_labels"), "HMM labels hash does not match Part 4")
    require(hashes["hmm4_state_labels"] == part5_hashes.get("hmm4_state_labels"), "HMM labels hash does not match Part 5")
    require(hashes["part4_allocation_rule_definition"] == part5_hashes.get("part4_allocation_rule_definition"), "Part 4 rule hash does not match Part 5 lineage")
    require(hashes["part4_risk_budget_cap_audit"] == part5_hashes.get("part4_risk_budget_cap_audit"), "Part 4 cap audit hash does not match Part 5 lineage")

    panel = asset.merge(labels[["date", "hmm4_state", "hmm4_state_id"]], on="date", how="inner", validate="one_to_one")
    robust_cols = [
        "date",
        "ret_btc_coinmetrics_usd",
        "ret_shy",
        "ret_ibit",
        "ret_fbtc",
        "macro_hy_oas_lag1",
        "macro_hy_oas_chg_4w_lag1",
    ]
    panel = panel.merge(robust[robust_cols], on="date", how="left", validate="one_to_one")
    panel = panel.merge(state_model[["date", "macro_credit_spread_baa10y_z"]], on="date", how="left", validate="one_to_one")
    panel = panel.sort_values("date").reset_index(drop=True)
    panel["lagged_hmm4_state"] = panel["hmm4_state"].shift(1)
    panel["lagged_hmm4_state_id"] = panel["hmm4_state_id"].shift(1)
    require(len(panel) == EXPECTED_STATE_ROWS, f"State-aligned panel has {len(panel)} rows")
    require(date_string(panel["date"], "min") == EXPECTED_STATE_START, "Unexpected panel start date")
    require(date_string(panel["date"], "max") == EXPECTED_STATE_END, "Unexpected panel end date")
    require(panel[["ret_btc_coinmetrics_usd", "ret_shy"]].isna().sum().sum() == 0, "Coin Metrics BTC or SHY missing in state sample")
    require(panel.dropna(subset=["lagged_hmm4_state"]).shape[0] == EXPECTED_LAGGED_ROWS, "Lagged panel row count mismatch")
    require(date_string(panel.dropna(subset=["lagged_hmm4_state"])["date"], "min") == EXPECTED_LAGGED_START, "Lagged panel start mismatch")

    availability = build_data_availability(panel, robust)
    availability.to_csv(dirs["results"] / "data_availability_and_missingness.csv", index=False)

    validation_summary = {
        "status": "passed",
        "asset_sample": {"rows": int(len(asset)), "start": date_string(asset["date"], "min"), "end": date_string(asset["date"], "max")},
        "state_aligned_sample": {"rows": int(len(panel)), "start": date_string(panel["date"], "min"), "end": date_string(panel["date"], "max")},
        "lagged_main_sample": {"rows": EXPECTED_LAGGED_ROWS, "start": EXPECTED_LAGGED_START, "end": EXPECTED_STATE_END},
        "hmm4_state_counts": {k: int(v) for k, v in state_counts.items()},
        "full_sample_replacements": {"ret_btc_coinmetrics_usd_complete": True, "ret_shy_complete": True},
        "short_sample_limits": {
            "ret_ibit_nonmissing": int(panel["ret_ibit"].notna().sum()),
            "ret_fbtc_nonmissing": int(panel["ret_fbtc"].notna().sum()),
            "macro_hy_oas_lag1_nonmissing": int(panel["macro_hy_oas_lag1"].notna().sum()),
            "macro_hy_oas_chg_4w_lag1_nonmissing": int(panel["macro_hy_oas_chg_4w_lag1"].notna().sum()),
        },
        "input_hashes": hashes,
        "upstream_runs": {
            "part1_run_id": inputs["part1_manifest"].get("run_id"),
            "part2_run_id": inputs["part2_manifest"].get("run_id"),
            "part3_run_id": inputs["part3_manifest"].get("run_id"),
            "part4_run_id": inputs["part4_manifest"].get("run_id"),
            "part5_run_id": inputs["part5_manifest"].get("run_id"),
        },
    }
    write_json(dirs["results"] / "input_validation_summary.json", validation_summary)
    logging.info("Input validation passed")
    return {"validation": validation_summary, "analysis_panel": panel, "availability": availability}


def build_data_availability(panel: pd.DataFrame, robust: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specs = [
        ("ret_btc_coinmetrics_usd", "full_state_sample", "BTC source replacement", True, "full_sample_replacement"),
        ("ret_shy", "full_state_sample", "BIL vs SHY cash proxy replacement", True, "full_sample_replacement"),
        ("ret_ibit", "full_state_sample", "ETF-era BTC proxy", False, "observational_appendix_only"),
        ("ret_fbtc", "full_state_sample", "ETF-era BTC proxy", False, "observational_appendix_only"),
        ("macro_hy_oas_lag1", "full_state_sample", "HY OAS credit proxy level", False, "short_overlap_only"),
        ("macro_hy_oas_chg_4w_lag1", "full_state_sample", "HY OAS credit proxy change", False, "short_overlap_only"),
    ]
    for col, sample_name, usage, required_full, allowed_role in specs:
        frame = panel[["date", col]].copy()
        nonmissing = frame.dropna(subset=[col])
        rows.append(
            {
                "variable": col,
                "sample_name": sample_name,
                "sample_rows": int(len(frame)),
                "nonmissing_rows": int(len(nonmissing)),
                "missing_rows": int(frame[col].isna().sum()),
                "first_nonmissing_date": date_string(nonmissing["date"], "min") if len(nonmissing) else "",
                "last_nonmissing_date": date_string(nonmissing["date"], "max") if len(nonmissing) else "",
                "required_for_full_sample_replacement": bool(required_full),
                "allowed_role": allowed_role,
                "usage": usage,
            }
        )
    for col in ["ret_ibit", "ret_fbtc", "macro_hy_oas_lag1", "macro_hy_oas_chg_4w_lag1"]:
        nonmissing = robust[["date", col]].dropna()
        rows.append(
            {
                "variable": col,
                "sample_name": "raw_robustness_panel",
                "sample_rows": int(len(robust)),
                "nonmissing_rows": int(len(nonmissing)),
                "missing_rows": int(robust[col].isna().sum()),
                "first_nonmissing_date": date_string(nonmissing["date"], "min") if len(nonmissing) else "",
                "last_nonmissing_date": date_string(nonmissing["date"], "max") if len(nonmissing) else "",
                "required_for_full_sample_replacement": False,
                "allowed_role": "availability_audit",
                "usage": "raw panel availability check",
            }
        )
    return pd.DataFrame(rows)


def base_weights(inputs: dict[str, Any]) -> dict[str, dict[str, float]]:
    weights: dict[str, dict[str, float]] = {}
    for family, frame in inputs["part2_baseline_weights"].groupby("portfolio_family"):
        weights[family] = {row["asset"]: float(row["weight"]) for _, row in frame.iterrows()}
    require(set(weights) == set(PORTFOLIO_FAMILIES), "Missing baseline portfolio families")
    return weights


def rule_weights(inputs: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    rules = inputs["part4_rule_definition"].copy()
    rules = rules[rules["rule_id"].isin(IMPLEMENTED_RULE_IDS) & (rules["constraint_stage"] == "risk_budget_executed")]
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for (rule_id, family), frame in rules.groupby(["rule_id", "portfolio_family"]):
        out.setdefault(rule_id, {})[family] = {
            "raw_rule_name": str(frame["raw_rule_name"].iloc[0]),
            "rule_role": str(frame["rule_role"].iloc[0]),
            "weights_by_state": {row["hmm4_state"]: float(row["selected_btc_weight"]) for _, row in frame.iterrows()},
            "max_sleeve": float(frame["selected_btc_weight"].max()),
        }
    require(set(out) == set(IMPLEMENTED_RULE_IDS), "Missing executed rules")
    return out


def fixed_weights(base: dict[str, float], btc_weight: float) -> dict[str, float]:
    weights = {"ret_btc": float(btc_weight)}
    for asset, weight in base.items():
        weights[asset] = float(weight * (1.0 - btc_weight))
    require(abs(sum(weights.values()) - 1.0) <= FLOAT_TOL, "Fixed weights do not sum to 1")
    return weights


def target_weights_for_state(
    base: dict[str, float],
    state_weights: dict[str, float],
    state: str,
    funding_convention: str,
    max_sleeve: float,
) -> dict[str, float]:
    btc_weight = float(state_weights[state])
    weights = {asset: 0.0 for asset in CANONICAL_ASSETS}
    weights["ret_btc"] = btc_weight
    if funding_convention == "pro_rata_base":
        for asset, base_weight in base.items():
            weights[asset] = float(base_weight * (1.0 - btc_weight))
        weights["ret_bil"] = 0.0
    elif funding_convention == "bil_sleeve":
        for asset, base_weight in base.items():
            weights[asset] = float(base_weight * (1.0 - max_sleeve))
        weights["ret_bil"] = float(max_sleeve - btc_weight)
    else:
        raise ValueError(f"Unknown funding convention: {funding_convention}")
    require(abs(sum(weights.values()) - 1.0) <= FLOAT_TOL, f"Target weights do not sum to 1: {weights}")
    require(min(weights.values()) >= -FLOAT_TOL, f"Negative target weight: {weights}")
    return weights


def portfolio_return(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    result = pd.Series(0.0, index=frame.index)
    for asset, weight in weights.items():
        result = result + weight * frame[asset]
    return result


def remap_panel_returns(panel: pd.DataFrame, btc_col: str = "ret_btc", cash_col: str = "ret_bil") -> pd.DataFrame:
    mapped = panel.copy()
    mapped["ret_btc"] = mapped[btc_col]
    mapped["ret_bil"] = mapped[cash_col]
    return mapped


def compute_tracking_summary(panel: pd.DataFrame, left_col: str, right_col: str, comparison: str, sample_role: str) -> dict[str, Any]:
    frame = panel[["date", left_col, right_col]].dropna().copy()
    diff = frame[left_col] - frame[right_col]
    abs_diff = diff.abs()
    max_idx = abs_diff.idxmax()
    row = {
        "comparison": comparison,
        "sample_role": sample_role,
        "left_series": left_col,
        "right_series": right_col,
        "n_overlap": int(len(frame)),
        "start_date": date_string(frame["date"], "min") if len(frame) else "",
        "end_date": date_string(frame["date"], "max") if len(frame) else "",
        "correlation": correlation(frame[left_col], frame[right_col]),
        "mean_difference": float(diff.mean()) if len(frame) else float("nan"),
        "median_difference": float(diff.median()) if len(frame) else float("nan"),
        "std_difference": float(diff.std(ddof=1)) if len(frame) > 1 else float("nan"),
        "mean_absolute_difference": float(abs_diff.mean()) if len(frame) else float("nan"),
        "max_absolute_difference": float(abs_diff.max()) if len(frame) else float("nan"),
        "max_difference_date": pd.Timestamp(frame.loc[max_idx, "date"]).strftime("%Y-%m-%d") if len(frame) else "",
    }
    row.update(performance_metrics(frame[left_col], prefix="left_"))
    row.update(performance_metrics(frame[right_col], prefix="right_"))
    return row


def compute_btc_source_robustness(validation_payload: dict[str, Any], inputs: dict[str, Any], dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    panel = validation_payload["analysis_panel"]
    tracking = pd.DataFrame(
        [
            compute_tracking_summary(
                panel,
                "ret_btc",
                "ret_btc_coinmetrics_usd",
                "BTCUSDT_minus_CoinMetrics",
                "full_state_sample",
            )
        ]
    )
    tracking.to_csv(dirs["results"] / "btc_source_tracking_summary.csv", index=False)

    fixed = compute_fixed_btc_robustness(panel, inputs)
    fixed.to_csv(dirs["results"] / "fixed_btc_risk_budget_robustness.csv", index=False)

    replacements = fixed.groupby(["btc_source", "portfolio_family", "btc_weight"], as_index=False).agg(
        annualized_mean_arithmetic=("annualized_mean_arithmetic", "first"),
        annualized_volatility=("annualized_volatility", "first"),
        cvar_95=("cvar_95", "first"),
        max_drawdown=("max_drawdown", "first"),
        btc_vol_share=("btc_component_share_vol", "first"),
        btc_cvar_share=("btc_component_share_cvar", "first"),
    )
    base = replacements[replacements["btc_source"] == "btc_usdt"]
    coin = replacements[replacements["btc_source"] == "coinmetrics"]
    replacement_diag = coin.merge(base, on=["portfolio_family", "btc_weight"], suffixes=("_coinmetrics", "_btc_usdt"), validate="one_to_one")
    for metric in ["annualized_mean_arithmetic", "annualized_volatility", "cvar_95", "max_drawdown", "btc_vol_share", "btc_cvar_share"]:
        replacement_diag[f"coinmetrics_minus_btc_usdt_{metric}"] = replacement_diag[f"{metric}_coinmetrics"] - replacement_diag[f"{metric}_btc_usdt"]
    replacement_diag.insert(0, "robustness_variant", "btc_source_replacement")
    replacement_diag.to_csv(dirs["results"] / "btc_source_replacement_diagnostics.csv", index=False)

    plot_btc_source_tracking(panel, dirs["figures"] / "btc_source_tracking.png")
    plot_fixed_btc_robustness(fixed, dirs["figures"] / "fixed_btc_robustness_summary.png")
    return {"tracking": tracking, "fixed": fixed, "replacement": replacement_diag}


def compute_fixed_btc_robustness(panel: pd.DataFrame, inputs: dict[str, Any]) -> pd.DataFrame:
    bases = base_weights(inputs)
    rows: list[dict[str, Any]] = []
    source_specs = [
        ("btc_usdt", "ret_btc"),
        ("coinmetrics", "ret_btc_coinmetrics_usd"),
    ]
    for btc_source, btc_col in source_specs:
        mapped = remap_panel_returns(panel, btc_col=btc_col, cash_col="ret_bil")
        for family, base in bases.items():
            for btc_weight in BTC_WEIGHT_GRID:
                weights = fixed_weights(base, btc_weight)
                returns = portfolio_return(mapped, weights)
                vol_contrib = risk_contributions_vol(mapped[list(weights.keys())], weights)
                cvar_contrib = risk_contributions_cvar(mapped[list(weights.keys())], weights)
                btc_vol_share = float(vol_contrib.loc[vol_contrib["asset"] == "ret_btc", "component_share_vol"].iloc[0])
                btc_cvar_share = float(cvar_contrib.loc[cvar_contrib["asset"] == "ret_btc", "component_share_cvar"].iloc[0])
                row = {
                    "btc_source": btc_source,
                    "portfolio_family": family,
                    "btc_weight": btc_weight,
                    "sample_rows": int(len(mapped)),
                    "start_date": date_string(mapped["date"], "min"),
                    "end_date": date_string(mapped["date"], "max"),
                    "btc_component_share_vol": btc_vol_share,
                    "btc_component_share_cvar": btc_cvar_share,
                    "vol_share_sum_check": float(vol_contrib["component_share_vol"].sum()),
                    "cvar_share_sum_check": float(cvar_contrib["component_share_cvar"].sum()),
                    "cvar_tail_scenario_count": int(cvar_contrib["tail_scenario_count"].iloc[0]),
                }
                row.update(performance_metrics(returns))
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["btc_source", "portfolio_family", "btc_weight"]).reset_index(drop=True)


def compute_cash_proxy_robustness(validation_payload: dict[str, Any], inputs: dict[str, Any], dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    panel = validation_payload["analysis_panel"]
    summary = pd.DataFrame(
        [
            compute_tracking_summary(panel, "ret_bil", "ret_shy", "BIL_minus_SHY", "full_state_sample"),
        ]
    )
    summary.to_csv(dirs["results"] / "cash_proxy_bil_shy_summary.csv", index=False)
    plot_cash_proxy(panel, dirs["figures"] / "cash_proxy_bil_shy.png")
    return {"summary": summary}


def compute_conditional_rule_robustness(validation_payload: dict[str, Any], inputs: dict[str, Any], dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    panel = validation_payload["analysis_panel"]
    bases = base_weights(inputs)
    rules = rule_weights(inputs)
    rows: list[dict[str, Any]] = []
    btc_sources = [("btc_usdt", "ret_btc"), ("coinmetrics", "ret_btc_coinmetrics_usd")]
    for btc_source, btc_col in btc_sources:
        mapped = remap_panel_returns(panel, btc_col=btc_col, cash_col="ret_bil")
        for rule_id, family_specs in rules.items():
            for family, spec in family_specs.items():
                returns = []
                btc_weights = []
                for _, row in mapped.iterrows():
                    target = target_weights_for_state(bases[family], spec["weights_by_state"], row["hmm4_state"], "pro_rata_base", spec["max_sleeve"])
                    returns.append(sum(target[asset] * float(row[asset]) for asset in ["ret_btc"] + BASE_ASSETS))
                    btc_weights.append(target["ret_btc"])
                ret_series = pd.Series(returns, index=mapped.index)
                out = {
                    "robustness_variant": "conditional_rule_btc_source",
                    "btc_source": btc_source,
                    "rule_id": rule_id,
                    "raw_rule_name": spec["raw_rule_name"],
                    "rule_role": spec["rule_role"],
                    "portfolio_family": family,
                    "funding_convention": "pro_rata_base_target_weight",
                    "sample_rows": int(len(mapped)),
                    "start_date": date_string(mapped["date"], "min"),
                    "end_date": date_string(mapped["date"], "max"),
                    "average_btc_weight": float(np.mean(btc_weights)),
                    "max_btc_weight": float(np.max(btc_weights)),
                }
                out.update(performance_metrics(ret_series))
                rows.append(out)
    summary = pd.DataFrame(rows).sort_values(["btc_source", "rule_id", "portfolio_family"]).reset_index(drop=True)
    base = summary[summary["btc_source"] == "btc_usdt"][["rule_id", "portfolio_family", "annualized_mean_arithmetic", "annualized_volatility", "cvar_95", "max_drawdown"]]
    coin = summary[summary["btc_source"] == "coinmetrics"]
    merged = coin.merge(base, on=["rule_id", "portfolio_family"], suffixes=("", "_btc_usdt"), validate="one_to_one")
    for metric in ["annualized_mean_arithmetic", "annualized_volatility", "cvar_95", "max_drawdown"]:
        merged[f"coinmetrics_minus_btc_usdt_{metric}"] = merged[metric] - merged[f"{metric}_btc_usdt"]
    summary = summary.merge(
        merged[["rule_id", "portfolio_family"] + [c for c in merged.columns if c.startswith("coinmetrics_minus_btc_usdt_")]],
        on=["rule_id", "portfolio_family"],
        how="left",
    )
    summary.to_csv(dirs["results"] / "conditional_rule_robustness_summary.csv", index=False)
    plot_conditional_rule(summary, dirs["figures"] / "conditional_rule_robustness_summary.png")
    return {"summary": summary}


def cost_rates() -> dict[str, float]:
    return {asset: (COST_RATES["ret_btc"] if asset == "ret_btc" else COST_RATES["etf_and_cash"]) for asset in CANONICAL_ASSETS}


def drift_weights(start_weights: dict[str, float], returns: pd.Series, gross_return: float) -> dict[str, float]:
    denominator = 1.0 + gross_return
    require(abs(denominator) > FLOAT_TOL, "Portfolio denominator is too close to zero")
    return {asset: float(start_weights[asset] * (1.0 + float(returns[asset])) / denominator) for asset in CANONICAL_ASSETS}


def simulate_rebalanced(
    panel: pd.DataFrame,
    base: dict[str, float],
    state_weights: dict[str, float],
    max_sleeve: float,
    funding_convention: str,
    rebalance_frequency: str,
) -> pd.DataFrame:
    data = panel[panel["lagged_hmm4_state"].notna()].copy().reset_index(drop=True)
    require(len(data) > 0, "No lagged signal rows available")
    data["year_month"] = data["date"].dt.to_period("M")
    data["year_quarter"] = data["date"].dt.to_period("Q")
    month_ends = set(data.groupby("year_month")["date"].max())
    quarter_ends = set(data.groupby("year_quarter")["date"].max())
    scheduled_dates = month_ends if rebalance_frequency == "monthly" else quarter_ends
    rates = cost_rates()
    previous_end_weights: dict[str, float] | None = None
    cumulative_gross = 1.0
    cumulative_net = 1.0
    rows: list[dict[str, Any]] = []
    for idx, row in data.iterrows():
        state = str(row["lagged_hmm4_state"])
        target = target_weights_for_state(base, state_weights, state, funding_convention, max_sleeve)
        is_formation = idx == 0
        is_rebalance = is_formation or row["date"] in scheduled_dates
        if previous_end_weights is None:
            beginning = target.copy()
            transaction_cost = 0.0
            turnover = 1.0
            event_type = "formation"
        elif is_rebalance:
            beginning = target.copy()
            transaction_cost = float(sum(abs(beginning[a] - previous_end_weights[a]) * rates[a] for a in CANONICAL_ASSETS))
            turnover = float(sum(abs(beginning[a] - previous_end_weights[a]) for a in CANONICAL_ASSETS))
            event_type = "scheduled_rebalance"
        else:
            beginning = previous_end_weights.copy()
            transaction_cost = 0.0
            turnover = 0.0
            event_type = "hold"
        asset_returns = row[CANONICAL_ASSETS]
        gross_return = float(sum(beginning[a] * float(asset_returns[a]) for a in CANONICAL_ASSETS))
        net_return = gross_return - transaction_cost
        cumulative_gross *= 1.0 + gross_return
        cumulative_net *= 1.0 + net_return
        end_weights = drift_weights(beginning, asset_returns, gross_return)
        rows.append(
            {
                "date": row["date"],
                "hmm4_state": row["hmm4_state"],
                "decision_hmm4_state": state,
                "btc_target_weight": target["ret_btc"],
                "btc_beginning_weight": beginning["ret_btc"],
                "cash_beginning_weight": beginning["ret_bil"],
                "event_type": event_type,
                "is_rebalance_date": bool(is_rebalance),
                "turnover": turnover if is_rebalance else 0.0,
                "transaction_cost": transaction_cost,
                "gross_return": gross_return,
                "net_return": net_return,
                "cumulative_gross_value": cumulative_gross,
                "cumulative_net_value": cumulative_net,
            }
        )
        previous_end_weights = end_weights
    return pd.DataFrame(rows)


def summarize_rebalanced(frame: pd.DataFrame) -> dict[str, Any]:
    out = {
        "start_date": date_string(frame["date"], "min"),
        "end_date": date_string(frame["date"], "max"),
        "average_btc_beginning_weight": float(frame["btc_beginning_weight"].mean()),
        "max_btc_beginning_weight": float(frame["btc_beginning_weight"].max()),
        "average_cash_beginning_weight": float(frame["cash_beginning_weight"].mean()),
        "total_transaction_cost": float(frame["transaction_cost"].sum()),
        "total_turnover_including_formation": float(frame["turnover"].sum()),
        "scheduled_rebalance_count": int((frame["event_type"] == "scheduled_rebalance").sum()),
        "final_cumulative_net_value": float(frame["cumulative_net_value"].iloc[-1]),
    }
    out.update(performance_metrics(frame["net_return"], prefix="net_"))
    return out


def compute_implementability_robustness(validation_payload: dict[str, Any], inputs: dict[str, Any], dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    panel = validation_payload["analysis_panel"]
    bases = base_weights(inputs)
    rules = rule_weights(inputs)
    variant_specs = [
        ("baseline", "ret_btc", "ret_bil", "full_state_sample", True),
        ("coinmetrics_btc", "ret_btc_coinmetrics_usd", "ret_bil", "full_state_sample", True),
        ("shy_cash_proxy", "ret_btc", "ret_shy", "full_state_sample", True),
    ]
    rows: list[dict[str, Any]] = []
    for variant, btc_col, cash_col, sample_role, full_sample in variant_specs:
        mapped = remap_panel_returns(panel, btc_col=btc_col, cash_col=cash_col)
        for rule_id, family_specs in rules.items():
            for family, spec in family_specs.items():
                for funding in FUNDING_CONVENTIONS:
                    for freq in REBALANCE_FREQUENCIES:
                        cash_applicable = funding == "bil_sleeve"
                        if variant == "shy_cash_proxy" and not cash_applicable:
                            rows.append(
                                {
                                    "robustness_variant": variant,
                                    "sample_role": sample_role,
                                    "rule_id": rule_id,
                                    "portfolio_family": family,
                                    "funding_convention": funding,
                                    "rebalance_frequency": freq,
                                    "signal_timing": MAIN_SIGNAL_TIMING,
                                    "cost_scenario": MAIN_COST_SCENARIO,
                                    "cash_proxy_applicable": False,
                                    "status": "not_applicable_pro_rata_has_no_cash_sleeve",
                                }
                            )
                            continue
                        sim = simulate_rebalanced(mapped, bases[family], spec["weights_by_state"], spec["max_sleeve"], funding, freq)
                        row = {
                            "robustness_variant": variant,
                            "btc_source_column": btc_col,
                            "cash_proxy_column": cash_col,
                            "sample_role": sample_role,
                            "uses_full_state_sample": bool(full_sample),
                            "rule_id": rule_id,
                            "raw_rule_name": spec["raw_rule_name"],
                            "rule_role": spec["rule_role"],
                            "portfolio_family": family,
                            "funding_convention": funding,
                            "rebalance_frequency": freq,
                            "signal_timing": MAIN_SIGNAL_TIMING,
                            "cost_scenario": MAIN_COST_SCENARIO,
                            "cash_proxy_applicable": bool(cash_applicable or variant != "shy_cash_proxy"),
                            "status": "computed",
                        }
                        row.update(summarize_rebalanced(sim))
                        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(dirs["results"] / "implementability_robustness_summary.csv", index=False)
    return {"summary": summary}


def monthly_compound(frame: pd.DataFrame, return_cols: list[str]) -> pd.DataFrame:
    data = frame.copy()
    data["month"] = data["date"].dt.to_period("M")
    rows: list[dict[str, Any]] = []
    for month, group in data.groupby("month", sort=True):
        row: dict[str, Any] = {
            "month": str(month),
            "date": group["date"].max(),
            "hmm4_state": group.sort_values("date")["hmm4_state"].iloc[-1],
            "hmm4_state_id": int(group.sort_values("date")["hmm4_state_id"].iloc[-1]),
            "weekly_rows_in_month": int(len(group)),
        }
        for col in return_cols:
            valid = group[col].dropna()
            row[col] = float((1.0 + valid).prod() - 1.0) if len(valid) == len(group) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def compute_monthly_robustness(validation_payload: dict[str, Any], inputs: dict[str, Any], dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    panel = validation_payload["analysis_panel"]
    monthly = monthly_compound(panel, ["ret_btc", "ret_btc_coinmetrics_usd", "ret_spy", "ret_tlt", "ret_ief", "ret_gld", "ret_dbc", "ret_bil", "ret_shy"])
    bases = base_weights(inputs)
    rules = rule_weights(inputs)
    state_rows: list[dict[str, Any]] = []
    for state, frame in monthly.groupby("hmm4_state", sort=True):
        row = {"hmm4_state": state, "hmm4_state_id": int(frame["hmm4_state_id"].iloc[0]), "monthly_count": int(len(frame))}
        row.update(performance_metrics(frame["ret_btc"], prefix="btc_monthly_", periods_per_year=TRADING_MONTHS_PER_YEAR))
        state_rows.append(row)
    monthly_state = pd.DataFrame(state_rows).sort_values("hmm4_state").reset_index(drop=True)
    monthly_state.to_csv(dirs["results"] / "monthly_state_conditioned_btc_summary.csv", index=False)

    rows: list[dict[str, Any]] = []
    for btc_source, btc_col in [("btc_usdt", "ret_btc"), ("coinmetrics", "ret_btc_coinmetrics_usd")]:
        mapped = remap_panel_returns(monthly, btc_col=btc_col, cash_col="ret_bil")
        for family, base in bases.items():
            for btc_weight in BTC_WEIGHT_GRID:
                weights = fixed_weights(base, btc_weight)
                ret = portfolio_return(mapped, weights)
                row = {
                    "robustness_variant": "monthly_fixed_btc",
                    "btc_source": btc_source,
                    "portfolio_family": family,
                    "btc_weight": btc_weight,
                    "sample_months": int(len(mapped)),
                    "start_month": str(mapped["month"].iloc[0]),
                    "end_month": str(mapped["month"].iloc[-1]),
                }
                row.update(performance_metrics(ret, periods_per_year=TRADING_MONTHS_PER_YEAR))
                rows.append(row)
    for rule_id, family_specs in rules.items():
        for family, spec in family_specs.items():
            mapped = remap_panel_returns(monthly, btc_col="ret_btc", cash_col="ret_bil")
            returns = []
            weights = []
            for _, row in mapped.iterrows():
                target = target_weights_for_state(bases[family], spec["weights_by_state"], row["hmm4_state"], "pro_rata_base", spec["max_sleeve"])
                returns.append(sum(target[asset] * float(row[asset]) for asset in ["ret_btc"] + BASE_ASSETS))
                weights.append(target["ret_btc"])
            out = {
                "robustness_variant": "monthly_conditional_rule_target_weight",
                "btc_source": "btc_usdt",
                "rule_id": rule_id,
                "portfolio_family": family,
                "funding_convention": "pro_rata_base_target_weight",
                "sample_months": int(len(mapped)),
                "average_btc_weight": float(np.mean(weights)),
                "max_btc_weight": float(np.max(weights)),
                "start_month": str(mapped["month"].iloc[0]),
                "end_month": str(mapped["month"].iloc[-1]),
            }
            out.update(performance_metrics(pd.Series(returns), periods_per_year=TRADING_MONTHS_PER_YEAR))
            rows.append(out)
    summary = pd.DataFrame(rows)
    summary.to_csv(dirs["results"] / "monthly_frequency_robustness_summary.csv", index=False)
    plot_monthly_robustness(summary, dirs["figures"] / "monthly_vs_weekly_robustness.png")
    return {"summary": summary, "state_summary": monthly_state, "monthly_panel": monthly}


def compute_etf_observation(validation_payload: dict[str, Any], inputs: dict[str, Any], dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    panel = validation_payload["analysis_panel"].copy()
    etf = panel[(panel["date"] >= ETF_RETURN_START) & panel["ret_ibit"].notna() & panel["ret_fbtc"].notna()].copy()
    require(len(etf) == ETF_RETURN_ROWS, f"Unexpected ETF return sample rows: {len(etf)}")
    tracking_rows = []
    for etf_col in ["ret_ibit", "ret_fbtc"]:
        tracking_rows.append(compute_tracking_summary(etf, etf_col, "ret_btc", f"{etf_col}_minus_BTCUSDT", "etf_post_observational"))
        tracking_rows.append(compute_tracking_summary(etf, etf_col, "ret_btc_coinmetrics_usd", f"{etf_col}_minus_CoinMetrics", "etf_post_observational"))
    tracking_rows.append(compute_tracking_summary(etf, "ret_ibit", "ret_fbtc", "IBIT_minus_FBTC", "etf_post_observational"))
    tracking = pd.DataFrame(tracking_rows)
    tracking.to_csv(dirs["results"] / "etf_post_tracking_summary.csv", index=False)

    bases = base_weights(inputs)
    rules = rule_weights(inputs)
    rows: list[dict[str, Any]] = []
    for etf_name, etf_col in [("ibit", "ret_ibit"), ("fbtc", "ret_fbtc")]:
        mapped = remap_panel_returns(etf, btc_col=etf_col, cash_col="ret_bil")
        for rule_id, family_specs in rules.items():
            for family, spec in family_specs.items():
                for funding in FUNDING_CONVENTIONS:
                    for freq in REBALANCE_FREQUENCIES:
                        sim = simulate_rebalanced(mapped, bases[family], spec["weights_by_state"], spec["max_sleeve"], funding, freq)
                        row = {
                            "robustness_variant": "etf_post_implementation_observation",
                            "etf_proxy": etf_name,
                            "btc_source_column": etf_col,
                            "sample_role": "observational_appendix_only",
                            "rule_id": rule_id,
                            "portfolio_family": family,
                            "funding_convention": funding,
                            "rebalance_frequency": freq,
                            "signal_timing": MAIN_SIGNAL_TIMING,
                            "cost_scenario": MAIN_COST_SCENARIO,
                            "status": "computed_short_sample_observation",
                        }
                        row.update(summarize_rebalanced(sim))
                        rows.append(row)
    observation = pd.DataFrame(rows)
    observation.to_csv(dirs["results"] / "etf_post_implementation_observation.csv", index=False)
    plot_etf_tracking(tracking, dirs["figures"] / "etf_post_tracking.png")
    return {"tracking": tracking, "observation": observation}


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=1)


def compute_credit_proxy_robustness(validation_payload: dict[str, Any], dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    panel = validation_payload["analysis_panel"].copy()
    rows: list[dict[str, Any]] = []
    beta_rows: list[dict[str, Any]] = []
    specs = [
        ("hy_oas_level", "macro_hy_oas_lag1"),
        ("hy_oas_change_4w", "macro_hy_oas_chg_4w_lag1"),
    ]
    for proxy_name, proxy_col in specs:
        frame = panel[["date", "ret_btc", "macro_credit_spread_baa10y_z", proxy_col]].dropna().copy()
        require(len(frame) >= 100, f"Credit proxy overlap too small for {proxy_name}: {len(frame)}")
        frame[f"{proxy_col}_z_overlap"] = zscore(frame[proxy_col])
        rows.append(
            {
                "credit_proxy_variant": proxy_name,
                "proxy_column": proxy_col,
                "n_overlap": int(len(frame)),
                "start_date": date_string(frame["date"], "min"),
                "end_date": date_string(frame["date"], "max"),
                "baa10y_z_vs_proxy_z_correlation": correlation(frame["macro_credit_spread_baa10y_z"], frame[f"{proxy_col}_z_overlap"]),
                "sample_role": "short_overlap_only",
            }
        )
        baa = ols_hac_single_predictor(frame["ret_btc"], frame["macro_credit_spread_baa10y_z"], f"baa10y_z_on_{proxy_name}_overlap")
        baa.update({"credit_proxy_variant": proxy_name, "predictor_family": "BAA10Y_overlap_comparator", "sample_role": "short_overlap_only"})
        hy = ols_hac_single_predictor(frame["ret_btc"], frame[f"{proxy_col}_z_overlap"], f"{proxy_col}_z_overlap")
        hy.update({"credit_proxy_variant": proxy_name, "predictor_family": "HY_OAS_overlap_proxy", "sample_role": "short_overlap_only"})
        beta_rows.extend([baa, hy])
    overlap = pd.DataFrame(rows)
    beta = pd.DataFrame(beta_rows)
    overlap.to_csv(dirs["results"] / "credit_proxy_overlap_summary.csv", index=False)
    beta.to_csv(dirs["results"] / "credit_proxy_beta_robustness.csv", index=False)
    plot_credit_proxy(overlap, beta, dirs["figures"] / "credit_proxy_overlap.png")
    return {"overlap": overlap, "beta": beta}


def build_variant_dictionary(validation_payload: dict[str, Any], dirs: dict[str, Path]) -> pd.DataFrame:
    rows = [
        {
            "variant_id": "btc_source_coinmetrics",
            "category": "BTC source",
            "sample_role": "full_state_sample",
            "primary_use": "Replace BTCUSDT with Coin Metrics BTC for full-sample robustness.",
            "allowed_as_main_robustness": True,
        },
        {
            "variant_id": "cash_proxy_shy",
            "category": "Cash proxy",
            "sample_role": "full_state_sample",
            "primary_use": "Replace BIL with SHY only where a cash sleeve exists.",
            "allowed_as_main_robustness": True,
        },
        {
            "variant_id": "monthly_frequency",
            "category": "Frequency",
            "sample_role": "monthly_aggregated",
            "primary_use": "Compound weekly returns to monthly returns and use month-end HMM labels.",
            "allowed_as_main_robustness": True,
        },
        {
            "variant_id": "etf_post_ibit_fbtc",
            "category": "ETF-era observation",
            "sample_role": "observational_appendix_only",
            "primary_use": "Inspect 2024-01-19 onward ETF tracking and implementation observations.",
            "allowed_as_main_robustness": False,
        },
        {
            "variant_id": "credit_proxy_hy_oas",
            "category": "Credit proxy",
            "sample_role": "short_overlap_only",
            "primary_use": "Compare BAA10Y and HY OAS beta diagnostics on overlap only.",
            "allowed_as_main_robustness": False,
        },
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(dirs["results"] / "robustness_variant_dictionary.csv", index=False)
    return frame


def build_decision_matrix(validation_payload: dict[str, Any], results: dict[str, Any], dirs: dict[str, Path]) -> pd.DataFrame:
    availability = validation_payload["availability"]
    lookup = availability.set_index(["variable", "sample_name"]).to_dict(orient="index")
    rows = [
        {
            "robustness_area": "BTC source replacement",
            "variant": "Coin Metrics BTC",
            "sample_rows": EXPECTED_STATE_ROWS,
            "missing_rows": int(lookup[("ret_btc_coinmetrics_usd", "full_state_sample")]["missing_rows"]),
            "diagnostic_status": "valid_full_sample_robustness",
            "use_in_main_robustness": True,
            "interpretation_limit": "Same HMM labels and same allocation rules; only BTC return source changes.",
        },
        {
            "robustness_area": "Cash proxy replacement",
            "variant": "SHY for BIL",
            "sample_rows": EXPECTED_STATE_ROWS,
            "missing_rows": int(lookup[("ret_shy", "full_state_sample")]["missing_rows"]),
            "diagnostic_status": "valid_full_sample_robustness",
            "use_in_main_robustness": True,
            "interpretation_limit": "Applies only to bil_sleeve funding convention.",
        },
        {
            "robustness_area": "Frequency",
            "variant": "Weekly compounded to monthly",
            "sample_rows": int(results["monthly"]["summary"]["sample_months"].dropna().max()),
            "missing_rows": 0,
            "diagnostic_status": "valid_frequency_robustness",
            "use_in_main_robustness": True,
            "interpretation_limit": "Month-end HMM labels are reused; no monthly HMM re-estimation.",
        },
        {
            "robustness_area": "ETF-era BTC proxy",
            "variant": "IBIT/FBTC",
            "sample_rows": ETF_RETURN_ROWS,
            "missing_rows": int(lookup[("ret_ibit", "full_state_sample")]["missing_rows"]),
            "diagnostic_status": "observational_appendix_only",
            "use_in_main_robustness": False,
            "interpretation_limit": "ETF returns start 2024-01-19; sample is too short for full-sample robustness.",
        },
        {
            "robustness_area": "Credit proxy",
            "variant": "HY OAS",
            "sample_rows": int(results["credit"]["overlap"]["n_overlap"].min()),
            "missing_rows": int(lookup[("macro_hy_oas_chg_4w_lag1", "full_state_sample")]["missing_rows"]),
            "diagnostic_status": "short_overlap_only",
            "use_in_main_robustness": False,
            "interpretation_limit": "HY OAS is evaluated only on overlap; not comparable to full-sample BAA10Y without matching dates.",
        },
    ]
    matrix = pd.DataFrame(rows)
    matrix.to_csv(dirs["results"] / "robustness_decision_matrix.csv", index=False)
    return matrix


def write_explainability_artifacts(
    inputs: dict[str, Any],
    validation_payload: dict[str, Any],
    variant_dictionary: pd.DataFrame,
    decision_matrix: pd.DataFrame,
    dirs: dict[str, Path],
) -> dict[str, Any]:
    methodology = """# Part 6 Methodology Audit

## Purpose
Part 6 evaluates whether the Part 1-5 findings are robust to data-source, cash-proxy, frequency, ETF-era, and credit-proxy substitutions. It is a diagnostic robustness layer, not a new strategy search and not a final thesis conclusion.

## Inputs
- Cleaned weekly returns and robustness panel from `data_2026/cleaned`.
- Part 1 HMM-4 state labels and state-model lineage.
- Part 2 no-BTC All Weather and ERC baseline weights.
- Part 4 executed conditional BTC allocation rules.
- Part 5 one-week-lagged implementability diagnostics.

## Full-Sample Robustness
Coin Metrics BTC and SHY are complete in the 425-week state-aligned sample and are valid full-sample substitutions. HMM states, ERC weights, and conditional rule weights are kept fixed.

## Short-Sample Diagnostics
IBIT/FBTC and HY OAS are not complete in the full sample. IBIT/FBTC begin as return series on 2024-01-19. HY OAS overlap begins in 2023. These outputs are marked as observational or overlap-only diagnostics and are not used as substitutes for the full-sample evidence.

## Monthly Frequency
Monthly robustness compounds weekly returns within each calendar month and assigns the month-end HMM state from the last available Friday. No monthly PCA/HMM model is estimated.

## Boundaries
- No HMM re-estimation.
- No ERC re-optimization.
- No new BTC weight selection.
- No final thesis conclusion.
"""
    (dirs["results"] / "methodology_audit.md").write_text(methodology, encoding="utf-8")

    assumptions = {
        "status": "documented",
        "state_labels": {
            "source": "Part 1 HMM-4 state labels",
            "estimation_type": "full-sample ex-post descriptive regime identification",
            "not_a_real_time_signal": True,
        },
        "fixed_objects": [
            "HMM-4 labels",
            "All Weather base weights",
            "ERC base weights",
            "Part 4 executed rule weights",
            "Part 5 main implementability timing and cost convention",
        ],
        "full_sample_replacements": ["Coin Metrics BTC", "SHY cash proxy"],
        "short_sample_outputs": ["IBIT/FBTC ETF-era observation", "HY OAS overlap-only credit proxy"],
        "excluded": [
            "new HMM estimation",
            "new BTC allocation threshold selection",
            "new ERC optimization",
            "final thesis conclusion",
        ],
    }
    write_json(dirs["results"] / "model_assumption_audit.json", assumptions)

    lineage_rows = []
    for name, path in inputs["paths"].items():
        lineage_rows.append(
            {
                "input_name": name,
                "path": str(path),
                "sha256": inputs["hashes"][name],
                "usage": lineage_usage(name),
            }
        )
    lineage = pd.DataFrame(lineage_rows)
    lineage.to_csv(dirs["results"] / "data_lineage.csv", index=False)
    return {"assumptions": assumptions, "lineage": lineage, "methodology": methodology}


def lineage_usage(name: str) -> str:
    if name.startswith("part"):
        return "Upstream experiment validation, lineage, or fixed rule/weight context."
    if name == "asset_returns_main_weekly":
        return "Primary weekly return sample for full-sample robustness."
    if name == "robustness_weekly_panel":
        return "Coin Metrics BTC, SHY, ETF-era, and HY OAS robustness variables."
    if name == "state_model_panel_weekly":
        return "BAA10Y credit predictor comparator for HY OAS overlap beta diagnostics."
    if name == "hmm4_state_labels":
        return "Fixed HMM-4 state labels and lagged state signals."
    return "Cleaning lineage and audit input."


def validate_outputs(
    dirs: dict[str, Path],
    validation_payload: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    result_checks = []
    for name in REQUIRED_RESULTS:
        path = dirs["results"] / name
        exists = True if name == "output_validation_summary.json" else path.exists()
        nonempty = True if name == "output_validation_summary.json" else (path.stat().st_size > 0 if path.exists() else False)
        result_checks.append({"file": name, "exists": exists, "nonempty": nonempty})
    figure_checks = []
    for name in REQUIRED_FIGURES:
        path = dirs["figures"] / name
        figure_checks.append({"file": name, "exists": path.exists(), "nonempty": path.stat().st_size > 0 if path.exists() else False})

    availability = validation_payload["availability"].set_index(["variable", "sample_name"])
    fixed = results["btc"]["fixed"]
    implementability = results["implementability"]["summary"]
    monthly = results["monthly"]["summary"]
    etf_tracking = results["etf"]["tracking"]
    credit = results["credit"]["beta"]

    checks = {
        "required_files_ok": all(row["exists"] and row["nonempty"] for row in result_checks),
        "required_figures_ok": all(row["exists"] and row["nonempty"] for row in figure_checks),
        "input_validation_passed": validation_payload["validation"]["status"] == "passed",
        "coinmetrics_full_sample_complete": int(availability.loc[("ret_btc_coinmetrics_usd", "full_state_sample"), "missing_rows"]) == 0,
        "shy_full_sample_complete": int(availability.loc[("ret_shy", "full_state_sample"), "missing_rows"]) == 0,
        "etf_short_sample_recorded": int(availability.loc[("ret_ibit", "full_state_sample"), "missing_rows"]) == 310,
        "hy_oas_short_sample_recorded": int(availability.loc[("macro_hy_oas_chg_4w_lag1", "full_state_sample"), "missing_rows"]) == 282,
        "fixed_btc_grid_complete": len(fixed) == 2 * len(PORTFOLIO_FAMILIES) * len(BTC_WEIGHT_GRID),
        "risk_contribution_sums_ok": bool(
            (fixed["vol_share_sum_check"].sub(1.0).abs() <= 1e-8).all()
            and (fixed["cvar_share_sum_check"].sub(1.0).abs() <= 1e-8).all()
        ),
        "implementability_variants_present": set(["baseline", "coinmetrics_btc", "shy_cash_proxy"]).issubset(set(implementability["robustness_variant"])),
        "monthly_outputs_present": len(monthly) > 0,
        "etf_sample_starts_2024_01_19": bool((etf_tracking["start_date"] == ETF_RETURN_START).all()),
        "credit_beta_finite": bool(np.isfinite(credit[["beta", "hac_se_beta", "p_beta"]].to_numpy(dtype=float)).all()),
    }
    summary = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "required_result_checks": result_checks,
        "required_figure_checks": figure_checks,
        "fixed_btc_rows": int(len(fixed)),
        "implementability_rows": int(len(implementability)),
        "monthly_rows": int(len(monthly)),
        "etf_tracking_rows": int(len(etf_tracking)),
        "credit_beta_rows": int(len(credit)),
    }
    write_json(dirs["results"] / "output_validation_summary.json", summary)
    require(summary["status"] == "passed", f"Output validation failed: {checks}")
    logging.info("Output validation passed")
    return summary


def build_manifest(
    args: argparse.Namespace,
    run_id: str,
    dirs: dict[str, Path],
    inputs: dict[str, Any],
    validation_payload: dict[str, Any],
    output_validation: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "objective": "Full robustness analysis for conditional BTC risk-budgeted allocation diagnostics",
        "input_dir": str(args.input_dir),
        "part1_run_dir": str(args.part1_run_dir),
        "part2_run_dir": str(args.part2_run_dir),
        "part3_run_dir": str(args.part3_run_dir),
        "part4_run_dir": str(args.part4_run_dir),
        "part5_run_dir": str(args.part5_run_dir),
        "output_dir": str(args.output_dir),
        "run_dir": str(dirs["root"]),
        "random_seed": int(args.seed),
        "sample": {
            "state_rows": EXPECTED_STATE_ROWS,
            "state_start": EXPECTED_STATE_START,
            "state_end": EXPECTED_STATE_END,
            "lagged_main_rows": EXPECTED_LAGGED_ROWS,
            "lagged_main_start": EXPECTED_LAGGED_START,
            "state_counts": EXPECTED_STATE_COUNTS,
            "etf_rows": ETF_RETURN_ROWS,
            "etf_start": ETF_RETURN_START,
        },
        "input_hashes": inputs["hashes"],
        "package_versions": package_versions(),
        "parameters": {
            "btc_weight_grid": BTC_WEIGHT_GRID,
            "implemented_rule_ids": IMPLEMENTED_RULE_IDS,
            "portfolio_families": PORTFOLIO_FAMILIES,
            "funding_conventions": FUNDING_CONVENTIONS,
            "rebalance_frequencies": REBALANCE_FREQUENCIES,
            "main_signal_timing": MAIN_SIGNAL_TIMING,
            "main_cost_scenario": MAIN_COST_SCENARIO,
            "cost_rates": COST_RATES,
            "tail_alpha": TAIL_ALPHA,
            "hac_lag_weeks": HAC_LAG_WEEKS,
            "monthly_frequency_method": "compound weekly returns by calendar month and use month-end HMM-4 label",
        },
        "lineage": {
            "part1_run_id": inputs["part1_manifest"].get("run_id"),
            "part2_run_id": inputs["part2_manifest"].get("run_id"),
            "part3_run_id": inputs["part3_manifest"].get("run_id"),
            "part4_run_id": inputs["part4_manifest"].get("run_id"),
            "part5_run_id": inputs["part5_manifest"].get("run_id"),
        },
        "output_validation": output_validation,
        "outputs": {name: str(path) for name, path in dirs.items() if name != "root"},
        "scope_notes": [
            "Part 6 is a robustness diagnostic layer.",
            "HMM labels, ERC weights, and conditional rule weights are not re-estimated.",
            "IBIT/FBTC and HY OAS outputs are short-sample diagnostics only.",
            "No final thesis conclusion is written by this runner.",
        ],
    }
    write_json(dirs["root"] / "run_manifest.json", manifest)
    return manifest


def plot_btc_source_tracking(panel: pd.DataFrame, path: Path) -> None:
    diff = panel["ret_btc"] - panel["ret_btc_coinmetrics_usd"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(panel["date"], (1 + panel["ret_btc"]).cumprod(), label="BTCUSDT")
    axes[0].plot(panel["date"], (1 + panel["ret_btc_coinmetrics_usd"]).cumprod(), label="Coin Metrics")
    axes[0].set_title("BTC Source Cumulative Return")
    axes[0].legend()
    axes[1].plot(panel["date"], diff)
    axes[1].set_title("Weekly Return Difference")
    axes[1].axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cash_proxy(panel: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(panel["date"], (1 + panel["ret_bil"]).cumprod(), label="BIL")
    axes[0].plot(panel["date"], (1 + panel["ret_shy"]).cumprod(), label="SHY")
    axes[0].set_title("Cash Proxy Cumulative Return")
    axes[0].legend()
    axes[1].plot(panel["date"], panel["ret_bil"] - panel["ret_shy"])
    axes[1].set_title("BIL Minus SHY Weekly Return")
    axes[1].axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_fixed_btc_robustness(fixed: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for source, frame in fixed.groupby("btc_source"):
        avg = frame.groupby("btc_weight")["btc_component_share_vol"].mean()
        axes[0].plot(avg.index * 100, avg.values, marker="o", label=source)
        avg_cvar = frame.groupby("btc_weight")["btc_component_share_cvar"].mean()
        axes[1].plot(avg_cvar.index * 100, avg_cvar.values, marker="o", label=source)
    axes[0].set_title("BTC Volatility Contribution")
    axes[0].set_xlabel("BTC weight (%)")
    axes[1].set_title("BTC CVaR Contribution")
    axes[1].set_xlabel("BTC weight (%)")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_conditional_rule(summary: pd.DataFrame, path: Path) -> None:
    frame = summary.pivot_table(
        index=["rule_id", "portfolio_family"],
        columns="btc_source",
        values="annualized_volatility",
        aggfunc="first",
    ).reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(frame))
    width = 0.35
    ax.bar(x - width / 2, frame.get("btc_usdt", pd.Series(np.nan, index=frame.index)), width, label="BTCUSDT")
    ax.bar(x + width / 2, frame.get("coinmetrics", pd.Series(np.nan, index=frame.index)), width, label="Coin Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels((frame["rule_id"] + " / " + frame["portfolio_family"]).tolist(), rotation=25, ha="right")
    ax.set_title("Conditional Rule Robustness: Annualized Volatility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_monthly_robustness(summary: pd.DataFrame, path: Path) -> None:
    frame = summary[summary["robustness_variant"] == "monthly_fixed_btc"].copy()
    fig, ax = plt.subplots(figsize=(10, 4))
    for source, sub in frame.groupby("btc_source"):
        avg = sub.groupby("btc_weight")["annualized_volatility"].mean()
        ax.plot(avg.index * 100, avg.values, marker="o", label=source)
    ax.set_title("Monthly Fixed BTC Robustness")
    ax.set_xlabel("BTC weight (%)")
    ax.set_ylabel("Annualized volatility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_etf_tracking(tracking: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    frame = tracking[tracking["comparison"].isin(["ret_ibit_minus_BTCUSDT", "ret_fbtc_minus_BTCUSDT", "IBIT_minus_FBTC"])].copy()
    ax.bar(frame["comparison"], frame["correlation"])
    ax.set_ylim(0, 1.05)
    ax.set_title("ETF-Era Tracking Correlations")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_credit_proxy(overlap: pd.DataFrame, beta: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(overlap["credit_proxy_variant"], overlap["baa10y_z_vs_proxy_z_correlation"])
    axes[0].set_title("BAA10Y vs HY OAS Overlap Correlation")
    hy = beta[beta["predictor_family"] == "HY_OAS_overlap_proxy"]
    axes[1].bar(hy["credit_proxy_variant"], hy["beta"], yerr=1.96 * hy["hac_se_beta"])
    axes[1].set_title("BTC Beta to HY OAS Proxy")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    run_id = args.run_id or now_run_id()
    run_dir = args.output_dir / run_id
    dirs = ensure_dirs(run_dir)
    setup_logging(dirs["logs"])
    logging.info("Starting Part 6 run: %s", run_id)
    guard_resume_input_hashes(args, dirs)

    inputs = load_or_run(dirs, "01_load_inputs", args.resume, lambda: load_inputs(args))
    validation_payload = load_or_run(dirs, "02_validate_inputs", args.resume, lambda: validate_inputs(inputs, dirs))
    variant_dictionary = load_or_run(dirs, "03_variant_dictionary", args.resume, lambda: build_variant_dictionary(validation_payload, dirs))
    btc_results = load_or_run(dirs, "04_btc_source_robustness", args.resume, lambda: compute_btc_source_robustness(validation_payload, inputs, dirs))
    cash_results = load_or_run(dirs, "05_cash_proxy_robustness", args.resume, lambda: compute_cash_proxy_robustness(validation_payload, inputs, dirs))
    conditional_results = load_or_run(dirs, "06_conditional_rule_robustness", args.resume, lambda: compute_conditional_rule_robustness(validation_payload, inputs, dirs))
    implementability_results = load_or_run(dirs, "07_implementability_robustness", args.resume, lambda: compute_implementability_robustness(validation_payload, inputs, dirs))
    monthly_results = load_or_run(dirs, "08_monthly_robustness", args.resume, lambda: compute_monthly_robustness(validation_payload, inputs, dirs))
    etf_results = load_or_run(dirs, "09_etf_observation", args.resume, lambda: compute_etf_observation(validation_payload, inputs, dirs))
    credit_results = load_or_run(dirs, "10_credit_proxy_robustness", args.resume, lambda: compute_credit_proxy_robustness(validation_payload, dirs))
    result_bundle = {
        "btc": btc_results,
        "cash": cash_results,
        "conditional": conditional_results,
        "implementability": implementability_results,
        "monthly": monthly_results,
        "etf": etf_results,
        "credit": credit_results,
    }
    decision_matrix = load_or_run(dirs, "11_decision_matrix", args.resume, lambda: build_decision_matrix(validation_payload, result_bundle, dirs))
    explainability = load_or_run(
        dirs,
        "12_explainability",
        args.resume,
        lambda: write_explainability_artifacts(inputs, validation_payload, variant_dictionary, decision_matrix, dirs),
    )
    output_validation = load_or_run(dirs, "13_output_validation", args.resume, lambda: validate_outputs(dirs, validation_payload, result_bundle))
    manifest = build_manifest(args, run_id, dirs, inputs, validation_payload, output_validation, result_bundle)
    save_pickle(dirs["models"] / "robustness_result_bundle.pkl", result_bundle)
    save_pickle(dirs["checkpoints"] / "14_run_manifest.pkl", manifest)
    logging.info("Part 6 run completed: %s", run_dir)


if __name__ == "__main__":
    main()
