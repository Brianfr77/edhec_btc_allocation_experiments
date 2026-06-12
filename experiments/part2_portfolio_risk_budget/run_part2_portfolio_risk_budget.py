#!/usr/bin/env python3
"""Part 2 experiment runner: portfolio baselines and BTC risk-budget diagnostics."""

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
from scipy.optimize import minimize


EXPECTED_ASSET_START = "2018-01-12"
EXPECTED_ASSET_END = "2026-03-27"
EXPECTED_ASSET_ROWS = 429
EXPECTED_STATE_START = "2018-02-09"
EXPECTED_STATE_END = "2026-03-27"
EXPECTED_STATE_ROWS = 425
TRADING_WEEKS_PER_YEAR = 52
TAIL_ALPHA = 0.05
BASE_ASSETS = ["ret_spy", "ret_tlt", "ret_ief", "ret_gld", "ret_dbc"]
PORTFOLIO_ASSETS = ["ret_btc"] + BASE_ASSETS
BTC_WEIGHTS = [0.0, 0.01, 0.02, 0.03, 0.05]
ALL_WEATHER_WEIGHTS = {
    "ret_spy": 0.30,
    "ret_tlt": 0.40,
    "ret_ief": 0.15,
    "ret_gld": 0.075,
    "ret_dbc": 0.075,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run All Weather/ERC baseline and fixed-BTC risk-budget diagnostics."
    )
    parser.add_argument("--input-dir", default="data_2026/cleaned", type=Path)
    parser.add_argument(
        "--part1-run-dir",
        default="outputs/part1_btc_macro_state/colab_part1_seed42",
        type=Path,
    )
    parser.add_argument("--output-dir", default="outputs/part2_portfolio_risk_budget", type=Path)
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


def read_inputs(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "asset_returns_main_weekly": args.input_dir / "asset_returns_main_weekly.csv",
        "cleaning_report": args.input_dir / "cleaning_report.json",
        "part1_manifest": args.part1_run_dir / "run_manifest.json",
        "hmm4_state_labels": args.part1_run_dir / "results" / "hmm4_state_labels.csv",
        "hmm4_state_profiles": args.part1_run_dir / "results" / "hmm4_state_profiles.csv",
        "part1_validation_summary": args.part1_run_dir / "results" / "validation_summary.json",
    }
    for name, path in paths.items():
        require(path.exists(), f"Missing required input file: {name} at {path}")

    asset = pd.read_csv(paths["asset_returns_main_weekly"], parse_dates=["date"])
    labels = pd.read_csv(paths["hmm4_state_labels"], parse_dates=["date"])
    profiles = pd.read_csv(paths["hmm4_state_profiles"])
    cleaning_report = json.loads(paths["cleaning_report"].read_text(encoding="utf-8"))
    part1_manifest = json.loads(paths["part1_manifest"].read_text(encoding="utf-8"))
    part1_validation = json.loads(paths["part1_validation_summary"].read_text(encoding="utf-8"))
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    return {
        "paths": paths,
        "asset": asset,
        "labels": labels,
        "profiles": profiles,
        "cleaning_report": cleaning_report,
        "part1_manifest": part1_manifest,
        "part1_validation": part1_validation,
        "input_hashes": hashes,
    }


def enforce_resume_input_hashes(args: argparse.Namespace, dirs: dict[str, Path], input_hashes: dict[str, str]) -> None:
    if not args.resume:
        return
    manifest_path = dirs["root"] / "run_manifest.json"
    if not manifest_path.exists():
        return
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_hashes = old_manifest.get("input_hashes", {})
    require(old_hashes == input_hashes, "Input hashes changed since the previous run manifest")
    logging.info("Resume input hash check passed")


def validate_inputs(inputs: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    asset = inputs["asset"].copy()
    labels = inputs["labels"].copy()
    profiles = inputs["profiles"].copy()
    cleaning_report = inputs["cleaning_report"]
    part1_manifest = inputs["part1_manifest"]
    input_hashes = inputs["input_hashes"]

    main_cols = list(cleaning_report["column_mapping"]["main_assets"].values())
    required_cols = ["date"] + PORTFOLIO_ASSETS
    require(all(col in asset.columns for col in required_cols), "Missing required asset return columns")
    require("ret_bil" in asset.columns, "BIL exists in cleaned data but is intentionally not used in Part 2")
    require(len(asset) == EXPECTED_ASSET_ROWS, f"Unexpected asset row count: {len(asset)}")
    require(date_string(asset["date"], "min") == EXPECTED_ASSET_START, "Unexpected asset start date")
    require(date_string(asset["date"], "max") == EXPECTED_ASSET_END, "Unexpected asset end date")
    require(asset["date"].dt.dayofweek.eq(4).all(), "Asset dates are not all Fridays")
    require(asset[required_cols[1:]].isna().sum().sum() == 0, "Missing values in required returns")

    require(inputs["input_hashes"]["asset_returns_main_weekly"] == part1_manifest["input_hashes"]["asset_returns_main_weekly"], "Asset file hash does not match Part 1 manifest")
    require(inputs["input_hashes"]["cleaning_report"] == part1_manifest["input_hashes"]["cleaning_report"], "Cleaning report hash does not match Part 1 manifest")
    require(part1_manifest["model_diagnostics"]["hmm4_converged"] is True, "Part 1 HMM-4 did not converge")

    required_label_cols = ["date", "hmm4_state", "hmm4_state_id", "hmm4_state_posterior_probability"]
    require(all(col in labels.columns for col in required_label_cols), "Missing HMM-4 label columns")
    require(len(labels) == EXPECTED_STATE_ROWS, f"Unexpected HMM-4 label rows: {len(labels)}")
    require(date_string(labels["date"], "min") == EXPECTED_STATE_START, "Unexpected HMM-4 label start date")
    require(date_string(labels["date"], "max") == EXPECTED_STATE_END, "Unexpected HMM-4 label end date")
    require(labels["date"].dt.dayofweek.eq(4).all(), "HMM-4 label dates are not all Fridays")
    state_counts = labels["hmm4_state"].value_counts().sort_index().to_dict()
    require(state_counts == {"state_0": 149, "state_1": 167, "state_2": 30, "state_3": 79}, f"Unexpected HMM-4 state counts: {state_counts}")
    require(len(profiles) == 4, "HMM-4 profiles must contain four states")

    panel = asset.merge(
        labels[["date", "hmm4_state", "hmm4_state_id", "hmm4_state_posterior_probability"]],
        on="date",
        how="inner",
    )
    require(len(panel) == EXPECTED_STATE_ROWS, f"Unexpected joined panel rows: {len(panel)}")
    require(date_string(panel["date"], "min") == EXPECTED_STATE_START, "Unexpected joined panel start date")
    require(date_string(panel["date"], "max") == EXPECTED_STATE_END, "Unexpected joined panel end date")
    require(panel[PORTFOLIO_ASSETS].isna().sum().sum() == 0, "Missing values after state join")

    excluded_asset_dates = sorted(set(asset["date"]) - set(labels["date"]))
    summary = {
        "status": "passed",
        "asset_sample": {
            "rows": int(len(asset)),
            "start": date_string(asset["date"], "min"),
            "end": date_string(asset["date"], "max"),
        },
        "part2_sample": {
            "rows": int(len(panel)),
            "start": date_string(panel["date"], "min"),
            "end": date_string(panel["date"], "max"),
        },
        "hmm4_state_counts": {key: int(value) for key, value in state_counts.items()},
        "hmm4_min_state_weeks": int(min(state_counts.values())),
        "excluded_asset_dates_without_state": [pd.Timestamp(value).strftime("%Y-%m-%d") for value in excluded_asset_dates],
        "used_assets": PORTFOLIO_ASSETS,
        "excluded_cleaned_asset_columns": [col for col in main_cols if col not in PORTFOLIO_ASSETS],
        "input_hashes": input_hashes,
        "part1_manifest_model_diagnostics": part1_manifest["model_diagnostics"],
    }
    write_json(dirs["results"] / "input_validation_summary.json", summary)
    logging.info("Input validation passed")
    return {"summary": summary, "panel": panel}


def portfolio_return(weights: dict[str, float], returns: pd.DataFrame) -> pd.Series:
    total = pd.Series(0.0, index=returns.index)
    for asset, weight in weights.items():
        total += weight * returns[asset]
    return total


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    require(total > 0, "Weight total must be positive")
    return {asset: weight / total for asset, weight in weights.items()}


def risk_contributions_vol(returns: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    assets = list(weights.keys())
    w = np.array([weights[asset] for asset in assets], dtype=float)
    sigma = returns[assets].cov().to_numpy(dtype=float)
    portfolio_variance = float(w @ sigma @ w)
    portfolio_vol = math.sqrt(max(portfolio_variance, 0.0))
    if portfolio_vol <= 0:
        marginal = np.zeros_like(w)
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
            "marginal_contribution_vol": marginal,
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
            "tail_loss_threshold": var_loss,
            "tail_scenario_count": tail_count,
            "portfolio_cvar_loss": cvar_loss,
            "component_contribution_cvar_loss": component,
            "component_share_cvar": share,
            "share_sum_check": float(share.sum()),
        }
    )


def var_cvar_from_returns(returns: pd.Series, alpha: float = TAIL_ALPHA) -> tuple[float, float, int]:
    clean = returns.dropna()
    if clean.empty:
        return np.nan, np.nan, 0
    var_return = float(clean.quantile(alpha))
    tail = clean[clean <= var_return]
    return var_return, float(tail.mean()), int(len(tail))


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    return float(drawdown.min())


def performance_metrics(portfolio_id: str, family: str, btc_weight: float, returns: pd.Series) -> dict[str, Any]:
    clean = returns.dropna()
    var_95, cvar_95, tail_count = var_cvar_from_returns(clean)
    weekly_mean = float(clean.mean())
    weekly_vol = float(clean.std(ddof=1))
    sharpe = weekly_mean / weekly_vol * math.sqrt(TRADING_WEEKS_PER_YEAR) if weekly_vol > 0 else np.nan
    return {
        "portfolio_id": portfolio_id,
        "portfolio_family": family,
        "btc_weight": btc_weight,
        "count": int(len(clean)),
        "mean_weekly": weekly_mean,
        "median_weekly": float(clean.median()),
        "volatility_weekly": weekly_vol,
        "annualized_mean_arithmetic": weekly_mean * TRADING_WEEKS_PER_YEAR,
        "annualized_volatility": weekly_vol * math.sqrt(TRADING_WEEKS_PER_YEAR),
        "min_weekly": float(clean.min()),
        "max_weekly": float(clean.max()),
        "var_95_weekly": var_95,
        "cvar_95_weekly": cvar_95,
        "tail_scenario_count": tail_count,
        "max_drawdown": max_drawdown(clean),
        "positive_week_share": float((clean > 0).mean()),
        "sharpe_annualized_zero_rf": float(sharpe),
    }


def erc_objective(weights: np.ndarray, cov: np.ndarray) -> float:
    portfolio_variance = float(weights @ cov @ weights)
    if portfolio_variance <= 0:
        return 1e9
    marginal = cov @ weights
    risk_contrib = weights * marginal / portfolio_variance
    target = np.full_like(risk_contrib, 1.0 / len(risk_contrib))
    return float(((risk_contrib - target) ** 2).sum())


def solve_erc_weights(returns: pd.DataFrame) -> tuple[dict[str, float], dict[str, Any]]:
    cov = returns[BASE_ASSETS].cov().to_numpy(dtype=float)
    n_assets = len(BASE_ASSETS)
    initial = np.full(n_assets, 1.0 / n_assets)
    result = minimize(
        erc_objective,
        initial,
        args=(cov,),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},),
        options={"ftol": 1e-12, "maxiter": 2000, "disp": False},
    )
    require(result.success, f"ERC optimization failed: {result.message}")
    weights = {asset: float(weight) for asset, weight in zip(BASE_ASSETS, result.x)}
    weights = normalize_weights(weights)
    vol_rc = risk_contributions_vol(returns[BASE_ASSETS], weights)
    target = 1.0 / n_assets
    diagnostics = {
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "objective_value": float(result.fun),
        "n_iterations": int(result.nit),
        "max_abs_rc_share_error_vs_equal": float((vol_rc["component_share_vol"] - target).abs().max()),
        "mean_abs_rc_share_error_vs_equal": float((vol_rc["component_share_vol"] - target).abs().mean()),
    }
    return weights, diagnostics


def build_portfolio_weights(validation: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    panel = validation["panel"]
    aw = normalize_weights(ALL_WEATHER_WEIGHTS)
    erc, erc_diagnostics = solve_erc_weights(panel[BASE_ASSETS])
    base_weights = {"all_weather": aw, "erc": erc}

    baseline_rows = []
    for family, weights in base_weights.items():
        for asset in BASE_ASSETS:
            baseline_rows.append(
                {
                    "portfolio_family": family,
                    "portfolio_id": f"{family}_btc_00pct",
                    "asset": asset,
                    "weight": weights[asset],
                    "btc_weight": 0.0,
                }
            )
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(dirs["results"] / "baseline_portfolio_weights.csv", index=False)

    fixed_rows = []
    wide_rows = []
    for family, weights in base_weights.items():
        for btc_weight in BTC_WEIGHTS:
            portfolio_id = f"{family}_btc_{int(round(btc_weight * 100)):02d}pct"
            portfolio_weights = {"ret_btc": btc_weight}
            for asset in BASE_ASSETS:
                portfolio_weights[asset] = (1.0 - btc_weight) * weights[asset]
            require(abs(sum(portfolio_weights.values()) - 1.0) < 1e-12, f"Weights do not sum to 1 for {portfolio_id}")
            require(min(portfolio_weights.values()) >= -1e-12, f"Negative weight found for {portfolio_id}")
            row = {
                "portfolio_id": portfolio_id,
                "portfolio_family": family,
                "btc_weight": btc_weight,
            }
            row.update(portfolio_weights)
            wide_rows.append(row)
            for asset, weight in portfolio_weights.items():
                fixed_rows.append(
                    {
                        "portfolio_id": portfolio_id,
                        "portfolio_family": family,
                        "btc_weight": btc_weight,
                        "asset": asset,
                        "weight": weight,
                        "weight_rule": "BTC fixed; non-BTC assets scaled from no-BTC base weights.",
                    }
                )
    fixed_df = pd.DataFrame(fixed_rows)
    fixed_wide = pd.DataFrame(wide_rows)
    fixed_df.to_csv(dirs["results"] / "fixed_btc_portfolio_weights.csv", index=False)
    fixed_wide.to_csv(dirs["results"] / "fixed_btc_portfolio_weights_wide.csv", index=False)

    erc_diag_df = pd.DataFrame([{**erc_diagnostics, **{f"erc_weight_{asset}": weight for asset, weight in erc.items()}}])
    erc_diag_df.to_csv(dirs["results"] / "erc_optimization_diagnostics.csv", index=False)
    save_pickle(dirs["models"] / "erc_static_weights.pkl", {"weights": erc, "diagnostics": erc_diagnostics})

    logging.info("Portfolio weights completed")
    return {
        "base_weights": base_weights,
        "baseline_weights": baseline_df,
        "fixed_weights_long": fixed_df,
        "fixed_weights_wide": fixed_wide,
        "erc_diagnostics": erc_diag_df,
    }


def compute_portfolio_outputs(validation: dict[str, Any], weights_payload: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    panel = validation["panel"].copy()
    fixed_wide = weights_payload["fixed_weights_wide"]
    returns_long = []
    performance_rows = []
    vol_rows = []
    cvar_rows = []
    btc_budget_rows = []

    for _, weight_row in fixed_wide.iterrows():
        portfolio_id = weight_row["portfolio_id"]
        family = weight_row["portfolio_family"]
        btc_weight = float(weight_row["btc_weight"])
        weights = {asset: float(weight_row[asset]) for asset in PORTFOLIO_ASSETS}
        returns = portfolio_return(weights, panel)
        for date, state, state_id, value in zip(panel["date"], panel["hmm4_state"], panel["hmm4_state_id"], returns):
            returns_long.append(
                {
                    "date": date,
                    "hmm4_state": state,
                    "hmm4_state_id": int(state_id),
                    "portfolio_id": portfolio_id,
                    "portfolio_family": family,
                    "btc_weight": btc_weight,
                    "portfolio_return": value,
                }
            )
        performance_rows.append(performance_metrics(portfolio_id, family, btc_weight, returns))

        vol = risk_contributions_vol(panel[PORTFOLIO_ASSETS], weights)
        vol.insert(0, "portfolio_id", portfolio_id)
        vol.insert(1, "portfolio_family", family)
        vol.insert(2, "btc_weight", btc_weight)
        cvar = risk_contributions_cvar(panel[PORTFOLIO_ASSETS], weights)
        cvar.insert(0, "portfolio_id", portfolio_id)
        cvar.insert(1, "portfolio_family", family)
        cvar.insert(2, "btc_weight", btc_weight)
        vol_rows.append(vol)
        cvar_rows.append(cvar)

        btc_vol = vol.loc[vol["asset"] == "ret_btc"].iloc[0]
        btc_cvar = cvar.loc[cvar["asset"] == "ret_btc"].iloc[0]
        btc_budget_rows.append(
            {
                "portfolio_id": portfolio_id,
                "portfolio_family": family,
                "btc_weight": btc_weight,
                "portfolio_volatility_weekly": float(btc_vol["portfolio_volatility"]),
                "portfolio_cvar_loss_weekly": float(btc_cvar["portfolio_cvar_loss"]),
                "btc_component_contribution_vol": float(btc_vol["component_contribution_vol"]),
                "btc_component_share_vol": float(btc_vol["component_share_vol"]),
                "btc_component_contribution_cvar_loss": float(btc_cvar["component_contribution_cvar_loss"]),
                "btc_component_share_cvar": float(btc_cvar["component_share_cvar"]),
                "cvar_tail_scenario_count": int(btc_cvar["tail_scenario_count"]),
            }
        )

    return_series = pd.DataFrame(returns_long)
    performance = pd.DataFrame(performance_rows)
    vol_rc = pd.concat(vol_rows, ignore_index=True)
    cvar_rc = pd.concat(cvar_rows, ignore_index=True)
    btc_budget = pd.DataFrame(btc_budget_rows)

    return_series.to_csv(dirs["results"] / "portfolio_return_series.csv", index=False)
    performance.to_csv(dirs["results"] / "portfolio_performance_summary.csv", index=False)
    vol_rc.to_csv(dirs["results"] / "full_sample_risk_contributions_vol.csv", index=False)
    cvar_rc.to_csv(dirs["results"] / "full_sample_risk_contributions_cvar.csv", index=False)
    btc_budget.to_csv(dirs["results"] / "btc_risk_budget_summary.csv", index=False)

    plot_portfolio_performance(performance, dirs["figures"] / "portfolio_risk_return_summary.png")
    plot_btc_risk_budget(btc_budget, dirs["figures"] / "btc_risk_contribution_by_weight.png")
    plot_drawdowns(return_series, dirs["figures"] / "portfolio_drawdowns.png")

    logging.info("Full-sample portfolio diagnostics completed")
    return {
        "return_series": return_series,
        "performance": performance,
        "vol_rc": vol_rc,
        "cvar_rc": cvar_rc,
        "btc_budget": btc_budget,
    }


def compute_state_conditioned_outputs(portfolio_outputs: dict[str, Any], weights_payload: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    returns_long = portfolio_outputs["return_series"]
    fixed_wide = weights_payload["fixed_weights_wide"]
    summary_rows = []
    btc_rc_rows = []
    drawdown_rows = []

    for _, weight_row in fixed_wide.iterrows():
        portfolio_id = weight_row["portfolio_id"]
        family = weight_row["portfolio_family"]
        btc_weight = float(weight_row["btc_weight"])
        weights = {asset: float(weight_row[asset]) for asset in PORTFOLIO_ASSETS}
        portfolio_frame = returns_long[returns_long["portfolio_id"] == portfolio_id].copy()
        for state_name, state_frame in portfolio_frame.groupby("hmm4_state", sort=True):
            state_id = int(state_frame["hmm4_state_id"].iloc[0])
            returns = state_frame["portfolio_return"]
            metrics = performance_metrics(portfolio_id, family, btc_weight, returns)
            metrics.update({"hmm4_state": state_name, "hmm4_state_id": state_id})
            metrics["state_sample_warning"] = "small_state_sample" if len(state_frame) < 40 else ""
            summary_rows.append(metrics)

            asset_state_returns = get_asset_returns_for_dates(
                state_frame["date"],
                portfolio_outputs["asset_panel"],
            )
            vol = risk_contributions_vol(asset_state_returns[PORTFOLIO_ASSETS], weights)
            cvar = risk_contributions_cvar(asset_state_returns[PORTFOLIO_ASSETS], weights)
            btc_vol = vol.loc[vol["asset"] == "ret_btc"].iloc[0]
            btc_cvar = cvar.loc[cvar["asset"] == "ret_btc"].iloc[0]
            btc_rc_rows.append(
                {
                    "portfolio_id": portfolio_id,
                    "portfolio_family": family,
                    "btc_weight": btc_weight,
                    "hmm4_state": state_name,
                    "hmm4_state_id": state_id,
                    "state_n_weeks": int(len(state_frame)),
                    "state_sample_warning": "small_state_sample" if len(state_frame) < 40 else "",
                    "btc_component_contribution_vol": float(btc_vol["component_contribution_vol"]),
                    "btc_component_share_vol": float(btc_vol["component_share_vol"]),
                    "vol_share_sum_check": float(vol["component_share_vol"].sum()),
                    "btc_component_contribution_cvar_loss": float(btc_cvar["component_contribution_cvar_loss"]),
                    "btc_component_share_cvar": float(btc_cvar["component_share_cvar"]),
                    "cvar_share_sum_check": float(cvar["component_share_cvar"].sum()),
                    "cvar_tail_scenario_count": int(btc_cvar["tail_scenario_count"]),
                    "portfolio_cvar_loss": float(btc_cvar["portfolio_cvar_loss"]),
                }
            )

            dd = state_drawdown_diagnostics(portfolio_frame, state_name)
            dd.update(
                {
                    "portfolio_id": portfolio_id,
                    "portfolio_family": family,
                    "btc_weight": btc_weight,
                    "hmm4_state": state_name,
                    "hmm4_state_id": state_id,
                }
            )
            drawdown_rows.append(dd)

    summary = pd.DataFrame(summary_rows).sort_values(["portfolio_family", "btc_weight", "hmm4_state_id"])
    btc_rc = pd.DataFrame(btc_rc_rows).sort_values(["portfolio_family", "btc_weight", "hmm4_state_id"])
    drawdowns = pd.DataFrame(drawdown_rows).sort_values(["portfolio_family", "btc_weight", "hmm4_state_id"])
    summary.to_csv(dirs["results"] / "state_conditioned_portfolio_summary.csv", index=False)
    btc_rc.to_csv(dirs["results"] / "state_conditioned_btc_risk_contributions.csv", index=False)
    drawdowns.to_csv(dirs["results"] / "state_conditioned_drawdowns.csv", index=False)
    plot_state_btc_contributions(btc_rc, dirs["figures"] / "state_conditioned_btc_risk_contribution.png")
    logging.info("State-conditioned portfolio diagnostics completed")
    return {"summary": summary, "btc_rc": btc_rc, "drawdowns": drawdowns}


def get_asset_returns_for_dates(dates: pd.Series, asset_panel: pd.DataFrame) -> pd.DataFrame:
    return asset_panel[asset_panel["date"].isin(set(dates))].sort_values("date").reset_index(drop=True)


def state_drawdown_diagnostics(portfolio_frame: pd.DataFrame, state_name: str) -> dict[str, Any]:
    state_only = portfolio_frame[portfolio_frame["hmm4_state"] == state_name].sort_values("date")
    noncontiguous_dd = max_drawdown(state_only["portfolio_return"])
    episodes = []
    current = []
    for _, row in portfolio_frame.sort_values("date").iterrows():
        if row["hmm4_state"] == state_name:
            current.append(row)
        elif current:
            episodes.append(pd.DataFrame(current))
            current = []
    if current:
        episodes.append(pd.DataFrame(current))

    worst_episode = {
        "worst_contiguous_episode_drawdown": np.nan,
        "worst_contiguous_episode_start": "",
        "worst_contiguous_episode_end": "",
        "worst_contiguous_episode_weeks": 0,
    }
    episode_lengths = []
    for episode in episodes:
        episode_lengths.append(len(episode))
        dd = max_drawdown(episode["portfolio_return"])
        if pd.isna(worst_episode["worst_contiguous_episode_drawdown"]) or dd < worst_episode["worst_contiguous_episode_drawdown"]:
            worst_episode = {
                "worst_contiguous_episode_drawdown": float(dd),
                "worst_contiguous_episode_start": pd.Timestamp(episode["date"].min()).strftime("%Y-%m-%d"),
                "worst_contiguous_episode_end": pd.Timestamp(episode["date"].max()).strftime("%Y-%m-%d"),
                "worst_contiguous_episode_weeks": int(len(episode)),
            }
    return {
        "state_n_weeks": int(len(state_only)),
        "noncontiguous_state_ordered_drawdown": float(noncontiguous_dd),
        "n_contiguous_episodes": int(len(episodes)),
        "average_episode_weeks": float(np.mean(episode_lengths)) if episode_lengths else np.nan,
        **worst_episode,
        "drawdown_interpretation_note": "Noncontiguous drawdown is a diagnostic grouping statistic; contiguous episode drawdown preserves calendar adjacency.",
    }


def run_state_conditioned_wrapper(validation: dict[str, Any], weights_payload: dict[str, Any], portfolio_outputs: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    portfolio_outputs = dict(portfolio_outputs)
    portfolio_outputs["asset_panel"] = validation["panel"]
    return compute_state_conditioned_outputs(portfolio_outputs, weights_payload, dirs)


def plot_portfolio_performance(performance: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for family, frame in performance.groupby("portfolio_family"):
        ax.plot(frame["btc_weight"], frame["annualized_volatility"], marker="o", label=f"{family} volatility")
    ax.set_title("Portfolio Annualized Volatility by Fixed BTC Weight")
    ax.set_xlabel("BTC Weight")
    ax.set_ylabel("Annualized Volatility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_btc_risk_budget(btc_budget: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for family, frame in btc_budget.groupby("portfolio_family"):
        ax.plot(frame["btc_weight"], frame["btc_component_share_vol"], marker="o", label=f"{family} vol share")
        ax.plot(frame["btc_weight"], frame["btc_component_share_cvar"], marker="s", linestyle="--", label=f"{family} CVaR share")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("BTC Risk Contribution Share by Fixed Weight")
    ax.set_xlabel("BTC Weight")
    ax.set_ylabel("Contribution Share")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_drawdowns(return_series: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for portfolio_id, frame in return_series.groupby("portfolio_id"):
        if not portfolio_id.endswith("05pct") and not portfolio_id.endswith("00pct"):
            continue
        ordered = frame.sort_values("date")
        wealth = (1.0 + ordered["portfolio_return"].fillna(0.0)).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        ax.plot(ordered["date"], drawdown, linewidth=1.1, label=portfolio_id)
    ax.set_title("Selected Portfolio Drawdowns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_state_btc_contributions(btc_rc: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_frame = btc_rc[btc_rc["btc_weight"] == 0.05]
    for family, frame in plot_frame.groupby("portfolio_family"):
        ax.plot(frame["hmm4_state"], frame["btc_component_share_cvar"], marker="o", label=f"{family} BTC CVaR share")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("BTC CVaR Contribution Share by HMM-4 State at 5% BTC Weight")
    ax.set_xlabel("HMM-4 State")
    ax.set_ylabel("BTC CVaR Contribution Share")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_explainability_artifacts(
    args: argparse.Namespace,
    inputs: dict[str, Any],
    validation: dict[str, Any],
    weights_payload: dict[str, Any],
    dirs: dict[str, Path],
) -> dict[str, str]:
    paths = inputs["paths"]
    data_lineage = pd.DataFrame(
        [
            {
                "artifact": name,
                "path": str(path),
                "sha256": inputs["input_hashes"][name],
                "role": role,
                "modified_by_part2_runner": False,
            }
            for name, path, role in [
                ("asset_returns_main_weekly", paths["asset_returns_main_weekly"], "Frozen cleaned weekly asset returns."),
                ("cleaning_report", paths["cleaning_report"], "Frozen data-cleaning audit report."),
                ("part1_manifest", paths["part1_manifest"], "Part 1 run manifest validating upstream state model."),
                ("hmm4_state_labels", paths["hmm4_state_labels"], "Full-sample descriptive HMM-4 macro-state labels used for grouping diagnostics."),
                ("hmm4_state_profiles", paths["hmm4_state_profiles"], "Part 1 state profile table used for state sample-size audit."),
            ]
        ]
    )
    data_lineage_path = dirs["results"] / "data_lineage.csv"
    data_lineage.to_csv(data_lineage_path, index=False)

    weight_dictionary = weights_payload["fixed_weights_long"].copy()
    weight_dictionary["interpretation_note"] = (
        "Fixed target weight used for risk-budget diagnostics only; not a trading or rebalancing rule."
    )
    weight_dictionary_path = dirs["results"] / "portfolio_weight_dictionary.csv"
    weight_dictionary.to_csv(weight_dictionary_path, index=False)

    risk_methodology = {
        "volatility_contribution": {
            "method": "Euler covariance contribution",
            "formula": "component_i = weight_i * (Sigma * weight)_i / portfolio_volatility",
            "share_formula": "component_i / portfolio_volatility",
            "sum_check": "Shares should sum to approximately 1 for each portfolio.",
        },
        "cvar_contribution": {
            "method": "Empirical left-tail loss contribution",
            "tail_alpha": TAIL_ALPHA,
            "formula": "component_i = mean(-weight_i * asset_return_i | portfolio_loss >= VaR_loss_95)",
            "share_formula": "component_i / portfolio_CVaR_loss",
            "note": "Contributions may be negative when an asset has positive average return in portfolio tail weeks.",
        },
        "drawdown": {
            "full_sample": "Computed on full chronological portfolio return path.",
            "state_conditioned": "Reports noncontiguous grouped drawdown and worst contiguous same-state episode drawdown separately.",
        },
    }
    risk_methodology_path = dirs["results"] / "risk_contribution_methodology.json"
    write_json(risk_methodology_path, risk_methodology)

    assumption_audit = {
        "purpose": "Document Part 2 portfolio and risk-budget assumptions before thesis interpretation.",
        "assumptions": [
            {
                "choice": "Static All Weather and static full-sample ERC baselines",
                "rationale": "Part 2 establishes long-term base portfolios before implementability and rolling-window experiments.",
                "implication": "Weights are diagnostic target weights, not dynamically estimated trading weights.",
            },
            {
                "choice": "Fixed BTC weights of 0%, 1%, 2%, 3%, and 5%",
                "rationale": "Tests BTC risk-budget impact at small allocation levels without optimizing on realized BTC returns.",
                "implication": "No BTC weight is selected as optimal in Part 2.",
            },
            {
                "choice": "HMM-4 states used only for grouping diagnostics",
                "rationale": "Part 1 state labels are full-sample descriptive labels.",
                "implication": "State-conditioned tables must not be described as real-time allocation rules.",
            },
            {
                "choice": "BIL excluded from Part 2 main portfolios",
                "rationale": "Part 2 focuses on risky All Weather/ERC base assets and BTC risk contribution.",
                "implication": "Cash parking and BIL robustness are deferred to later conditional allocation work.",
            },
        ],
        "limitations_to_discuss": [
            "State_2 has only 30 weeks, so state-conditioned tail metrics are sample-limited.",
            "Fixed target-weight returns do not include transaction costs, turnover, or rebalancing implementation.",
            "Static ERC uses full-sample covariance and is descriptive, not a real-time rolling estimate.",
        ],
    }
    assumption_path = dirs["results"] / "model_assumption_audit.json"
    write_json(assumption_path, assumption_audit)

    methodology_text = f"""# Part 2 Methodology Audit

## Purpose
Part 2 builds no-BTC All Weather and ERC baseline portfolios, then adds fixed BTC weights of 0%, 1%, 2%, 3%, and 5%. It reports return and risk-budget diagnostics only. It does not estimate a conditional allocation rule, trading strategy, transaction costs, or turnover.

## Inputs
- Cleaned asset returns: `{paths["asset_returns_main_weekly"]}`.
- Cleaning report: `{paths["cleaning_report"]}`.
- Part 1 HMM-4 state labels: `{paths["hmm4_state_labels"]}`.
- Part 1 manifest: `{paths["part1_manifest"]}`.

The effective Part 2 sample is the inner join between asset returns and HMM-4 labels: 425 weekly observations from 2018-02-09 to 2026-03-27.

## Portfolio Construction
All Weather no-BTC weights are fixed at SPY 30%, TLT 40%, IEF 15%, GLD 7.5%, and DBC 7.5%. ERC uses the same five assets and one static full-sample covariance matrix. Fixed BTC portfolios scale the no-BTC base weights by `1 - btc_weight` and assign the remaining fixed share to BTC.

## Risk Contribution
Volatility contribution uses Euler covariance contribution. CVaR contribution uses empirical portfolio left-tail weeks and reports loss contributions, which can be negative for hedging assets. State-conditioned results group the fixed portfolios by HMM-4 state and do not change weights.

## Discussion Boundaries
- No result in Part 2 is a recommended BTC allocation.
- State-conditioned results are diagnostic and full-sample descriptive.
- BIL, transaction costs, turnover, rolling ERC, and conditional BTC de-risking are outside Part 2.
"""
    methodology_path = dirs["results"] / "methodology_audit.md"
    methodology_path.write_text(methodology_text, encoding="utf-8")
    logging.info("Explainability artifacts completed")
    return {
        "data_lineage": str(data_lineage_path),
        "portfolio_weight_dictionary": str(weight_dictionary_path),
        "risk_contribution_methodology": str(risk_methodology_path),
        "model_assumption_audit": str(assumption_path),
        "methodology_audit": str(methodology_path),
    }


def validate_outputs(dirs: dict[str, Path]) -> dict[str, Any]:
    fixed = pd.read_csv(dirs["results"] / "fixed_btc_portfolio_weights.csv")
    vol = pd.read_csv(dirs["results"] / "full_sample_risk_contributions_vol.csv")
    cvar = pd.read_csv(dirs["results"] / "full_sample_risk_contributions_cvar.csv")
    state_btc = pd.read_csv(dirs["results"] / "state_conditioned_btc_risk_contributions.csv")
    erc = pd.read_csv(dirs["results"] / "erc_optimization_diagnostics.csv")

    weight_checks = []
    for portfolio_id, frame in fixed.groupby("portfolio_id"):
        weight_sum = float(frame["weight"].sum())
        btc_weight = float(frame.loc[frame["asset"] == "ret_btc", "weight"].iloc[0])
        weight_checks.append(
            {
                "portfolio_id": portfolio_id,
                "weight_sum": weight_sum,
                "min_weight": float(frame["weight"].min()),
                "btc_weight": btc_weight,
                "weight_sum_ok": abs(weight_sum - 1.0) < 1e-12,
                "nonnegative_ok": float(frame["weight"].min()) >= -1e-12,
            }
        )
    vol_share_errors = vol.groupby("portfolio_id")["component_share_vol"].sum().sub(1.0).abs()
    cvar_share_errors = cvar.groupby("portfolio_id")["component_share_cvar"].sum().sub(1.0).abs()
    state_cvar_errors = state_btc.groupby(["portfolio_id", "hmm4_state"])["cvar_share_sum_check"].first().sub(1.0).abs()
    state_vol_errors = state_btc.groupby(["portfolio_id", "hmm4_state"])["vol_share_sum_check"].first().sub(1.0).abs()

    summary = {
        "status": "passed",
        "portfolio_count": int(fixed["portfolio_id"].nunique()),
        "weight_checks": weight_checks,
        "erc_optimizer_success": bool(erc["optimizer_success"].iloc[0]),
        "erc_max_abs_rc_share_error_vs_equal": float(erc["max_abs_rc_share_error_vs_equal"].iloc[0]),
        "max_full_sample_vol_share_sum_error": float(vol_share_errors.max()),
        "max_full_sample_cvar_share_sum_error": float(cvar_share_errors.max()),
        "max_state_vol_share_sum_error": float(state_vol_errors.max()),
        "max_state_cvar_share_sum_error": float(state_cvar_errors.max()),
        "minimum_state_tail_scenario_count": int(state_btc["cvar_tail_scenario_count"].min()),
    }
    require(all(row["weight_sum_ok"] and row["nonnegative_ok"] for row in weight_checks), "Portfolio weight check failed")
    require(summary["erc_optimizer_success"], "ERC optimizer did not succeed")
    require(summary["max_full_sample_vol_share_sum_error"] < 1e-10, "Full-sample volatility contribution shares do not sum to 1")
    require(summary["max_full_sample_cvar_share_sum_error"] < 1e-10, "Full-sample CVaR contribution shares do not sum to 1")
    require(summary["max_state_vol_share_sum_error"] < 1e-10, "State volatility contribution shares do not sum to 1")
    require(summary["max_state_cvar_share_sum_error"] < 1e-10, "State CVaR contribution shares do not sum to 1")
    write_json(dirs["results"] / "output_validation_summary.json", summary)
    logging.info("Output validation completed")
    return summary


def build_manifest(
    args: argparse.Namespace,
    run_id: str,
    dirs: dict[str, Path],
    inputs: dict[str, Any],
    validation: dict[str, Any],
    weights_payload: dict[str, Any],
    output_validation: dict[str, Any],
    explainability: dict[str, str],
) -> dict[str, Any]:
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "objective": "Part 2 All Weather/ERC baselines and fixed BTC risk-budget diagnostics",
        "input_dir": str(args.input_dir),
        "part1_run_dir": str(args.part1_run_dir),
        "output_dir": str(args.output_dir),
        "run_dir": str(dirs["root"]),
        "input_hashes": inputs["input_hashes"],
        "package_versions": package_versions(),
        "random_seed": args.seed,
        "sample": validation["summary"]["part2_sample"],
        "parameters": {
            "base_assets": BASE_ASSETS,
            "portfolio_assets": PORTFOLIO_ASSETS,
            "all_weather_weights": ALL_WEATHER_WEIGHTS,
            "btc_weights": BTC_WEIGHTS,
            "erc": {
                "method": "static full-sample long-only ERC",
                "optimizer": "scipy.optimize.minimize SLSQP",
                "covariance_sample": "Part 2 state-aligned sample",
            },
            "risk_contribution": {
                "volatility": "Euler covariance contribution",
                "cvar": "Empirical left-tail loss contribution",
                "tail_alpha": TAIL_ALPHA,
            },
            "state_usage": "HMM-4 states are diagnostic groups only; weights do not depend on state.",
        },
        "erc_diagnostics": weights_payload["erc_diagnostics"].iloc[0].to_dict(),
        "output_validation": output_validation,
        "outputs": {
            "checkpoints": str(dirs["checkpoints"]),
            "results": str(dirs["results"]),
            "figures": str(dirs["figures"]),
            "models": str(dirs["models"]),
            "logs": str(dirs["logs"]),
            "explainability_artifacts": explainability,
        },
        "scope_notes": [
            "No conditional allocation rule in Part 2.",
            "No trading strategy, transaction cost, turnover, or rolling ERC in Part 2.",
            "BIL is excluded from Part 2 main portfolios.",
            "State-conditioned outputs are descriptive diagnostics based on Part 1 full-sample HMM-4 labels.",
        ],
    }
    write_json(dirs["root"] / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    run_id = args.run_id or now_run_id()
    dirs = ensure_dirs(args.output_dir / run_id)
    setup_logging(dirs["logs"])
    logging.info("Starting Part 2 run: %s", run_id)

    inputs = read_inputs(args)
    enforce_resume_input_hashes(args, dirs, inputs["input_hashes"])
    validation = load_or_run(dirs, "01_input_validation", args.resume, lambda: validate_inputs(inputs, dirs))
    weights_payload = load_or_run(dirs, "02_portfolio_weights", args.resume, lambda: build_portfolio_weights(validation, dirs))
    portfolio_outputs = load_or_run(
        dirs,
        "03_portfolio_outputs",
        args.resume,
        lambda: compute_portfolio_outputs(validation, weights_payload, dirs),
    )
    state_outputs = load_or_run(
        dirs,
        "04_state_conditioned_outputs",
        args.resume,
        lambda: run_state_conditioned_wrapper(validation, weights_payload, portfolio_outputs, dirs),
    )
    explainability = load_or_run(
        dirs,
        "05_explainability_artifacts",
        args.resume,
        lambda: write_explainability_artifacts(args, inputs, validation, weights_payload, dirs),
    )
    output_validation = load_or_run(dirs, "06_output_validation", args.resume, lambda: validate_outputs(dirs))
    manifest = build_manifest(args, run_id, dirs, inputs, validation, weights_payload, output_validation, explainability)
    logging.info("Completed Part 2 run: %s", run_id)
    logging.info("Results directory: %s", dirs["results"])
    _ = state_outputs, manifest


if __name__ == "__main__":
    main()
