#!/usr/bin/env python3
"""Part 10 experiment runner: matched benchmarks and BTC risk-cap sensitivity.

This runner extends the frozen Part 1-9 evidence chain without overwriting it.
It compares the existing HMM-conditioned BTC satellite rule against simpler
matched-exposure and cap-only alternatives.
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


EXPECTED_ASSET_START = "2018-01-12"
EXPECTED_ASSET_END = "2026-03-27"
EXPECTED_ASSET_ROWS = 429
EXPECTED_STATE_START = "2018-02-09"
EXPECTED_STATE_END = "2026-03-27"
EXPECTED_STATE_ROWS = 425
EXPECTED_STATE_COUNTS = {"state_0": 149, "state_1": 167, "state_2": 30, "state_3": 79}

TRADING_WEEKS_PER_YEAR = 52
TAIL_ALPHA = 0.05
FLOAT_TOL = 1e-10

BASE_ASSETS = ["ret_spy", "ret_tlt", "ret_ief", "ret_gld", "ret_dbc"]
PORTFOLIO_ASSETS = ["ret_btc"] + BASE_ASSETS
ASSET_RETURN_COLS = PORTFOLIO_ASSETS + ["ret_bil"]
PORTFOLIO_FAMILIES = ["all_weather", "erc"]

ORIGINAL_BTC_GRID = [0.0, 0.01, 0.02, 0.03, 0.05]
FINE_BTC_GRID = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
RISK_BUDGET_CAPS = [0.05, 0.10, 0.15, 0.20]
MATCHED_FIXED_BTC_WEIGHT = 0.010941176470588234
RAW_MAIN_RULE = {"state_0": 0.03, "state_1": 0.01, "state_2": 0.0, "state_3": 0.0}

REQUIRED_RESULTS = [
    "input_validation_summary.json",
    "output_validation_summary.json",
    "data_lineage.csv",
    "model_assumption_audit.json",
    "methodology_audit.md",
    "part10_scenario_dictionary.csv",
    "part10_target_weight_performance_summary.csv",
    "part10_risk_contribution_summary.csv",
    "part10_executed_state_weights.csv",
    "part10_weekly_weights.csv",
    "part10_return_series.csv",
    "part10_pairwise_benchmark_comparison.csv",
    "part10_key_findings.json",
]

REQUIRED_FIGURES = [
    "part10_cap_sensitivity_btc_weight.png",
    "part10_cap_sensitivity_risk_contribution.png",
    "part10_conditional_vs_matched_fixed_performance.png",
    "part10_conditional_vs_cap_only_performance.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 10 benchmark and cap-sensitivity diagnostics.")
    parser.add_argument("--input-dir", default="data_2026/cleaned", type=Path)
    parser.add_argument(
        "--part1-run-dir",
        default="outputs/part1_btc_macro_state_outputs/part1_btc_macro_state/colab_part1_seed42",
        type=Path,
    )
    parser.add_argument(
        "--part2-run-dir",
        default="outputs/part2_portfolio_risk_budget_outputs/part2_portfolio_risk_budget/colab_part2_seed42",
        type=Path,
    )
    parser.add_argument(
        "--part4-run-dir",
        default="outputs/part4_conditional_btc_allocation_outputs/part4_conditional_btc_allocation/colab_part4_seed42",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/part10_benchmark_cap_sensitivity_outputs/part10_benchmark_cap_sensitivity",
        type=Path,
    )
    parser.add_argument("--run-id", default="colab_part10_seed42")
    parser.add_argument("--seed", default=42, type=int)
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.replace("\n", " "), "platform": platform.platform()}
    for package in ["numpy", "pandas", "matplotlib"]:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def date_string(series: pd.Series, fn: str) -> str:
    value = series.min() if fn == "min" else series.max()
    return pd.Timestamp(value).strftime("%Y-%m-%d")


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


def read_inputs(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "asset_returns_main_weekly": args.input_dir / "asset_returns_main_weekly.csv",
        "cleaning_report": args.input_dir / "cleaning_report.json",
        "part1_manifest": args.part1_run_dir / "run_manifest.json",
        "part1_validation_summary": args.part1_run_dir / "results" / "validation_summary.json",
        "hmm4_state_labels": args.part1_run_dir / "results" / "hmm4_state_labels.csv",
        "hmm4_state_profiles": args.part1_run_dir / "results" / "hmm4_state_profiles.csv",
        "part2_manifest": args.part2_run_dir / "run_manifest.json",
        "part2_input_validation_summary": args.part2_run_dir / "results" / "input_validation_summary.json",
        "part2_output_validation_summary": args.part2_run_dir / "results" / "output_validation_summary.json",
        "part2_baseline_portfolio_weights": args.part2_run_dir / "results" / "baseline_portfolio_weights.csv",
        "part4_manifest": args.part4_run_dir / "run_manifest.json",
        "part4_input_validation_summary": args.part4_run_dir / "results" / "input_validation_summary.json",
        "part4_output_validation_summary": args.part4_run_dir / "results" / "output_validation_summary.json",
        "part4_allocation_rule_definition": args.part4_run_dir / "results" / "allocation_rule_definition.csv",
        "part4_performance_summary": args.part4_run_dir / "results" / "conditional_portfolio_performance_summary.csv",
    }
    for name, path in paths.items():
        require(path.exists(), f"Missing required input file: {name} at {path}")

    payload = {
        "paths": paths,
        "asset": pd.read_csv(paths["asset_returns_main_weekly"], parse_dates=["date"]),
        "cleaning_report": read_json(paths["cleaning_report"]),
        "part1_manifest": read_json(paths["part1_manifest"]),
        "part1_validation": read_json(paths["part1_validation_summary"]),
        "labels": pd.read_csv(paths["hmm4_state_labels"], parse_dates=["date"]),
        "profiles": pd.read_csv(paths["hmm4_state_profiles"]),
        "part2_manifest": read_json(paths["part2_manifest"]),
        "part2_input_validation": read_json(paths["part2_input_validation_summary"]),
        "part2_output_validation": read_json(paths["part2_output_validation_summary"]),
        "part2_baseline_weights": pd.read_csv(paths["part2_baseline_portfolio_weights"]),
        "part4_manifest": read_json(paths["part4_manifest"]),
        "part4_input_validation": read_json(paths["part4_input_validation_summary"]),
        "part4_output_validation": read_json(paths["part4_output_validation_summary"]),
        "part4_rule_definition": pd.read_csv(paths["part4_allocation_rule_definition"]),
        "part4_performance": pd.read_csv(paths["part4_performance_summary"]),
        "input_hashes": {name: file_sha256(path) for name, path in paths.items()},
    }
    return payload


def enforce_resume_input_hashes(args: argparse.Namespace, dirs: dict[str, Path], input_hashes: dict[str, str]) -> None:
    if not args.resume:
        return
    manifest_path = dirs["root"] / "run_manifest.json"
    if not manifest_path.exists():
        return
    old_manifest = read_json(manifest_path)
    require(old_manifest.get("input_hashes", {}) == input_hashes, "Input hashes changed since previous run")
    logging.info("Resume input hash check passed")


def build_base_weights(baseline: pd.DataFrame) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for family, frame in baseline.groupby("portfolio_family"):
        weights = frame.set_index("asset")["weight"].astype(float).to_dict()
        require(set(weights) == set(BASE_ASSETS), f"Unexpected base assets for {family}: {weights.keys()}")
        require(abs(sum(weights.values()) - 1.0) < 1e-8, f"Base weights do not sum to one for {family}")
        payload[str(family)] = weights
    require(set(payload) == set(PORTFOLIO_FAMILIES), "Missing base portfolio families")
    return payload


def validate_inputs(inputs: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    asset = inputs["asset"].copy()
    labels = inputs["labels"].copy()
    profiles = inputs["profiles"].copy()

    require(inputs["part1_validation"].get("status") == "passed", "Part 1 validation did not pass")
    require(inputs["part2_input_validation"].get("status") == "passed", "Part 2 input validation did not pass")
    require(inputs["part2_output_validation"].get("status") == "passed", "Part 2 output validation did not pass")
    require(inputs["part4_input_validation"].get("status") == "passed", "Part 4 input validation did not pass")
    require(inputs["part4_output_validation"].get("status") == "passed", "Part 4 output validation did not pass")

    require(len(asset) == EXPECTED_ASSET_ROWS, f"Unexpected asset row count: {len(asset)}")
    require(date_string(asset["date"], "min") == EXPECTED_ASSET_START, "Unexpected asset start date")
    require(date_string(asset["date"], "max") == EXPECTED_ASSET_END, "Unexpected asset end date")
    require(asset["date"].dt.dayofweek.eq(4).all(), "Asset dates are not all Fridays")
    require(all(col in asset.columns for col in ASSET_RETURN_COLS), "Missing required asset return columns")
    require(asset[ASSET_RETURN_COLS].isna().sum().sum() == 0, "Missing values in asset returns")

    require(len(labels) == EXPECTED_STATE_ROWS, f"Unexpected HMM label row count: {len(labels)}")
    require(date_string(labels["date"], "min") == EXPECTED_STATE_START, "Unexpected label start date")
    require(date_string(labels["date"], "max") == EXPECTED_STATE_END, "Unexpected label end date")
    require(all(col in labels.columns for col in ["date", "hmm4_state", "hmm4_state_id"]), "Missing HMM state columns")
    state_counts = labels["hmm4_state"].value_counts().sort_index().to_dict()
    require(state_counts == EXPECTED_STATE_COUNTS, f"Unexpected state counts: {state_counts}")
    require(len(profiles) == 4, "Expected four HMM profiles")

    panel = asset[["date"] + PORTFOLIO_ASSETS].merge(
        labels[["date", "hmm4_state", "hmm4_state_id"]], on="date", how="inner", validate="one_to_one"
    )
    require(len(panel) == EXPECTED_STATE_ROWS, f"Unexpected analysis panel rows: {len(panel)}")
    require(date_string(panel["date"], "min") == EXPECTED_STATE_START, "Unexpected panel start date")
    require(date_string(panel["date"], "max") == EXPECTED_STATE_END, "Unexpected panel end date")

    base_weights = build_base_weights(inputs["part2_baseline_weights"])

    part4_main = inputs["part4_performance"]
    part4_main = part4_main[part4_main["rule_id"].eq("main_executed")]
    require(len(part4_main) == 2, "Expected two Part 4 main_executed rows")
    avg_weights = part4_main["average_btc_weight"].astype(float).to_numpy()
    require(np.allclose(avg_weights, MATCHED_FIXED_BTC_WEIGHT, atol=1e-12), "Part 4 average BTC weight changed")

    rule = inputs["part4_rule_definition"]
    main_exec = rule[rule["rule_id"].eq("main_executed")]
    require(len(main_exec) == 8, "Expected 8 Part 4 main_executed rule rows")

    summary = {
        "status": "passed",
        "sample_frozen": True,
        "asset_sample": {"rows": len(asset), "start": EXPECTED_ASSET_START, "end": EXPECTED_ASSET_END},
        "state_aligned_sample": {"rows": len(panel), "start": EXPECTED_STATE_START, "end": EXPECTED_STATE_END},
        "hmm4_state_counts": {k: int(v) for k, v in state_counts.items()},
        "original_btc_grid": ORIGINAL_BTC_GRID,
        "fine_btc_grid": FINE_BTC_GRID,
        "risk_budget_caps": RISK_BUDGET_CAPS,
        "matched_fixed_btc_weight": MATCHED_FIXED_BTC_WEIGHT,
        "raw_main_rule": RAW_MAIN_RULE,
        "base_weights": base_weights,
        "input_hashes": inputs["input_hashes"],
        "upstream_runs": {
            "part1_run_id": inputs["part1_manifest"].get("run_id"),
            "part2_run_id": inputs["part2_manifest"].get("run_id"),
            "part4_run_id": inputs["part4_manifest"].get("run_id"),
        },
    }
    write_json(dirs["results"] / "input_validation_summary.json", normalize_for_json(summary))
    logging.info("Input validation passed")
    return {"summary": summary, "analysis_panel": panel, "base_weights": base_weights}


def var_cvar(returns: pd.Series, alpha: float = TAIL_ALPHA) -> tuple[float, float, int]:
    clean = returns.dropna()
    require(len(clean) > 0, "No returns for VaR/CVaR")
    var = float(clean.quantile(alpha))
    tail = clean[clean <= var]
    require(len(tail) > 0, "No tail observations")
    return var, float(tail.mean()), int(len(tail))


def drawdown_from_returns(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    return float(drawdown.min())


def performance_metrics(returns: pd.Series) -> dict[str, Any]:
    clean = returns.dropna()
    var, cvar, tail_count = var_cvar(clean)
    vol = float(clean.std(ddof=1))
    return {
        "count": int(len(clean)),
        "mean_weekly": float(clean.mean()),
        "median_weekly": float(clean.median()),
        "volatility_weekly": vol,
        "annualized_mean_arithmetic": float(clean.mean() * TRADING_WEEKS_PER_YEAR),
        "annualized_volatility": float(vol * math.sqrt(TRADING_WEEKS_PER_YEAR)),
        "min_weekly": float(clean.min()),
        "max_weekly": float(clean.max()),
        "var_95_weekly": var,
        "cvar_95_weekly": cvar,
        "tail_scenario_count": tail_count,
        "max_drawdown": drawdown_from_returns(clean),
        "positive_week_share": float((clean > 0.0).mean()),
        "sharpe_annualized_zero_rf": float(clean.mean() / vol * math.sqrt(TRADING_WEEKS_PER_YEAR)) if vol > 0 else float("nan"),
    }


def weights_from_btc(base_weights: dict[str, float], btc_weight: float) -> dict[str, float]:
    weights = {"ret_btc": float(btc_weight)}
    for asset, base_weight in base_weights.items():
        weights[asset] = float((1.0 - btc_weight) * base_weight)
    require(min(weights.values()) >= -FLOAT_TOL, f"Negative weight found for BTC weight {btc_weight}")
    require(abs(sum(weights.values()) - 1.0) < 1e-8, f"Weights do not sum to one for BTC weight {btc_weight}")
    return weights


def compute_portfolio_return(panel: pd.DataFrame, weights: dict[str, float]) -> tuple[pd.Series, pd.DataFrame]:
    components = pd.DataFrame({"date": panel["date"]})
    for asset in PORTFOLIO_ASSETS:
        components[asset] = float(weights.get(asset, 0.0)) * panel[asset].astype(float).to_numpy()
    returns = components[PORTFOLIO_ASSETS].sum(axis=1)
    return pd.Series(returns.to_numpy(), index=panel.index), components


def compute_vol_contributions(component_frame: pd.DataFrame) -> pd.DataFrame:
    pivot = component_frame.set_index("date")[PORTFOLIO_ASSETS]
    portfolio = pivot.sum(axis=1)
    vol = float(portfolio.std(ddof=1))
    require(vol > 0, "Portfolio volatility is zero")
    rows = []
    for asset in PORTFOLIO_ASSETS:
        cov = float(np.cov(pivot[asset], portfolio, ddof=1)[0, 1])
        contribution = cov / vol
        rows.append(
            {
                "asset": asset,
                "portfolio_volatility_weekly": vol,
                "component_contribution_vol": contribution,
                "component_share_vol": contribution / vol,
            }
        )
    share_sum = float(sum(row["component_share_vol"] for row in rows))
    for row in rows:
        row["share_sum_check_vol"] = share_sum
    return pd.DataFrame(rows)


def compute_cvar_contributions(component_frame: pd.DataFrame) -> pd.DataFrame:
    pivot = component_frame.set_index("date")[PORTFOLIO_ASSETS]
    portfolio = pivot.sum(axis=1)
    var = float(portfolio.quantile(TAIL_ALPHA))
    tail_mask = portfolio <= var
    tail = pivot.loc[tail_mask]
    portfolio_cvar_loss = float((-portfolio.loc[tail_mask]).mean())
    require(abs(portfolio_cvar_loss) > 1e-15, "Portfolio CVaR loss is zero")
    rows = []
    for asset in PORTFOLIO_ASSETS:
        contribution = float((-tail[asset]).mean())
        rows.append(
            {
                "asset": asset,
                "portfolio_cvar_95_weekly": float(portfolio.loc[tail_mask].mean()),
                "portfolio_cvar_loss": portfolio_cvar_loss,
                "tail_scenario_count": int(tail_mask.sum()),
                "component_contribution_cvar_loss": contribution,
                "component_share_cvar": contribution / portfolio_cvar_loss,
            }
        )
    share_sum = float(sum(row["component_share_cvar"] for row in rows))
    for row in rows:
        row["share_sum_check_cvar"] = share_sum
    return pd.DataFrame(rows)


def btc_risk_shares(component_frame: pd.DataFrame) -> dict[str, Any]:
    vol = compute_vol_contributions(component_frame)
    cvar = compute_cvar_contributions(component_frame)
    btc_vol = vol[vol["asset"].eq("ret_btc")].iloc[0]
    btc_cvar = cvar[cvar["asset"].eq("ret_btc")].iloc[0]
    return {
        "btc_share_vol": float(btc_vol["component_share_vol"]),
        "btc_share_cvar": float(btc_cvar["component_share_cvar"]),
        "portfolio_volatility_weekly": float(btc_vol["portfolio_volatility_weekly"]),
        "portfolio_cvar_95_weekly": float(btc_cvar["portfolio_cvar_95_weekly"]),
        "tail_scenario_count": int(btc_cvar["tail_scenario_count"]),
    }


def scenario_slug(value: float) -> str:
    return f"{int(round(value * 10000)):04d}bp"


def cap_slug(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "nocap"
    return f"{int(round(value * 100)):02d}pct"


def constant_scenario_id(benchmark_family: str, family: str, btc_weight: float) -> str:
    if benchmark_family == "no_btc_baseline":
        return f"part10__no_btc_baseline__{family}"
    if benchmark_family == "matched_fixed_btc":
        return f"part10__matched_fixed_btc__{family}__{scenario_slug(btc_weight)}"
    return f"part10__fixed_btc_grid__{family}__{scenario_slug(btc_weight)}"


def cap_only_scenario_id(family: str, cap: float) -> str:
    return f"part10__cap_only__{family}__cap_{cap_slug(cap)}"


def conditional_scenario_id(family: str, cap: float) -> str:
    return f"part10__conditional_cap__{family}__cap_{cap_slug(cap)}"


def conditional_fine_grid_scenario_id(family: str, cap: float) -> str:
    return f"part10__conditional_cap_fine_grid__{family}__cap_{cap_slug(cap)}"


def evaluate_constant_scenario(
    panel: pd.DataFrame,
    family: str,
    base_weights: dict[str, float],
    btc_weight: float,
    scenario_id: str,
) -> dict[str, Any]:
    weights = weights_from_btc(base_weights, btc_weight)
    returns, components = compute_portfolio_return(panel, weights)
    component_long = components.melt(id_vars=["date"], value_vars=PORTFOLIO_ASSETS, var_name="asset", value_name="component_return")
    weight_rows = []
    for _, obs in panel[["date", "hmm4_state", "hmm4_state_id"]].iterrows():
        for asset, weight in weights.items():
            weight_rows.append(
                {
                    "date": obs["date"],
                    "scenario_id": scenario_id,
                    "portfolio_family": family,
                    "hmm4_state": obs["hmm4_state"],
                    "hmm4_state_id": int(obs["hmm4_state_id"]),
                    "asset": asset,
                    "weight": weight,
                    "btc_weight": btc_weight,
                }
            )
    return {
        "returns": pd.DataFrame(
            {
                "date": panel["date"],
                "scenario_id": scenario_id,
                "portfolio_family": family,
                "hmm4_state": panel["hmm4_state"],
                "hmm4_state_id": panel["hmm4_state_id"],
                "btc_weight": btc_weight,
                "portfolio_return": returns.to_numpy(),
                "ret_btc_component": components["ret_btc"].to_numpy(),
            }
        ),
        "components": component_long.assign(scenario_id=scenario_id, portfolio_family=family),
        "weights": pd.DataFrame(weight_rows),
        "full_components": components,
    }


def candidate_constant_caps(panel: pd.DataFrame, family: str, base_weights: dict[str, float], btc_weight: float) -> dict[str, Any]:
    scenario = evaluate_constant_scenario(panel, family, base_weights, btc_weight, "candidate")
    risk = btc_risk_shares(scenario["full_components"])
    return risk


def select_cap_only_weight(panel: pd.DataFrame, family: str, base_weights: dict[str, float], cap: float) -> tuple[float, dict[str, Any]]:
    audit = []
    for btc_weight in sorted(FINE_BTC_GRID, reverse=True):
        risk = candidate_constant_caps(panel, family, base_weights, btc_weight)
        row = {
            "candidate_btc_weight": btc_weight,
            **risk,
            "full_sample_vol_cap_ok": risk["btc_share_vol"] <= cap + FLOAT_TOL,
            "full_sample_cvar_cap_ok": risk["btc_share_cvar"] <= cap + FLOAT_TOL,
        }
        row["all_caps_ok"] = bool(row["full_sample_vol_cap_ok"] and row["full_sample_cvar_cap_ok"])
        audit.append(row)
        if row["all_caps_ok"]:
            return btc_weight, {"candidate_audit": audit, **row}
    return 0.0, {"candidate_audit": audit, **audit[-1]}


def evaluate_conditional_rule(
    panel: pd.DataFrame,
    family: str,
    base_weights: dict[str, float],
    state_weight_map: dict[str, float],
    scenario_id: str,
) -> dict[str, Any]:
    returns_rows = []
    component_rows = []
    weight_rows = []
    for _, obs in panel.sort_values("date").iterrows():
        state = str(obs["hmm4_state"])
        btc_weight = float(state_weight_map[state])
        weights = weights_from_btc(base_weights, btc_weight)
        component_returns = {asset: weights[asset] * float(obs[asset]) for asset in PORTFOLIO_ASSETS}
        portfolio_return = float(sum(component_returns.values()))
        returns_rows.append(
            {
                "date": obs["date"],
                "scenario_id": scenario_id,
                "portfolio_family": family,
                "hmm4_state": state,
                "hmm4_state_id": int(obs["hmm4_state_id"]),
                "btc_weight": btc_weight,
                "portfolio_return": portfolio_return,
                "ret_btc_component": component_returns["ret_btc"],
            }
        )
        for asset, weight in weights.items():
            weight_rows.append(
                {
                    "date": obs["date"],
                    "scenario_id": scenario_id,
                    "portfolio_family": family,
                    "hmm4_state": state,
                    "hmm4_state_id": int(obs["hmm4_state_id"]),
                    "asset": asset,
                    "weight": weight,
                    "btc_weight": btc_weight,
                }
            )
            component_rows.append(
                {
                    "date": obs["date"],
                    "scenario_id": scenario_id,
                    "portfolio_family": family,
                    "hmm4_state": state,
                    "asset": asset,
                    "component_return": component_returns[asset],
                }
            )
    returns = pd.DataFrame(returns_rows)
    components_long = pd.DataFrame(component_rows)
    components_wide = components_long.pivot_table(index="date", columns="asset", values="component_return", aggfunc="first").reset_index()
    return {
        "returns": returns,
        "components": components_long,
        "weights": pd.DataFrame(weight_rows),
        "full_components": components_wide,
    }


def conditional_candidate_risk(
    panel: pd.DataFrame,
    family: str,
    base_weights: dict[str, float],
    state: str,
    candidate: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    full_scenario = evaluate_constant_scenario(panel, family, base_weights, candidate, "candidate_full")
    state_panel = panel[panel["hmm4_state"].eq(state)].copy()
    state_scenario = evaluate_constant_scenario(state_panel, family, base_weights, candidate, "candidate_state")
    return btc_risk_shares(full_scenario["full_components"]), btc_risk_shares(state_scenario["full_components"])


def select_conditional_weight(
    panel: pd.DataFrame,
    family: str,
    base_weights: dict[str, float],
    state: str,
    raw_weight: float,
    cap: float,
    grid: list[float],
) -> tuple[float, dict[str, Any]]:
    candidates = sorted([w for w in grid if w <= raw_weight + FLOAT_TOL], reverse=True)
    audit = []
    for candidate in candidates:
        full_risk, state_risk = conditional_candidate_risk(panel, family, base_weights, state, candidate)
        row = {
            "candidate_btc_weight": candidate,
            "full_sample_btc_share_vol": full_risk["btc_share_vol"],
            "full_sample_btc_share_cvar": full_risk["btc_share_cvar"],
            "state_btc_share_vol": state_risk["btc_share_vol"],
            "state_btc_share_cvar": state_risk["btc_share_cvar"],
            "state_n_weeks": int((panel["hmm4_state"] == state).sum()),
            "state_cvar_tail_scenario_count": int(state_risk["tail_scenario_count"]),
        }
        row.update(
            {
                "full_sample_vol_cap_ok": row["full_sample_btc_share_vol"] <= cap + FLOAT_TOL,
                "full_sample_cvar_cap_ok": row["full_sample_btc_share_cvar"] <= cap + FLOAT_TOL,
                "state_vol_cap_ok": row["state_btc_share_vol"] <= cap + FLOAT_TOL,
                "state_cvar_cap_ok": row["state_btc_share_cvar"] <= cap + FLOAT_TOL,
                "candidate_le_raw_ok": candidate <= raw_weight + FLOAT_TOL,
            }
        )
        row["all_caps_ok"] = bool(
            row["full_sample_vol_cap_ok"]
            and row["full_sample_cvar_cap_ok"]
            and row["state_vol_cap_ok"]
            and row["state_cvar_cap_ok"]
            and row["candidate_le_raw_ok"]
        )
        audit.append(row)
        if row["all_caps_ok"]:
            return candidate, {"candidate_audit": audit, **row}
    if audit:
        return 0.0, {"candidate_audit": audit, **audit[-1]}
    return 0.0, {"candidate_audit": [], "all_caps_ok": True}


def sample_scope_components(payload: dict[str, Any], returns: pd.DataFrame, scope: str, state: str | None = None) -> pd.DataFrame:
    components = payload["components"]
    if scope == "full_sample":
        return payload["full_components"]
    if scope == "active_state":
        active_dates = returns[returns["btc_weight"] > FLOAT_TOL]["date"]
        frame = components[components["date"].isin(active_dates)]
    elif state is not None:
        dates = returns[returns["hmm4_state"].eq(state)]["date"]
        frame = components[components["date"].isin(dates)]
    else:
        raise ValueError(f"Unknown risk scope: {scope}")
    if frame.empty:
        return pd.DataFrame(columns=["date"] + PORTFOLIO_ASSETS)
    return frame.pivot_table(index="date", columns="asset", values="component_return", aggfunc="first").reset_index()


def build_all_scenarios(validation_payload: dict[str, Any], dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    panel = validation_payload["analysis_panel"].copy()
    base_weights_all = validation_payload["base_weights"]

    scenario_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    state_weight_rows: list[dict[str, Any]] = []
    weekly_weights: list[pd.DataFrame] = []
    return_frames: list[pd.DataFrame] = []

    scenario_payloads: dict[str, dict[str, Any]] = {}

    def register_scenario(
        scenario_id: str,
        benchmark_family: str,
        rule_id: str,
        family: str,
        risk_budget_cap: float | None,
        btc_grid_weight: float | None,
        uses_hmm_state: bool,
        uses_risk_cap: bool,
        description: str,
        payload: dict[str, Any],
        matched_average_btc_weight: float = MATCHED_FIXED_BTC_WEIGHT,
    ) -> None:
        returns = payload["returns"].copy()
        perf = performance_metrics(returns["portfolio_return"])
        perf_row = {
            "scenario_id": scenario_id,
            "benchmark_family": benchmark_family,
            "rule_id": rule_id,
            "portfolio_family": family,
            "risk_budget_cap": risk_budget_cap,
            "btc_grid_weight": btc_grid_weight,
            "average_btc_weight": float(returns["btc_weight"].mean()),
            "max_btc_weight": float(returns["btc_weight"].max()),
            "active_week_share": float((returns["btc_weight"] > FLOAT_TOL).mean()),
            **perf,
        }
        performance_rows.append(perf_row)
        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "benchmark_family": benchmark_family,
                "rule_id": rule_id,
                "portfolio_family": family,
                "risk_budget_cap": risk_budget_cap,
                "btc_grid_weight": btc_grid_weight,
                "matched_average_btc_weight": matched_average_btc_weight,
                "uses_hmm_state": uses_hmm_state,
                "uses_risk_cap": uses_risk_cap,
                "funding_rule": "BTC funded pro rata from the non-BTC base portfolio.",
                "sample_start": EXPECTED_STATE_START,
                "sample_end": EXPECTED_STATE_END,
                "sample_count": int(len(returns)),
                "description": description,
            }
        )
        returns = returns.assign(benchmark_family=benchmark_family, rule_id=rule_id, risk_budget_cap=risk_budget_cap)
        return_frames.append(returns)
        weekly_weights.append(payload["weights"].assign(benchmark_family=benchmark_family, rule_id=rule_id, risk_budget_cap=risk_budget_cap))
        scenario_payloads[scenario_id] = {"payload": payload, "returns": returns, "metadata": scenario_rows[-1], "performance": perf_row}

        for scope, state in [("full_sample", None), ("active_state", None)] + [(f"state_{i}", f"state_{i}") for i in range(4)]:
            component_scope = sample_scope_components(payload, returns, scope, state)
            if component_scope.empty or len(component_scope) < 2:
                risk = {
                    "btc_share_vol": np.nan,
                    "btc_share_cvar": np.nan,
                    "portfolio_volatility_weekly": np.nan,
                    "portfolio_cvar_95_weekly": np.nan,
                    "tail_scenario_count": 0,
                }
            else:
                risk = btc_risk_shares(component_scope)
            risk_rows.append(
                {
                    "scenario_id": scenario_id,
                    "benchmark_family": benchmark_family,
                    "rule_id": rule_id,
                    "portfolio_family": family,
                    "risk_budget_cap": risk_budget_cap,
                    "sample_scope": scope,
                    "hmm_state": state if state is not None else "",
                    "btc_weight_context": "time_varying" if uses_hmm_state else "constant",
                    **risk,
                    "cap_vol_ok": bool(risk["btc_share_vol"] <= risk_budget_cap + FLOAT_TOL) if risk_budget_cap is not None and not pd.isna(risk["btc_share_vol"]) else "",
                    "cap_cvar_ok": bool(risk["btc_share_cvar"] <= risk_budget_cap + FLOAT_TOL) if risk_budget_cap is not None and not pd.isna(risk["btc_share_cvar"]) else "",
                    "all_caps_ok": (
                        bool(risk["btc_share_vol"] <= risk_budget_cap + FLOAT_TOL and risk["btc_share_cvar"] <= risk_budget_cap + FLOAT_TOL)
                        if risk_budget_cap is not None and not pd.isna(risk["btc_share_vol"]) and not pd.isna(risk["btc_share_cvar"])
                        else ""
                    ),
                }
            )

    for family in PORTFOLIO_FAMILIES:
        base_weights = base_weights_all[family]

        # no-BTC baseline
        sid = constant_scenario_id("no_btc_baseline", family, 0.0)
        register_scenario(
            sid,
            "no_btc_baseline",
            "no_btc_baseline",
            family,
            None,
            0.0,
            False,
            False,
            "No-BTC baseline portfolio.",
            evaluate_constant_scenario(panel, family, base_weights, 0.0, sid),
        )

        # fixed fine grid
        for btc_weight in FINE_BTC_GRID:
            sid = constant_scenario_id("fixed_btc_grid", family, btc_weight)
            register_scenario(
                sid,
                "fixed_btc_grid",
                f"fixed_btc_{scenario_slug(btc_weight)}",
                family,
                None,
                btc_weight,
                False,
                False,
                "Fixed BTC grid benchmark; non-BTC assets scaled pro rata.",
                evaluate_constant_scenario(panel, family, base_weights, btc_weight, sid),
            )

        # matched fixed
        sid = constant_scenario_id("matched_fixed_btc", family, MATCHED_FIXED_BTC_WEIGHT)
        register_scenario(
            sid,
            "matched_fixed_btc",
            "matched_fixed_btc",
            family,
            None,
            MATCHED_FIXED_BTC_WEIGHT,
            False,
            False,
            "Fixed BTC benchmark matching the average BTC exposure of the original Part 4 main executed rule.",
            evaluate_constant_scenario(panel, family, base_weights, MATCHED_FIXED_BTC_WEIGHT, sid),
        )

        # cap-only
        for cap in RISK_BUDGET_CAPS:
            selected, audit = select_cap_only_weight(panel, family, base_weights, cap)
            sid = cap_only_scenario_id(family, cap)
            register_scenario(
                sid,
                "cap_only",
                f"cap_only_{cap_slug(cap)}",
                family,
                cap,
                selected,
                False,
                True,
                "Cap-only benchmark: no HMM state, highest fine-grid BTC weight satisfying full-sample BTC vol/CVaR caps.",
                evaluate_constant_scenario(panel, family, base_weights, selected, sid),
            )
            state_weight_rows.append(
                {
                    "scenario_id": sid,
                    "rule_id": f"cap_only_{cap_slug(cap)}",
                    "portfolio_family": family,
                    "risk_budget_cap": cap,
                    "hmm_state": "",
                    "raw_btc_weight": np.nan,
                    "selected_btc_weight": selected,
                    "adjustment_reason": "highest_constant_grid_weight_satisfying_full_sample_cap",
                    "state_n_weeks": len(panel),
                    "state_cvar_tail_scenario_count": int(audit.get("tail_scenario_count", 0)),
                }
            )

        # conditional cap on the original Part 4 grid, retained as the main
        # benchmark so Part 10 does not silently redefine the existing thesis rule.
        for cap in RISK_BUDGET_CAPS:
            state_map: dict[str, float] = {}
            for state, raw_weight in RAW_MAIN_RULE.items():
                selected, audit = select_conditional_weight(panel, family, base_weights, state, raw_weight, cap, ORIGINAL_BTC_GRID)
                state_map[state] = selected
                if raw_weight == 0.0:
                    reason = "raw_rule_zero_allocation"
                elif abs(selected - raw_weight) < FLOAT_TOL:
                    reason = "raw_weight_within_full_and_active_state_risk_budget_caps"
                else:
                    reason = "reduced_to_highest_fine_grid_weight_satisfying_full_and_active_state_caps"
                state_weight_rows.append(
                    {
                        "scenario_id": conditional_scenario_id(family, cap),
                        "rule_id": f"conditional_cap_{cap_slug(cap)}",
                        "portfolio_family": family,
                        "risk_budget_cap": cap,
                        "hmm_state": state,
                        "raw_btc_weight": raw_weight,
                        "selected_btc_weight": selected,
                        "adjustment_reason": reason,
                        "state_n_weeks": int(audit.get("state_n_weeks", (panel["hmm4_state"] == state).sum())),
                        "state_cvar_tail_scenario_count": int(audit.get("state_cvar_tail_scenario_count", 0)),
                    }
                )
            sid = conditional_scenario_id(family, cap)
            register_scenario(
                sid,
                "conditional_cap",
                f"conditional_cap_{cap_slug(cap)}",
                family,
                cap,
                None,
                True,
                True,
                "HMM-conditioned raw rule with BTC weight capped by full-sample and active-state vol/CVaR caps.",
                evaluate_conditional_rule(panel, family, base_weights, state_map, sid),
            )

        # conditional cap on the finer grid, reported as sensitivity evidence.
        for cap in RISK_BUDGET_CAPS:
            state_map = {}
            for state, raw_weight in RAW_MAIN_RULE.items():
                selected, audit = select_conditional_weight(panel, family, base_weights, state, raw_weight, cap, FINE_BTC_GRID)
                state_map[state] = selected
                if raw_weight == 0.0:
                    reason = "raw_rule_zero_allocation"
                elif abs(selected - raw_weight) < FLOAT_TOL:
                    reason = "raw_weight_within_full_and_active_state_risk_budget_caps_on_fine_grid"
                else:
                    reason = "reduced_to_highest_fine_grid_weight_satisfying_full_and_active_state_caps"
                state_weight_rows.append(
                    {
                        "scenario_id": conditional_fine_grid_scenario_id(family, cap),
                        "rule_id": f"conditional_cap_fine_grid_{cap_slug(cap)}",
                        "portfolio_family": family,
                        "risk_budget_cap": cap,
                        "hmm_state": state,
                        "raw_btc_weight": raw_weight,
                        "selected_btc_weight": selected,
                        "adjustment_reason": reason,
                        "state_n_weeks": int(audit.get("state_n_weeks", (panel["hmm4_state"] == state).sum())),
                        "state_cvar_tail_scenario_count": int(audit.get("state_cvar_tail_scenario_count", 0)),
                    }
                )
            sid = conditional_fine_grid_scenario_id(family, cap)
            register_scenario(
                sid,
                "conditional_cap_fine_grid",
                f"conditional_cap_fine_grid_{cap_slug(cap)}",
                family,
                cap,
                None,
                True,
                True,
                "Fine-grid sensitivity for the HMM-conditioned raw rule; not the frozen Part 4 main specification.",
                evaluate_conditional_rule(panel, family, base_weights, state_map, sid),
            )

    scenario_df = pd.DataFrame(scenario_rows).sort_values(["portfolio_family", "benchmark_family", "scenario_id"]).reset_index(drop=True)
    performance_df = pd.DataFrame(performance_rows).sort_values(["portfolio_family", "benchmark_family", "scenario_id"]).reset_index(drop=True)
    risk_df = pd.DataFrame(risk_rows).sort_values(["portfolio_family", "benchmark_family", "scenario_id", "sample_scope"]).reset_index(drop=True)
    state_weights_df = pd.DataFrame(state_weight_rows).sort_values(["portfolio_family", "risk_budget_cap", "rule_id", "hmm_state"]).reset_index(drop=True)
    weekly_weights_df = pd.concat(weekly_weights, ignore_index=True)
    returns_df = pd.concat(return_frames, ignore_index=True)

    scenario_df.to_csv(dirs["results"] / "part10_scenario_dictionary.csv", index=False)
    performance_df.to_csv(dirs["results"] / "part10_target_weight_performance_summary.csv", index=False)
    risk_df.to_csv(dirs["results"] / "part10_risk_contribution_summary.csv", index=False)
    state_weights_df.to_csv(dirs["results"] / "part10_executed_state_weights.csv", index=False)
    weekly_weights_df.to_csv(dirs["results"] / "part10_weekly_weights.csv", index=False)
    returns_df.to_csv(dirs["results"] / "part10_return_series.csv", index=False)

    logging.info("Built %d Part 10 scenarios", len(scenario_df))
    return {
        "scenarios": scenario_df,
        "performance": performance_df,
        "risk": risk_df,
        "state_weights": state_weights_df,
        "weekly_weights": weekly_weights_df,
        "returns": returns_df,
        "scenario_payloads": scenario_payloads,
    }


def metric_value(performance: pd.DataFrame, risk: pd.DataFrame, scenario_id: str, metric: str) -> float:
    perf_metrics = set(performance.columns)
    if metric in perf_metrics:
        row = performance[performance["scenario_id"].eq(scenario_id)]
        require(len(row) == 1, f"Missing performance row for {scenario_id}")
        return float(row.iloc[0][metric])
    if metric in {"btc_share_vol", "btc_share_cvar"}:
        row = risk[(risk["scenario_id"].eq(scenario_id)) & (risk["sample_scope"].eq("full_sample"))]
        require(len(row) == 1, f"Missing risk row for {scenario_id}")
        return float(row.iloc[0][metric])
    raise ValueError(f"Unsupported metric: {metric}")


def build_pairwise_comparisons(payload: dict[str, pd.DataFrame], dirs: dict[str, Path]) -> pd.DataFrame:
    performance = payload["performance"]
    risk = payload["risk"]
    rows = []
    metrics = [
        "annualized_mean_arithmetic",
        "annualized_volatility",
        "cvar_95_weekly",
        "max_drawdown",
        "average_btc_weight",
        "max_btc_weight",
        "btc_share_vol",
        "btc_share_cvar",
    ]
    pair_specs: list[tuple[str, str, str]] = []
    for family in PORTFOLIO_FAMILIES:
        conditional_10 = conditional_scenario_id(family, 0.10)
        matched = constant_scenario_id("matched_fixed_btc", family, MATCHED_FIXED_BTC_WEIGHT)
        no_btc = constant_scenario_id("no_btc_baseline", family, 0.0)
        cap_only_10 = cap_only_scenario_id(family, 0.10)
        pair_specs.extend(
            [
                (f"{family}__conditional_cap_10pct_vs_matched_fixed_btc", conditional_10, matched),
                (f"{family}__conditional_cap_10pct_vs_cap_only_10pct", conditional_10, cap_only_10),
                (f"{family}__conditional_cap_10pct_vs_no_btc_baseline", conditional_10, no_btc),
                (f"{family}__cap_only_10pct_vs_matched_fixed_btc", cap_only_10, matched),
            ]
        )
        for cap in RISK_BUDGET_CAPS:
            if abs(cap - 0.10) < FLOAT_TOL:
                continue
            pair_specs.append(
                (
                    f"{family}__conditional_cap_{cap_slug(cap)}_vs_cap_only_{cap_slug(cap)}",
                    conditional_scenario_id(family, cap),
                    cap_only_scenario_id(family, cap),
                )
            )

    for comparison_id, left, right in pair_specs:
        left_family = performance.loc[performance["scenario_id"].eq(left), "portfolio_family"].iloc[0]
        for metric in metrics:
            left_value = metric_value(performance, risk, left, metric)
            right_value = metric_value(performance, risk, right, metric)
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "portfolio_family": left_family,
                    "sample_scope": "full_sample",
                    "left_scenario_id": left,
                    "right_scenario_id": right,
                    "metric": metric,
                    "left_value": left_value,
                    "right_value": right_value,
                    "difference_left_minus_right": left_value - right_value,
                    "interpretation_direction": interpretation_direction(metric),
                }
            )
    comparisons = pd.DataFrame(rows)
    require(
        not comparisons.duplicated(["comparison_id", "metric"]).any(),
        "Duplicate pairwise comparison/metric rows found",
    )
    comparisons.to_csv(dirs["results"] / "part10_pairwise_benchmark_comparison.csv", index=False)
    return comparisons


def interpretation_direction(metric: str) -> str:
    if metric in {"annualized_mean_arithmetic", "positive_week_share", "sharpe_annualized_zero_rf"}:
        return "higher_is_better"
    if metric in {"annualized_volatility", "btc_share_vol", "btc_share_cvar", "average_btc_weight", "max_btc_weight"}:
        return "lower_is_better_or_smaller_exposure"
    if metric in {"cvar_95_weekly", "max_drawdown"}:
        return "less_negative_is_better"
    return "context_dependent"


def build_key_findings(payload: dict[str, pd.DataFrame], comparisons: pd.DataFrame, dirs: dict[str, Path]) -> dict[str, Any]:
    performance = payload["performance"]
    risk = payload["risk"]
    state_weights = payload["state_weights"]
    findings: dict[str, Any] = {
        "matched_fixed_weight": MATCHED_FIXED_BTC_WEIGHT,
        "sample_end": EXPECTED_STATE_END,
        "conditional_vs_matched_fixed": {},
        "conditional_vs_cap_only": {},
        "risk_cap_sensitivity": {},
        "cap_only_selected_weights": {},
        "main_text_recommendation": "",
    }
    for family in PORTFOLIO_FAMILIES:
        cond_10 = conditional_scenario_id(family, 0.10)
        matched = constant_scenario_id("matched_fixed_btc", family, MATCHED_FIXED_BTC_WEIGHT)
        cap_only_10 = cap_only_scenario_id(family, 0.10)
        comp_matched = comparisons[comparisons["comparison_id"].eq(f"{family}__conditional_cap_10pct_vs_matched_fixed_btc")]
        comp_cap = comparisons[comparisons["comparison_id"].eq(f"{family}__conditional_cap_10pct_vs_cap_only_10pct")]
        findings["conditional_vs_matched_fixed"][family] = {
            row["metric"]: row["difference_left_minus_right"] for _, row in comp_matched.iterrows()
        }
        findings["conditional_vs_cap_only"][family] = {
            row["metric"]: row["difference_left_minus_right"] for _, row in comp_cap.iterrows()
        }
        findings["cap_only_selected_weights"][family] = {}
        for cap in RISK_BUDGET_CAPS:
            cap_id = cap_only_scenario_id(family, cap)
            row = performance[performance["scenario_id"].eq(cap_id)].iloc[0]
            findings["cap_only_selected_weights"][family][cap_slug(cap)] = {
                "selected_btc_weight": float(row["average_btc_weight"]),
                "btc_share_vol": float(
                    risk[(risk["scenario_id"].eq(cap_id)) & (risk["sample_scope"].eq("full_sample"))].iloc[0]["btc_share_vol"]
                ),
                "btc_share_cvar": float(
                    risk[(risk["scenario_id"].eq(cap_id)) & (risk["sample_scope"].eq("full_sample"))].iloc[0]["btc_share_cvar"]
                ),
            }
        cap_rows = state_weights[
            (state_weights["portfolio_family"].eq(family))
            & (state_weights["rule_id"].str.match(r"conditional_cap_\d{2}pct$"))
        ]
        findings["risk_cap_sensitivity"][family] = (
            cap_rows.pivot_table(index="risk_budget_cap", columns="hmm_state", values="selected_btc_weight", aggfunc="first")
            .reset_index()
            .to_dict(orient="records")
        )
        findings["conditional_vs_matched_fixed"][family]["left_scenario_id"] = cond_10
        findings["conditional_vs_matched_fixed"][family]["right_scenario_id"] = matched
        findings["conditional_vs_cap_only"][family]["left_scenario_id"] = cond_10
        findings["conditional_vs_cap_only"][family]["right_scenario_id"] = cap_only_10

    findings[
        "main_text_recommendation"
    ] = "Use Part 10 primarily to compare the HMM-conditioned rule with matched fixed exposure and cap-only benchmarks; do not interpret the 2%/10% choices as optimal."
    write_json(dirs["results"] / "part10_key_findings.json", normalize_for_json(findings))
    return findings


def make_figures(payload: dict[str, pd.DataFrame], comparisons: pd.DataFrame, dirs: dict[str, Path]) -> None:
    performance = payload["performance"]
    risk = payload["risk"]
    state_weights = payload["state_weights"]

    cap_cond = performance[performance["benchmark_family"].eq("conditional_cap")].copy()
    cap_only = performance[performance["benchmark_family"].eq("cap_only")].copy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for family in PORTFOLIO_FAMILIES:
        sub = cap_cond[cap_cond["portfolio_family"].eq(family)].sort_values("risk_budget_cap")
        ax.plot(sub["risk_budget_cap"] * 100, sub["average_btc_weight"] * 100, marker="o", label=f"{family} conditional")
        sub2 = cap_only[cap_only["portfolio_family"].eq(family)].sort_values("risk_budget_cap")
        ax.plot(sub2["risk_budget_cap"] * 100, sub2["average_btc_weight"] * 100, marker="x", linestyle="--", label=f"{family} cap-only")
    ax.set_xlabel("Risk-budget cap (%)")
    ax.set_ylabel("Average BTC weight (%)")
    ax.set_title("Part 10 cap sensitivity: average BTC weight")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "part10_cap_sensitivity_btc_weight.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    full_risk = risk[risk["sample_scope"].eq("full_sample") & risk["benchmark_family"].isin(["conditional_cap", "cap_only"])].copy()
    for family in PORTFOLIO_FAMILIES:
        for bench, style in [("conditional_cap", "-"), ("cap_only", "--")]:
            sub = full_risk[(full_risk["portfolio_family"].eq(family)) & (full_risk["benchmark_family"].eq(bench))].sort_values("risk_budget_cap")
            ax.plot(sub["risk_budget_cap"] * 100, sub["btc_share_vol"] * 100, linestyle=style, marker="o", label=f"{family} {bench} vol")
    ax.set_xlabel("Risk-budget cap (%)")
    ax.set_ylabel("Full-sample BTC volatility contribution (%)")
    ax.set_title("Part 10 cap sensitivity: BTC volatility contribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "part10_cap_sensitivity_risk_contribution.png", dpi=160)
    plt.close(fig)

    def comparison_bar(filename: str, comparison_filter: str, title: str) -> None:
        metrics = ["annualized_mean_arithmetic", "annualized_volatility", "btc_share_vol", "btc_share_cvar"]
        sub = comparisons[comparisons["comparison_id"].str.contains(comparison_filter) & comparisons["metric"].isin(metrics)].copy()
        sub["label"] = sub["portfolio_family"] + " " + sub["metric"]
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.barh(sub["label"], sub["difference_left_minus_right"] * 100)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Difference, left minus right (percentage points)")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(dirs["figures"] / filename, dpi=160)
        plt.close(fig)

    comparison_bar(
        "part10_conditional_vs_matched_fixed_performance.png",
        "conditional_cap_10pct_vs_matched_fixed_btc",
        "Conditional cap 10% vs matched fixed BTC",
    )
    comparison_bar(
        "part10_conditional_vs_cap_only_performance.png",
        "conditional_cap_10pct_vs_cap_only_10pct",
        "Conditional cap 10% vs cap-only 10%",
    )


def write_audits(args: argparse.Namespace, inputs: dict[str, Any], payload: dict[str, pd.DataFrame], dirs: dict[str, Path]) -> None:
    paths = inputs["paths"]
    lineage_rows = []
    for name, path in paths.items():
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
        "uses_hmm_state": "conditional_cap scenarios use frozen full-sample HMM-4 labels from Part 1; cap-only scenarios do not use HMM states.",
        "matched_fixed_weight": MATCHED_FIXED_BTC_WEIGHT,
        "original_btc_grid": ORIGINAL_BTC_GRID,
        "fine_btc_grid": FINE_BTC_GRID,
        "risk_budget_caps": RISK_BUDGET_CAPS,
        "raw_main_rule": RAW_MAIN_RULE,
        "risk_cap_scope": "full-sample and active-state BTC volatility/CVaR contribution shares for conditional scenarios; full-sample shares only for cap-only scenarios.",
        "interpretation_boundary": "Part 10 is a benchmark and sensitivity extension, not a new out-of-sample trading validation.",
    }
    write_json(dirs["results"] / "model_assumption_audit.json", normalize_for_json(assumption_audit))

    methodology = f"""# Part 10 Methodology Audit

Part 10 extends the frozen Part 1-9 evidence chain with benchmark and BTC risk-cap sensitivity diagnostics.
It does not update the data sample, re-estimate Part 1 HMM labels, or overwrite previous outputs.

Inputs:
- State-aligned weekly sample: {EXPECTED_STATE_START} to {EXPECTED_STATE_END}, {EXPECTED_STATE_ROWS} rows.
- HMM-4 labels from Part 1.
- All Weather and ERC base weights from Part 2.
- Main executed rule exposure from Part 4, used only to define the matched fixed BTC benchmark.

Scenario families:
- no-BTC baseline.
- fixed BTC fine grid: {FINE_BTC_GRID}.
- matched fixed BTC weight: {MATCHED_FIXED_BTC_WEIGHT:.12f}.
- cap-only rules with caps {RISK_BUDGET_CAPS}; these do not use HMM states.
- conditional-cap rules with the existing raw state mapping {RAW_MAIN_RULE}; the main conditional-cap scenarios use the original Part 4 grid {ORIGINAL_BTC_GRID} to preserve the frozen evidence chain.
- conditional-cap fine-grid scenarios use {FINE_BTC_GRID} as sensitivity evidence only.

Interpretation boundary:
Part 10 is diagnostic and full-sample benchmark evidence. It is designed to test whether the HMM-conditioned
rule differs from simpler matched-exposure and cap-only alternatives. It should not be described as independent
out-of-sample validation or as evidence that the 10% cap or 2% maximum BTC weight is optimal.
"""
    (dirs["results"] / "methodology_audit.md").write_text(methodology, encoding="utf-8")


def validate_outputs(
    args: argparse.Namespace,
    validation: dict[str, Any],
    payload: dict[str, pd.DataFrame],
    comparisons: pd.DataFrame,
    dirs: dict[str, Path],
) -> dict[str, Any]:
    scenarios = payload["scenarios"]
    performance = payload["performance"]
    risk = payload["risk"]
    state_weights = payload["state_weights"]
    weekly_weights = payload["weekly_weights"]

    required_paths = [dirs["results"] / name for name in REQUIRED_RESULTS if name != "output_validation_summary.json"] + [
        dirs["figures"] / name for name in REQUIRED_FIGURES
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    require(not missing, f"Missing output files: {missing}")

    require(scenarios["scenario_id"].is_unique, "Scenario ids are not unique")
    require(len(performance) == len(scenarios), "Performance rows do not match scenarios")
    require(not performance.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).isna().all(axis=None), "Numeric outputs all NaN")
    require(set(scenarios["portfolio_family"]) == set(PORTFOLIO_FAMILIES), "Missing portfolio families")

    sums = weekly_weights.groupby(["date", "scenario_id"])["weight"].sum()
    require(float((sums - 1.0).abs().max()) < 1e-8, "Weekly weights do not sum to one")
    require(float(weekly_weights["weight"].min()) >= -1e-10, "Negative weekly weight found")

    matched_rows = performance[performance["benchmark_family"].eq("matched_fixed_btc")]
    require(len(matched_rows) == 2, "Expected two matched fixed rows")
    require(np.allclose(matched_rows["average_btc_weight"].to_numpy(dtype=float), MATCHED_FIXED_BTC_WEIGHT, atol=1e-12), "Matched fixed BTC weight mismatch")

    cond_10 = state_weights[state_weights["rule_id"].eq("conditional_cap_10pct")]
    require(len(cond_10) == 8, "Expected 8 conditional cap 10% state-weight rows")
    expected_state_weights = {"state_0": 0.02, "state_1": 0.01, "state_2": 0.0, "state_3": 0.0}
    for family in PORTFOLIO_FAMILIES:
        frame = cond_10[cond_10["portfolio_family"].eq(family)]
        got = frame.set_index("hmm_state")["selected_btc_weight"].astype(float).to_dict()
        require(all(abs(got[state] - weight) < 1e-12 for state, weight in expected_state_weights.items()), f"Part 10 cap 10% does not reproduce Part 4 state weights for {family}: {got}")

    cap_only = scenarios[scenarios["benchmark_family"].eq("cap_only")]
    require(not cap_only["uses_hmm_state"].any(), "Cap-only scenarios should not use HMM state")
    require(cap_only["uses_risk_cap"].all(), "Cap-only scenarios should use risk cap")

    pair_expect = {
        f"{family}__conditional_cap_10pct_vs_matched_fixed_btc" for family in PORTFOLIO_FAMILIES
    } | {f"{family}__conditional_cap_10pct_vs_cap_only_10pct" for family in PORTFOLIO_FAMILIES}
    require(pair_expect.issubset(set(comparisons["comparison_id"])), "Missing core pairwise comparisons")
    require(not comparisons.duplicated(["comparison_id", "metric"]).any(), "Duplicate pairwise comparison rows found")

    summary = {
        "status": "passed",
        "scenario_count": int(len(scenarios)),
        "performance_rows": int(len(performance)),
        "risk_rows": int(len(risk)),
        "state_weight_rows": int(len(state_weights)),
        "weekly_weight_rows": int(len(weekly_weights)),
        "pairwise_rows": int(len(comparisons)),
        "weights_sum_max_abs_error": float((sums - 1.0).abs().max()),
        "matched_fixed_weight_ok": True,
        "conditional_cap_10pct_reproduces_part4_state_weights": True,
        "cap_only_uses_hmm_state": False,
        "required_files_present": True,
    }
    write_json(dirs["results"] / "output_validation_summary.json", normalize_for_json(summary))
    logging.info("Output validation passed")
    return summary


def write_manifest(
    args: argparse.Namespace,
    inputs: dict[str, Any],
    output_validation: dict[str, Any],
    dirs: dict[str, Path],
) -> None:
    manifest = {
        "part_id": "part10_benchmark_cap_sensitivity",
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_frozen": True,
        "sample_end": EXPECTED_STATE_END,
        "seed": args.seed,
        "inputs": {name: str(path) for name, path in inputs["paths"].items()},
        "input_hashes": inputs["input_hashes"],
        "parameters": {
            "fine_btc_grid": FINE_BTC_GRID,
            "original_btc_grid": ORIGINAL_BTC_GRID,
            "risk_budget_caps": RISK_BUDGET_CAPS,
            "matched_fixed_btc_weight": MATCHED_FIXED_BTC_WEIGHT,
            "raw_main_rule": RAW_MAIN_RULE,
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

    logging.info("Starting Part 10 run in %s", run_dir)
    inputs = read_inputs(args)
    enforce_resume_input_hashes(args, dirs, inputs["input_hashes"])
    validation = load_or_run(dirs, "01_input_validation", args.resume, lambda: validate_inputs(inputs, dirs))
    scenario_payload = load_or_run(dirs, "02_scenarios", args.resume, lambda: build_all_scenarios(validation, dirs))
    comparisons = load_or_run(dirs, "03_pairwise_comparisons", args.resume, lambda: build_pairwise_comparisons(scenario_payload, dirs))
    _ = build_key_findings(scenario_payload, comparisons, dirs)
    make_figures(scenario_payload, comparisons, dirs)
    write_audits(args, inputs, scenario_payload, dirs)
    output_validation = validate_outputs(args, validation, scenario_payload, comparisons, dirs)
    write_manifest(args, inputs, output_validation, dirs)
    logging.info("Part 10 completed successfully")


if __name__ == "__main__":
    main()
