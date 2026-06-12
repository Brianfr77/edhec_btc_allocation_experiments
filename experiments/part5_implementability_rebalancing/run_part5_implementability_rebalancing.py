#!/usr/bin/env python3
"""Part 5 experiment runner: implementability, rebalancing, and turnover diagnostics."""

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


EXPECTED_ASSET_START = "2018-01-12"
EXPECTED_ASSET_END = "2026-03-27"
EXPECTED_ASSET_ROWS = 429
EXPECTED_STATE_START = "2018-02-09"
EXPECTED_STATE_END = "2026-03-27"
EXPECTED_STATE_ROWS = 425
EXPECTED_LAGGED_START = "2018-02-16"
EXPECTED_LAGGED_ROWS = 424
EXPECTED_STATE_COUNTS = {"state_0": 149, "state_1": 167, "state_2": 30, "state_3": 79}
TRADING_WEEKS_PER_YEAR = 52
TAIL_ALPHA = 0.05
FLOAT_TOL = 1e-10

BASE_ASSETS = ["ret_spy", "ret_tlt", "ret_ief", "ret_gld", "ret_dbc"]
ASSETS = ["ret_btc"] + BASE_ASSETS + ["ret_bil"]
IMPLEMENTED_RULE_IDS = ["main_executed", "sensitivity_state2_low_executed"]
PORTFOLIO_FAMILIES = ["all_weather", "erc"]
FUNDING_CONVENTIONS = ["pro_rata_base", "bil_sleeve"]
REBALANCE_FREQUENCIES = ["monthly", "quarterly"]
SIGNAL_TIMINGS = ["lagged_one_week", "same_week_bridge"]
COST_SCENARIOS = {
    "no_cost": {"ret_btc": 0.0, "etf_and_bil": 0.0},
    "low_cost": {"ret_btc": 0.0010, "etf_and_bil": 0.0002},
    "moderate_cost": {"ret_btc": 0.0025, "etf_and_bil": 0.0005},
}
MAIN_SIGNAL_TIMING = "lagged_one_week"
MAIN_COST_SCENARIO = "moderate_cost"

REQUIRED_RESULTS = [
    "input_validation_summary.json",
    "rebalance_calendar.csv",
    "implementation_scenario_dictionary.csv",
    "rebalance_event_log.csv",
    "rebalanced_weight_series.csv",
    "rebalanced_portfolio_return_series.csv",
    "rebalanced_performance_summary.csv",
    "turnover_summary.csv",
    "transaction_cost_summary.csv",
    "frequency_comparison_summary.csv",
    "funding_convention_comparison.csv",
    "signal_lag_bridge_summary.csv",
    "state_conditioned_implementability_summary.csv",
    "weight_drift_diagnostics.csv",
    "cost_assumption_dictionary.csv",
    "methodology_audit.md",
    "data_lineage.csv",
    "model_assumption_audit.json",
    "output_validation_summary.json",
]
REQUIRED_FIGURES = [
    "rebalanced_drawdowns.png",
    "turnover_by_frequency.png",
    "transaction_cost_impact.png",
    "target_vs_actual_btc_weight.png",
    "funding_convention_comparison.png",
    "signal_lag_bridge.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run conditional BTC implementability, rebalancing, and turnover diagnostics."
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
    parser.add_argument(
        "--part3-run-dir",
        default="outputs/part3_btc_state_dependence/colab_part3_seed42",
        type=Path,
    )
    parser.add_argument(
        "--part4-run-dir",
        default="outputs/part4_conditional_btc_allocation/colab_part4_seed42",
        type=Path,
    )
    parser.add_argument("--output-dir", default="outputs/part5_implementability_rebalancing", type=Path)
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1.0


def drawdown_from_returns(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def var_cvar(returns: pd.Series, alpha: float = TAIL_ALPHA) -> tuple[float, float, int]:
    clean = returns.dropna()
    var = float(clean.quantile(alpha))
    tail = clean[clean <= var]
    return var, float(tail.mean()), int(len(tail))


def performance_metrics(returns: pd.Series, prefix: str = "") -> dict[str, Any]:
    clean = returns.dropna()
    var, cvar, tail_count = var_cvar(clean)
    vol = float(clean.std(ddof=1))
    return {
        f"{prefix}count": int(len(clean)),
        f"{prefix}mean_weekly": float(clean.mean()),
        f"{prefix}median_weekly": float(clean.median()),
        f"{prefix}volatility_weekly": vol,
        f"{prefix}annualized_mean_arithmetic": float(clean.mean() * TRADING_WEEKS_PER_YEAR),
        f"{prefix}annualized_volatility": float(vol * math.sqrt(TRADING_WEEKS_PER_YEAR)),
        f"{prefix}var_95_weekly": var,
        f"{prefix}cvar_95_weekly": cvar,
        f"{prefix}tail_scenario_count": tail_count,
        f"{prefix}max_drawdown": drawdown_from_returns(clean),
        f"{prefix}positive_week_share": float((clean > 0.0).mean()),
        f"{prefix}sharpe_annualized_zero_rf": float(clean.mean() / vol * math.sqrt(TRADING_WEEKS_PER_YEAR)) if vol > 0 else float("nan"),
    }


def read_inputs(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "asset_returns_main_weekly": args.input_dir / "asset_returns_main_weekly.csv",
        "cleaning_report": args.input_dir / "cleaning_report.json",
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
        "part4_manifest": args.part4_run_dir / "run_manifest.json",
        "part4_input_validation_summary": args.part4_run_dir / "results" / "input_validation_summary.json",
        "part4_output_validation_summary": args.part4_run_dir / "results" / "output_validation_summary.json",
        "part4_allocation_rule_definition": args.part4_run_dir / "results" / "allocation_rule_definition.csv",
        "part4_weekly_conditional_weights": args.part4_run_dir / "results" / "weekly_conditional_weights.csv",
        "part4_conditional_return_series": args.part4_run_dir / "results" / "conditional_portfolio_return_series.csv",
        "part4_risk_budget_cap_audit": args.part4_run_dir / "results" / "risk_budget_cap_audit.csv",
    }
    for name, path in paths.items():
        require(path.exists(), f"Missing required input file: {name} at {path}")

    return {
        "paths": paths,
        "asset": pd.read_csv(paths["asset_returns_main_weekly"], parse_dates=["date"]),
        "cleaning_report": read_json(paths["cleaning_report"]),
        "labels": pd.read_csv(paths["hmm4_state_labels"], parse_dates=["date"]),
        "part1_manifest": read_json(paths["part1_manifest"]),
        "part1_validation": read_json(paths["part1_validation_summary"]),
        "part2_manifest": read_json(paths["part2_manifest"]),
        "part2_input_validation": read_json(paths["part2_input_validation_summary"]),
        "part2_output_validation": read_json(paths["part2_output_validation_summary"]),
        "part2_baseline_weights": pd.read_csv(paths["part2_baseline_portfolio_weights"]),
        "part3_manifest": read_json(paths["part3_manifest"]),
        "part3_input_validation": read_json(paths["part3_input_validation_summary"]),
        "part3_output_validation": read_json(paths["part3_output_validation_summary"]),
        "part4_manifest": read_json(paths["part4_manifest"]),
        "part4_input_validation": read_json(paths["part4_input_validation_summary"]),
        "part4_output_validation": read_json(paths["part4_output_validation_summary"]),
        "part4_rule_definition": pd.read_csv(paths["part4_allocation_rule_definition"]),
        "part4_weekly_weights": pd.read_csv(paths["part4_weekly_conditional_weights"], parse_dates=["date"]),
        "part4_returns": pd.read_csv(paths["part4_conditional_return_series"], parse_dates=["date"]),
        "part4_cap_audit": pd.read_csv(paths["part4_risk_budget_cap_audit"]),
        "input_hashes": {name: file_sha256(path) for name, path in paths.items()},
    }


def enforce_resume_input_hashes(args: argparse.Namespace, dirs: dict[str, Path], input_hashes: dict[str, str]) -> None:
    if not args.resume:
        return
    manifest_path = dirs["root"] / "run_manifest.json"
    if not manifest_path.exists():
        return
    old_manifest = read_json(manifest_path)
    require(old_manifest.get("input_hashes", {}) == input_hashes, "Input hashes changed since the previous run manifest")
    logging.info("Resume input hash check passed")


def validate_inputs(inputs: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    asset = inputs["asset"].copy()
    labels = inputs["labels"].copy()
    hashes = inputs["input_hashes"]
    part1_manifest = inputs["part1_manifest"]
    part2_manifest = inputs["part2_manifest"]
    part3_manifest = inputs["part3_manifest"]
    part4_manifest = inputs["part4_manifest"]

    require(inputs["part1_validation"].get("status") == "passed", "Part 1 validation did not pass")
    require(part1_manifest.get("model_diagnostics", {}).get("hmm4_converged") is True, "Part 1 HMM-4 did not converge")
    require(inputs["part2_input_validation"].get("status") == "passed", "Part 2 input validation did not pass")
    require(inputs["part2_output_validation"].get("status") == "passed", "Part 2 output validation did not pass")
    require(inputs["part3_input_validation"].get("status") == "passed", "Part 3 input validation did not pass")
    require(inputs["part3_output_validation"].get("status") == "passed", "Part 3 output validation did not pass")
    require(inputs["part4_input_validation"].get("status") == "passed", "Part 4 input validation did not pass")
    require(inputs["part4_output_validation"].get("status") == "passed", "Part 4 output validation did not pass")

    require(len(asset) == EXPECTED_ASSET_ROWS, f"Unexpected asset row count: {len(asset)}")
    require(date_string(asset["date"], "min") == EXPECTED_ASSET_START, "Unexpected asset start date")
    require(date_string(asset["date"], "max") == EXPECTED_ASSET_END, "Unexpected asset end date")
    require(asset["date"].dt.dayofweek.eq(4).all(), "Asset dates are not all Fridays")
    require(all(col in asset.columns for col in ASSETS), "Missing required asset return columns")
    require(asset[ASSETS].isna().sum().sum() == 0, "Missing required asset returns")

    require(len(labels) == EXPECTED_STATE_ROWS, f"Unexpected HMM-4 label rows: {len(labels)}")
    require(date_string(labels["date"], "min") == EXPECTED_STATE_START, "Unexpected HMM-4 label start date")
    require(date_string(labels["date"], "max") == EXPECTED_STATE_END, "Unexpected HMM-4 label end date")
    require(labels["date"].dt.dayofweek.eq(4).all(), "HMM-4 label dates are not all Fridays")
    require(all(col in labels.columns for col in ["date", "hmm4_state", "hmm4_state_id"]), "Missing HMM-4 label columns")
    state_counts = labels["hmm4_state"].value_counts().sort_index().to_dict()
    require(state_counts == EXPECTED_STATE_COUNTS, f"Unexpected HMM-4 state counts: {state_counts}")

    require(hashes["asset_returns_main_weekly"] == part1_manifest["input_hashes"]["asset_returns_main_weekly"], "Asset hash does not match Part 1")
    require(hashes["cleaning_report"] == part1_manifest["input_hashes"]["cleaning_report"], "Cleaning report hash does not match Part 1")
    require(hashes["asset_returns_main_weekly"] == part2_manifest["input_hashes"]["asset_returns_main_weekly"], "Asset hash does not match Part 2")
    require(hashes["hmm4_state_labels"] == part2_manifest["input_hashes"]["hmm4_state_labels"], "HMM labels hash does not match Part 2")
    require(hashes["asset_returns_main_weekly"] == part3_manifest["input_hashes"]["asset_returns_main_weekly"], "Asset hash does not match Part 3")
    require(hashes["hmm4_state_labels"] == part3_manifest["input_hashes"]["hmm4_state_labels"], "HMM labels hash does not match Part 3")
    require(hashes["asset_returns_main_weekly"] == part4_manifest["input_hashes"]["asset_returns_main_weekly"], "Asset hash does not match Part 4")
    require(hashes["hmm4_state_labels"] == part4_manifest["input_hashes"]["hmm4_state_labels"], "HMM labels hash does not match Part 4")
    require(hashes["part1_manifest"] == part4_manifest["input_hashes"]["part1_manifest"], "Part 1 manifest hash does not match Part 4 lineage")
    require(hashes["part2_manifest"] == part4_manifest["input_hashes"]["part2_manifest"], "Part 2 manifest hash does not match Part 4 lineage")
    require(hashes["part3_manifest"] == part4_manifest["input_hashes"]["part3_manifest"], "Part 3 manifest hash does not match Part 4 lineage")

    panel = asset[["date"] + ASSETS].merge(
        labels[["date", "hmm4_state", "hmm4_state_id"]],
        on="date",
        how="inner",
        validate="one_to_one",
    )
    panel = panel.sort_values("date").reset_index(drop=True)
    panel["lagged_hmm4_state"] = panel["hmm4_state"].shift(1)
    panel["lagged_hmm4_state_id"] = panel["hmm4_state_id"].shift(1)
    require(len(panel) == EXPECTED_STATE_ROWS, f"Asset/HMM inner join produced {len(panel)} rows")
    require(date_string(panel["date"], "min") == EXPECTED_STATE_START, "Unexpected Part 5 reference start date")
    require(date_string(panel["date"], "max") == EXPECTED_STATE_END, "Unexpected Part 5 reference end date")
    lagged_panel = panel.dropna(subset=["lagged_hmm4_state"]).copy()
    require(len(lagged_panel) == EXPECTED_LAGGED_ROWS, f"Lagged signal panel produced {len(lagged_panel)} rows")
    require(date_string(lagged_panel["date"], "min") == EXPECTED_LAGGED_START, "Unexpected lagged signal start date")

    baseline = inputs["part2_baseline_weights"].copy()
    require(set(baseline["portfolio_family"]) == set(PORTFOLIO_FAMILIES), "Missing baseline portfolio families")
    for family, frame in baseline.groupby("portfolio_family"):
        require(set(frame["asset"]) == set(BASE_ASSETS), f"Unexpected baseline assets for {family}")
        require(abs(float(frame["weight"].sum()) - 1.0) <= FLOAT_TOL, f"Baseline weights do not sum to 1 for {family}")
        require((frame["weight"] >= -FLOAT_TOL).all(), f"Negative baseline weight for {family}")

    rules = inputs["part4_rule_definition"].copy()
    require(set(IMPLEMENTED_RULE_IDS).issubset(set(rules["rule_id"])), "Missing implemented Part 4 rule ids")
    implemented_rules = rules[rules["rule_id"].isin(IMPLEMENTED_RULE_IDS)].copy()
    require(set(implemented_rules["portfolio_family"]) == set(PORTFOLIO_FAMILIES), "Implemented rules missing portfolio families")
    require(set(implemented_rules["hmm4_state"]) == set(EXPECTED_STATE_COUNTS), "Implemented rules missing HMM states")
    require((implemented_rules["selected_btc_weight"] >= -FLOAT_TOL).all(), "Negative selected BTC weight")
    require((implemented_rules["selected_btc_weight"] <= implemented_rules["raw_btc_weight"] + FLOAT_TOL).all(), "Executed BTC weight exceeds raw rule weight")
    cap = inputs["part4_cap_audit"]
    cap_ok = cap[cap["rule_id"].isin(IMPLEMENTED_RULE_IDS)]["all_caps_ok"].astype(bool).all()
    require(bool(cap_ok), "Implemented Part 4 rules did not all pass risk cap audit")

    summary = {
        "status": "passed",
        "asset_sample": {"rows": int(len(asset)), "start": date_string(asset["date"], "min"), "end": date_string(asset["date"], "max")},
        "part5_reference_sample": {"rows": int(len(panel)), "start": date_string(panel["date"], "min"), "end": date_string(panel["date"], "max")},
        "lagged_main_sample": {"rows": int(len(lagged_panel)), "start": date_string(lagged_panel["date"], "min"), "end": date_string(lagged_panel["date"], "max")},
        "same_week_bridge_sample": {"rows": int(len(panel)), "start": date_string(panel["date"], "min"), "end": date_string(panel["date"], "max")},
        "hmm4_state_counts": {k: int(v) for k, v in state_counts.items()},
        "implemented_rule_ids": IMPLEMENTED_RULE_IDS,
        "portfolio_families": PORTFOLIO_FAMILIES,
        "funding_conventions": FUNDING_CONVENTIONS,
        "rebalance_frequencies": REBALANCE_FREQUENCIES,
        "signal_timings": SIGNAL_TIMINGS,
        "cost_scenarios": COST_SCENARIOS,
        "input_hashes": hashes,
        "upstream_runs": {
            "part1_run_id": part1_manifest.get("run_id"),
            "part2_run_id": part2_manifest.get("run_id"),
            "part3_run_id": part3_manifest.get("run_id"),
            "part4_run_id": part4_manifest.get("run_id"),
        },
    }
    write_json(dirs["results"] / "input_validation_summary.json", summary)
    logging.info("Input validation passed")
    return {"validation": summary, "analysis_panel": panel, "lagged_panel": lagged_panel}


def build_base_weights(inputs: dict[str, Any]) -> dict[str, dict[str, float]]:
    baseline = inputs["part2_baseline_weights"].copy()
    result: dict[str, dict[str, float]] = {}
    for family, frame in baseline.groupby("portfolio_family"):
        result[family] = {row["asset"]: float(row["weight"]) for _, row in frame.iterrows()}
    return result


def cost_rates_for_scenario(cost_scenario: str) -> dict[str, float]:
    spec = COST_SCENARIOS[cost_scenario]
    return {asset: (spec["ret_btc"] if asset == "ret_btc" else spec["etf_and_bil"]) for asset in ASSETS}


def target_weights_for_state(
    rule_weights: dict[str, float],
    base_weights: dict[str, float],
    state: str,
    funding_convention: str,
    max_sleeve: float,
) -> dict[str, float]:
    btc_weight = float(rule_weights[state])
    weights = {asset: 0.0 for asset in ASSETS}
    weights["ret_btc"] = btc_weight
    if funding_convention == "pro_rata_base":
        for asset, base_weight in base_weights.items():
            weights[asset] = float(base_weight * (1.0 - btc_weight))
        weights["ret_bil"] = 0.0
    elif funding_convention == "bil_sleeve":
        for asset, base_weight in base_weights.items():
            weights[asset] = float(base_weight * (1.0 - max_sleeve))
        weights["ret_bil"] = float(max_sleeve - btc_weight)
    else:
        raise ValueError(f"Unknown funding convention: {funding_convention}")
    require(abs(sum(weights.values()) - 1.0) <= FLOAT_TOL, f"Target weights do not sum to 1: {weights}")
    require(min(weights.values()) >= -FLOAT_TOL, f"Negative target weight: {weights}")
    return weights


def build_scenarios(inputs: dict[str, Any], validation_payload: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    rules = inputs["part4_rule_definition"]
    base_weights = build_base_weights(inputs)
    scenario_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    model_spec: dict[str, Any] = {"rules": {}, "base_weights": base_weights}

    for cost_name, spec in COST_SCENARIOS.items():
        cost_rows.append(
            {
                "cost_scenario": cost_name,
                "btc_one_way_cost_bps": spec["ret_btc"] * 10000.0,
                "etf_and_bil_one_way_cost_bps": spec["etf_and_bil"] * 10000.0,
                "cost_formula": "sum(abs(target_weight - pre_trade_weight) * asset_cost_rate)",
                "initial_setup_cost_in_net_returns": False,
            }
        )

    for rule_id in IMPLEMENTED_RULE_IDS:
        model_spec["rules"][rule_id] = {}
        for family in PORTFOLIO_FAMILIES:
            frame = rules[(rules["rule_id"] == rule_id) & (rules["portfolio_family"] == family)].copy()
            require(len(frame) == 4, f"Expected four states for {rule_id}/{family}")
            rule_weights = {row["hmm4_state"]: float(row["selected_btc_weight"]) for _, row in frame.iterrows()}
            raw_rule_name = str(frame["raw_rule_name"].iloc[0])
            rule_role = str(frame["rule_role"].iloc[0])
            max_sleeve = max(rule_weights.values())
            model_spec["rules"][rule_id][family] = {"selected_btc_weights": rule_weights, "max_sleeve": max_sleeve}
            for funding in FUNDING_CONVENTIONS:
                for state in EXPECTED_STATE_COUNTS:
                    target = target_weights_for_state(rule_weights, base_weights[family], state, funding, max_sleeve)
                    for asset, weight in target.items():
                        target_rows.append(
                            {
                                "rule_id": rule_id,
                                "raw_rule_name": raw_rule_name,
                                "rule_role": rule_role,
                                "portfolio_family": family,
                                "funding_convention": funding,
                                "hmm4_state": state,
                                "asset": asset,
                                "target_weight": weight,
                                "state_btc_weight": rule_weights[state],
                                "max_btc_sleeve": max_sleeve,
                            }
                        )
                for frequency in REBALANCE_FREQUENCIES:
                    for timing in SIGNAL_TIMINGS:
                        for cost_name in COST_SCENARIOS:
                            scenario_id = "__".join([rule_id, family, funding, frequency, timing, cost_name])
                            scenario_rows.append(
                                {
                                    "scenario_id": scenario_id,
                                    "rule_id": rule_id,
                                    "raw_rule_name": raw_rule_name,
                                    "rule_role": rule_role,
                                    "portfolio_family": family,
                                    "funding_convention": funding,
                                    "rebalance_frequency": frequency,
                                    "signal_timing": timing,
                                    "cost_scenario": cost_name,
                                    "is_main_specification": bool(timing == MAIN_SIGNAL_TIMING and cost_name == MAIN_COST_SCENARIO),
                                    "max_btc_sleeve": max_sleeve,
                                    "sample_start": EXPECTED_LAGGED_START if timing == "lagged_one_week" else EXPECTED_STATE_START,
                                    "expected_return_rows": EXPECTED_LAGGED_ROWS if timing == "lagged_one_week" else EXPECTED_STATE_ROWS,
                                }
                            )

    scenario_df = pd.DataFrame(scenario_rows).sort_values("scenario_id").reset_index(drop=True)
    target_df = pd.DataFrame(target_rows).sort_values(
        ["rule_id", "portfolio_family", "funding_convention", "hmm4_state", "asset"]
    )
    cost_df = pd.DataFrame(cost_rows).sort_values("cost_scenario")
    scenario_df.to_csv(dirs["results"] / "implementation_scenario_dictionary.csv", index=False)
    target_df.to_csv(dirs["models"] / "target_weight_dictionary.csv", index=False)
    cost_df.to_csv(dirs["results"] / "cost_assumption_dictionary.csv", index=False)
    with (dirs["models"] / "implementation_scenario_specification.pkl").open("wb") as handle:
        pickle.dump({"scenarios": scenario_df, "targets": target_df, "model_spec": model_spec}, handle)
    logging.info("Implementation scenarios completed")
    return {
        "scenarios": scenario_df,
        "target_weights": target_df,
        "cost_assumptions": cost_df,
        "base_weights": base_weights,
        "model_spec": model_spec,
    }


def build_rebalance_calendar(validation_payload: dict[str, Any], dirs: dict[str, Path]) -> pd.DataFrame:
    panel = validation_payload["analysis_panel"][["date", "hmm4_state", "hmm4_state_id", "lagged_hmm4_state", "lagged_hmm4_state_id"]].copy()
    panel["year_month"] = panel["date"].dt.to_period("M").astype(str)
    panel["year_quarter"] = panel["date"].dt.to_period("Q").astype(str)
    panel["is_month_end_rebalance"] = panel["date"].eq(panel.groupby("year_month")["date"].transform("max"))
    panel["is_quarter_end_rebalance"] = panel["date"].eq(panel.groupby("year_quarter")["date"].transform("max"))
    panel["is_same_week_formation_date"] = panel.index == 0
    panel["is_lagged_formation_date"] = panel.index == 1
    panel["lagged_signal_available"] = panel["lagged_hmm4_state"].notna()
    out = panel[
        [
            "date",
            "hmm4_state",
            "hmm4_state_id",
            "lagged_hmm4_state",
            "lagged_hmm4_state_id",
            "is_month_end_rebalance",
            "is_quarter_end_rebalance",
            "is_same_week_formation_date",
            "is_lagged_formation_date",
            "lagged_signal_available",
        ]
    ].copy()
    out.to_csv(dirs["results"] / "rebalance_calendar.csv", index=False)
    logging.info("Rebalance calendar completed")
    return out


def target_for_scenario_state(
    scenario: pd.Series,
    state: str,
    scenarios_payload: dict[str, Any],
) -> dict[str, float]:
    rule_spec = scenarios_payload["model_spec"]["rules"][scenario["rule_id"]][scenario["portfolio_family"]]
    return target_weights_for_state(
        rule_spec["selected_btc_weights"],
        scenarios_payload["base_weights"][scenario["portfolio_family"]],
        state,
        scenario["funding_convention"],
        float(rule_spec["max_sleeve"]),
    )


def drift_weights(start_weights: dict[str, float], returns: pd.Series, gross_return: float) -> dict[str, float]:
    denominator = 1.0 + gross_return
    require(abs(denominator) > FLOAT_TOL, "Portfolio gross return denominator is too close to zero")
    return {asset: float(start_weights[asset] * (1.0 + float(returns[asset])) / denominator) for asset in ASSETS}


def simulate_one_scenario(
    scenario: pd.Series,
    panel: pd.DataFrame,
    scenarios_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if scenario["signal_timing"] == "lagged_one_week":
        data = panel[panel["lagged_hmm4_state"].notna()].copy()
        decision_state_col = "lagged_hmm4_state"
        decision_state_id_col = "lagged_hmm4_state_id"
    else:
        data = panel.copy()
        decision_state_col = "hmm4_state"
        decision_state_id_col = "hmm4_state_id"
    data = data.reset_index(drop=True)

    frequency_col = "is_month_end_rebalance" if scenario["rebalance_frequency"] == "monthly" else "is_quarter_end_rebalance"
    formation_col = "is_lagged_formation_date" if scenario["signal_timing"] == "lagged_one_week" else "is_same_week_formation_date"
    cost_rates = cost_rates_for_scenario(str(scenario["cost_scenario"]))
    previous_end_weights: dict[str, float] | None = None
    return_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    cumulative_gross = 1.0
    cumulative_net = 1.0

    for idx, row in data.iterrows():
        date = pd.Timestamp(row["date"])
        decision_state = str(row[decision_state_col])
        decision_state_id = int(row[decision_state_id_col])
        target = target_for_scenario_state(scenario, decision_state, scenarios_payload)
        is_formation = bool(row[formation_col])
        is_scheduled_rebalance = bool(row[frequency_col])
        is_rebalance = is_formation or is_scheduled_rebalance

        if previous_end_weights is None:
            beginning_weights = target.copy()
            pre_trade_weights = {asset: 0.0 for asset in ASSETS}
            turnover = float(sum(abs(beginning_weights[asset] - pre_trade_weights[asset]) for asset in ASSETS))
            setup_cost_estimate = float(sum(abs(beginning_weights[asset]) * cost_rates[asset] for asset in ASSETS))
            transaction_cost = 0.0
            event_type = "formation"
        elif is_rebalance:
            pre_trade_weights = previous_end_weights.copy()
            beginning_weights = target.copy()
            turnover = float(sum(abs(beginning_weights[asset] - pre_trade_weights[asset]) for asset in ASSETS))
            setup_cost_estimate = 0.0
            transaction_cost = float(sum(abs(beginning_weights[asset] - pre_trade_weights[asset]) * cost_rates[asset] for asset in ASSETS))
            event_type = "scheduled_rebalance"
        else:
            pre_trade_weights = previous_end_weights.copy()
            beginning_weights = previous_end_weights.copy()
            turnover = 0.0
            setup_cost_estimate = 0.0
            transaction_cost = 0.0
            event_type = "hold"

        asset_returns = row[ASSETS]
        gross_return = float(sum(beginning_weights[asset] * float(asset_returns[asset]) for asset in ASSETS))
        net_return = gross_return - transaction_cost
        cumulative_gross *= 1.0 + gross_return
        cumulative_net *= 1.0 + net_return
        end_weights = drift_weights(beginning_weights, asset_returns, gross_return)

        common = {
            "scenario_id": scenario["scenario_id"],
            "rule_id": scenario["rule_id"],
            "raw_rule_name": scenario["raw_rule_name"],
            "rule_role": scenario["rule_role"],
            "portfolio_family": scenario["portfolio_family"],
            "funding_convention": scenario["funding_convention"],
            "rebalance_frequency": scenario["rebalance_frequency"],
            "signal_timing": scenario["signal_timing"],
            "cost_scenario": scenario["cost_scenario"],
            "date": date,
            "hmm4_state": row["hmm4_state"],
            "hmm4_state_id": int(row["hmm4_state_id"]),
            "decision_hmm4_state": decision_state,
            "decision_hmm4_state_id": decision_state_id,
            "is_rebalance_date": bool(is_rebalance),
            "event_type": event_type,
        }
        return_rows.append(
            {
                **common,
                "btc_target_weight": target["ret_btc"],
                "btc_beginning_weight": beginning_weights["ret_btc"],
                "btc_ending_weight": end_weights["ret_btc"],
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "turnover": turnover if is_rebalance else 0.0,
                "setup_cost_estimate": setup_cost_estimate,
                "cumulative_gross_value": cumulative_gross,
                "cumulative_net_value": cumulative_net,
            }
        )
        if is_rebalance:
            event_rows.append(
                {
                    **common,
                    "turnover": turnover,
                    "transaction_cost": transaction_cost,
                    "setup_cost_estimate": setup_cost_estimate,
                    "btc_pre_trade_weight": pre_trade_weights["ret_btc"],
                    "btc_target_weight": beginning_weights["ret_btc"],
                    "bil_pre_trade_weight": pre_trade_weights["ret_bil"],
                    "bil_target_weight": beginning_weights["ret_bil"],
                    "cost_is_deducted_from_net_return": bool(event_type == "scheduled_rebalance"),
                }
            )
        for asset in ASSETS:
            weight_rows.append(
                {
                    **common,
                    "asset": asset,
                    "asset_return": float(asset_returns[asset]),
                    "target_weight": target[asset],
                    "pre_trade_weight": pre_trade_weights[asset],
                    "beginning_weight": beginning_weights[asset],
                    "component_return": beginning_weights[asset] * float(asset_returns[asset]),
                    "ending_weight_before_next_trade": end_weights[asset],
                    "abs_target_beginning_drift": abs(beginning_weights[asset] - target[asset]),
                    "abs_ending_target_drift": abs(end_weights[asset] - target[asset]),
                }
            )
        previous_end_weights = end_weights

    return return_rows, weight_rows, event_rows


def run_rebalancing_simulations(
    validation_payload: dict[str, Any],
    scenarios_payload: dict[str, Any],
    calendar: pd.DataFrame,
    dirs: dict[str, Path],
) -> dict[str, pd.DataFrame]:
    panel = validation_payload["analysis_panel"].merge(
        calendar[
            [
                "date",
                "is_month_end_rebalance",
                "is_quarter_end_rebalance",
                "is_same_week_formation_date",
                "is_lagged_formation_date",
            ]
        ],
        on="date",
        how="left",
        validate="one_to_one",
    )
    all_returns: list[dict[str, Any]] = []
    all_weights: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    for _, scenario in scenarios_payload["scenarios"].iterrows():
        returns, weights, events = simulate_one_scenario(scenario, panel, scenarios_payload)
        all_returns.extend(returns)
        all_weights.extend(weights)
        all_events.extend(events)

    return_df = pd.DataFrame(all_returns).sort_values(["scenario_id", "date"]).reset_index(drop=True)
    weight_df = pd.DataFrame(all_weights).sort_values(["scenario_id", "date", "asset"]).reset_index(drop=True)
    event_df = pd.DataFrame(all_events).sort_values(["scenario_id", "date"]).reset_index(drop=True)
    return_df.to_csv(dirs["results"] / "rebalanced_portfolio_return_series.csv", index=False)
    weight_df.to_csv(dirs["results"] / "rebalanced_weight_series.csv", index=False)
    event_df.to_csv(dirs["results"] / "rebalance_event_log.csv", index=False)
    logging.info("Rebalancing simulations completed")
    return {"returns": return_df, "weights": weight_df, "events": event_df}


def compute_diagnostics(
    simulations: dict[str, pd.DataFrame],
    scenarios_payload: dict[str, Any],
    dirs: dict[str, Path],
) -> dict[str, pd.DataFrame]:
    returns = simulations["returns"]
    weights = simulations["weights"]
    events = simulations["events"]

    perf_rows: list[dict[str, Any]] = []
    for scenario_id, frame in returns.groupby("scenario_id", sort=True):
        first = frame.iloc[0]
        row = {col: first[col] for col in [
            "scenario_id",
            "rule_id",
            "raw_rule_name",
            "rule_role",
            "portfolio_family",
            "funding_convention",
            "rebalance_frequency",
            "signal_timing",
            "cost_scenario",
        ]}
        row.update(
            {
                "start_date": pd.Timestamp(frame["date"].min()).strftime("%Y-%m-%d"),
                "end_date": pd.Timestamp(frame["date"].max()).strftime("%Y-%m-%d"),
                "average_btc_beginning_weight": float(frame["btc_beginning_weight"].mean()),
                "max_btc_beginning_weight": float(frame["btc_beginning_weight"].max()),
                "average_btc_target_weight": float(frame["btc_target_weight"].mean()),
                "total_transaction_cost": float(frame["transaction_cost"].sum()),
                "average_weekly_transaction_cost": float(frame["transaction_cost"].mean()),
                "total_turnover_including_formation": float(frame["turnover"].sum()),
                "final_cumulative_gross_value": float(frame["cumulative_gross_value"].iloc[-1]),
                "final_cumulative_net_value": float(frame["cumulative_net_value"].iloc[-1]),
            }
        )
        row.update(performance_metrics(frame["gross_return"], prefix="gross_"))
        row.update(performance_metrics(frame["net_return"], prefix="net_"))
        perf_rows.append(row)
    performance = pd.DataFrame(perf_rows).sort_values("scenario_id").reset_index(drop=True)

    nonformation = events[events["event_type"] == "scheduled_rebalance"].copy()
    turnover_rows: list[dict[str, Any]] = []
    for scenario_id, scenario in scenarios_payload["scenarios"].set_index("scenario_id").iterrows():
        frame = nonformation[nonformation["scenario_id"] == scenario_id]
        ret_frame = returns[returns["scenario_id"] == scenario_id]
        years = len(ret_frame) / TRADING_WEEKS_PER_YEAR
        turnover_rows.append(
            {
                "scenario_id": scenario_id,
                "rule_id": scenario["rule_id"],
                "portfolio_family": scenario["portfolio_family"],
                "funding_convention": scenario["funding_convention"],
                "rebalance_frequency": scenario["rebalance_frequency"],
                "signal_timing": scenario["signal_timing"],
                "cost_scenario": scenario["cost_scenario"],
                "scheduled_rebalance_count": int(len(frame)),
                "total_scheduled_turnover": float(frame["turnover"].sum()),
                "annualized_scheduled_turnover": float(frame["turnover"].sum() / years) if years > 0 else float("nan"),
                "average_turnover_per_scheduled_rebalance": float(frame["turnover"].mean()) if len(frame) else 0.0,
                "max_turnover_per_scheduled_rebalance": float(frame["turnover"].max()) if len(frame) else 0.0,
                "formation_setup_turnover": float(events[(events["scenario_id"] == scenario_id) & (events["event_type"] == "formation")]["turnover"].sum()),
            }
        )
    turnover = pd.DataFrame(turnover_rows).sort_values("scenario_id").reset_index(drop=True)

    cost_rows: list[dict[str, Any]] = []
    for scenario_id, scenario in scenarios_payload["scenarios"].set_index("scenario_id").iterrows():
        frame = returns[returns["scenario_id"] == scenario_id]
        event_frame = events[events["scenario_id"] == scenario_id]
        years = len(frame) / TRADING_WEEKS_PER_YEAR
        cost_rows.append(
            {
                "scenario_id": scenario_id,
                "rule_id": scenario["rule_id"],
                "portfolio_family": scenario["portfolio_family"],
                "funding_convention": scenario["funding_convention"],
                "rebalance_frequency": scenario["rebalance_frequency"],
                "signal_timing": scenario["signal_timing"],
                "cost_scenario": scenario["cost_scenario"],
                "total_transaction_cost": float(frame["transaction_cost"].sum()),
                "annualized_transaction_cost": float(frame["transaction_cost"].sum() / years) if years > 0 else float("nan"),
                "average_weekly_transaction_cost": float(frame["transaction_cost"].mean()),
                "average_cost_on_scheduled_rebalance": float(nonformation[nonformation["scenario_id"] == scenario_id]["transaction_cost"].mean()) if len(nonformation[nonformation["scenario_id"] == scenario_id]) else 0.0,
                "max_cost_on_scheduled_rebalance": float(nonformation[nonformation["scenario_id"] == scenario_id]["transaction_cost"].max()) if len(nonformation[nonformation["scenario_id"] == scenario_id]) else 0.0,
                "initial_setup_cost_estimate": float(event_frame[event_frame["event_type"] == "formation"]["setup_cost_estimate"].sum()),
            }
        )
    transaction_cost = pd.DataFrame(cost_rows).sort_values("scenario_id").reset_index(drop=True)

    drift_rows: list[dict[str, Any]] = []
    for (scenario_id, asset), frame in weights.groupby(["scenario_id", "asset"], sort=True):
        first = frame.iloc[0]
        drift_rows.append(
            {
                "scenario_id": scenario_id,
                "rule_id": first["rule_id"],
                "portfolio_family": first["portfolio_family"],
                "funding_convention": first["funding_convention"],
                "rebalance_frequency": first["rebalance_frequency"],
                "signal_timing": first["signal_timing"],
                "cost_scenario": first["cost_scenario"],
                "asset": asset,
                "average_abs_beginning_target_drift": float(frame["abs_target_beginning_drift"].mean()),
                "max_abs_beginning_target_drift": float(frame["abs_target_beginning_drift"].max()),
                "average_abs_ending_target_drift": float(frame["abs_ending_target_drift"].mean()),
                "max_abs_ending_target_drift": float(frame["abs_ending_target_drift"].max()),
                "average_beginning_weight": float(frame["beginning_weight"].mean()),
                "average_target_weight": float(frame["target_weight"].mean()),
            }
        )
    drift = pd.DataFrame(drift_rows).sort_values(["scenario_id", "asset"]).reset_index(drop=True)

    state_rows: list[dict[str, Any]] = []
    for (scenario_id, state), frame in returns.groupby(["scenario_id", "hmm4_state"], sort=True):
        first = frame.iloc[0]
        row = {col: first[col] for col in [
            "scenario_id",
            "rule_id",
            "portfolio_family",
            "funding_convention",
            "rebalance_frequency",
            "signal_timing",
            "cost_scenario",
            "hmm4_state",
            "hmm4_state_id",
        ]}
        row.update(
            {
                "state_sample_warning": "small_state_sample" if len(frame) < 52 else "",
                "average_btc_beginning_weight": float(frame["btc_beginning_weight"].mean()),
                "average_turnover": float(frame["turnover"].mean()),
                "total_transaction_cost": float(frame["transaction_cost"].sum()),
            }
        )
        row.update(performance_metrics(frame["net_return"], prefix="net_"))
        state_rows.append(row)
    state_summary = pd.DataFrame(state_rows).sort_values(["scenario_id", "hmm4_state"]).reset_index(drop=True)

    frequency_comparison = build_pairwise_comparison(
        performance,
        key_cols=["rule_id", "portfolio_family", "funding_convention", "signal_timing", "cost_scenario"],
        compare_col="rebalance_frequency",
        left_value="monthly",
        right_value="quarterly",
        metric_cols=["net_annualized_mean_arithmetic", "net_annualized_volatility", "net_cvar_95_weekly", "net_max_drawdown", "total_transaction_cost"],
        label="monthly_minus_quarterly",
    )
    funding_comparison = build_pairwise_comparison(
        performance,
        key_cols=["rule_id", "portfolio_family", "rebalance_frequency", "signal_timing", "cost_scenario"],
        compare_col="funding_convention",
        left_value="bil_sleeve",
        right_value="pro_rata_base",
        metric_cols=["net_annualized_mean_arithmetic", "net_annualized_volatility", "net_cvar_95_weekly", "net_max_drawdown", "average_btc_beginning_weight"],
        label="bil_sleeve_minus_pro_rata_base",
    )
    signal_bridge = build_pairwise_comparison(
        performance,
        key_cols=["rule_id", "portfolio_family", "funding_convention", "rebalance_frequency", "cost_scenario"],
        compare_col="signal_timing",
        left_value="same_week_bridge",
        right_value="lagged_one_week",
        metric_cols=["net_annualized_mean_arithmetic", "net_annualized_volatility", "net_cvar_95_weekly", "net_max_drawdown", "average_btc_beginning_weight"],
        label="same_week_minus_lagged",
    )

    performance.to_csv(dirs["results"] / "rebalanced_performance_summary.csv", index=False)
    turnover.to_csv(dirs["results"] / "turnover_summary.csv", index=False)
    transaction_cost.to_csv(dirs["results"] / "transaction_cost_summary.csv", index=False)
    frequency_comparison.to_csv(dirs["results"] / "frequency_comparison_summary.csv", index=False)
    funding_comparison.to_csv(dirs["results"] / "funding_convention_comparison.csv", index=False)
    signal_bridge.to_csv(dirs["results"] / "signal_lag_bridge_summary.csv", index=False)
    state_summary.to_csv(dirs["results"] / "state_conditioned_implementability_summary.csv", index=False)
    drift.to_csv(dirs["results"] / "weight_drift_diagnostics.csv", index=False)
    logging.info("Implementability diagnostics completed")
    return {
        "performance": performance,
        "turnover": turnover,
        "transaction_cost": transaction_cost,
        "frequency_comparison": frequency_comparison,
        "funding_comparison": funding_comparison,
        "signal_bridge": signal_bridge,
        "state_summary": state_summary,
        "drift": drift,
    }


def build_pairwise_comparison(
    frame: pd.DataFrame,
    key_cols: list[str],
    compare_col: str,
    left_value: str,
    right_value: str,
    metric_cols: list[str],
    label: str,
) -> pd.DataFrame:
    left = frame[frame[compare_col] == left_value][key_cols + metric_cols].copy()
    right = frame[frame[compare_col] == right_value][key_cols + metric_cols].copy()
    merged = left.merge(right, on=key_cols, suffixes=(f"_{left_value}", f"_{right_value}"), how="inner", validate="one_to_one")
    for metric in metric_cols:
        merged[f"{label}_{metric}"] = merged[f"{metric}_{left_value}"] - merged[f"{metric}_{right_value}"]
    merged.insert(0, "comparison", label)
    return merged


def write_explainability_artifacts(
    inputs: dict[str, Any],
    validation_payload: dict[str, Any],
    scenarios_payload: dict[str, Any],
    dirs: dict[str, Path],
) -> dict[str, Any]:
    methodology = """# Part 5 Methodology Audit

## Purpose
Part 5 evaluates whether the Part 4 conditional BTC allocation rules remain operationally interpretable under scheduled rebalancing, turnover, transaction costs, and BIL cash parking conventions. It is not a live trading strategy and does not write a final thesis conclusion.

## Inputs
- Cleaned weekly asset returns from `data_2026/cleaned`.
- Part 1 full-sample ex-post HMM-4 state labels.
- Part 2 All Weather and ERC no-BTC baseline weights.
- Part 4 executed conditional BTC rules and risk-budget audit.

## Signal Timing
The main specification uses `lagged_one_week`: the prior weekly HMM state is used to choose the current target allocation. The first realized return is therefore 2018-02-16. `same_week_bridge` is retained only to explain differences versus Part 4 same-week target-weight diagnostics.

## Rebalancing and Costs
Monthly and quarterly schedules use the final available Friday in each calendar month or quarter. The first eligible sample week is a formation event. Initial setup cost is estimated but not deducted from net returns. Subsequent scheduled rebalance costs are deducted from weekly net returns using one-way turnover by asset.

## Funding Conventions
`pro_rata_base` returns unused BTC allocation to the non-BTC base portfolio. `bil_sleeve` keeps a fixed BTC allocation budget and places unused allocation in BIL.

## Boundaries
- No HMM re-estimation.
- No new BTC weight optimization or risk-budget threshold selection.
- No turnover or transaction-cost conclusion beyond descriptive diagnostics.
- No BTCUSDT/Coin Metrics, BIL/SHY, frequency, or ETF-era robustness; those belong to later robustness work.
"""
    (dirs["results"] / "methodology_audit.md").write_text(methodology, encoding="utf-8")

    model_assumptions = {
        "status": "documented",
        "state_labels": {
            "source": "Part 1 HMM-4 state labels",
            "estimation_type": "full-sample ex-post descriptive regime identification",
            "not_a_real_time_signal": True,
        },
        "implemented_rules": IMPLEMENTED_RULE_IDS,
        "signal_timing": {
            "main": MAIN_SIGNAL_TIMING,
            "bridge": "same_week_bridge",
            "lagged_main_first_return_date": EXPECTED_LAGGED_START,
        },
        "rebalancing": {
            "frequencies": REBALANCE_FREQUENCIES,
            "calendar_rule": "last available Friday in each calendar month or quarter",
            "initial_setup_cost_deducted": False,
        },
        "transaction_costs": COST_SCENARIOS,
        "funding_conventions": FUNDING_CONVENTIONS,
        "excluded": [
            "new HMM estimation",
            "new BTC allocation threshold selection",
            "final thesis conclusion",
            "Coin Metrics BTC robustness",
            "BIL versus SHY robustness",
            "monthly source-data resampling robustness",
            "ETF-era-only implementation study",
        ],
    }
    write_json(dirs["results"] / "model_assumption_audit.json", model_assumptions)

    lineage_rows = []
    for name, path in inputs["paths"].items():
        lineage_rows.append(
            {
                "input_name": name,
                "path": str(path),
                "sha256": inputs["input_hashes"][name],
                "usage": lineage_usage(name),
            }
        )
    pd.DataFrame(lineage_rows).to_csv(dirs["results"] / "data_lineage.csv", index=False)
    logging.info("Explainability artifacts completed")
    return {"methodology_audit": methodology, "model_assumption_audit": model_assumptions}


def lineage_usage(name: str) -> str:
    mapping = {
        "asset_returns_main_weekly": "Weekly simple returns for portfolio realization, drift, BIL parking, and costs.",
        "cleaning_report": "Frozen cleaning lineage and hash verification.",
        "part1_manifest": "State-model lineage and upstream hash verification.",
        "part1_validation_summary": "Part 1 validity check.",
        "hmm4_state_labels": "HMM-4 states and one-week-lagged state signals.",
        "part2_manifest": "Baseline portfolio lineage and upstream hash verification.",
        "part2_input_validation_summary": "Part 2 input validity check.",
        "part2_output_validation_summary": "Part 2 output validity check.",
        "part2_baseline_portfolio_weights": "No-BTC All Weather and ERC base weights.",
        "part3_manifest": "Part 3 lineage and upstream hash verification.",
        "part3_input_validation_summary": "Part 3 input validity check.",
        "part3_output_validation_summary": "Part 3 output validity check.",
        "part4_manifest": "Conditional allocation rule lineage.",
        "part4_input_validation_summary": "Part 4 input validity check.",
        "part4_output_validation_summary": "Part 4 output validity check.",
        "part4_allocation_rule_definition": "Executed main and sensitivity BTC state weights.",
        "part4_weekly_conditional_weights": "Part 4 reference target-weight diagnostics; not reused as realized drift weights.",
        "part4_conditional_return_series": "Part 4 same-week bridge reference.",
        "part4_risk_budget_cap_audit": "Confirms implemented rules passed Part 4 risk-budget cap.",
    }
    return mapping.get(name, "Input used for lineage verification.")


def validate_outputs(
    validation_payload: dict[str, Any],
    scenarios_payload: dict[str, Any],
    calendar: pd.DataFrame,
    simulations: dict[str, pd.DataFrame],
    diagnostics: dict[str, pd.DataFrame],
    dirs: dict[str, Path],
) -> dict[str, Any]:
    result_checks = [
        {
            "file": name,
            "exists": True if name == "output_validation_summary.json" else (dirs["results"] / name).exists(),
            "nonempty": True
            if name == "output_validation_summary.json"
            else (dirs["results"] / name).exists() and (dirs["results"] / name).stat().st_size > 0,
        }
        for name in REQUIRED_RESULTS
    ]
    figure_checks = [
        {"file": name, "exists": (dirs["figures"] / name).exists(), "nonempty": (dirs["figures"] / name).exists() and (dirs["figures"] / name).stat().st_size > 0}
        for name in REQUIRED_FIGURES
    ]
    scenarios = scenarios_payload["scenarios"]
    weights = simulations["weights"]
    returns = simulations["returns"]
    events = simulations["events"]
    performance = diagnostics["performance"]

    target_sum = weights.groupby(["scenario_id", "date"])["target_weight"].sum()
    beginning_sum = weights.groupby(["scenario_id", "date"])["beginning_weight"].sum()
    ending_sum = weights.groupby(["scenario_id", "date"])["ending_weight_before_next_trade"].sum()
    weight_sum_ok = bool(
        (target_sum.sub(1.0).abs().max() <= FLOAT_TOL)
        and (beginning_sum.sub(1.0).abs().max() <= FLOAT_TOL)
        and (ending_sum.sub(1.0).abs().max() <= FLOAT_TOL)
    )
    nonnegative_ok = bool(
        (weights[["target_weight", "pre_trade_weight", "beginning_weight", "ending_weight_before_next_trade"]] >= -FLOAT_TOL).all().all()
    )
    target_pivot = weights.pivot_table(
        index=["scenario_id", "date"],
        columns="asset",
        values="target_weight",
        aggfunc="first",
    ).reset_index()
    scenario_lookup = scenarios.set_index("scenario_id")
    bil_sleeve_ok = True
    pro_rata_ok = True
    for _, row in target_pivot.iterrows():
        scenario = scenario_lookup.loc[row["scenario_id"]]
        if scenario["funding_convention"] == "bil_sleeve":
            if abs(float(row["ret_btc"] + row["ret_bil"]) - float(scenario["max_btc_sleeve"])) > FLOAT_TOL:
                bil_sleeve_ok = False
                break
        if scenario["funding_convention"] == "pro_rata_base":
            if abs(float(row["ret_bil"])) > FLOAT_TOL:
                pro_rata_ok = False
                break

    counts = returns.groupby("scenario_id").size()
    expected_counts = scenarios.set_index("scenario_id")["expected_return_rows"]
    return_counts_ok = bool(counts.sort_index().equals(expected_counts.sort_index().astype(int)))

    net_formula_ok = bool((returns["net_return"] - (returns["gross_return"] - returns["transaction_cost"])).abs().max() <= FLOAT_TOL)
    nonzero_costs = returns[returns["transaction_cost"].abs() > FLOAT_TOL]
    cost_only_on_rebalance_ok = bool(nonzero_costs["is_rebalance_date"].all()) if len(nonzero_costs) else True
    no_cost_zero_ok = bool((returns.loc[returns["cost_scenario"] == "no_cost", "transaction_cost"].abs() <= FLOAT_TOL).all())
    finite_ok = bool(np.isfinite(returns[["gross_return", "net_return", "transaction_cost", "turnover"]].to_numpy()).all())
    implemented_rules_only = bool(set(returns["rule_id"].unique()) == set(IMPLEMENTED_RULE_IDS))
    frequency_ok = bool(set(returns["rebalance_frequency"].unique()) == set(REBALANCE_FREQUENCIES))
    signal_timing_ok = bool(set(returns["signal_timing"].unique()) == set(SIGNAL_TIMINGS))
    main_spec_present = bool(
        len(
            performance[
                (performance["signal_timing"] == MAIN_SIGNAL_TIMING)
                & (performance["cost_scenario"] == MAIN_COST_SCENARIO)
            ]
        )
        == len(IMPLEMENTED_RULE_IDS) * len(PORTFOLIO_FAMILIES) * len(FUNDING_CONVENTIONS) * len(REBALANCE_FREQUENCIES)
    )
    formation_events_ok = bool(
        (events.groupby("scenario_id")["event_type"].apply(lambda x: (x == "formation").sum()) == 1).all()
    )
    scheduled_event_dates_ok = validate_scheduled_event_dates(events, calendar)
    files_ok = all(item["exists"] and item["nonempty"] for item in result_checks + figure_checks)

    checks = {
        "required_files_ok": files_ok,
        "implemented_rules_only": implemented_rules_only,
        "weight_sum_ok": weight_sum_ok,
        "nonnegative_weights_ok": nonnegative_ok,
        "bil_sleeve_target_ok": bil_sleeve_ok,
        "pro_rata_bil_zero_ok": pro_rata_ok,
        "return_counts_ok": return_counts_ok,
        "net_return_formula_ok": net_formula_ok,
        "cost_only_on_rebalance_ok": cost_only_on_rebalance_ok,
        "no_cost_zero_ok": no_cost_zero_ok,
        "finite_outputs_ok": finite_ok,
        "frequency_set_ok": frequency_ok,
        "signal_timing_set_ok": signal_timing_ok,
        "main_spec_present": main_spec_present,
        "formation_events_ok": formation_events_ok,
        "scheduled_event_dates_ok": scheduled_event_dates_ok,
    }
    status = "passed" if all(checks.values()) else "failed"
    summary = {
        "status": status,
        "checks": checks,
        "required_result_checks": result_checks,
        "required_figure_checks": figure_checks,
        "scenario_count": int(len(scenarios)),
        "return_rows": int(len(returns)),
        "weight_rows": int(len(weights)),
        "event_rows": int(len(events)),
        "return_counts_by_signal_timing": {str(k): int(v) for k, v in returns.groupby("signal_timing").size().to_dict().items()},
        "main_specification": {"signal_timing": MAIN_SIGNAL_TIMING, "cost_scenario": MAIN_COST_SCENARIO},
        "lagged_main_sample": validation_payload["validation"]["lagged_main_sample"],
        "same_week_bridge_sample": validation_payload["validation"]["same_week_bridge_sample"],
    }
    require(status == "passed", f"Output validation failed: {summary}")
    write_json(dirs["results"] / "output_validation_summary.json", summary)
    logging.info("Output validation completed")
    return summary


def validate_scheduled_event_dates(events: pd.DataFrame, calendar: pd.DataFrame) -> bool:
    cal = calendar.set_index("date")
    scheduled = events[events["event_type"] == "scheduled_rebalance"].copy()
    for _, row in scheduled.iterrows():
        date = pd.Timestamp(row["date"])
        if row["rebalance_frequency"] == "monthly" and not bool(cal.loc[date, "is_month_end_rebalance"]):
            return False
        if row["rebalance_frequency"] == "quarterly" and not bool(cal.loc[date, "is_quarter_end_rebalance"]):
            return False
    return True


def write_manifest(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    inputs: dict[str, Any],
    validation_payload: dict[str, Any],
    scenarios_payload: dict[str, Any],
    output_validation: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "run_id": dirs["root"].name,
        "objective": "Implementability, rebalancing, turnover, transaction cost, and BIL sleeve diagnostics",
        "input_dir": str(args.input_dir),
        "part1_run_dir": str(args.part1_run_dir),
        "part2_run_dir": str(args.part2_run_dir),
        "part3_run_dir": str(args.part3_run_dir),
        "part4_run_dir": str(args.part4_run_dir),
        "output_dir": str(args.output_dir),
        "run_dir": str(dirs["root"]),
        "random_seed": int(args.seed),
        "sample": {
            "reference_rows": EXPECTED_STATE_ROWS,
            "reference_start": EXPECTED_STATE_START,
            "reference_end": EXPECTED_STATE_END,
            "lagged_main_rows": EXPECTED_LAGGED_ROWS,
            "lagged_main_start": EXPECTED_LAGGED_START,
            "state_counts": validation_payload["validation"]["hmm4_state_counts"],
        },
        "input_hashes": inputs["input_hashes"],
        "package_versions": package_versions(),
        "parameters": {
            "implemented_rule_ids": IMPLEMENTED_RULE_IDS,
            "portfolio_families": PORTFOLIO_FAMILIES,
            "funding_conventions": FUNDING_CONVENTIONS,
            "rebalance_frequencies": REBALANCE_FREQUENCIES,
            "signal_timings": SIGNAL_TIMINGS,
            "main_signal_timing": MAIN_SIGNAL_TIMING,
            "main_cost_scenario": MAIN_COST_SCENARIO,
            "cost_scenarios": COST_SCENARIOS,
            "tail_alpha": TAIL_ALPHA,
            "trading_weeks_per_year": TRADING_WEEKS_PER_YEAR,
        },
        "lineage": {
            "part1_run_id": inputs["part1_manifest"].get("run_id"),
            "part2_run_id": inputs["part2_manifest"].get("run_id"),
            "part3_run_id": inputs["part3_manifest"].get("run_id"),
            "part4_run_id": inputs["part4_manifest"].get("run_id"),
        },
        "scenario_count": int(len(scenarios_payload["scenarios"])),
        "output_validation": output_validation,
        "outputs": {
            "checkpoints": str(dirs["checkpoints"]),
            "results": str(dirs["results"]),
            "figures": str(dirs["figures"]),
            "models": str(dirs["models"]),
            "logs": str(dirs["logs"]),
        },
        "scope_notes": [
            "Part 5 is an ex-post descriptive implementability diagnostic.",
            "Lagged one-week state timing is the main specification; same-week bridge is diagnostic only.",
            "Initial setup costs are estimated but not deducted from net returns.",
            "No final thesis conclusion is written by this runner.",
        ],
    }
    write_json(dirs["root"] / "run_manifest.json", manifest)
    return manifest


def plot_rebalanced_drawdowns(returns: pd.DataFrame, output_path: Path) -> None:
    frame = returns[
        (returns["rule_id"] == "main_executed")
        & (returns["signal_timing"] == MAIN_SIGNAL_TIMING)
        & (returns["cost_scenario"] == MAIN_COST_SCENARIO)
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, family in zip(axes, PORTFOLIO_FAMILIES):
        sub = frame[frame["portfolio_family"] == family]
        for (funding, freq), group in sub.groupby(["funding_convention", "rebalance_frequency"]):
            group = group.sort_values("date")
            ax.plot(group["date"], drawdown_series(group["net_return"]), linewidth=1.4, label=f"{funding} {freq}")
        ax.set_title(f"{family}: main executed net drawdown")
        ax.set_ylabel("Drawdown")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_turnover_by_frequency(turnover: pd.DataFrame, output_path: Path) -> None:
    frame = turnover[
        (turnover["rule_id"] == "main_executed")
        & (turnover["signal_timing"] == MAIN_SIGNAL_TIMING)
        & (turnover["cost_scenario"] == MAIN_COST_SCENARIO)
    ].copy()
    frame["label"] = frame["portfolio_family"] + "\n" + frame["funding_convention"] + "\n" + frame["rebalance_frequency"]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(frame["label"], frame["annualized_scheduled_turnover"], color="#4777aa")
    ax.set_title("Annualized Scheduled Turnover")
    ax.set_ylabel("Annualized turnover")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_transaction_cost_impact(performance: pd.DataFrame, output_path: Path) -> None:
    frame = performance[
        (performance["rule_id"] == "main_executed")
        & (performance["signal_timing"] == MAIN_SIGNAL_TIMING)
        & (performance["rebalance_frequency"] == "monthly")
    ].copy()
    frame["label"] = frame["portfolio_family"] + "\n" + frame["funding_convention"] + "\n" + frame["cost_scenario"]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(frame["label"], frame["gross_annualized_mean_arithmetic"] - frame["net_annualized_mean_arithmetic"], color="#b55b52")
    ax.set_title("Transaction Cost Impact on Annualized Mean Return")
    ax.set_ylabel("Gross minus net annualized mean")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_target_vs_actual_btc(weights: pd.DataFrame, output_path: Path) -> None:
    frame = weights[
        (weights["rule_id"] == "main_executed")
        & (weights["portfolio_family"] == "all_weather")
        & (weights["funding_convention"] == "bil_sleeve")
        & (weights["rebalance_frequency"] == "monthly")
        & (weights["signal_timing"] == MAIN_SIGNAL_TIMING)
        & (weights["cost_scenario"] == MAIN_COST_SCENARIO)
        & (weights["asset"] == "ret_btc")
    ].sort_values("date")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.step(frame["date"], frame["target_weight"], where="post", linewidth=1.4, label="Target BTC weight")
    ax.plot(frame["date"], frame["beginning_weight"], linewidth=1.1, label="Beginning BTC weight after drift/rebalance")
    ax.plot(frame["date"], frame["ending_weight_before_next_trade"], linewidth=1.0, label="Ending BTC weight before next trade", alpha=0.8)
    ax.set_title("Target vs Actual BTC Weight Drift")
    ax.set_ylabel("BTC weight")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_funding_convention_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    metric = "bil_sleeve_minus_pro_rata_base_net_annualized_volatility"
    frame = summary[
        (summary["signal_timing"] == MAIN_SIGNAL_TIMING)
        & (summary["cost_scenario"] == MAIN_COST_SCENARIO)
    ].copy()
    frame["label"] = frame["rule_id"] + "\n" + frame["portfolio_family"] + "\n" + frame["rebalance_frequency"]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(frame["label"], frame[metric], color="#5f8f5f")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("BIL Sleeve Minus Pro-Rata Base: Net Volatility")
    ax.set_ylabel("Annualized volatility difference")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_signal_lag_bridge(summary: pd.DataFrame, output_path: Path) -> None:
    metric = "same_week_minus_lagged_net_annualized_mean_arithmetic"
    frame = summary[
        (summary["cost_scenario"] == MAIN_COST_SCENARIO)
        & (summary["funding_convention"] == "pro_rata_base")
    ].copy()
    frame["label"] = frame["rule_id"] + "\n" + frame["portfolio_family"] + "\n" + frame["rebalance_frequency"]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(frame["label"], frame[metric], color="#d19a3a")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Same-Week Bridge Minus Lagged Main: Net Annualized Mean")
    ax.set_ylabel("Annualized mean difference")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_figures(simulations: dict[str, pd.DataFrame], diagnostics: dict[str, pd.DataFrame], dirs: dict[str, Path]) -> None:
    plot_rebalanced_drawdowns(simulations["returns"], dirs["figures"] / "rebalanced_drawdowns.png")
    plot_turnover_by_frequency(diagnostics["turnover"], dirs["figures"] / "turnover_by_frequency.png")
    plot_transaction_cost_impact(diagnostics["performance"], dirs["figures"] / "transaction_cost_impact.png")
    plot_target_vs_actual_btc(simulations["weights"], dirs["figures"] / "target_vs_actual_btc_weight.png")
    plot_funding_convention_comparison(diagnostics["funding_comparison"], dirs["figures"] / "funding_convention_comparison.png")
    plot_signal_lag_bridge(diagnostics["signal_bridge"], dirs["figures"] / "signal_lag_bridge.png")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    run_id = args.run_id or now_run_id()
    run_dir = args.output_dir / run_id
    dirs = ensure_dirs(run_dir)
    setup_logging(dirs["logs"])
    logging.info("Starting Part 5 run: %s", run_id)

    inputs = read_inputs(args)
    enforce_resume_input_hashes(args, dirs, inputs["input_hashes"])
    validation_payload = load_or_run(
        dirs,
        "01_input_validation",
        args.resume,
        lambda: validate_inputs(inputs, dirs),
    )
    scenarios_payload = load_or_run(
        dirs,
        "02_implementation_scenarios",
        args.resume,
        lambda: build_scenarios(inputs, validation_payload, dirs),
    )
    calendar = load_or_run(
        dirs,
        "03_rebalance_calendar",
        args.resume,
        lambda: build_rebalance_calendar(validation_payload, dirs),
    )
    simulations = load_or_run(
        dirs,
        "04_rebalancing_simulations",
        args.resume,
        lambda: run_rebalancing_simulations(validation_payload, scenarios_payload, calendar, dirs),
    )
    diagnostics = load_or_run(
        dirs,
        "05_implementability_diagnostics",
        args.resume,
        lambda: compute_diagnostics(simulations, scenarios_payload, dirs),
    )
    explainability = load_or_run(
        dirs,
        "06_explainability_artifacts",
        args.resume,
        lambda: write_explainability_artifacts(inputs, validation_payload, scenarios_payload, dirs),
    )
    write_figures(simulations, diagnostics, dirs)
    output_validation = load_or_run(
        dirs,
        "07_output_validation",
        args.resume,
        lambda: validate_outputs(validation_payload, scenarios_payload, calendar, simulations, diagnostics, dirs),
    )
    manifest = write_manifest(args, dirs, inputs, validation_payload, scenarios_payload, output_validation)
    _ = explainability, manifest
    logging.info("Completed Part 5 run: %s", run_id)
    logging.info("Results directory: %s", dirs["results"])


if __name__ == "__main__":
    main()
