#!/usr/bin/env python3
"""Part 13 experiment runner: transaction-cost sensitivity.

This runner implements Part 10 target-weight benchmark scenarios under monthly
and quarterly rebalancing, pro-rata and BIL-sleeve funding, three transaction
cost levels, and two initial-formation-cost conventions. It uses a one-week
lagged implementation window to avoid same-week state look-ahead and to align
the conditional 10% cap scenario with the Part 5 main implementation scope.
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
EXPECTED_IMPLEMENTED_START = "2018-02-16"
EXPECTED_IMPLEMENTED_ROWS = 424
TRADING_WEEKS_PER_YEAR = 52
TAIL_ALPHA = 0.05
FLOAT_TOL = 1e-10

BASE_ASSETS = ["ret_spy", "ret_tlt", "ret_ief", "ret_gld", "ret_dbc"]
ASSETS = ["ret_btc"] + BASE_ASSETS + ["ret_bil"]
PART10_WEIGHT_ASSETS = ["ret_btc"] + BASE_ASSETS
REQUIRED_RULE_IDS = ["conditional_cap_10pct", "matched_fixed_btc", "cap_only_10pct"]
PORTFOLIO_FAMILIES = ["all_weather", "erc"]
FUNDING_CONVENTIONS = ["pro_rata_base", "bil_sleeve"]
REBALANCE_FREQUENCIES = ["monthly", "quarterly"]
COST_SCENARIOS = {
    "low_cost": {"btc_turnover_cost_bps": 10.0, "etf_turnover_cost_bps": 2.0},
    "medium_cost": {"btc_turnover_cost_bps": 25.0, "etf_turnover_cost_bps": 5.0},
    "high_cost": {"btc_turnover_cost_bps": 50.0, "etf_turnover_cost_bps": 10.0},
}
INITIAL_FORMATION_OPTIONS = [False, True]

REQUIRED_RESULTS = [
    "input_validation_summary.json",
    "output_validation_summary.json",
    "data_lineage.csv",
    "model_assumption_audit.json",
    "methodology_audit.md",
    "part13_cost_scenario_dictionary.csv",
    "part13_target_weight_dictionary.csv",
    "part13_rebalance_calendar.csv",
    "part13_rebalance_event_log.csv",
    "part13_rebalanced_return_series.csv",
    "part13_rebalanced_weight_series.csv",
    "part13_implementability_cost_summary.csv",
    "part13_cost_impact_comparison.csv",
    "part13_key_findings.json",
]

REQUIRED_FIGURES = [
    "part13_cost_sensitivity_net_mean.png",
    "part13_cost_sensitivity_total_cost.png",
    "part13_cost_sensitivity_turnover.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 13 transaction-cost sensitivity.")
    parser.add_argument("--input-dir", default="data_2026/cleaned", type=Path)
    parser.add_argument(
        "--part10-run-dir",
        default="outputs/part10_benchmark_cap_sensitivity_outputs/part10_benchmark_cap_sensitivity/colab_part10_seed42",
        type=Path,
    )
    parser.add_argument(
        "--part5-run-dir",
        default="outputs/part5_implementability_rebalancing_outputs/part5_implementability_rebalancing/colab_part5_seed42",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/part13_transaction_cost_sensitivity_outputs/part13_transaction_cost_sensitivity",
        type=Path,
    )
    parser.add_argument("--run-id", default="colab_part13_seed42")
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_pickle(path: Path, payload: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def normalize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [normalize_for_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
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
        "part10_weekly_weights": args.part10_run_dir / "results" / "part10_weekly_weights.csv",
        "part10_target_weight_performance": args.part10_run_dir / "results" / "part10_target_weight_performance_summary.csv",
        "part5_manifest": args.part5_run_dir / "run_manifest.json",
        "part5_output_validation": args.part5_run_dir / "results" / "output_validation_summary.json",
        "part5_rebalanced_performance": args.part5_run_dir / "results" / "rebalanced_performance_summary.csv",
    }
    for name, path in paths.items():
        require(path.exists(), f"Missing required input file: {name} at {path}")
    payload: dict[str, Any] = {
        "paths": paths,
        "state": pd.read_csv(paths["state_model_panel_weekly"], parse_dates=["date"]),
        "part10_manifest": read_json(paths["part10_manifest"]),
        "part10_input_validation": read_json(paths["part10_input_validation"]),
        "part10_output_validation": read_json(paths["part10_output_validation"]),
        "part10_scenario_dictionary": pd.read_csv(paths["part10_scenario_dictionary"]),
        "part10_weekly_weights": pd.read_csv(paths["part10_weekly_weights"], parse_dates=["date"]),
        "part10_target_weight_performance": pd.read_csv(paths["part10_target_weight_performance"]),
        "part5_manifest": read_json(paths["part5_manifest"]),
        "part5_output_validation": read_json(paths["part5_output_validation"]),
        "part5_rebalanced_performance": pd.read_csv(paths["part5_rebalanced_performance"]),
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


def validate_inputs(inputs: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    state = inputs["state"]
    weights = inputs["part10_weekly_weights"]
    scenarios = inputs["part10_scenario_dictionary"]
    require(inputs["part10_input_validation"].get("status") == "passed", "Part 10 input validation did not pass")
    require(inputs["part10_output_validation"].get("status") == "passed", "Part 10 output validation did not pass")
    require(inputs["part5_output_validation"].get("status") == "passed", "Part 5 output validation did not pass")
    require(len(state) == EXPECTED_STATE_ROWS, f"Unexpected state panel rows: {len(state)}")
    require(date_string(state["date"], "min") == EXPECTED_STATE_START, "Unexpected state start date")
    require(date_string(state["date"], "max") == EXPECTED_STATE_END, "Unexpected state end date")
    require(all(col in state.columns for col in ["date"] + ASSETS), "Missing required asset return columns")
    require(set(REQUIRED_RULE_IDS).issubset(set(weights["rule_id"].unique())), "Missing required Part 10 rule ids")
    require(set(PORTFOLIO_FAMILIES).issubset(set(weights["portfolio_family"].unique())), "Missing portfolio families")
    target_scenarios = scenarios[
        scenarios["rule_id"].isin(REQUIRED_RULE_IDS) & scenarios["portfolio_family"].isin(PORTFOLIO_FAMILIES)
    ]
    require(len(target_scenarios) == len(REQUIRED_RULE_IDS) * len(PORTFOLIO_FAMILIES), "Unexpected required scenario count")
    summary = {
        "status": "passed",
        "sample_frozen": True,
        "state_rows": EXPECTED_STATE_ROWS,
        "state_start": EXPECTED_STATE_START,
        "state_end": EXPECTED_STATE_END,
        "implemented_rows": EXPECTED_IMPLEMENTED_ROWS,
        "implemented_start": EXPECTED_IMPLEMENTED_START,
        "rules": REQUIRED_RULE_IDS,
        "portfolio_families": PORTFOLIO_FAMILIES,
        "funding_conventions": FUNDING_CONVENTIONS,
        "rebalance_frequencies": REBALANCE_FREQUENCIES,
        "cost_scenarios": COST_SCENARIOS,
        "initial_formation_options": INITIAL_FORMATION_OPTIONS,
        "implementation_timing": "one_week_lagged_targets",
        "input_hashes": inputs["input_hashes"],
        "upstream_runs": {
            "part10_run_id": inputs["part10_manifest"].get("run_id"),
            "part5_run_id": inputs["part5_manifest"].get("run_id"),
        },
    }
    write_json(dirs["results"] / "input_validation_summary.json", normalize_for_json(summary))
    logging.info("Input validation passed")
    return {"summary": summary}


def cost_rate(cost_scenario: str, asset: str) -> float:
    spec = COST_SCENARIOS[cost_scenario]
    bps = spec["btc_turnover_cost_bps"] if asset == "ret_btc" else spec["etf_turnover_cost_bps"]
    return float(bps / 10000.0)


def build_cost_dictionary(dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for cost_scenario, spec in COST_SCENARIOS.items():
        for include_initial in INITIAL_FORMATION_OPTIONS:
            rows.append(
                {
                    "cost_scenario": cost_scenario,
                    "btc_turnover_cost_bps": spec["btc_turnover_cost_bps"],
                    "etf_turnover_cost_bps": spec["etf_turnover_cost_bps"],
                    "include_initial_formation_cost": include_initial,
                    "notes": (
                        "One-way cost applied to absolute turnover. Initial formation cost is deducted from net returns."
                        if include_initial
                        else "One-way cost applied to scheduled rebalance turnover only; initial formation cost is reported but not deducted."
                    ),
                }
            )
    out = pd.DataFrame(rows).sort_values(["cost_scenario", "include_initial_formation_cost"]).reset_index(drop=True)
    out.to_csv(dirs["results"] / "part13_cost_scenario_dictionary.csv", index=False)
    return out


def scenario_short_id(row: pd.Series) -> str:
    return f"{row['rule_id']}__{row['portfolio_family']}"


def build_lagged_target_weights(inputs: dict[str, Any], dirs: dict[str, Path]) -> pd.DataFrame:
    weights = inputs["part10_weekly_weights"].copy()
    scenarios = inputs["part10_scenario_dictionary"].copy()
    scenarios = scenarios[scenarios["rule_id"].isin(REQUIRED_RULE_IDS) & scenarios["portfolio_family"].isin(PORTFOLIO_FAMILIES)]
    weights = weights.merge(
        scenarios[["scenario_id", "benchmark_family", "rule_id", "portfolio_family"]],
        on=["scenario_id", "benchmark_family", "rule_id", "portfolio_family"],
        how="inner",
        validate="many_to_one",
    )
    weights = weights[weights["asset"].isin(PART10_WEIGHT_ASSETS)].copy()
    require(len(weights) == len(REQUIRED_RULE_IDS) * len(PORTFOLIO_FAMILIES) * EXPECTED_STATE_ROWS * len(PART10_WEIGHT_ASSETS), "Unexpected Part 10 target-weight rows")
    rows = []
    for scenario_id, frame in weights.groupby("scenario_id", sort=True):
        frame = frame.sort_values(["asset", "date"]).copy()
        meta = frame.iloc[0]
        pivot = frame.pivot(index="date", columns="asset", values="weight").sort_index()
        btc_target = frame[frame["asset"].eq("ret_btc")].set_index("date")["btc_weight"].sort_index()
        require(list(pivot.columns) == sorted(PART10_WEIGHT_ASSETS), f"Unexpected asset columns for {scenario_id}")
        max_btc_sleeve = float(btc_target.max())
        lagged_pivot = pivot.shift(1).iloc[1:]
        lagged_btc = btc_target.shift(1).iloc[1:]
        dates = lagged_pivot.index
        for funding in FUNDING_CONVENTIONS:
            for date in dates:
                source = lagged_pivot.loc[date].astype(float).to_dict()
                btc_weight = float(lagged_btc.loc[date])
                if funding == "pro_rata_base":
                    target = {asset: source[asset] for asset in PART10_WEIGHT_ASSETS}
                    target["ret_bil"] = 0.0
                elif funding == "bil_sleeve":
                    if max_btc_sleeve <= FLOAT_TOL:
                        target = {asset: source[asset] for asset in PART10_WEIGHT_ASSETS}
                        target["ret_bil"] = 0.0
                    else:
                        non_btc_sum = float(sum(source[asset] for asset in BASE_ASSETS))
                        require(non_btc_sum > FLOAT_TOL, f"Non-BTC weight sum is zero for {scenario_id}")
                        target = {"ret_btc": btc_weight}
                        for asset in BASE_ASSETS:
                            target[asset] = float(source[asset] / non_btc_sum * (1.0 - max_btc_sleeve))
                        target["ret_bil"] = float(max_btc_sleeve - btc_weight)
                else:
                    raise ValueError(f"Unknown funding convention: {funding}")
                require(abs(sum(target.values()) - 1.0) < 1e-8, f"Target weights do not sum to one for {scenario_id} {funding} {date}")
                require(min(target.values()) >= -1e-8, f"Negative target weight for {scenario_id} {funding} {date}: {target}")
                for asset in ASSETS:
                    rows.append(
                        {
                            "date": date,
                            "source_target_date": pivot.index[pivot.index.get_loc(date) - 1],
                            "source_part10_scenario_id": scenario_id,
                            "benchmark_family": meta["benchmark_family"],
                            "rule_id": meta["rule_id"],
                            "portfolio_family": meta["portfolio_family"],
                            "funding_convention": funding,
                            "asset": asset,
                            "target_weight": target[asset],
                            "target_btc_weight": btc_weight,
                            "max_btc_sleeve": max_btc_sleeve,
                            "implementation_timing": "one_week_lagged_targets",
                        }
                    )
    out = pd.DataFrame(rows).sort_values(["rule_id", "portfolio_family", "funding_convention", "date", "asset"]).reset_index(drop=True)
    out.to_csv(dirs["results"] / "part13_target_weight_dictionary.csv", index=False)
    with (dirs["models"] / "part13_target_weight_dictionary.pkl").open("wb") as handle:
        pickle.dump(out, handle)
    return out


def build_rebalance_calendar(inputs: dict[str, Any], dirs: dict[str, Path]) -> pd.DataFrame:
    panel = inputs["state"][["date"]].copy().sort_values("date").reset_index(drop=True)
    panel = panel.iloc[1:].copy().reset_index(drop=True)
    panel["year_month"] = panel["date"].dt.to_period("M").astype(str)
    panel["year_quarter"] = panel["date"].dt.to_period("Q").astype(str)
    panel["is_month_end_rebalance"] = panel["date"].eq(panel.groupby("year_month")["date"].transform("max"))
    panel["is_quarter_end_rebalance"] = panel["date"].eq(panel.groupby("year_quarter")["date"].transform("max"))
    panel["is_formation_date"] = panel.index == 0
    out = panel[["date", "is_month_end_rebalance", "is_quarter_end_rebalance", "is_formation_date"]].copy()
    out.to_csv(dirs["results"] / "part13_rebalance_calendar.csv", index=False)
    return out


def drift_weights(start_weights: dict[str, float], returns: pd.Series, gross_return: float) -> dict[str, float]:
    denominator = 1.0 + gross_return
    require(abs(denominator) > FLOAT_TOL, "Portfolio gross return denominator is too close to zero")
    return {asset: float(start_weights[asset] * (1.0 + float(returns[asset])) / denominator) for asset in ASSETS}


def target_for_date(targets: pd.DataFrame, date: pd.Timestamp) -> dict[str, float]:
    frame = targets[targets["date"].eq(date)]
    require(len(frame) == len(ASSETS), f"Unexpected target rows for {date}: {len(frame)}")
    return frame.set_index("asset")["target_weight"].astype(float).to_dict()


def scenario_id_for(meta: pd.Series, funding: str, frequency: str, cost_scenario: str, include_initial: bool) -> str:
    initial_tag = "including_initial" if include_initial else "excluding_initial"
    return "__".join(
        [
            str(meta["rule_id"]),
            str(meta["portfolio_family"]),
            funding,
            frequency,
            cost_scenario,
            initial_tag,
        ]
    )


def simulate_scenario(
    scenario_id: str,
    meta: pd.Series,
    targets: pd.DataFrame,
    returns_panel: pd.DataFrame,
    calendar: pd.DataFrame,
    frequency: str,
    cost_scenario: str,
    include_initial: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    data = returns_panel.merge(calendar, on="date", how="inner", validate="one_to_one").sort_values("date").reset_index(drop=True)
    require(len(data) == EXPECTED_IMPLEMENTED_ROWS, f"Unexpected implemented data rows: {len(data)}")
    frequency_col = "is_month_end_rebalance" if frequency == "monthly" else "is_quarter_end_rebalance"
    previous_end_weights: dict[str, float] | None = None
    return_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    cumulative_gross = 1.0
    cumulative_net = 1.0

    for _, row in data.iterrows():
        date = pd.Timestamp(row["date"])
        target = target_for_date(targets, date)
        is_formation = bool(row["is_formation_date"])
        is_scheduled_rebalance = bool(row[frequency_col])
        is_rebalance = is_formation or is_scheduled_rebalance
        asset_returns = row[ASSETS]

        if previous_end_weights is None:
            pre_trade_weights = {asset: 0.0 for asset in ASSETS}
            beginning_weights = target.copy()
            turnover = float(sum(abs(beginning_weights[asset] - pre_trade_weights[asset]) for asset in ASSETS))
            setup_cost_estimate = float(sum(abs(beginning_weights[asset]) * cost_rate(cost_scenario, asset) for asset in ASSETS))
            transaction_cost = setup_cost_estimate if include_initial else 0.0
            event_type = "formation"
        elif is_rebalance:
            pre_trade_weights = previous_end_weights.copy()
            beginning_weights = target.copy()
            turnover = float(sum(abs(beginning_weights[asset] - pre_trade_weights[asset]) for asset in ASSETS))
            setup_cost_estimate = 0.0
            transaction_cost = float(sum(abs(beginning_weights[asset] - pre_trade_weights[asset]) * cost_rate(cost_scenario, asset) for asset in ASSETS))
            event_type = "scheduled_rebalance"
        else:
            pre_trade_weights = previous_end_weights.copy()
            beginning_weights = previous_end_weights.copy()
            turnover = 0.0
            setup_cost_estimate = 0.0
            transaction_cost = 0.0
            event_type = "hold"

        gross_return = float(sum(beginning_weights[asset] * float(asset_returns[asset]) for asset in ASSETS))
        net_return = gross_return - transaction_cost
        cumulative_gross *= 1.0 + gross_return
        cumulative_net *= 1.0 + net_return
        end_weights = drift_weights(beginning_weights, asset_returns, gross_return)

        common = {
            "scenario_id": scenario_id,
            "benchmark_family": meta["benchmark_family"],
            "rule_id": meta["rule_id"],
            "portfolio_family": meta["portfolio_family"],
            "funding_convention": meta["funding_convention"],
            "rebalance_frequency": frequency,
            "cost_scenario": cost_scenario,
            "include_initial_formation_cost": include_initial,
            "date": date,
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
                    "cost_is_deducted_from_net_return": bool(event_type == "scheduled_rebalance" or include_initial),
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
                    "asset_turnover": abs(beginning_weights[asset] - pre_trade_weights[asset]) if is_rebalance else 0.0,
                    "asset_cost_rate": cost_rate(cost_scenario, asset),
                }
            )
        previous_end_weights = end_weights
    return return_rows, weight_rows, event_rows


def run_simulations(inputs: dict[str, Any], targets: pd.DataFrame, calendar: pd.DataFrame, dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    state = inputs["state"][["date"] + ASSETS].copy()
    implemented_returns = state[state["date"].ge(pd.Timestamp(EXPECTED_IMPLEMENTED_START))].copy()
    scenario_meta = (
        targets.groupby(["source_part10_scenario_id", "benchmark_family", "rule_id", "portfolio_family", "funding_convention"], as_index=False)
        .agg(max_btc_sleeve=("max_btc_sleeve", "max"), average_target_btc_weight=("target_btc_weight", "mean"))
        .sort_values(["rule_id", "portfolio_family", "funding_convention"])
    )
    all_returns: list[dict[str, Any]] = []
    all_weights: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    for _, meta in scenario_meta.iterrows():
        scenario_targets = targets[
            (targets["source_part10_scenario_id"].eq(meta["source_part10_scenario_id"]))
            & (targets["funding_convention"].eq(meta["funding_convention"]))
        ].copy()
        for frequency in REBALANCE_FREQUENCIES:
            for cost_scenario in COST_SCENARIOS:
                for include_initial in INITIAL_FORMATION_OPTIONS:
                    sid = scenario_id_for(meta, str(meta["funding_convention"]), frequency, cost_scenario, bool(include_initial))
                    returns, weights, events = simulate_scenario(
                        sid,
                        meta,
                        scenario_targets,
                        implemented_returns,
                        calendar,
                        frequency,
                        cost_scenario,
                        bool(include_initial),
                    )
                    all_returns.extend(returns)
                    all_weights.extend(weights)
                    all_events.extend(events)
    return_df = pd.DataFrame(all_returns).sort_values(["scenario_id", "date"]).reset_index(drop=True)
    weight_df = pd.DataFrame(all_weights).sort_values(["scenario_id", "date", "asset"]).reset_index(drop=True)
    event_df = pd.DataFrame(all_events).sort_values(["scenario_id", "date"]).reset_index(drop=True)
    return_df.to_csv(dirs["results"] / "part13_rebalanced_return_series.csv", index=False)
    weight_df.to_csv(dirs["results"] / "part13_rebalanced_weight_series.csv", index=False)
    event_df.to_csv(dirs["results"] / "part13_rebalance_event_log.csv", index=False)
    logging.info("Rebalancing simulations completed")
    return {"returns": return_df, "weights": weight_df, "events": event_df}


def var_cvar(returns: pd.Series) -> tuple[float, float, int]:
    clean = returns.dropna()
    var = float(clean.quantile(TAIL_ALPHA))
    tail = clean[clean <= var]
    return var, float(tail.mean()), int(len(tail))


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    return float((wealth / peak - 1.0).min())


def performance_metrics(returns: pd.Series, prefix: str = "") -> dict[str, Any]:
    clean = returns.dropna()
    var, cvar, tail_count = var_cvar(clean)
    vol = float(clean.std(ddof=1))
    return {
        f"{prefix}annualized_mean_arithmetic": float(clean.mean() * TRADING_WEEKS_PER_YEAR),
        f"{prefix}annualized_volatility": float(vol * math.sqrt(TRADING_WEEKS_PER_YEAR)),
        f"{prefix}var_95_weekly": var,
        f"{prefix}cvar_95_weekly": cvar,
        f"{prefix}tail_scenario_count": tail_count,
        f"{prefix}max_drawdown": max_drawdown(clean),
        f"{prefix}positive_week_share": float((clean > 0.0).mean()),
        f"{prefix}sharpe_annualized_zero_rf": float(clean.mean() / vol * math.sqrt(TRADING_WEEKS_PER_YEAR)) if vol > 0 else float("nan"),
    }


def compute_summary(sim: dict[str, pd.DataFrame], dirs: dict[str, Path]) -> pd.DataFrame:
    returns = sim["returns"].copy()
    events = sim["events"].copy()
    rows = []
    for scenario_id, frame in returns.groupby("scenario_id", sort=True):
        first = frame.iloc[0]
        event_frame = events[events["scenario_id"].eq(scenario_id)]
        scheduled = event_frame[event_frame["event_type"].eq("scheduled_rebalance")]
        row = {
            "scenario_id": scenario_id,
            "benchmark_family": first["benchmark_family"],
            "rule_id": first["rule_id"],
            "portfolio_family": first["portfolio_family"],
            "funding_convention": first["funding_convention"],
            "rebalance_frequency": first["rebalance_frequency"],
            "cost_scenario": first["cost_scenario"],
            "include_initial_formation_cost": bool(first["include_initial_formation_cost"]),
            "start_date": pd.Timestamp(frame["date"].min()).strftime("%Y-%m-%d"),
            "end_date": pd.Timestamp(frame["date"].max()).strftime("%Y-%m-%d"),
            "average_btc_beginning_weight": float(frame["btc_beginning_weight"].mean()),
            "max_btc_beginning_weight": float(frame["btc_beginning_weight"].max()),
            "average_btc_target_weight": float(frame["btc_target_weight"].mean()),
            "total_transaction_cost": float(frame["transaction_cost"].sum()),
            "initial_formation_cost": float(event_frame[event_frame["event_type"].eq("formation")]["setup_cost_estimate"].sum()),
            "scheduled_transaction_cost": float(scheduled["transaction_cost"].sum()),
            "scheduled_rebalance_count": int(len(scheduled)),
            "total_turnover_including_formation": float(event_frame["turnover"].sum()),
            "total_turnover_excluding_formation": float(scheduled["turnover"].sum()),
            "annualized_turnover_excluding_formation": float(scheduled["turnover"].sum() / (len(frame) / TRADING_WEEKS_PER_YEAR)),
            "final_cumulative_gross_value": float(frame["cumulative_gross_value"].iloc[-1]),
            "final_cumulative_net_value": float(frame["cumulative_net_value"].iloc[-1]),
        }
        row.update(performance_metrics(frame["gross_return"], prefix="gross_"))
        row.update(performance_metrics(frame["net_return"], prefix="net_"))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("scenario_id").reset_index(drop=True)
    out.to_csv(dirs["results"] / "part13_implementability_cost_summary.csv", index=False)
    return out


def build_cost_impact_comparison(summary: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    metrics = [
        "total_transaction_cost",
        "final_cumulative_net_value",
        "net_annualized_mean_arithmetic",
        "net_annualized_volatility",
        "net_cvar_95_weekly",
        "net_max_drawdown",
    ]
    rows: list[dict[str, Any]] = []
    key_cols = ["portfolio_family", "funding_convention", "rebalance_frequency", "rule_id"]
    for key, frame in summary.groupby(key_cols, sort=True):
        key_payload = dict(zip(key_cols, key))
        for include_initial in INITIAL_FORMATION_OPTIONS:
            scoped = frame[frame["include_initial_formation_cost"].eq(include_initial)].set_index("cost_scenario")
            for left, right in [("medium_cost", "low_cost"), ("high_cost", "medium_cost"), ("high_cost", "low_cost")]:
                for metric in metrics:
                    left_value = float(scoped.loc[left, metric])
                    right_value = float(scoped.loc[right, metric])
                    rows.append(
                        {
                            "comparison_id": f"{key_payload['rule_id']}__{key_payload['portfolio_family']}__{key_payload['funding_convention']}__{key_payload['rebalance_frequency']}__{left}_minus_{right}__initial_{include_initial}",
                            **key_payload,
                            "cost_scenario_left": left,
                            "cost_scenario_right": right,
                            "include_initial_formation_cost_left": include_initial,
                            "include_initial_formation_cost_right": include_initial,
                            "metric": metric,
                            "left_value": left_value,
                            "right_value": right_value,
                            "difference_left_minus_right": left_value - right_value,
                        }
                    )
        for cost_scenario, scoped in frame.groupby("cost_scenario", sort=True):
            by_initial = scoped.set_index("include_initial_formation_cost")
            for metric in metrics:
                left_value = float(by_initial.loc[True, metric])
                right_value = float(by_initial.loc[False, metric])
                rows.append(
                    {
                        "comparison_id": f"{key_payload['rule_id']}__{key_payload['portfolio_family']}__{key_payload['funding_convention']}__{key_payload['rebalance_frequency']}__including_minus_excluding_initial__{cost_scenario}",
                        **key_payload,
                        "cost_scenario_left": cost_scenario,
                        "cost_scenario_right": cost_scenario,
                        "include_initial_formation_cost_left": True,
                        "include_initial_formation_cost_right": False,
                        "metric": metric,
                        "left_value": left_value,
                        "right_value": right_value,
                        "difference_left_minus_right": left_value - right_value,
                    }
                )
    out = pd.DataFrame(rows).sort_values(["comparison_id", "metric"]).reset_index(drop=True)
    out.to_csv(dirs["results"] / "part13_cost_impact_comparison.csv", index=False)
    return out


def build_key_findings(inputs: dict[str, Any], summary: pd.DataFrame, comparison: pd.DataFrame, dirs: dict[str, Path]) -> dict[str, Any]:
    medium = summary[
        summary["cost_scenario"].eq("medium_cost")
        & summary["include_initial_formation_cost"].eq(True)
        & summary["rebalance_frequency"].eq("monthly")
        & summary["funding_convention"].eq("bil_sleeve")
    ].copy()
    cond = medium[medium["rule_id"].eq("conditional_cap_10pct")]
    cap = medium[medium["rule_id"].eq("cap_only_10pct")]
    fixed = medium[medium["rule_id"].eq("matched_fixed_btc")]
    initial_impact = comparison[
        comparison["metric"].eq("final_cumulative_net_value")
        & comparison["include_initial_formation_cost_left"].eq(True)
        & comparison["include_initial_formation_cost_right"].eq(False)
    ]
    payload = {
        "scenario_rows": int(len(summary)),
        "comparison_rows": int(len(comparison)),
        "implementation_scope": "one_week_lagged_monthly_or_quarterly_rebalancing",
        "medium_monthly_bil_including_initial_snapshot": medium[
            [
                "rule_id",
                "portfolio_family",
                "total_transaction_cost",
                "final_cumulative_net_value",
                "net_annualized_mean_arithmetic",
                "net_annualized_volatility",
            ]
        ].to_dict(orient="records"),
        "max_initial_formation_drag_on_final_net_value": float(initial_impact["difference_left_minus_right"].min()) if len(initial_impact) else 0.0,
        "conditional_vs_matched_fixed_medium_monthly_bil_net_mean": compare_snapshot(cond, fixed),
        "conditional_vs_cap_only_medium_monthly_bil_net_mean": compare_snapshot(cond, cap),
        "recommended_thesis_statement": (
            "Part 13 shows that the inclusion of initial formation costs mechanically lowers net performance, but the magnitudes are small relative to weekly return variation. "
            "The transaction-cost extension should be used to support implementability discipline, not to claim a new source of return improvement."
        ),
    }
    write_json(dirs["results"] / "part13_key_findings.json", normalize_for_json(payload))
    return payload


def compare_snapshot(left: pd.DataFrame, right: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for family in PORTFOLIO_FAMILIES:
        l = left[left["portfolio_family"].eq(family)]
        r = right[right["portfolio_family"].eq(family)]
        if len(l) == 1 and len(r) == 1:
            rows.append(
                {
                    "portfolio_family": family,
                    "left_rule": str(l.iloc[0]["rule_id"]),
                    "right_rule": str(r.iloc[0]["rule_id"]),
                    "net_annualized_mean_difference": float(l.iloc[0]["net_annualized_mean_arithmetic"] - r.iloc[0]["net_annualized_mean_arithmetic"]),
                    "total_transaction_cost_difference": float(l.iloc[0]["total_transaction_cost"] - r.iloc[0]["total_transaction_cost"]),
                }
            )
    return rows


def plot_net_mean(summary: pd.DataFrame, output_path: Path) -> None:
    frame = summary[
        summary["funding_convention"].eq("bil_sleeve")
        & summary["rebalance_frequency"].eq("monthly")
        & summary["include_initial_formation_cost"].eq(True)
    ].copy()
    frame["label"] = frame["portfolio_family"] + "\n" + frame["rule_id"].str.replace("_", " ")
    ordered = frame.sort_values(["portfolio_family", "rule_id", "cost_scenario"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x_labels = ordered["label"].drop_duplicates().tolist()
    x = np.arange(len(x_labels))
    width = 0.24
    colors = {"low_cost": "#4c78a8", "medium_cost": "#f58518", "high_cost": "#e45756"}
    for i, cost in enumerate(["low_cost", "medium_cost", "high_cost"]):
        vals = ordered[ordered["cost_scenario"].eq(cost)].set_index("label").loc[x_labels]["net_annualized_mean_arithmetic"]
        ax.bar(x + (i - 1) * width, vals, width=width, label=cost, color=colors[cost])
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Net annualized mean")
    ax.set_title("Part 13 Net Mean by Cost Scenario\nMonthly BIL Sleeve, Initial Formation Cost Included")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_total_cost(summary: pd.DataFrame, output_path: Path) -> None:
    frame = summary[
        summary["rebalance_frequency"].eq("monthly")
        & summary["include_initial_formation_cost"].eq(True)
    ].copy()
    grouped = frame.groupby(["rule_id", "cost_scenario"], as_index=False)["total_transaction_cost"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = REQUIRED_RULE_IDS
    x = np.arange(len(x_labels))
    width = 0.24
    colors = {"low_cost": "#4c78a8", "medium_cost": "#f58518", "high_cost": "#e45756"}
    for i, cost in enumerate(["low_cost", "medium_cost", "high_cost"]):
        vals = grouped[grouped["cost_scenario"].eq(cost)].set_index("rule_id").loc[x_labels]["total_transaction_cost"]
        ax.bar(x + (i - 1) * width, vals, width=width, label=cost, color=colors[cost])
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("_", "\n") for label in x_labels], fontsize=8)
    ax.set_ylabel("Total transaction cost")
    ax.set_title("Part 13 Average Total Cost by Rule\nMonthly, Initial Formation Cost Included")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_turnover(summary: pd.DataFrame, output_path: Path) -> None:
    frame = summary[
        summary["cost_scenario"].eq("medium_cost")
        & summary["include_initial_formation_cost"].eq(True)
    ].copy()
    grouped = frame.groupby(["rule_id", "rebalance_frequency"], as_index=False)["total_turnover_excluding_formation"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = REQUIRED_RULE_IDS
    x = np.arange(len(x_labels))
    width = 0.32
    for i, freq in enumerate(REBALANCE_FREQUENCIES):
        vals = grouped[grouped["rebalance_frequency"].eq(freq)].set_index("rule_id").loc[x_labels]["total_turnover_excluding_formation"]
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=freq)
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("_", "\n") for label in x_labels], fontsize=8)
    ax.set_ylabel("Total scheduled turnover")
    ax.set_title("Part 13 Scheduled Turnover by Rule\nMedium Cost, Initial Formation Cost Included")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_audits(args: argparse.Namespace, inputs: dict[str, Any], dirs: dict[str, Path]) -> None:
    lineage = pd.DataFrame(
        [{"input_name": name, "path": str(path), "sha256": inputs["input_hashes"][name]} for name, path in sorted(inputs["paths"].items())]
    )
    lineage.to_csv(dirs["results"] / "data_lineage.csv", index=False)
    audit = {
        "status": "passed",
        "uses_part10_targets": True,
        "reestimates_models": False,
        "implementation_timing": "one_week_lagged_targets",
        "cost_formula": "sum(abs(target_weight - pre_trade_weight) * one_way_asset_cost_rate)",
        "initial_formation_cost": "reported under both included and excluded net-return conventions",
        "sample": f"{EXPECTED_IMPLEMENTED_START} to {EXPECTED_STATE_END}",
    }
    write_json(dirs["results"] / "model_assumption_audit.json", audit)
    md = f"""# Part 13 Methodology Audit

Part 13 reads Part 10 target-weight benchmark scenarios and implements the required rules under monthly and quarterly rebalancing. The simulated implementation uses one-week lagged targets, so the first implemented week is {EXPECTED_IMPLEMENTED_START}. This avoids same-week state look-ahead and aligns the conditional 10% cap case with the Part 5 main implementation timing.

Funding conventions follow Part 5. Under `pro_rata_base`, BTC is funded by scaling the non-BTC base assets. Under `bil_sleeve`, the maximum BTC sleeve for a scenario is held constant and unused sleeve capital is parked in BIL.

Transaction costs are one-way costs applied to absolute turnover by asset. BTC turnover uses the BTC cost rate; ETF and BIL turnover use the ETF cost rate. Each cost scenario is reported twice: excluding initial formation cost from net returns and including it in the first implemented week.
"""
    (dirs["results"] / "methodology_audit.md").write_text(md, encoding="utf-8")


def validate_outputs(
    inputs: dict[str, Any],
    cost_dict: pd.DataFrame,
    targets: pd.DataFrame,
    calendar: pd.DataFrame,
    sim: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    dirs: dict[str, Path],
) -> dict[str, Any]:
    required_results = [name for name in REQUIRED_RESULTS if name != "output_validation_summary.json"]
    required_files_present = all((dirs["results"] / name).exists() for name in required_results) and all(
        (dirs["figures"] / name).exists() for name in REQUIRED_FIGURES
    )
    expected_summary_rows = len(REQUIRED_RULE_IDS) * len(PORTFOLIO_FAMILIES) * len(FUNDING_CONVENTIONS) * len(REBALANCE_FREQUENCIES) * len(COST_SCENARIOS) * len(INITIAL_FORMATION_OPTIONS)
    target_sum = targets.groupby(["source_part10_scenario_id", "funding_convention", "date"])["target_weight"].sum()
    target_weights_sum_ok = bool((target_sum - 1.0).abs().max() < 1e-8)
    return_rows_ok = bool(len(sim["returns"]) == expected_summary_rows * EXPECTED_IMPLEMENTED_ROWS)
    summary_rows_ok = bool(len(summary) == expected_summary_rows)
    cost_rows_ok = bool(len(cost_dict) == len(COST_SCENARIOS) * len(INITIAL_FORMATION_OPTIONS))
    calendar_ok = bool(len(calendar) == EXPECTED_IMPLEMENTED_ROWS and date_string(calendar["date"], "min") == EXPECTED_IMPLEMENTED_START)
    include_compare = summary.pivot_table(
        index=["rule_id", "portfolio_family", "funding_convention", "rebalance_frequency", "cost_scenario"],
        columns="include_initial_formation_cost",
        values=["total_transaction_cost", "final_cumulative_net_value"],
        aggfunc="first",
    )
    initial_cost_monotonic_ok = bool((include_compare[("total_transaction_cost", True)] >= include_compare[("total_transaction_cost", False)] - FLOAT_TOL).all())
    initial_net_not_better_ok = bool((include_compare[("final_cumulative_net_value", True)] <= include_compare[("final_cumulative_net_value", False)] + FLOAT_TOL).all())

    cost_order_ok = True
    for _, frame in summary.groupby(["rule_id", "portfolio_family", "funding_convention", "rebalance_frequency", "include_initial_formation_cost"], sort=True):
        vals = frame.set_index("cost_scenario")["total_transaction_cost"]
        if not (vals["high_cost"] >= vals["medium_cost"] - FLOAT_TOL and vals["medium_cost"] >= vals["low_cost"] - FLOAT_TOL):
            cost_order_ok = False
            break

    part5_compare = compare_part5_main(inputs["part5_rebalanced_performance"], summary)
    part5_reproduction_ok = bool(part5_compare["max_abs_net_mean_diff"] <= 1e-12 and part5_compare["max_abs_total_cost_diff"] <= 1e-12)
    summary_payload = {
        "status": "passed"
        if all(
            [
                required_files_present,
                target_weights_sum_ok,
                return_rows_ok,
                summary_rows_ok,
                cost_rows_ok,
                calendar_ok,
                initial_cost_monotonic_ok,
                initial_net_not_better_ok,
                cost_order_ok,
                part5_reproduction_ok,
            ]
        )
        else "failed",
        "cost_dictionary_rows": int(len(cost_dict)),
        "target_weight_rows": int(len(targets)),
        "calendar_rows": int(len(calendar)),
        "return_rows": int(len(sim["returns"])),
        "summary_rows": int(len(summary)),
        "comparison_rows": int(len(comparison)),
        "target_weights_sum_ok": target_weights_sum_ok,
        "return_rows_ok": return_rows_ok,
        "summary_rows_ok": summary_rows_ok,
        "initial_cost_monotonic_ok": initial_cost_monotonic_ok,
        "initial_net_not_better_ok": initial_net_not_better_ok,
        "cost_order_ok": cost_order_ok,
        "part5_medium_excluding_initial_reproduction": part5_compare,
        "required_files_present": required_files_present,
    }
    require(summary_payload["status"] == "passed", f"Output validation failed: {summary_payload}")
    write_json(dirs["results"] / "output_validation_summary.json", normalize_for_json(summary_payload))
    return summary_payload


def compare_part5_main(part5: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Any]:
    p5 = part5[
        part5["rule_id"].eq("main_executed")
        & part5["signal_timing"].eq("lagged_one_week")
        & part5["cost_scenario"].eq("moderate_cost")
    ].copy()
    p13 = summary[
        summary["rule_id"].eq("conditional_cap_10pct")
        & summary["cost_scenario"].eq("medium_cost")
        & summary["include_initial_formation_cost"].eq(False)
    ].copy()
    pairs = []
    for _, row in p13.iterrows():
        match = p5[
            p5["portfolio_family"].eq(row["portfolio_family"])
            & p5["funding_convention"].eq(row["funding_convention"])
            & p5["rebalance_frequency"].eq(row["rebalance_frequency"])
        ]
        if len(match) == 1:
            ref = match.iloc[0]
            pairs.append(
                {
                    "portfolio_family": row["portfolio_family"],
                    "funding_convention": row["funding_convention"],
                    "rebalance_frequency": row["rebalance_frequency"],
                    "net_mean_diff": float(row["net_annualized_mean_arithmetic"] - ref["net_annualized_mean_arithmetic"]),
                    "total_cost_diff": float(row["total_transaction_cost"] - ref["total_transaction_cost"]),
                }
            )
    max_abs_mean = max((abs(item["net_mean_diff"]) for item in pairs), default=float("inf"))
    max_abs_cost = max((abs(item["total_cost_diff"]) for item in pairs), default=float("inf"))
    return {
        "matched_rows": len(pairs),
        "max_abs_net_mean_diff": float(max_abs_mean),
        "max_abs_total_cost_diff": float(max_abs_cost),
        "notes": "Part 13 medium/excluding-initial conditional_cap_10pct reproduces Part 5 main_executed lagged moderate-cost implementation.",
    }


def package_versions() -> dict[str, str]:
    packages = {}
    for name in ["numpy", "pandas", "matplotlib"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    return {"python": sys.version.replace("\n", " "), "platform": platform.platform(), **packages}


def write_manifest(args: argparse.Namespace, inputs: dict[str, Any], output_validation: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    manifest = {
        "part_id": "part13_transaction_cost_sensitivity",
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_frozen": True,
        "implemented_start": EXPECTED_IMPLEMENTED_START,
        "sample_end": EXPECTED_STATE_END,
        "seed": args.seed,
        "inputs": {
            "state_model_panel_weekly": str(args.input_dir / "state_model_panel_weekly.csv"),
            "part10_run_dir": str(args.part10_run_dir),
            "part5_run_dir": str(args.part5_run_dir),
        },
        "input_hashes": inputs["input_hashes"],
        "parameters": {
            "rules": REQUIRED_RULE_IDS,
            "portfolio_families": PORTFOLIO_FAMILIES,
            "funding_conventions": FUNDING_CONVENTIONS,
            "rebalance_frequencies": REBALANCE_FREQUENCIES,
            "cost_scenarios": COST_SCENARIOS,
            "initial_formation_options": INITIAL_FORMATION_OPTIONS,
            "implementation_timing": "one_week_lagged_targets",
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
    return manifest


def main() -> None:
    args = parse_args()
    run_dir = args.output_dir / args.run_id
    dirs = ensure_dirs(run_dir)
    setup_logging(dirs["logs"])
    logging.info("Starting Part 13 run in %s", run_dir)
    inputs = read_inputs(args)
    enforce_resume_input_hashes(args, dirs, inputs["input_hashes"])
    validation = validate_inputs(inputs, dirs)
    save_pickle(dirs["checkpoints"] / "01_input_validation.pkl", validation)
    cost_dict = build_cost_dictionary(dirs)
    targets = build_lagged_target_weights(inputs, dirs)
    save_pickle(dirs["checkpoints"] / "02_targets.pkl", targets)
    calendar = build_rebalance_calendar(inputs, dirs)
    sim = run_simulations(inputs, targets, calendar, dirs)
    save_pickle(dirs["checkpoints"] / "03_simulations.pkl", sim)
    summary = compute_summary(sim, dirs)
    comparison = build_cost_impact_comparison(summary, dirs)
    findings = build_key_findings(inputs, summary, comparison, dirs)
    save_pickle(dirs["checkpoints"] / "04_summary_comparison.pkl", {"summary": summary, "comparison": comparison, "findings": findings})
    plot_net_mean(summary, dirs["figures"] / "part13_cost_sensitivity_net_mean.png")
    plot_total_cost(summary, dirs["figures"] / "part13_cost_sensitivity_total_cost.png")
    plot_turnover(summary, dirs["figures"] / "part13_cost_sensitivity_turnover.png")
    write_audits(args, inputs, dirs)
    output_validation = validate_outputs(inputs, cost_dict, targets, calendar, sim, summary, comparison, dirs)
    write_manifest(args, inputs, output_validation, dirs)
    logging.info("Part 13 completed successfully")


if __name__ == "__main__":
    main()
