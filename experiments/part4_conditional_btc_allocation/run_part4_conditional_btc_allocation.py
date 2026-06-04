#!/usr/bin/env python3
"""Part 4 experiment runner: conditional BTC allocation and risk-budget diagnostics."""

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
EXPECTED_STATE_COUNTS = {"state_0": 149, "state_1": 167, "state_2": 30, "state_3": 79}
TRADING_WEEKS_PER_YEAR = 52
TAIL_ALPHA = 0.05
RISK_BUDGET_CAP = 0.10
FLOAT_TOL = 1e-10
SMALL_STATE_THRESHOLD_WEEKS = 52

BASE_ASSETS = ["ret_spy", "ret_tlt", "ret_ief", "ret_gld", "ret_dbc"]
PORTFOLIO_ASSETS = ["ret_btc"] + BASE_ASSETS
ASSET_RETURN_COLS = PORTFOLIO_ASSETS + ["ret_bil"]
BTC_WEIGHT_GRID = [0.0, 0.01, 0.02, 0.03, 0.05]

RAW_RULES = {
    "main": {
        "description": "Conservative evidence-gated rule; state_2 is excluded from the main rule because it has only 30 weeks.",
        "role": "main",
        "raw_weights": {"state_0": 0.03, "state_1": 0.01, "state_2": 0.0, "state_3": 0.0},
    },
    "sensitivity_state2_low": {
        "description": "Sensitivity rule that assigns a low 1% BTC weight to the small state_2 sample.",
        "role": "sensitivity",
        "raw_weights": {"state_0": 0.03, "state_1": 0.01, "state_2": 0.01, "state_3": 0.0},
    },
}

REQUIRED_RESULTS = [
    "input_validation_summary.json",
    "allocation_rule_definition.csv",
    "risk_budget_cap_audit.csv",
    "rule_evidence_by_state.csv",
    "weekly_conditional_weights.csv",
    "conditional_portfolio_return_series.csv",
    "conditional_portfolio_performance_summary.csv",
    "conditional_risk_contributions_vol.csv",
    "conditional_risk_contributions_cvar.csv",
    "state_conditioned_conditional_portfolio_summary.csv",
    "state_conditioned_conditional_risk_contributions.csv",
    "state_activation_summary.csv",
    "sensitivity_state2_low_allocation_summary.csv",
    "methodology_audit.md",
    "data_lineage.csv",
    "model_assumption_audit.json",
    "output_validation_summary.json",
]
REQUIRED_FIGURES = [
    "conditional_rule_timeline.png",
    "conditional_portfolio_drawdowns.png",
    "conditional_vs_fixed_btc_summary.png",
    "risk_budget_cap_audit.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run conditional BTC allocation and risk-budget satellite diagnostics."
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
    parser.add_argument("--output-dir", default="outputs/part4_conditional_btc_allocation", type=Path)
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


def drawdown_from_returns(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    return float(drawdown.min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1.0


def var_cvar(returns: pd.Series, alpha: float = TAIL_ALPHA) -> tuple[float, float, int]:
    clean = returns.dropna()
    var = float(clean.quantile(alpha))
    tail = clean[clean <= var]
    cvar = float(tail.mean())
    return var, cvar, int(len(tail))


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        "part2_btc_risk_budget_summary": args.part2_run_dir / "results" / "btc_risk_budget_summary.csv",
        "part2_state_btc_risk_contributions": args.part2_run_dir
        / "results"
        / "state_conditioned_btc_risk_contributions.csv",
        "part2_fixed_btc_portfolio_weights": args.part2_run_dir / "results" / "fixed_btc_portfolio_weights.csv",
        "part2_baseline_portfolio_weights": args.part2_run_dir / "results" / "baseline_portfolio_weights.csv",
        "part2_portfolio_performance_summary": args.part2_run_dir / "results" / "portfolio_performance_summary.csv",
        "part3_manifest": args.part3_run_dir / "run_manifest.json",
        "part3_input_validation_summary": args.part3_run_dir / "results" / "input_validation_summary.json",
        "part3_output_validation_summary": args.part3_run_dir / "results" / "output_validation_summary.json",
        "part3_state_btc_performance": args.part3_run_dir / "results" / "state_conditioned_btc_performance.csv",
        "part3_state_beta_diagnostics": args.part3_run_dir / "results" / "state_conditioned_beta_diagnostics.csv",
        "part3_state_sample_warnings": args.part3_run_dir / "results" / "state_sample_warnings.csv",
    }
    for name, path in paths.items():
        require(path.exists(), f"Missing required input file: {name} at {path}")

    return {
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
        "part2_btc_risk_budget": pd.read_csv(paths["part2_btc_risk_budget_summary"]),
        "part2_state_btc_risk": pd.read_csv(paths["part2_state_btc_risk_contributions"]),
        "part2_fixed_weights": pd.read_csv(paths["part2_fixed_btc_portfolio_weights"]),
        "part2_baseline_weights": pd.read_csv(paths["part2_baseline_portfolio_weights"]),
        "part2_portfolio_performance": pd.read_csv(paths["part2_portfolio_performance_summary"]),
        "part3_manifest": read_json(paths["part3_manifest"]),
        "part3_input_validation": read_json(paths["part3_input_validation_summary"]),
        "part3_output_validation": read_json(paths["part3_output_validation_summary"]),
        "part3_state_btc_performance": pd.read_csv(paths["part3_state_btc_performance"]),
        "part3_state_beta": pd.read_csv(paths["part3_state_beta_diagnostics"]),
        "part3_state_warnings": pd.read_csv(paths["part3_state_sample_warnings"]),
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
    profiles = inputs["profiles"].copy()
    hashes = inputs["input_hashes"]
    part1_manifest = inputs["part1_manifest"]
    part2_manifest = inputs["part2_manifest"]
    part3_manifest = inputs["part3_manifest"]

    require(inputs["part1_validation"].get("status") == "passed", "Part 1 validation did not pass")
    require(part1_manifest.get("model_diagnostics", {}).get("hmm4_converged") is True, "Part 1 HMM-4 did not converge")
    require(inputs["part2_input_validation"].get("status") == "passed", "Part 2 input validation did not pass")
    require(inputs["part2_output_validation"].get("status") == "passed", "Part 2 output validation did not pass")
    require(inputs["part3_input_validation"].get("status") == "passed", "Part 3 input validation did not pass")
    require(inputs["part3_output_validation"].get("status") == "passed", "Part 3 output validation did not pass")

    require(len(asset) == EXPECTED_ASSET_ROWS, f"Unexpected asset row count: {len(asset)}")
    require(date_string(asset["date"], "min") == EXPECTED_ASSET_START, "Unexpected asset start date")
    require(date_string(asset["date"], "max") == EXPECTED_ASSET_END, "Unexpected asset end date")
    require(asset["date"].dt.dayofweek.eq(4).all(), "Asset dates are not all Fridays")
    require(all(col in asset.columns for col in ASSET_RETURN_COLS), "Missing required return columns")
    require(asset[ASSET_RETURN_COLS].isna().sum().sum() == 0, "Missing values in asset returns")

    require(len(labels) == EXPECTED_STATE_ROWS, f"Unexpected HMM-4 label rows: {len(labels)}")
    require(date_string(labels["date"], "min") == EXPECTED_STATE_START, "Unexpected HMM-4 label start date")
    require(date_string(labels["date"], "max") == EXPECTED_STATE_END, "Unexpected HMM-4 label end date")
    require(labels["date"].dt.dayofweek.eq(4).all(), "HMM-4 label dates are not all Fridays")
    require(all(col in labels.columns for col in ["date", "hmm4_state", "hmm4_state_id"]), "Missing HMM-4 state columns")
    state_counts = labels["hmm4_state"].value_counts().sort_index().to_dict()
    require(state_counts == EXPECTED_STATE_COUNTS, f"Unexpected HMM-4 state counts: {state_counts}")
    require(len(profiles) == 4, "HMM-4 profiles must contain four states")

    require(hashes["asset_returns_main_weekly"] == part1_manifest["input_hashes"]["asset_returns_main_weekly"], "Asset hash does not match Part 1")
    require(hashes["cleaning_report"] == part1_manifest["input_hashes"]["cleaning_report"], "Cleaning report hash does not match Part 1")
    require(hashes["asset_returns_main_weekly"] == part2_manifest["input_hashes"]["asset_returns_main_weekly"], "Asset hash does not match Part 2")
    require(hashes["cleaning_report"] == part2_manifest["input_hashes"]["cleaning_report"], "Cleaning report hash does not match Part 2")
    require(hashes["hmm4_state_labels"] == part2_manifest["input_hashes"]["hmm4_state_labels"], "HMM labels hash does not match Part 2")
    require(hashes["asset_returns_main_weekly"] == part3_manifest["input_hashes"]["asset_returns_main_weekly"], "Asset hash does not match Part 3")
    require(hashes["hmm4_state_labels"] == part3_manifest["input_hashes"]["hmm4_state_labels"], "HMM labels hash does not match Part 3")
    require(hashes["part2_manifest"] == part3_manifest["input_hashes"]["part2_manifest"], "Part 2 manifest hash does not match Part 3 lineage")

    panel = asset[["date"] + PORTFOLIO_ASSETS].merge(
        labels[["date", "hmm4_state", "hmm4_state_id"]],
        on="date",
        how="inner",
        validate="one_to_one",
    )
    require(len(panel) == EXPECTED_STATE_ROWS, f"Asset/HMM inner join produced {len(panel)} rows")
    require(date_string(panel["date"], "min") == EXPECTED_STATE_START, "Unexpected Part 4 sample start date")
    require(date_string(panel["date"], "max") == EXPECTED_STATE_END, "Unexpected Part 4 sample end date")

    baseline = inputs["part2_baseline_weights"]
    fixed = inputs["part2_fixed_weights"]
    require(set(baseline["portfolio_family"]) == {"all_weather", "erc"}, "Missing baseline portfolio families")
    require(set(fixed["portfolio_family"]) == {"all_weather", "erc"}, "Missing fixed BTC portfolio families")
    require(sorted(round(float(x), 2) for x in fixed["btc_weight"].unique()) == BTC_WEIGHT_GRID, "Unexpected fixed BTC weight grid")

    summary = {
        "status": "passed",
        "asset_sample": {"rows": int(len(asset)), "start": date_string(asset["date"], "min"), "end": date_string(asset["date"], "max")},
        "part4_sample": {"rows": int(len(panel)), "start": date_string(panel["date"], "min"), "end": date_string(panel["date"], "max")},
        "hmm4_state_counts": {k: int(v) for k, v in state_counts.items()},
        "btc_weight_grid": BTC_WEIGHT_GRID,
        "risk_budget_cap": RISK_BUDGET_CAP,
        "base_assets": BASE_ASSETS,
        "portfolio_assets": PORTFOLIO_ASSETS,
        "input_hashes": hashes,
        "upstream_runs": {
            "part1_run_id": part1_manifest.get("run_id"),
            "part2_run_id": part2_manifest.get("run_id"),
            "part3_run_id": part3_manifest.get("run_id"),
        },
    }
    write_json(dirs["results"] / "input_validation_summary.json", summary)
    logging.info("Input validation passed")
    return {"validation": summary, "analysis_panel": panel}


def build_base_weights(inputs: dict[str, Any]) -> dict[str, dict[str, float]]:
    baseline = inputs["part2_baseline_weights"]
    payload: dict[str, dict[str, float]] = {}
    for family, frame in baseline.groupby("portfolio_family"):
        weights = frame.set_index("asset")["weight"].astype(float).to_dict()
        require(set(weights) == set(BASE_ASSETS), f"Unexpected base assets for {family}: {weights.keys()}")
        require(abs(sum(weights.values()) - 1.0) < FLOAT_TOL, f"Base weights do not sum to one for {family}")
        payload[family] = weights
    return payload


def lookup_full_cap(inputs: dict[str, Any], family: str, btc_weight: float) -> dict[str, float]:
    frame = inputs["part2_btc_risk_budget"]
    row = frame[(frame["portfolio_family"] == family) & (np.isclose(frame["btc_weight"], btc_weight))]
    require(len(row) == 1, f"Missing full-sample risk-budget row for {family} {btc_weight}")
    item = row.iloc[0]
    return {
        "full_sample_btc_share_vol": float(item["btc_component_share_vol"]),
        "full_sample_btc_share_cvar": float(item["btc_component_share_cvar"]),
    }


def lookup_state_cap(inputs: dict[str, Any], family: str, state: str, btc_weight: float) -> dict[str, float]:
    frame = inputs["part2_state_btc_risk"]
    row = frame[
        (frame["portfolio_family"] == family)
        & (frame["hmm4_state"] == state)
        & (np.isclose(frame["btc_weight"], btc_weight))
    ]
    require(len(row) == 1, f"Missing state risk-budget row for {family} {state} {btc_weight}")
    item = row.iloc[0]
    return {
        "state_btc_share_vol": float(item["btc_component_share_vol"]),
        "state_btc_share_cvar": float(item["btc_component_share_cvar"]),
        "state_n_weeks": int(item["state_n_weeks"]),
        "state_sample_warning": "" if pd.isna(item["state_sample_warning"]) else str(item["state_sample_warning"]),
        "state_cvar_tail_scenario_count": int(item["cvar_tail_scenario_count"]),
    }


def candidate_cap_row(inputs: dict[str, Any], family: str, state: str, raw_weight: float, candidate: float) -> dict[str, Any]:
    full = lookup_full_cap(inputs, family, candidate)
    state_cap = lookup_state_cap(inputs, family, state, candidate)
    checks = {
        **full,
        **state_cap,
    }
    checks.update(
        {
            "full_sample_vol_cap_ok": full["full_sample_btc_share_vol"] <= RISK_BUDGET_CAP + FLOAT_TOL,
            "full_sample_cvar_cap_ok": full["full_sample_btc_share_cvar"] <= RISK_BUDGET_CAP + FLOAT_TOL,
            "state_vol_cap_ok": state_cap["state_btc_share_vol"] <= RISK_BUDGET_CAP + FLOAT_TOL,
            "state_cvar_cap_ok": state_cap["state_btc_share_cvar"] <= RISK_BUDGET_CAP + FLOAT_TOL,
            "candidate_le_raw_ok": candidate <= raw_weight + FLOAT_TOL,
        }
    )
    checks["all_caps_ok"] = bool(
        checks["full_sample_vol_cap_ok"]
        and checks["full_sample_cvar_cap_ok"]
        and checks["state_vol_cap_ok"]
        and checks["state_cvar_cap_ok"]
        and checks["candidate_le_raw_ok"]
    )
    return checks


def select_executed_weight(inputs: dict[str, Any], family: str, state: str, raw_weight: float) -> tuple[float, dict[str, Any]]:
    candidates = sorted([w for w in BTC_WEIGHT_GRID if w <= raw_weight + FLOAT_TOL], reverse=True)
    audit_by_candidate = []
    for candidate in candidates:
        checks = candidate_cap_row(inputs, family, state, raw_weight, candidate)
        audit_by_candidate.append({"candidate_btc_weight": candidate, **checks})
        if checks["all_caps_ok"]:
            return candidate, checks | {"candidate_audit": audit_by_candidate}
    fallback = candidate_cap_row(inputs, family, state, raw_weight, 0.0)
    return 0.0, fallback | {"candidate_audit": audit_by_candidate}


def build_rules(inputs: dict[str, Any], validation_payload: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    base_weights = build_base_weights(inputs)
    profiles = inputs["profiles"].copy()
    state_perf = inputs["part3_state_btc_performance"].copy()
    state_beta = inputs["part3_state_beta"].copy()
    state_warnings = inputs["part3_state_warnings"].copy()

    rule_rows = []
    cap_rows = []
    for raw_rule_name, spec in RAW_RULES.items():
        for constraint_stage in ["raw", "risk_budget_executed"]:
            rule_id = f"{raw_rule_name}_{'executed' if constraint_stage == 'risk_budget_executed' else 'raw'}"
            for family in sorted(base_weights):
                for state, raw_weight in spec["raw_weights"].items():
                    if constraint_stage == "raw":
                        selected_weight = raw_weight
                        cap_info = candidate_cap_row(inputs, family, state, raw_weight, selected_weight)
                        adjustment_reason = "raw_evidence_rule_not_cap_adjusted"
                    else:
                        selected_weight, cap_info = select_executed_weight(inputs, family, state, raw_weight)
                        if raw_weight == 0.0:
                            adjustment_reason = "raw_rule_zero_allocation"
                        elif abs(selected_weight - raw_weight) < FLOAT_TOL:
                            adjustment_reason = "raw_weight_within_full_and_state_risk_budget_caps"
                        else:
                            adjustment_reason = "reduced_to_highest_grid_weight_satisfying_full_and_active_state_caps"

                    rule_rows.append(
                        {
                            "rule_id": rule_id,
                            "raw_rule_name": raw_rule_name,
                            "rule_role": spec["role"],
                            "constraint_stage": constraint_stage,
                            "portfolio_family": family,
                            "hmm4_state": state,
                            "raw_btc_weight": raw_weight,
                            "selected_btc_weight": selected_weight,
                            "risk_budget_cap": RISK_BUDGET_CAP,
                            "funding_rule": "Scale non-BTC base weights by 1 - BTC weight.",
                            "adjustment_reason": adjustment_reason,
                            "rule_description": spec["description"],
                        }
                    )
                    cap_rows.append(
                        {
                            "rule_id": rule_id,
                            "raw_rule_name": raw_rule_name,
                            "rule_role": spec["role"],
                            "constraint_stage": constraint_stage,
                            "portfolio_family": family,
                            "hmm4_state": state,
                            "raw_btc_weight": raw_weight,
                            "selected_btc_weight": selected_weight,
                            "risk_budget_cap": RISK_BUDGET_CAP,
                            "adjustment_reason": adjustment_reason,
                            **{k: v for k, v in cap_info.items() if k != "candidate_audit"},
                        }
                    )

    rule_def = pd.DataFrame(rule_rows).sort_values(["rule_id", "portfolio_family", "hmm4_state"]).reset_index(drop=True)
    cap_audit = pd.DataFrame(cap_rows).sort_values(["rule_id", "portfolio_family", "hmm4_state"]).reset_index(drop=True)

    evidence = build_rule_evidence(profiles, state_perf, state_beta, state_warnings, rule_def)
    rule_def.to_csv(dirs["results"] / "allocation_rule_definition.csv", index=False)
    cap_audit.to_csv(dirs["results"] / "risk_budget_cap_audit.csv", index=False)
    evidence.to_csv(dirs["results"] / "rule_evidence_by_state.csv", index=False)

    rule_spec = {
        "raw_rules": RAW_RULES,
        "btc_weight_grid": BTC_WEIGHT_GRID,
        "risk_budget_cap": RISK_BUDGET_CAP,
        "risk_cap_scope": "full-sample and active-state BTC volatility/CVaR contribution shares",
        "funding_rule": "Released BTC allocation scales back to the non-BTC base portfolio; BIL is not used.",
    }
    save_pickle(dirs["models"] / "conditional_allocation_rule_specification.pkl", rule_spec)
    logging.info("Allocation rules completed")
    return {
        "base_weights": base_weights,
        "allocation_rule_definition": rule_def,
        "risk_budget_cap_audit": cap_audit,
        "rule_evidence_by_state": evidence,
        "rule_specification": rule_spec,
    }


def beta_value(state_beta: pd.DataFrame, state: str, predictor: str, col: str) -> float:
    row = state_beta[(state_beta["state"] == state) & (state_beta["predictor"] == predictor)]
    require(len(row) == 1, f"Missing Part 3 beta row for {state} {predictor}")
    return float(row.iloc[0][col])


def build_rule_evidence(
    profiles: pd.DataFrame,
    state_perf: pd.DataFrame,
    state_beta: pd.DataFrame,
    state_warnings: pd.DataFrame,
    rule_def: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    profile_idx = profiles.set_index("state")
    perf_idx = state_perf.set_index("state")
    warn_idx = state_warnings.set_index("state")
    main_exec = rule_def[rule_def["rule_id"] == "main_executed"]
    for _, row in main_exec.iterrows():
        state = row["hmm4_state"]
        prof = profile_idx.loc[state]
        perf = perf_idx.loc[state]
        warn = warn_idx.loc[state]
        rows.append(
            {
                "portfolio_family": row["portfolio_family"],
                "hmm4_state": state,
                "hmm4_state_id": int(prof["state_id"]),
                "candidate_profile": prof["candidate_profile"],
                "n_weeks": int(prof["n_weeks"]),
                "state_sample_warning": "" if pd.isna(warn["state_sample_warning"]) else str(warn["state_sample_warning"]),
                "macro_stress_composite_mean": float(prof["macro_stress_composite_mean"]),
                "macro_net_liquidity_chg_4w_z_mean": float(prof["macro_net_liquidity_chg_4w_z_mean"]),
                "macro_vix_z_mean": float(prof["macro_vix_z_mean"]),
                "macro_credit_spread_baa10y_z_mean": float(prof["macro_credit_spread_baa10y_z_mean"]),
                "macro_real_yield_10y_z_mean": float(prof["macro_real_yield_10y_z_mean"]),
                "btc_mean_weekly": float(perf["btc_mean_weekly"]),
                "btc_volatility_weekly": float(perf["btc_volatility_weekly"]),
                "btc_cvar_95_weekly": float(perf["btc_cvar_95_weekly"]),
                "btc_max_drawdown": float(perf["btc_max_drawdown"]),
                "btc_positive_week_share": float(perf["btc_positive_week_share"]),
                "btc_spy_beta": beta_value(state_beta, state, "ret_spy", "beta"),
                "btc_spy_beta_p": beta_value(state_beta, state, "ret_spy", "p_beta"),
                "btc_real_yield_beta": beta_value(state_beta, state, "macro_real_yield_10y_z", "beta"),
                "btc_real_yield_beta_p": beta_value(state_beta, state, "macro_real_yield_10y_z", "p_beta"),
                "main_raw_btc_weight": float(row["raw_btc_weight"]),
                "main_executed_btc_weight": float(row["selected_btc_weight"]),
                "evidence_note": evidence_note_for_state(state),
            }
        )
    return pd.DataFrame(rows).sort_values(["portfolio_family", "hmm4_state"]).reset_index(drop=True)


def evidence_note_for_state(state: str) -> str:
    notes = {
        "state_0": "Lower-stress state with positive BTC diagnostics; raw 3% is cap-adjusted when needed.",
        "state_1": "Moderate-stress state with weaker BTC return evidence; limited to 1%.",
        "state_2": "Small elevated-stress state; excluded from the main rule despite high historical BTC mean.",
        "state_3": "Highest-stress state with negative BTC mean and deeper drawdown; BTC is set to zero.",
    }
    return notes[state]


def state_warning(state: str, state_counts: dict[str, int]) -> str:
    return "small_state_sample" if state_counts[state] < SMALL_STATE_THRESHOLD_WEEKS else ""


def build_weekly_weights_and_returns(
    validation_payload: dict[str, Any],
    rules_payload: dict[str, Any],
    dirs: dict[str, Path],
) -> dict[str, Any]:
    panel = validation_payload["analysis_panel"].copy()
    rule_def = rules_payload["allocation_rule_definition"].copy()
    base_weights = rules_payload["base_weights"]
    state_counts = panel["hmm4_state"].value_counts().sort_index().to_dict()

    weight_rows = []
    return_rows = []
    component_rows = []
    activation_rows = []

    for (rule_id, family), frame in rule_def.groupby(["rule_id", "portfolio_family"], sort=True):
        state_weight_map = frame.set_index("hmm4_state")["selected_btc_weight"].astype(float).to_dict()
        active_states = [state for state, weight in state_weight_map.items() if weight > 0]
        active_weeks = int(panel["hmm4_state"].isin(active_states).sum())
        activation_rows.append(
            {
                "rule_id": rule_id,
                "portfolio_family": family,
                "active_states": ",".join(active_states),
                "active_weeks": active_weeks,
                "active_week_share": float(active_weeks / len(panel)),
                "average_target_btc_weight": float(panel["hmm4_state"].map(state_weight_map).mean()),
                "max_target_btc_weight": float(max(state_weight_map.values())),
                "state_0_weight": state_weight_map["state_0"],
                "state_1_weight": state_weight_map["state_1"],
                "state_2_weight": state_weight_map["state_2"],
                "state_3_weight": state_weight_map["state_3"],
            }
        )
        for _, obs in panel.sort_values("date").iterrows():
            state = obs["hmm4_state"]
            btc_weight = float(state_weight_map[state])
            weights = {"ret_btc": btc_weight}
            for asset, base_weight in base_weights[family].items():
                weights[asset] = float(base_weight * (1.0 - btc_weight))
            weight_sum = float(sum(weights.values()))
            require(abs(weight_sum - 1.0) < FLOAT_TOL, f"Weight sum failed for {rule_id} {family} {obs['date']}")

            component_returns = {asset: weights[asset] * float(obs[asset]) for asset in PORTFOLIO_ASSETS}
            portfolio_return = float(sum(component_returns.values()))
            return_rows.append(
                {
                    "date": obs["date"],
                    "rule_id": rule_id,
                    "raw_rule_name": frame["raw_rule_name"].iloc[0],
                    "rule_role": frame["rule_role"].iloc[0],
                    "constraint_stage": frame["constraint_stage"].iloc[0],
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
                        "rule_id": rule_id,
                        "raw_rule_name": frame["raw_rule_name"].iloc[0],
                        "rule_role": frame["rule_role"].iloc[0],
                        "constraint_stage": frame["constraint_stage"].iloc[0],
                        "portfolio_family": family,
                        "hmm4_state": state,
                        "hmm4_state_id": int(obs["hmm4_state_id"]),
                        "asset": asset,
                        "weight": weight,
                        "btc_weight": btc_weight,
                        "state_sample_warning": state_warning(state, state_counts),
                    }
                )
                component_rows.append(
                    {
                        "date": obs["date"],
                        "rule_id": rule_id,
                        "portfolio_family": family,
                        "hmm4_state": state,
                        "asset": asset,
                        "weight": weight,
                        "asset_return": float(obs[asset]),
                        "component_return": component_returns[asset],
                        "portfolio_return": portfolio_return,
                    }
                )

    weekly_weights = pd.DataFrame(weight_rows)
    returns = pd.DataFrame(return_rows)
    components = pd.DataFrame(component_rows)
    activation = pd.DataFrame(activation_rows).sort_values(["rule_id", "portfolio_family"]).reset_index(drop=True)

    weekly_weights.to_csv(dirs["results"] / "weekly_conditional_weights.csv", index=False)
    returns.to_csv(dirs["results"] / "conditional_portfolio_return_series.csv", index=False)
    activation.to_csv(dirs["results"] / "state_activation_summary.csv", index=False)
    logging.info("Weekly conditional weights and returns completed")
    return {
        "weekly_conditional_weights": weekly_weights,
        "conditional_portfolio_return_series": returns,
        "component_return_series": components,
        "state_activation_summary": activation,
    }


def compute_vol_contributions(component_frame: pd.DataFrame) -> pd.DataFrame:
    pivot = component_frame.pivot_table(index="date", columns="asset", values="component_return", aggfunc="first")
    pivot = pivot[PORTFOLIO_ASSETS]
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
                "portfolio_volatility": vol,
                "component_contribution_vol": contribution,
                "component_share_vol": contribution / vol,
            }
        )
    share_sum = float(sum(row["component_share_vol"] for row in rows))
    for row in rows:
        row["share_sum_check"] = share_sum
    return pd.DataFrame(rows)


def compute_cvar_contributions(component_frame: pd.DataFrame) -> pd.DataFrame:
    pivot = component_frame.pivot_table(index="date", columns="asset", values="component_return", aggfunc="first")
    pivot = pivot[PORTFOLIO_ASSETS]
    portfolio = pivot.sum(axis=1)
    var = float(portfolio.quantile(TAIL_ALPHA))
    tail_mask = portfolio <= var
    tail = pivot.loc[tail_mask]
    portfolio_cvar_loss = float((-portfolio.loc[tail_mask]).mean())
    require(portfolio_cvar_loss != 0, "Portfolio CVaR loss is zero")
    rows = []
    for asset in PORTFOLIO_ASSETS:
        contribution = float((-tail[asset]).mean())
        rows.append(
            {
                "asset": asset,
                "portfolio_var_95": var,
                "portfolio_cvar_loss": portfolio_cvar_loss,
                "tail_scenario_count": int(tail_mask.sum()),
                "component_contribution_cvar_loss": contribution,
                "component_share_cvar": contribution / portfolio_cvar_loss,
            }
        )
    share_sum = float(sum(row["component_share_cvar"] for row in rows))
    for row in rows:
        row["share_sum_check"] = share_sum
    return pd.DataFrame(rows)


def compute_portfolio_diagnostics(
    validation_payload: dict[str, Any],
    rules_payload: dict[str, Any],
    weekly_payload: dict[str, Any],
    inputs: dict[str, Any],
    dirs: dict[str, Path],
) -> dict[str, Any]:
    returns = weekly_payload["conditional_portfolio_return_series"].copy()
    components = weekly_payload["component_return_series"].copy()
    panel = validation_payload["analysis_panel"].copy()
    state_counts = panel["hmm4_state"].value_counts().sort_index().to_dict()

    perf_rows = []
    vol_rows = []
    cvar_rows = []
    state_perf_rows = []
    state_rc_rows = []

    for (rule_id, family), frame in returns.groupby(["rule_id", "portfolio_family"], sort=True):
        metrics = performance_metrics(frame["portfolio_return"])
        perf_rows.append(
            {
                "rule_id": rule_id,
                "raw_rule_name": frame["raw_rule_name"].iloc[0],
                "rule_role": frame["rule_role"].iloc[0],
                "constraint_stage": frame["constraint_stage"].iloc[0],
                "portfolio_family": family,
                "average_btc_weight": float(frame["btc_weight"].mean()),
                "max_btc_weight": float(frame["btc_weight"].max()),
                "active_week_share": float((frame["btc_weight"] > 0).mean()),
                **metrics,
            }
        )
        comp = components[(components["rule_id"] == rule_id) & (components["portfolio_family"] == family)]
        vol = compute_vol_contributions(comp)
        cvar = compute_cvar_contributions(comp)
        for _, row in vol.iterrows():
            vol_rows.append({"rule_id": rule_id, "portfolio_family": family, **row.to_dict()})
        for _, row in cvar.iterrows():
            cvar_rows.append({"rule_id": rule_id, "portfolio_family": family, **row.to_dict()})

        for state, state_frame in frame.groupby("hmm4_state", sort=True):
            state_metrics = performance_metrics(state_frame["portfolio_return"])
            state_perf_rows.append(
                {
                    "rule_id": rule_id,
                    "portfolio_family": family,
                    "hmm4_state": state,
                    "hmm4_state_id": int(state_frame["hmm4_state_id"].iloc[0]),
                    "state_sample_warning": state_warning(state, state_counts),
                    "btc_weight": float(state_frame["btc_weight"].iloc[0]),
                    **state_metrics,
                }
            )
            state_comp = comp[comp["hmm4_state"] == state]
            state_vol = compute_vol_contributions(state_comp)
            state_cvar = compute_cvar_contributions(state_comp)
            for _, row in state_vol.merge(state_cvar, on="asset", suffixes=("_vol", "_cvar")).iterrows():
                state_rc_rows.append(
                    {
                        "rule_id": rule_id,
                        "portfolio_family": family,
                        "hmm4_state": state,
                        "hmm4_state_id": int(state_frame["hmm4_state_id"].iloc[0]),
                        "state_n_weeks": int(len(state_frame)),
                        "state_sample_warning": state_warning(state, state_counts),
                        "btc_weight": float(state_frame["btc_weight"].iloc[0]),
                        **row.to_dict(),
                    }
                )

    performance = pd.DataFrame(perf_rows).sort_values(["rule_id", "portfolio_family"]).reset_index(drop=True)
    vol_contrib = pd.DataFrame(vol_rows).sort_values(["rule_id", "portfolio_family", "asset"]).reset_index(drop=True)
    cvar_contrib = pd.DataFrame(cvar_rows).sort_values(["rule_id", "portfolio_family", "asset"]).reset_index(drop=True)
    state_perf = pd.DataFrame(state_perf_rows).sort_values(["rule_id", "portfolio_family", "hmm4_state_id"]).reset_index(drop=True)
    state_rc = pd.DataFrame(state_rc_rows).sort_values(["rule_id", "portfolio_family", "hmm4_state_id", "asset"]).reset_index(drop=True)

    sensitivity = build_sensitivity_summary(performance, vol_contrib, cvar_contrib, state_perf, state_rc)
    performance.to_csv(dirs["results"] / "conditional_portfolio_performance_summary.csv", index=False)
    vol_contrib.to_csv(dirs["results"] / "conditional_risk_contributions_vol.csv", index=False)
    cvar_contrib.to_csv(dirs["results"] / "conditional_risk_contributions_cvar.csv", index=False)
    state_perf.to_csv(dirs["results"] / "state_conditioned_conditional_portfolio_summary.csv", index=False)
    state_rc.to_csv(dirs["results"] / "state_conditioned_conditional_risk_contributions.csv", index=False)
    sensitivity.to_csv(dirs["results"] / "sensitivity_state2_low_allocation_summary.csv", index=False)

    plot_rule_timeline(returns, dirs["figures"] / "conditional_rule_timeline.png")
    plot_conditional_drawdowns(returns, dirs["figures"] / "conditional_portfolio_drawdowns.png")
    plot_conditional_vs_fixed(performance, inputs["part2_portfolio_performance"], dirs["figures"] / "conditional_vs_fixed_btc_summary.png")
    plot_risk_budget_cap_audit(rules_payload["risk_budget_cap_audit"], dirs["figures"] / "risk_budget_cap_audit.png")
    logging.info("Conditional portfolio diagnostics completed")
    return {
        "performance": performance,
        "vol_contrib": vol_contrib,
        "cvar_contrib": cvar_contrib,
        "state_performance": state_perf,
        "state_risk_contrib": state_rc,
        "sensitivity": sensitivity,
    }


def build_sensitivity_summary(
    performance: pd.DataFrame,
    vol_contrib: pd.DataFrame,
    cvar_contrib: pd.DataFrame,
    state_perf: pd.DataFrame,
    state_rc: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for family in sorted(performance["portfolio_family"].unique()):
        main = performance[(performance["rule_id"] == "main_executed") & (performance["portfolio_family"] == family)].iloc[0]
        sens = performance[
            (performance["rule_id"] == "sensitivity_state2_low_executed")
            & (performance["portfolio_family"] == family)
        ].iloc[0]
        main_btc_vol = vol_contrib[
            (vol_contrib["rule_id"] == "main_executed")
            & (vol_contrib["portfolio_family"] == family)
            & (vol_contrib["asset"] == "ret_btc")
        ].iloc[0]
        sens_btc_vol = vol_contrib[
            (vol_contrib["rule_id"] == "sensitivity_state2_low_executed")
            & (vol_contrib["portfolio_family"] == family)
            & (vol_contrib["asset"] == "ret_btc")
        ].iloc[0]
        main_btc_cvar = cvar_contrib[
            (cvar_contrib["rule_id"] == "main_executed")
            & (cvar_contrib["portfolio_family"] == family)
            & (cvar_contrib["asset"] == "ret_btc")
        ].iloc[0]
        sens_btc_cvar = cvar_contrib[
            (cvar_contrib["rule_id"] == "sensitivity_state2_low_executed")
            & (cvar_contrib["portfolio_family"] == family)
            & (cvar_contrib["asset"] == "ret_btc")
        ].iloc[0]
        rows.append(
            {
                "portfolio_family": family,
                "main_rule_id": "main_executed",
                "sensitivity_rule_id": "sensitivity_state2_low_executed",
                "main_average_btc_weight": float(main["average_btc_weight"]),
                "sensitivity_average_btc_weight": float(sens["average_btc_weight"]),
                "delta_average_btc_weight": float(sens["average_btc_weight"] - main["average_btc_weight"]),
                "main_annualized_volatility": float(main["annualized_volatility"]),
                "sensitivity_annualized_volatility": float(sens["annualized_volatility"]),
                "delta_annualized_volatility": float(sens["annualized_volatility"] - main["annualized_volatility"]),
                "main_cvar_95_weekly": float(main["cvar_95_weekly"]),
                "sensitivity_cvar_95_weekly": float(sens["cvar_95_weekly"]),
                "delta_cvar_95_weekly": float(sens["cvar_95_weekly"] - main["cvar_95_weekly"]),
                "main_max_drawdown": float(main["max_drawdown"]),
                "sensitivity_max_drawdown": float(sens["max_drawdown"]),
                "delta_max_drawdown": float(sens["max_drawdown"] - main["max_drawdown"]),
                "main_btc_vol_share": float(main_btc_vol["component_share_vol"]),
                "sensitivity_btc_vol_share": float(sens_btc_vol["component_share_vol"]),
                "main_btc_cvar_share": float(main_btc_cvar["component_share_cvar"]),
                "sensitivity_btc_cvar_share": float(sens_btc_cvar["component_share_cvar"]),
                "interpretation_note": "Sensitivity only; state_2 remains a 30-week small sample.",
            }
        )
    return pd.DataFrame(rows)


def write_explainability_artifacts(
    inputs: dict[str, Any],
    validation_payload: dict[str, Any],
    rules_payload: dict[str, Any],
    diagnostics_payload: dict[str, Any],
    dirs: dict[str, Path],
) -> dict[str, Any]:
    paths = inputs["paths"]
    hashes = inputs["input_hashes"]
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
                "sha256": hashes[name],
                "rows": rows,
                "start_date": start,
                "end_date": end,
                "part4_usage": lineage_usage(name),
            }
        )
    lineage = pd.DataFrame(lineage_rows)
    lineage.to_csv(dirs["results"] / "data_lineage.csv", index=False)

    audit_md = f"""# Part 4 Methodology Audit

## Purpose
Part 4 evaluates conditional BTC allocation rules as risk-budgeted satellite diagnostics. It does not estimate a trading strategy, transaction costs, turnover, real-time state signal, or final thesis conclusion.

## Inputs
- Cleaned weekly asset returns from `data_2026/cleaned`.
- Part 1 HMM-4 full-sample ex-post state labels.
- Part 2 fixed BTC risk-budget diagnostics and base portfolio weights.
- Part 3 BTC state-dependence and conditional beta diagnostics.

The effective sample is {EXPECTED_STATE_ROWS} weekly observations from {EXPECTED_STATE_START} to {EXPECTED_STATE_END}.

## Allocation Rules
The main raw rule is `state_0=3%`, `state_1=1%`, `state_2=0%`, and `state_3=0%`. The executed rule applies a {RISK_BUDGET_CAP:.0%} BTC risk-budget cap to both full-sample and active-state BTC volatility and CVaR contribution shares. Released BTC allocation is returned pro rata to the non-BTC base portfolio.

## Risk Contributions
Conditional rules use time-varying target weights. Component returns are computed weekly as `weight_i,t * asset_return_i,t`. Volatility contribution is `cov(component_i, portfolio) / portfolio_volatility`; CVaR contribution is the empirical mean component loss in portfolio left-tail weeks.

## Discussion Boundaries
- HMM-4 states are ex-post descriptive groups and not real-time signals.
- The main rule excludes `state_2` because it has only 30 weeks despite strong historical BTC returns.
- BIL cash parking, transaction costs, rebalancing frequency, turnover, and implementability are outside Part 4.
- The sensitivity state_2 low-allocation rule is not the main conclusion rule.
"""
    (dirs["results"] / "methodology_audit.md").write_text(audit_md, encoding="utf-8")

    assumptions = {
        "status": "documented",
        "state_labels": {
            "source": "Part 1 HMM-4 state labels",
            "estimation_type": "full-sample ex-post descriptive regime identification",
            "not_a_real_time_signal": True,
        },
        "rules": {
            "main_raw": RAW_RULES["main"]["raw_weights"],
            "sensitivity_state2_low_raw": RAW_RULES["sensitivity_state2_low"]["raw_weights"],
            "btc_weight_grid": BTC_WEIGHT_GRID,
            "risk_budget_cap": RISK_BUDGET_CAP,
            "risk_cap_scope": "full-sample and active-state volatility/CVaR BTC contribution shares",
            "funding_rule": "Scale non-BTC base weights by 1 - BTC weight; no BIL parking.",
        },
        "excluded": [
            "transaction costs",
            "turnover",
            "monthly or quarterly rebalancing",
            "cash parking in BIL",
            "real-time state-signal construction",
            "final thesis conclusion",
        ],
    }
    write_json(dirs["results"] / "model_assumption_audit.json", assumptions)
    logging.info("Explainability artifacts completed")
    return {
        "data_lineage": lineage,
        "methodology_audit": audit_md,
        "model_assumption_audit": assumptions,
    }


def lineage_usage(name: str) -> str:
    usage = {
        "asset_returns_main_weekly": "Primary weekly return panel for conditional target-weight returns.",
        "cleaning_report": "Input lineage and column mapping.",
        "part1_manifest": "Part 1 input lineage and HMM model status.",
        "part1_validation_summary": "Part 1 validation status.",
        "hmm4_state_labels": "Required conditional allocation state labels.",
        "hmm4_state_profiles": "Rule evidence by macro state.",
        "part2_manifest": "Part 2 lineage.",
        "part2_input_validation_summary": "Part 2 validation status.",
        "part2_output_validation_summary": "Part 2 output validation status.",
        "part2_btc_risk_budget_summary": "Full-sample BTC risk-budget cap reference.",
        "part2_state_btc_risk_contributions": "Active-state BTC risk-budget cap reference.",
        "part2_fixed_btc_portfolio_weights": "Fixed-weight grid and proportional scaling audit.",
        "part2_baseline_portfolio_weights": "No-BTC All Weather/ERC base weights.",
        "part2_portfolio_performance_summary": "Context for conditional-vs-fixed figure.",
        "part3_manifest": "Part 3 lineage.",
        "part3_input_validation_summary": "Part 3 validation status.",
        "part3_output_validation_summary": "Part 3 output validation status.",
        "part3_state_btc_performance": "Rule evidence by BTC state performance.",
        "part3_state_beta_diagnostics": "Rule evidence by conditional beta diagnostics.",
        "part3_state_sample_warnings": "Small-sample state warnings.",
    }
    return usage.get(name, "Supporting audit input.")


def validate_outputs(
    validation_payload: dict[str, Any],
    rules_payload: dict[str, Any],
    weekly_payload: dict[str, Any],
    diagnostics_payload: dict[str, Any],
    dirs: dict[str, Path],
) -> dict[str, Any]:
    result_checks = []
    for name in REQUIRED_RESULTS:
        if name == "output_validation_summary.json":
            continue
        path = dirs["results"] / name
        result_checks.append({"file": name, "exists": path.exists(), "nonempty": path.exists() and path.stat().st_size > 0})
    figure_checks = []
    for name in REQUIRED_FIGURES:
        path = dirs["figures"] / name
        figure_checks.append({"file": name, "exists": path.exists(), "nonempty": path.exists() and path.stat().st_size > 0})

    rule_def = rules_payload["allocation_rule_definition"]
    cap_audit = rules_payload["risk_budget_cap_audit"]
    weights = weekly_payload["weekly_conditional_weights"]
    returns = weekly_payload["conditional_portfolio_return_series"]
    vol = diagnostics_payload["vol_contrib"]
    cvar = diagnostics_payload["cvar_contrib"]
    state_rc = diagnostics_payload["state_risk_contrib"]

    weight_grid_ok = bool(rule_def["selected_btc_weight"].round(10).isin(BTC_WEIGHT_GRID).all())
    executed_le_raw_ok = bool(
        rule_def[rule_def["constraint_stage"] == "risk_budget_executed"].apply(
            lambda row: row["selected_btc_weight"] <= row["raw_btc_weight"] + FLOAT_TOL, axis=1
        ).all()
    )
    main_exec_cap_audit = cap_audit[(cap_audit["rule_id"] == "main_executed") & (cap_audit["selected_btc_weight"] > 0)]
    cap_reference_ok = bool(
        (main_exec_cap_audit["full_sample_btc_share_vol"] <= RISK_BUDGET_CAP + FLOAT_TOL).all()
        and (main_exec_cap_audit["full_sample_btc_share_cvar"] <= RISK_BUDGET_CAP + FLOAT_TOL).all()
        and (main_exec_cap_audit["state_btc_share_vol"] <= RISK_BUDGET_CAP + FLOAT_TOL).all()
        and (main_exec_cap_audit["state_btc_share_cvar"] <= RISK_BUDGET_CAP + FLOAT_TOL).all()
    )

    weight_sum = weights.groupby(["rule_id", "portfolio_family", "date"])["weight"].sum()
    weight_sum_ok = bool((weight_sum - 1.0).abs().max() < FLOAT_TOL)
    nonnegative_weights_ok = bool(weights["weight"].min() >= -FLOAT_TOL)
    scaling_ok = check_weight_scaling(weights, rules_payload["base_weights"])
    return_counts = returns.groupby(["rule_id", "portfolio_family"]).size()
    return_counts_ok = bool(return_counts.eq(EXPECTED_STATE_ROWS).all())

    vol_sum_ok = bool((vol.groupby(["rule_id", "portfolio_family"])["component_share_vol"].sum() - 1.0).abs().max() < 1e-9)
    cvar_sum_ok = bool((cvar.groupby(["rule_id", "portfolio_family"])["component_share_cvar"].sum() - 1.0).abs().max() < 1e-9)
    state_vol_sum_ok = bool((state_rc.groupby(["rule_id", "portfolio_family", "hmm4_state"])["component_share_vol"].sum() - 1.0).abs().max() < 1e-9)
    state_cvar_sum_ok = bool((state_rc.groupby(["rule_id", "portfolio_family", "hmm4_state"])["component_share_cvar"].sum() - 1.0).abs().max() < 1e-9)

    main_full_btc_vol = vol[(vol["rule_id"] == "main_executed") & (vol["asset"] == "ret_btc")]
    main_full_btc_cvar = cvar[(cvar["rule_id"] == "main_executed") & (cvar["asset"] == "ret_btc")]
    main_full_dynamic_cap_ok = bool(
        (main_full_btc_vol["component_share_vol"] <= RISK_BUDGET_CAP + FLOAT_TOL).all()
        and (main_full_btc_cvar["component_share_cvar"] <= RISK_BUDGET_CAP + FLOAT_TOL).all()
    )
    active_state_btc = state_rc[
        (state_rc["rule_id"] == "main_executed")
        & (state_rc["asset"] == "ret_btc")
        & (state_rc["btc_weight"] > 0)
    ]
    main_active_state_dynamic_cap_ok = bool(
        (active_state_btc["component_share_vol"] <= RISK_BUDGET_CAP + FLOAT_TOL).all()
        and (active_state_btc["component_share_cvar"] <= RISK_BUDGET_CAP + FLOAT_TOL).all()
    )

    state2_present = bool(
        "state_2"
        in rules_payload["rule_evidence_by_state"].loc[
            rules_payload["rule_evidence_by_state"]["state_sample_warning"].fillna("") != "",
            "hmm4_state",
        ].tolist()
    )
    sensitivity_present = bool(
        {"sensitivity_state2_low_raw", "sensitivity_state2_low_executed"}.issubset(set(rule_def["rule_id"]))
    )
    files_ok = all(item["exists"] and item["nonempty"] for item in result_checks + figure_checks)
    checks = {
        "required_files_ok": files_ok,
        "weight_grid_ok": weight_grid_ok,
        "executed_le_raw_ok": executed_le_raw_ok,
        "cap_reference_ok": cap_reference_ok,
        "weight_sum_ok": weight_sum_ok,
        "nonnegative_weights_ok": nonnegative_weights_ok,
        "nonbtc_scaling_ok": scaling_ok,
        "return_counts_ok": return_counts_ok,
        "full_vol_contribution_sum_ok": vol_sum_ok,
        "full_cvar_contribution_sum_ok": cvar_sum_ok,
        "state_vol_contribution_sum_ok": state_vol_sum_ok,
        "state_cvar_contribution_sum_ok": state_cvar_sum_ok,
        "main_full_dynamic_cap_ok": main_full_dynamic_cap_ok,
        "main_active_state_dynamic_cap_ok": main_active_state_dynamic_cap_ok,
        "state2_small_sample_warning_present": state2_present,
        "sensitivity_rule_present": sensitivity_present,
    }
    status = "passed" if all(checks.values()) else "failed"
    summary = {
        "status": status,
        "checks": checks,
        "required_result_checks": result_checks,
        "required_figure_checks": figure_checks,
        "rule_count": int(len(rule_def)),
        "weekly_weight_rows": int(len(weights)),
        "conditional_return_rows": int(len(returns)),
        "return_counts_by_rule_family": {str(k): int(v) for k, v in return_counts.to_dict().items()},
        "risk_budget_cap": RISK_BUDGET_CAP,
        "main_executed_dynamic_btc_full_sample_vol_shares": main_full_btc_vol[["portfolio_family", "component_share_vol"]].to_dict(orient="records"),
        "main_executed_dynamic_btc_full_sample_cvar_shares": main_full_btc_cvar[["portfolio_family", "component_share_cvar"]].to_dict(orient="records"),
        "main_executed_active_state_dynamic_btc_shares": active_state_btc[
            ["portfolio_family", "hmm4_state", "btc_weight", "component_share_vol", "component_share_cvar"]
        ].to_dict(orient="records"),
    }
    require(status == "passed", f"Output validation failed: {summary}")
    write_json(dirs["results"] / "output_validation_summary.json", summary)
    logging.info("Output validation completed")
    return summary


def check_weight_scaling(weights: pd.DataFrame, base_weights: dict[str, dict[str, float]]) -> bool:
    for (rule_id, family, date), frame in weights.groupby(["rule_id", "portfolio_family", "date"]):
        btc_weight = float(frame.loc[frame["asset"] == "ret_btc", "weight"].iloc[0])
        for asset, base_weight in base_weights[family].items():
            actual = float(frame.loc[frame["asset"] == asset, "weight"].iloc[0])
            expected = base_weight * (1.0 - btc_weight)
            if abs(actual - expected) > FLOAT_TOL:
                return False
    return True


def write_manifest(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    inputs: dict[str, Any],
    validation_payload: dict[str, Any],
    rules_payload: dict[str, Any],
    output_validation: dict[str, Any],
) -> dict[str, Any]:
    panel = validation_payload["analysis_panel"]
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "run_id": dirs["root"].name,
        "objective": "Conditional BTC allocation rule and risk-budgeted satellite diagnostics",
        "input_dir": str(args.input_dir),
        "part1_run_dir": str(args.part1_run_dir),
        "part2_run_dir": str(args.part2_run_dir),
        "part3_run_dir": str(args.part3_run_dir),
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
            "btc_weight_grid": BTC_WEIGHT_GRID,
            "risk_budget_cap": RISK_BUDGET_CAP,
            "risk_cap_scope": "full-sample and active-state volatility/CVaR BTC contribution shares",
            "tail_alpha": TAIL_ALPHA,
            "trading_weeks_per_year": TRADING_WEEKS_PER_YEAR,
            "funding_rule": "Released BTC weight returns pro rata to the non-BTC base portfolio.",
            "raw_rules": RAW_RULES,
        },
        "lineage": {
            "part1_run_id": inputs["part1_manifest"].get("run_id"),
            "part2_run_id": inputs["part2_manifest"].get("run_id"),
            "part3_run_id": inputs["part3_manifest"].get("run_id"),
        },
        "rule_specification": rules_payload["rule_specification"],
        "output_validation": output_validation,
        "outputs": {
            "checkpoints": str(dirs["checkpoints"]),
            "results": str(dirs["results"]),
            "figures": str(dirs["figures"]),
            "models": str(dirs["models"]),
            "logs": str(dirs["logs"]),
        },
        "scope_notes": [
            "Part 4 is an ex-post descriptive conditional target-weight diagnostic.",
            "No real-time signal, transaction cost, turnover, or implementability conclusion.",
            "BIL is not used as a cash parking asset in Part 4.",
            "Sensitivity state_2 allocation is not the main rule.",
        ],
    }
    write_json(dirs["root"] / "run_manifest.json", manifest)
    return manifest


def plot_rule_timeline(returns: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    plot_ids = ["main_raw", "main_executed", "sensitivity_state2_low_executed"]
    styles = {"all_weather": "-", "erc": "--"}
    for (rule_id, family), frame in returns[returns["rule_id"].isin(plot_ids)].groupby(["rule_id", "portfolio_family"]):
        label = f"{family} {rule_id}"
        ax.step(frame["date"], frame["btc_weight"], where="post", linestyle=styles[family], linewidth=1.6, label=label)
    ax.set_title("Conditional BTC Target Weight Timeline")
    ax.set_ylabel("BTC target weight")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_conditional_drawdowns(returns: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    for (rule_id, family), frame in returns.groupby(["rule_id", "portfolio_family"]):
        if rule_id.endswith("_raw") and rule_id != "main_raw":
            continue
        dd = drawdown_series(frame.sort_values("date")["portfolio_return"]).to_numpy()
        ax.plot(frame.sort_values("date")["date"], dd, linewidth=1.4, label=f"{family} {rule_id}")
    ax.set_title("Conditional Portfolio Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_conditional_vs_fixed(performance: pd.DataFrame, fixed_perf: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, family in zip(axes, ["all_weather", "erc"]):
        fixed = fixed_perf[fixed_perf["portfolio_family"] == family].copy()
        fixed["label"] = fixed["btc_weight"].map(lambda x: f"fixed {int(round(x * 100))}%")
        cond = performance[
            (performance["portfolio_family"] == family)
            & (performance["rule_id"].isin(["main_executed", "sensitivity_state2_low_executed"]))
        ].copy()
        cond["label"] = cond["rule_id"]
        labels = fixed["label"].tolist() + cond["label"].tolist()
        vols = fixed["annualized_volatility"].tolist() + cond["annualized_volatility"].tolist()
        ax.bar(labels, vols, color=["#8aa6c8"] * len(fixed) + ["#2f7d59", "#d99b2b"])
        ax.set_title(f"{family}: Conditional vs Fixed BTC Volatility")
        ax.set_ylabel("Annualized volatility")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_risk_budget_cap_audit(cap_audit: pd.DataFrame, output_path: Path) -> None:
    frame = cap_audit[
        (cap_audit["rule_id"].isin(["main_raw", "main_executed"]))
        & (cap_audit["selected_btc_weight"] > 0)
    ].copy()
    frame["label"] = frame["portfolio_family"] + " " + frame["rule_id"] + " " + frame["hmm4_state"]
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(13, 5))
    width = 0.38
    ax.bar(x - width / 2, frame["state_btc_share_vol"], width=width, label="Active-state vol share", color="#3b7ddd")
    ax.bar(x + width / 2, frame["state_btc_share_cvar"], width=width, label="Active-state CVaR share", color="#d98b3a")
    ax.axhline(RISK_BUDGET_CAP, color="red", linestyle="--", linewidth=1.2, label="10% cap")
    ax.set_xticks(x, frame["label"], rotation=45, ha="right")
    ax.set_title("Risk Budget Cap Audit for Main Rule")
    ax.set_ylabel("BTC contribution share")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
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
    logging.info("Starting Part 4 run: %s", run_id)

    inputs = read_inputs(args)
    enforce_resume_input_hashes(args, dirs, inputs["input_hashes"])
    validation_payload = load_or_run(
        dirs,
        "01_input_validation",
        args.resume,
        lambda: validate_inputs(inputs, dirs),
    )
    rules_payload = load_or_run(
        dirs,
        "02_allocation_rules",
        args.resume,
        lambda: build_rules(inputs, validation_payload, dirs),
    )
    weekly_payload = load_or_run(
        dirs,
        "03_weekly_weights_and_returns",
        args.resume,
        lambda: build_weekly_weights_and_returns(validation_payload, rules_payload, dirs),
    )
    diagnostics_payload = load_or_run(
        dirs,
        "04_portfolio_diagnostics",
        args.resume,
        lambda: compute_portfolio_diagnostics(validation_payload, rules_payload, weekly_payload, inputs, dirs),
    )
    explainability = load_or_run(
        dirs,
        "05_explainability_artifacts",
        args.resume,
        lambda: write_explainability_artifacts(inputs, validation_payload, rules_payload, diagnostics_payload, dirs),
    )
    output_validation = load_or_run(
        dirs,
        "06_output_validation",
        args.resume,
        lambda: validate_outputs(validation_payload, rules_payload, weekly_payload, diagnostics_payload, dirs),
    )
    manifest = write_manifest(args, dirs, inputs, validation_payload, rules_payload, output_validation)
    _ = explainability, manifest
    logging.info("Completed Part 4 run: %s", run_id)
    logging.info("Results directory: %s", dirs["results"])


if __name__ == "__main__":
    main()
