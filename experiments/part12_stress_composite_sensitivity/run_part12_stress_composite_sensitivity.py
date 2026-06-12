#!/usr/bin/env python3
"""Part 12 experiment runner: stress composite and state-ordering sensitivity.

This runner audits alternative orderings of the frozen Part 1 HMM-4 states. It
does not re-estimate PCA or HMM models. The only object that changes is the
mapping from original HMM-4 states to stress-ranked state labels.
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

BASE_ASSETS = ["ret_spy", "ret_tlt", "ret_ief", "ret_gld", "ret_dbc"]
PORTFOLIO_ASSETS = ["ret_btc"] + BASE_ASSETS
PORTFOLIO_FAMILIES = ["all_weather", "erc"]
HMM4_STATES = [f"state_{i}" for i in range(4)]
ORIGINAL_BTC_GRID = [0.0, 0.01, 0.02, 0.03, 0.05]
RISK_BUDGET_CAP = 0.10

STRESS_COMPONENTS = {
    "vix": {"column": "macro_vix_z", "sign": 1, "label": "VIX"},
    "credit_spread": {"column": "macro_credit_spread_baa10y_z", "sign": 1, "label": "BAA10Y credit spread"},
    "financial_conditions": {
        "column": "macro_adjusted_financial_conditions_z",
        "sign": 1,
        "label": "adjusted financial conditions",
    },
    "real_yield": {"column": "macro_real_yield_10y_z", "sign": 1, "label": "10Y real yield"},
    "dollar": {"column": "macro_dollar_chg_4w_z", "sign": 1, "label": "dollar 4-week change"},
    "liquidity": {"column": "macro_net_liquidity_chg_4w_z", "sign": -1, "label": "net liquidity 4-week change"},
    "yield_curve": {"column": "macro_yield_curve_10y_2y_z", "sign": -1, "label": "10Y-2Y yield curve"},
}

SORTING_METHODS = [
    "original_equal_weight_stress",
    "leave_one_out_vix",
    "leave_one_out_credit_spread",
    "leave_one_out_financial_conditions",
    "leave_one_out_real_yield",
    "leave_one_out_dollar",
    "leave_one_out_liquidity",
    "leave_one_out_yield_curve",
    "vix_credit_conditions_only",
    "pca_first_component_order",
]

REQUIRED_RESULTS = [
    "input_validation_summary.json",
    "output_validation_summary.json",
    "data_lineage.csv",
    "model_assumption_audit.json",
    "methodology_audit.md",
    "part12_sorting_method_dictionary.csv",
    "part12_state_relabeling_summary.csv",
    "part12_label_stability_vs_original.csv",
    "part12_rule_sensitivity_summary.csv",
    "part12_rule_weight_by_sorting_method.csv",
    "part12_key_findings.json",
]

REQUIRED_FIGURES = [
    "part12_state_relabeling_heatmap.png",
    "part12_rule_sensitivity_by_sorting_method.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 12 stress composite sensitivity.")
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
        default="outputs/part12_stress_composite_sensitivity_outputs/part12_stress_composite_sensitivity",
        type=Path,
    )
    parser.add_argument("--run-id", default="colab_part12_seed42")
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
    paths: dict[str, Path] = {
        "state_model_panel_weekly": args.input_dir / "state_model_panel_weekly.csv",
        "part1_manifest": args.part1_run_dir / "run_manifest.json",
        "part1_validation": args.part1_run_dir / "results" / "validation_summary.json",
        "hmm4_state_labels": args.part1_run_dir / "results" / "hmm4_state_labels.csv",
        "hmm4_state_profiles": args.part1_run_dir / "results" / "hmm4_state_profiles.csv",
        "pca_loadings": args.part1_run_dir / "results" / "pca_loadings.csv",
        "pca_explained_variance": args.part1_run_dir / "results" / "pca_explained_variance.csv",
        "part2_manifest": args.part2_run_dir / "run_manifest.json",
        "part2_input_validation": args.part2_run_dir / "results" / "input_validation_summary.json",
        "part2_output_validation": args.part2_run_dir / "results" / "output_validation_summary.json",
        "part2_baseline_portfolio_weights": args.part2_run_dir / "results" / "baseline_portfolio_weights.csv",
        "part4_manifest": args.part4_run_dir / "run_manifest.json",
        "part4_input_validation": args.part4_run_dir / "results" / "input_validation_summary.json",
        "part4_output_validation": args.part4_run_dir / "results" / "output_validation_summary.json",
        "part4_allocation_rule_definition": args.part4_run_dir / "results" / "allocation_rule_definition.csv",
        "part4_performance_summary": args.part4_run_dir / "results" / "conditional_portfolio_performance_summary.csv",
    }
    for name, path in paths.items():
        require(path.exists(), f"Missing required input file: {name} at {path}")

    payload: dict[str, Any] = {
        "paths": paths,
        "state": pd.read_csv(paths["state_model_panel_weekly"], parse_dates=["date"]),
        "part1_manifest": read_json(paths["part1_manifest"]),
        "part1_validation": read_json(paths["part1_validation"]),
        "labels": pd.read_csv(paths["hmm4_state_labels"], parse_dates=["date"]),
        "profiles": pd.read_csv(paths["hmm4_state_profiles"]),
        "pca_loadings": pd.read_csv(paths["pca_loadings"]),
        "pca_explained_variance": pd.read_csv(paths["pca_explained_variance"]),
        "part2_manifest": read_json(paths["part2_manifest"]),
        "part2_input_validation": read_json(paths["part2_input_validation"]),
        "part2_output_validation": read_json(paths["part2_output_validation"]),
        "part2_baseline_weights": pd.read_csv(paths["part2_baseline_portfolio_weights"]),
        "part4_manifest": read_json(paths["part4_manifest"]),
        "part4_input_validation": read_json(paths["part4_input_validation"]),
        "part4_output_validation": read_json(paths["part4_output_validation"]),
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


def validate_inputs(inputs: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    state = inputs["state"].copy()
    labels = inputs["labels"].copy()
    profiles = inputs["profiles"].copy()
    require(inputs["part1_validation"].get("status") == "passed", "Part 1 validation did not pass")
    require(inputs["part2_input_validation"].get("status") == "passed", "Part 2 input validation did not pass")
    require(inputs["part2_output_validation"].get("status") == "passed", "Part 2 output validation did not pass")
    require(inputs["part4_input_validation"].get("status") == "passed", "Part 4 input validation did not pass")
    require(inputs["part4_output_validation"].get("status") == "passed", "Part 4 output validation did not pass")
    require(len(state) == EXPECTED_STATE_ROWS, f"Unexpected state panel rows: {len(state)}")
    require(len(labels) == EXPECTED_STATE_ROWS, f"Unexpected HMM4 label rows: {len(labels)}")
    require(date_string(state["date"], "min") == EXPECTED_STATE_START, "Unexpected state start date")
    require(date_string(state["date"], "max") == EXPECTED_STATE_END, "Unexpected state end date")
    require(date_string(labels["date"], "min") == EXPECTED_STATE_START, "Unexpected HMM4 label start date")
    require(date_string(labels["date"], "max") == EXPECTED_STATE_END, "Unexpected HMM4 label end date")
    require(all(col in state.columns for col in ["date"] + PORTFOLIO_ASSETS), "Missing return columns")
    require(set(labels["hmm4_state"].unique()) == set(HMM4_STATES), "Unexpected HMM4 state labels")
    require(set(profiles["state"]) == set(HMM4_STATES), "Unexpected HMM4 profile states")
    for component in STRESS_COMPONENTS.values():
        require(f"{component['column']}_mean" in profiles.columns, f"Missing profile column for {component['column']}")
    require(set(inputs["pca_loadings"]["predictor"]) >= {v["column"] for v in STRESS_COMPONENTS.values()}, "PCA loadings missing stress predictors")

    base_weights = build_base_weights(inputs["part2_baseline_weights"])
    summary = {
        "status": "passed",
        "sample_frozen": True,
        "sample_rows": EXPECTED_STATE_ROWS,
        "sample_start": EXPECTED_STATE_START,
        "sample_end": EXPECTED_STATE_END,
        "model": "hmm4",
        "reestimates_hmm": False,
        "sorting_methods": SORTING_METHODS,
        "base_weights": base_weights,
        "risk_budget_cap": RISK_BUDGET_CAP,
        "btc_grid": ORIGINAL_BTC_GRID,
        "input_hashes": inputs["input_hashes"],
        "part1_run_id": inputs["part1_manifest"].get("run_id"),
        "part2_run_id": inputs["part2_manifest"].get("run_id"),
        "part4_run_id": inputs["part4_manifest"].get("run_id"),
    }
    write_json(dirs["results"] / "input_validation_summary.json", normalize_for_json(summary))
    logging.info("Input validation passed")
    return {"summary": summary, "base_weights": base_weights}


def build_base_weights(baseline: pd.DataFrame) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for family, frame in baseline.groupby("portfolio_family"):
        weights = frame.set_index("asset")["weight"].astype(float).to_dict()
        require(set(weights) == set(BASE_ASSETS), f"Unexpected base assets for {family}: {weights.keys()}")
        require(abs(sum(weights.values()) - 1.0) < 1e-8, f"Base weights do not sum to one for {family}")
        payload[str(family)] = weights
    require(set(payload) == set(PORTFOLIO_FAMILIES), "Missing portfolio families")
    return payload


def method_components(method_id: str) -> tuple[list[str], list[str], str]:
    all_components = list(STRESS_COMPONENTS)
    if method_id == "original_equal_weight_stress":
        return all_components, [], "Original Part 1 equal-weight stress composite."
    if method_id.startswith("leave_one_out_"):
        excluded = method_id.replace("leave_one_out_", "")
        included = [item for item in all_components if item != excluded]
        return included, [excluded], f"Equal-weight stress composite excluding {excluded}."
    if method_id == "vix_credit_conditions_only":
        included = ["vix", "credit_spread", "financial_conditions"]
        return included, [item for item in all_components if item not in included], "Market stress-only composite."
    if method_id == "pca_first_component_order":
        return [], [], "Part 1 PC1 score based on saved PCA loadings."
    raise ValueError(f"Unknown sorting method: {method_id}")


def component_sign_string(components: list[str]) -> str:
    return ";".join(f"{name}:{STRESS_COMPONENTS[name]['sign']:+d}" for name in components)


def build_sorting_method_dictionary(inputs: dict[str, Any], dirs: dict[str, Path]) -> pd.DataFrame:
    pca = inputs["pca_loadings"].set_index("predictor")["PC1"].astype(float)
    anchor_cols = [
        "macro_vix_z",
        "macro_credit_spread_baa10y_z",
        "macro_adjusted_financial_conditions_z",
    ]
    pca_anchor = float(pca.loc[anchor_cols].sum())
    pca_direction = 1.0 if pca_anchor >= 0 else -1.0
    rows = []
    for method_id in SORTING_METHODS:
        included, excluded, notes = method_components(method_id)
        if method_id == "pca_first_component_order":
            signs = ";".join(f"{name}:{pca_direction * float(value):+.6f}" for name, value in pca.items())
            normalization = "state_profile_z_means_dot_PC1_loadings"
            direction = "Higher PC1 score is interpreted as higher stress because VIX, credit spread, and financial-conditions loadings are jointly positive after orientation."
            included_components = ",".join(pca.index)
            excluded_components = ""
            if pca_direction < 0:
                notes += " PC1 sign was flipped by the runner to keep the stress anchor positive."
        else:
            signs = component_sign_string(included)
            normalization = f"equal_weight_mean_over_{len(included)}_components"
            direction = "Higher score means higher macro stress under the stated component signs."
            included_components = ",".join(included)
            excluded_components = ",".join(excluded)
        rows.append(
            {
                "sorting_method_id": method_id,
                "included_components": included_components,
                "excluded_components": excluded_components,
                "component_signs": signs,
                "normalization_method": normalization,
                "direction_interpretation": direction,
                "notes": notes,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(dirs["results"] / "part12_sorting_method_dictionary.csv", index=False)
    return out


def state_score(profiles: pd.DataFrame, pca_loadings: pd.DataFrame, method_id: str) -> pd.Series:
    profile_idx = profiles.set_index("state")
    if method_id == "pca_first_component_order":
        load = pca_loadings.set_index("predictor")["PC1"].astype(float)
        anchor = float(load.loc[["macro_vix_z", "macro_credit_spread_baa10y_z", "macro_adjusted_financial_conditions_z"]].sum())
        direction = 1.0 if anchor >= 0 else -1.0
        scores = pd.Series(0.0, index=profile_idx.index)
        for predictor, loading in load.items():
            col = f"{predictor}_mean"
            if col in profile_idx.columns:
                scores += direction * float(loading) * profile_idx[col].astype(float)
        return scores

    included, _, _ = method_components(method_id)
    scores = pd.Series(0.0, index=profile_idx.index)
    for name in included:
        spec = STRESS_COMPONENTS[name]
        scores += float(spec["sign"]) * profile_idx[f"{spec['column']}_mean"].astype(float)
    return scores / float(len(included))


def build_relabeling(inputs: dict[str, Any], dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    profiles = inputs["profiles"].copy()
    pca_loadings = inputs["pca_loadings"].copy()
    relabel_rows = []
    mapping: dict[str, dict[str, str]] = {}
    for method_id in SORTING_METHODS:
        scores = state_score(profiles, pca_loadings, method_id)
        ordered_states = list(scores.sort_values(kind="mergesort").index)
        state_to_new_label = {state: f"state_{rank}" for rank, state in enumerate(ordered_states)}
        mapping[method_id] = state_to_new_label
        for state in HMM4_STATES:
            prof = profiles[profiles["state"].eq(state)].iloc[0]
            new_label = state_to_new_label[state]
            relabel_rows.append(
                {
                    "sorting_method_id": method_id,
                    "original_state": state,
                    "new_state_rank": int(new_label.split("_")[-1]),
                    "new_state_label": new_label,
                    "stress_score_mean": float(scores.loc[state]),
                    "state_count": int(prof["n_weeks"]),
                    "state_share": float(prof["sample_share"]),
                }
            )
    relabel = pd.DataFrame(relabel_rows).sort_values(["sorting_method_id", "new_state_rank"]).reset_index(drop=True)
    relabel.to_csv(dirs["results"] / "part12_state_relabeling_summary.csv", index=False)
    save_pickle(dirs["models"] / "part12_state_relabeling_mapping.pkl", mapping)
    return {"relabeling": relabel, "mapping": mapping}


def contingency(labels_a: np.ndarray, labels_b: np.ndarray) -> np.ndarray:
    a_vals = {v: i for i, v in enumerate(sorted(set(labels_a)))}
    b_vals = {v: i for i, v in enumerate(sorted(set(labels_b)))}
    table = np.zeros((len(a_vals), len(b_vals)), dtype=int)
    for a, b in zip(labels_a, labels_b):
        table[a_vals[a], b_vals[b]] += 1
    return table


def comb2(x: np.ndarray | float | int) -> np.ndarray | float:
    return np.asarray(x) * (np.asarray(x) - 1) / 2.0


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    table = contingency(labels_a, labels_b)
    n = table.sum()
    if n < 2:
        return np.nan
    sum_comb = comb2(table).sum()
    row_comb = comb2(table.sum(axis=1)).sum()
    col_comb = comb2(table.sum(axis=0)).sum()
    total_comb = comb2(n)
    expected = row_comb * col_comb / total_comb if total_comb else 0.0
    max_index = 0.5 * (row_comb + col_comb)
    denom = max_index - expected
    return float((sum_comb - expected) / denom) if abs(denom) > FLOAT_TOL else 0.0


def build_label_stability(inputs: dict[str, Any], relabel_payload: dict[str, Any], dirs: dict[str, Path]) -> pd.DataFrame:
    labels = inputs["labels"].copy()
    original_ids = labels["hmm4_state_id"].astype(int).to_numpy()
    state_counts = labels["hmm4_state"].value_counts().to_dict()
    rows = []
    for method_id, state_to_new_label in relabel_payload["mapping"].items():
        new_labels = labels["hmm4_state"].map(state_to_new_label)
        new_ids = new_labels.str.split("_").str[-1].astype(int).to_numpy()
        changed_pairs = [
            f"{state}->{state_to_new_label[state]}"
            for state in HMM4_STATES
            if state_to_new_label[state] != state
        ]
        identity_count = sum(state_counts[state] for state in HMM4_STATES if state_to_new_label[state] == state)
        state_0_same = bool(state_to_new_label["state_0"] == "state_0")
        state_3_same = bool(state_to_new_label["state_3"] == "state_3")
        notes = "Partition unchanged; ARI remains one under pure relabeling. Agreement tracks label identity."
        if method_id == "pca_first_component_order":
            notes += " PC1 direction is interpreted through VIX, credit-spread, and financial-conditions loadings."
        rows.append(
            {
                "sorting_method_id": method_id,
                "agreement_with_original_ranked_labels": float(identity_count / len(labels)),
                "states_reordered": ";".join(changed_pairs),
                "state_0_same_as_original": state_0_same,
                "state_3_same_as_original": state_3_same,
                "adjusted_rand_index": adjusted_rand_index(original_ids, new_ids),
                "notes": notes,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(dirs["results"] / "part12_label_stability_vs_original.csv", index=False)
    return out


def weights_from_btc(base_weights: dict[str, float], btc_weight: float) -> dict[str, float]:
    weights = {"ret_btc": float(btc_weight)}
    for asset, base_weight in base_weights.items():
        weights[asset] = float((1.0 - btc_weight) * base_weight)
    require(min(weights.values()) >= -FLOAT_TOL, "Negative weight found")
    require(abs(sum(weights.values()) - 1.0) < 1e-8, "Weights do not sum to one")
    return weights


def component_matrix_for_panel(panel: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    return np.column_stack([weights.get(asset, 0.0) * panel[asset].astype(float).to_numpy() for asset in PORTFOLIO_ASSETS])


def risk_shares(component_matrix: np.ndarray) -> dict[str, Any]:
    portfolio = component_matrix.sum(axis=1)
    vol = float(np.std(portfolio, ddof=1))
    if vol <= FLOAT_TOL:
        vol_share = np.nan
    else:
        cov = float(np.cov(component_matrix[:, 0], portfolio, ddof=1)[0, 1])
        vol_share = cov / (vol * vol)
    var_value = float(np.quantile(portfolio, TAIL_ALPHA))
    tail_mask = portfolio <= var_value
    tail_count = int(tail_mask.sum())
    cvar_loss = float((-portfolio[tail_mask]).mean()) if tail_count else np.nan
    if not math.isfinite(cvar_loss) or abs(cvar_loss) <= FLOAT_TOL:
        cvar_share = np.nan
    else:
        cvar_share = float((-component_matrix[tail_mask, 0]).mean() / cvar_loss)
    return {
        "btc_share_vol": vol_share,
        "btc_share_cvar": cvar_share,
        "portfolio_volatility_weekly": vol,
        "portfolio_cvar_95_weekly": float(portfolio[tail_mask].mean()) if tail_count else np.nan,
        "tail_scenario_count": tail_count,
    }


def var_cvar(returns: pd.Series) -> tuple[float, float, int]:
    clean = returns.dropna()
    var = float(clean.quantile(TAIL_ALPHA))
    tail = clean[clean <= var]
    return var, float(tail.mean()), int(len(tail))


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    return float((wealth / peak - 1.0).min())


def performance_metrics(returns: pd.Series) -> dict[str, Any]:
    clean = returns.dropna()
    var, cvar, tail_count = var_cvar(clean)
    vol = float(clean.std(ddof=1))
    return {
        "annualized_mean_arithmetic": float(clean.mean() * TRADING_WEEKS_PER_YEAR),
        "annualized_volatility": float(vol * math.sqrt(TRADING_WEEKS_PER_YEAR)),
        "cvar_95_weekly": cvar,
        "max_drawdown": max_drawdown(clean),
        "positive_week_share": float((clean > 0.0).mean()),
        "tail_scenario_count": tail_count,
    }


def raw_ranked_rule_weights() -> dict[str, float]:
    return {"state_0": 0.03, "state_1": 0.01, "state_2": 0.0, "state_3": 0.0}


def select_state_weight(panel: pd.DataFrame, base_weights: dict[str, float], state_col: str, state: str, raw_weight: float) -> tuple[float, str, dict[str, Any]]:
    candidates = sorted([w for w in ORIGINAL_BTC_GRID if w <= raw_weight + FLOAT_TOL], reverse=True)
    state_panel = panel[panel[state_col].eq(state)]
    for candidate in candidates:
        weights = weights_from_btc(base_weights, candidate)
        full_risk = risk_shares(component_matrix_for_panel(panel, weights))
        state_risk = risk_shares(component_matrix_for_panel(state_panel, weights)) if len(state_panel) >= 2 else {
            "btc_share_vol": np.nan,
            "btc_share_cvar": np.nan,
            "tail_scenario_count": 0,
        }
        ok = (
            full_risk["btc_share_vol"] <= RISK_BUDGET_CAP + FLOAT_TOL
            and full_risk["btc_share_cvar"] <= RISK_BUDGET_CAP + FLOAT_TOL
            and state_risk["btc_share_vol"] <= RISK_BUDGET_CAP + FLOAT_TOL
            and state_risk["btc_share_cvar"] <= RISK_BUDGET_CAP + FLOAT_TOL
        )
        if ok:
            if raw_weight == 0:
                reason = "raw_rule_zero_allocation"
            elif abs(candidate - raw_weight) < FLOAT_TOL:
                reason = "raw_weight_within_full_and_active_state_caps"
            else:
                reason = "reduced_to_highest_original_grid_weight_satisfying_caps"
            return candidate, reason, {
                "full_sample_btc_share_vol": full_risk["btc_share_vol"],
                "full_sample_btc_share_cvar": full_risk["btc_share_cvar"],
                "state_btc_share_vol": state_risk["btc_share_vol"],
                "state_btc_share_cvar": state_risk["btc_share_cvar"],
                "state_cvar_tail_scenario_count": state_risk["tail_scenario_count"],
            }
    return 0.0, "fallback_zero_weight_no_candidate_satisfied_caps", {}


def build_rule_sensitivity(
    inputs: dict[str, Any],
    validation: dict[str, Any],
    relabel_payload: dict[str, Any],
    stability: pd.DataFrame,
    dirs: dict[str, Path],
) -> dict[str, pd.DataFrame]:
    state_returns = inputs["state"][["date"] + PORTFOLIO_ASSETS].copy()
    labels = inputs["labels"][["date", "hmm4_state", "hmm4_state_id"]].copy()
    base_weights_all = validation["base_weights"]
    raw_map = raw_ranked_rule_weights()
    stability_idx = stability.set_index("sorting_method_id")
    weight_rows = []
    perf_rows = []

    for method_id, state_to_new_label in relabel_payload["mapping"].items():
        panel = state_returns.merge(labels, on="date", how="inner", validate="one_to_one")
        panel["stress_ranked_state"] = panel["hmm4_state"].map(state_to_new_label)
        panel["stress_ranked_state_id"] = panel["stress_ranked_state"].str.split("_").str[-1].astype(int)
        original_state_by_new_label = {new_label: old_state for old_state, new_label in state_to_new_label.items()}

        for family in PORTFOLIO_FAMILIES:
            base_weights = base_weights_all[family]
            executed_map: dict[str, float] = {}
            for ranked_state, raw_weight in raw_map.items():
                selected, reason, audit = select_state_weight(panel, base_weights, "stress_ranked_state", ranked_state, raw_weight)
                executed_map[ranked_state] = selected
                original_state = original_state_by_new_label[ranked_state]
                weight_rows.append(
                    {
                        "sorting_method_id": method_id,
                        "portfolio_family": family,
                        "risk_budget_cap": RISK_BUDGET_CAP,
                        "new_state_label": ranked_state,
                        "original_hmm4_state": original_state,
                        "original_hmm4_state_id": int(original_state.split("_")[-1]),
                        "state_n_weeks": int((panel["stress_ranked_state"] == ranked_state).sum()),
                        "raw_btc_weight": raw_weight,
                        "selected_btc_weight": selected,
                        "adjustment_reason": reason,
                        **audit,
                    }
                )

            component_rows = []
            btc_weights = []
            for _, obs in panel.sort_values("date").iterrows():
                btc_weight = executed_map[str(obs["stress_ranked_state"])]
                weights = weights_from_btc(base_weights, btc_weight)
                btc_weights.append(btc_weight)
                component_rows.append([weights[asset] * float(obs[asset]) for asset in PORTFOLIO_ASSETS])
            component = np.asarray(component_rows, dtype=float)
            returns = pd.Series(component.sum(axis=1))
            perf = performance_metrics(returns)
            risk = risk_shares(component)
            row_stability = stability_idx.loc[method_id]
            conclusion_changed = bool(row_stability["states_reordered"] != "")
            change_reason = conclusion_change_reason(row_stability)
            perf_rows.append(
                {
                    "sorting_method_id": method_id,
                    "portfolio_family": family,
                    "risk_budget_cap": RISK_BUDGET_CAP,
                    "average_btc_weight": float(np.mean(btc_weights)),
                    "max_btc_weight": float(np.max(btc_weights)),
                    "active_week_share": float(np.mean(np.asarray(btc_weights) > FLOAT_TOL)),
                    **perf,
                    "btc_share_vol": risk["btc_share_vol"],
                    "btc_share_cvar": risk["btc_share_cvar"],
                    "conclusion_changed": conclusion_changed,
                    "conclusion_change_reason": change_reason,
                }
            )

    weights_out = pd.DataFrame(weight_rows).sort_values(["sorting_method_id", "portfolio_family", "new_state_label"]).reset_index(drop=True)
    perf_out = pd.DataFrame(perf_rows).sort_values(["portfolio_family", "sorting_method_id"]).reset_index(drop=True)
    weights_out.to_csv(dirs["results"] / "part12_rule_weight_by_sorting_method.csv", index=False)
    perf_out.to_csv(dirs["results"] / "part12_rule_sensitivity_summary.csv", index=False)
    return {"weights": weights_out, "performance": perf_out}


def conclusion_change_reason(stability_row: pd.Series) -> str:
    reasons = []
    if str(stability_row["states_reordered"]) != "":
        reasons.append("state_order_relabeling_changed")
    if not bool(stability_row["state_0_same_as_original"]):
        reasons.append("low-stress_state_0_identity_changed")
    if not bool(stability_row["state_3_same_as_original"]):
        reasons.append("highest-stress_state_3_identity_changed")
    if not reasons:
        return "no_label_order_change"
    if reasons == ["state_order_relabeling_changed", "highest-stress_state_3_identity_changed"]:
        return "highest-stress_state_identity_changed_but_zero-BTC_tail_states_keep_rule_performance_unchanged"
    return ";".join(reasons)


def build_key_findings(stability: pd.DataFrame, rule_results: dict[str, pd.DataFrame], dirs: dict[str, Path]) -> dict[str, Any]:
    perf = rule_results["performance"].copy()
    changed = stability[stability["states_reordered"].astype(str).ne("")]
    state0_changed = stability[~stability["state_0_same_as_original"].astype(bool)]
    state3_changed = stability[~stability["state_3_same_as_original"].astype(bool)]
    original_perf = perf[perf["sorting_method_id"].eq("original_equal_weight_stress")].set_index("portfolio_family")
    deltas = []
    for _, row in perf.iterrows():
        base = original_perf.loc[row["portfolio_family"]]
        deltas.append(
            {
                "sorting_method_id": row["sorting_method_id"],
                "portfolio_family": row["portfolio_family"],
                "delta_annualized_mean_vs_original": float(row["annualized_mean_arithmetic"] - base["annualized_mean_arithmetic"]),
                "delta_btc_share_cvar_vs_original": float(row["btc_share_cvar"] - base["btc_share_cvar"]),
                "delta_average_btc_weight_vs_original": float(row["average_btc_weight"] - base["average_btc_weight"]),
            }
        )
    delta_frame = pd.DataFrame(deltas)
    non_original = delta_frame[delta_frame["sorting_method_id"].ne("original_equal_weight_stress")]
    max_abs_mean_delta = float(non_original["delta_annualized_mean_vs_original"].abs().max())
    max_abs_cvar_share_delta = float(non_original["delta_btc_share_cvar_vs_original"].abs().max())
    performance_changed = non_original[
        (non_original["delta_annualized_mean_vs_original"].abs() > 1e-12)
        | (non_original["delta_btc_share_cvar_vs_original"].abs() > 1e-12)
        | (non_original["delta_average_btc_weight_vs_original"].abs() > 1e-12)
    ]["sorting_method_id"].drop_duplicates().tolist()
    payload = {
        "methods_tested": SORTING_METHODS,
        "methods_with_any_reordering": changed["sorting_method_id"].tolist(),
        "methods_changing_state_0": state0_changed["sorting_method_id"].tolist(),
        "methods_changing_state_3": state3_changed["sorting_method_id"].tolist(),
        "methods_changing_rule_performance_or_exposure": performance_changed,
        "max_abs_annualized_mean_delta_vs_original": max_abs_mean_delta,
        "max_abs_btc_cvar_share_delta_vs_original": max_abs_cvar_share_delta,
        "rule_performance_snapshot": perf.to_dict(orient="records"),
        "recommended_thesis_statement": (
            "Part 12 keeps the HMM-4 partition fixed and varies only the economic stress ordering. "
            "If alternative composites reorder state_0 or state_3, the paper should frame the conditional BTC rule as sensitive to the chosen stress interpretation rather than as a uniquely identified allocation rule."
        ),
    }
    write_json(dirs["results"] / "part12_key_findings.json", normalize_for_json(payload))
    return payload


def plot_state_relabeling_heatmap(relabeling: pd.DataFrame, output_path: Path) -> None:
    pivot = relabeling.pivot(index="sorting_method_id", columns="original_state", values="new_state_rank").loc[SORTING_METHODS, HMM4_STATES]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis_r", vmin=0, vmax=3)
    ax.set_xticks(range(len(HMM4_STATES)))
    ax.set_xticklabels(HMM4_STATES)
    ax.set_yticks(range(len(SORTING_METHODS)))
    ax.set_yticklabels(SORTING_METHODS, fontsize=7)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{int(pivot.iloc[i, j])}", ha="center", va="center", color="white", fontsize=8)
    ax.set_title("Part 12 State Relabeling by Stress Sorting Method")
    ax.set_xlabel("Original HMM-4 state")
    ax.set_ylabel("Sorting method")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("New stress rank")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_rule_sensitivity(performance: pd.DataFrame, output_path: Path) -> None:
    perf = performance.copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    colors = {"all_weather": "#4c78a8", "erc": "#f58518"}
    y_positions = np.arange(len(SORTING_METHODS))
    for family in PORTFOLIO_FAMILIES:
        frame = perf[perf["portfolio_family"].eq(family)].set_index("sorting_method_id").loc[SORTING_METHODS]
        offset = -0.18 if family == "all_weather" else 0.18
        axes[0].barh(y_positions + offset, frame["average_btc_weight"], height=0.32, color=colors[family], label=family)
        axes[1].barh(y_positions + offset, frame["btc_share_cvar"], height=0.32, color=colors[family], label=family)
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(SORTING_METHODS, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Average BTC weight")
    axes[0].set_title("Exposure")
    axes[1].axvline(RISK_BUDGET_CAP, color="#d62728", linestyle="--", linewidth=1, label="10% cap")
    axes[1].set_xlabel("BTC CVaR contribution share")
    axes[1].set_title("Risk Contribution")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Part 12 Rule Sensitivity to State Ordering")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_audits(args: argparse.Namespace, inputs: dict[str, Any], dirs: dict[str, Path]) -> None:
    lineage_rows = [
        {"input_name": name, "path": str(path), "sha256": inputs["input_hashes"][name]}
        for name, path in sorted(inputs["paths"].items())
    ]
    pd.DataFrame(lineage_rows).to_csv(dirs["results"] / "data_lineage.csv", index=False)
    audit = {
        "status": "passed",
        "reestimates_hmm": False,
        "reestimates_pca": False,
        "state_partition_scope": "Frozen Part 1 HMM-4 labels; only state ranking labels change.",
        "rule_scope": "Common stress-ranked rule state_0=3%, state_1=1%, state_2=0%, state_3=0%, then original-grid 10% risk cap.",
        "sample_frozen": f"{EXPECTED_STATE_START} to {EXPECTED_STATE_END}",
        "interpretation_caveat": "Alternative stress composites test label-ordering sensitivity, not a newly optimized trading rule.",
    }
    write_json(dirs["results"] / "model_assumption_audit.json", audit)
    md = f"""# Part 12 Methodology Audit

Part 12 audits the economic ordering of the frozen Part 1 HMM-4 states. It does not re-estimate PCA, HMM, KMeans, portfolio weights, or state posterior probabilities.

The original Part 1 ordering uses an equal-weight stress composite with positive signs for VIX, BAA10Y credit spread, adjusted financial conditions, 10-year real yield, and dollar 4-week change, and negative signs for net liquidity and the 10Y-2Y yield curve. Leave-one-out methods recompute this ordering after dropping one component. The VIX-credit-conditions method uses only market stress and financial conditions. The PC1 method uses saved Part 1 PCA loadings and interprets higher PC1 as higher stress because the VIX, credit-spread, and financial-conditions loadings are jointly positive after orientation.

For each ordering, the same ranked rule is applied: `state_0=3%`, `state_1=1%`, `state_2=0%`, and `state_3=0%`, followed by the original-grid {RISK_BUDGET_CAP:.0%} BTC volatility and CVaR contribution cap. This is a sensitivity audit, not rule optimization.
"""
    (dirs["results"] / "methodology_audit.md").write_text(md, encoding="utf-8")


def validate_outputs(
    inputs: dict[str, Any],
    relabeling: pd.DataFrame,
    stability: pd.DataFrame,
    rule_results: dict[str, pd.DataFrame],
    dirs: dict[str, Path],
) -> dict[str, Any]:
    weights = rule_results["weights"].copy()
    performance = rule_results["performance"].copy()
    required_results_already_written = [name for name in REQUIRED_RESULTS if name != "output_validation_summary.json"]
    required_files_present = all((dirs["results"] / name).exists() for name in required_results_already_written) and all(
        (dirs["figures"] / name).exists() for name in REQUIRED_FIGURES
    )
    original_stability = stability[stability["sorting_method_id"].eq("original_equal_weight_stress")].iloc[0]
    original_identity_ok = bool(
        abs(float(original_stability["agreement_with_original_ranked_labels"]) - 1.0) < FLOAT_TOL
        and str(original_stability["states_reordered"]) == ""
        and bool(original_stability["state_0_same_as_original"])
        and bool(original_stability["state_3_same_as_original"])
    )
    ranks_ok = bool(
        relabeling.groupby("sorting_method_id")["new_state_rank"].apply(lambda s: sorted(s.astype(int).tolist()) == [0, 1, 2, 3]).all()
    )
    part4_main = inputs["part4_rule_definition"][
        (inputs["part4_rule_definition"]["rule_id"].eq("main_executed"))
        & (inputs["part4_rule_definition"]["constraint_stage"].eq("risk_budget_executed"))
    ]
    original_weights = weights[weights["sorting_method_id"].eq("original_equal_weight_stress")]
    merged = original_weights.merge(
        part4_main,
        left_on=["portfolio_family", "new_state_label"],
        right_on=["portfolio_family", "hmm4_state"],
        how="inner",
        suffixes=("_part12", "_part4"),
    )
    part4_weight_reproduction_ok = bool(
        len(merged) == 8
        and np.allclose(
            merged["selected_btc_weight_part12"].astype(float),
            merged["selected_btc_weight_part4"].astype(float),
            atol=FLOAT_TOL,
        )
    )
    state0_or_state3_reordered = stability[
        (~stability["state_0_same_as_original"].astype(bool)) | (~stability["state_3_same_as_original"].astype(bool))
    ]
    flagged = performance[performance["sorting_method_id"].isin(state0_or_state3_reordered["sorting_method_id"])]
    flagged_ok = bool(flagged.empty or flagged["conclusion_changed"].astype(bool).all())
    cvar_cap_ok = bool((performance["btc_share_cvar"].astype(float) <= RISK_BUDGET_CAP + 1e-8).all())
    vol_cap_ok = bool((performance["btc_share_vol"].astype(float) <= RISK_BUDGET_CAP + 1e-8).all())
    summary = {
        "status": "passed" if all([required_files_present, original_identity_ok, ranks_ok, part4_weight_reproduction_ok, flagged_ok, cvar_cap_ok, vol_cap_ok]) else "failed",
        "sorting_method_rows": len(SORTING_METHODS),
        "relabeling_rows": int(len(relabeling)),
        "stability_rows": int(len(stability)),
        "rule_weight_rows": int(len(weights)),
        "rule_sensitivity_rows": int(len(performance)),
        "original_identity_reproduced": original_identity_ok,
        "new_ranks_complete_by_method": ranks_ok,
        "part4_main_weight_reproduction_ok": part4_weight_reproduction_ok,
        "state0_or_state3_reordering_flagged": flagged_ok,
        "vol_cap_ok": vol_cap_ok,
        "cvar_cap_ok": cvar_cap_ok,
        "required_files_present": required_files_present,
    }
    require(summary["status"] == "passed", f"Output validation failed: {summary}")
    write_json(dirs["results"] / "output_validation_summary.json", normalize_for_json(summary))
    return summary


def package_versions() -> dict[str, str]:
    packages = {}
    for name in ["numpy", "pandas", "matplotlib"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    return {"python": sys.version, "platform": platform.platform(), **packages}


def write_manifest(args: argparse.Namespace, output_validation: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    manifest = {
        "part_id": "part12_stress_composite_sensitivity",
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_frozen": True,
        "sample_end": EXPECTED_STATE_END,
        "seed": args.seed,
        "inputs": {
            "state_model_panel_weekly": str(args.input_dir / "state_model_panel_weekly.csv"),
            "part1_run_dir": str(args.part1_run_dir),
            "part2_run_dir": str(args.part2_run_dir),
            "part4_run_dir": str(args.part4_run_dir),
        },
        "parameters": {
            "sorting_methods": SORTING_METHODS,
            "risk_budget_cap": RISK_BUDGET_CAP,
            "btc_grid": ORIGINAL_BTC_GRID,
            "reestimates_hmm": False,
            "rule_mapping": "state_0=3%, state_1=1%, state_2=0%, state_3=0%, then original-grid 10% risk cap",
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
    logging.info("Starting Part 12 run in %s", run_dir)
    inputs = read_inputs(args)
    enforce_resume_input_hashes(args, dirs, inputs["input_hashes"])
    validation = validate_inputs(inputs, dirs)
    save_pickle(dirs["checkpoints"] / "01_input_validation.pkl", validation)
    sorting = build_sorting_method_dictionary(inputs, dirs)
    relabel_payload = build_relabeling(inputs, dirs)
    save_pickle(dirs["checkpoints"] / "02_relabeling.pkl", relabel_payload)
    stability = build_label_stability(inputs, relabel_payload, dirs)
    save_pickle(dirs["checkpoints"] / "03_label_stability.pkl", stability)
    rule_results = build_rule_sensitivity(inputs, validation, relabel_payload, stability, dirs)
    save_pickle(dirs["checkpoints"] / "04_rule_sensitivity.pkl", rule_results)
    findings = build_key_findings(stability, rule_results, dirs)
    save_pickle(dirs["checkpoints"] / "05_key_findings.pkl", findings)
    plot_state_relabeling_heatmap(relabel_payload["relabeling"], dirs["figures"] / "part12_state_relabeling_heatmap.png")
    plot_rule_sensitivity(rule_results["performance"], dirs["figures"] / "part12_rule_sensitivity_by_sorting_method.png")
    write_audits(args, inputs, dirs)
    output_validation = validate_outputs(inputs, relabel_payload["relabeling"], stability, rule_results, dirs)
    write_manifest(args, output_validation, dirs)
    logging.info("Part 12 completed successfully")


if __name__ == "__main__":
    main()
