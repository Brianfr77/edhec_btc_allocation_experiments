#!/usr/bin/env python3
"""Part 11 experiment runner: frozen HMM/KMeans model comparison.

This runner audits the model variants already produced by Part 1. It does not
re-estimate PCA, HMM, or KMeans models. It compares model-selection diagnostics,
label stability, and the effect of applying a common stress-ranked BTC rule.
"""

from __future__ import annotations

import argparse
import ast
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
MODEL_IDS = ["hmm3", "hmm4", "hmm5", "kmeans4"]
ORIGINAL_BTC_GRID = [0.0, 0.01, 0.02, 0.03, 0.05]
RISK_BUDGET_CAP = 0.10

REQUIRED_RESULTS = [
    "input_validation_summary.json",
    "output_validation_summary.json",
    "data_lineage.csv",
    "model_assumption_audit.json",
    "methodology_audit.md",
    "part11_model_selection_summary.csv",
    "part11_state_profile_summary.csv",
    "part11_state_mapping_stability.csv",
    "part11_rule_weights_by_model.csv",
    "part11_rule_performance_by_model.csv",
    "part11_model_comparison_key_findings.json",
]

REQUIRED_FIGURES = [
    "part11_aic_bic_by_model.png",
    "part11_min_state_share_by_model.png",
    "part11_hmm4_vs_hmm5_state_timeline.png",
    "part11_rule_result_by_model.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 11 HMM/KMeans model comparison.")
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
        "--output-dir",
        default="outputs/part11_hmm_model_comparison_outputs/part11_hmm_model_comparison",
        type=Path,
    )
    parser.add_argument("--run-id", default="colab_part11_seed42")
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
    paths: dict[str, Path] = {
        "state_model_panel_weekly": args.input_dir / "state_model_panel_weekly.csv",
        "part1_manifest": args.part1_run_dir / "run_manifest.json",
        "part1_validation": args.part1_run_dir / "results" / "validation_summary.json",
        "model_robustness_summary": args.part1_run_dir / "results" / "model_robustness_summary.csv",
        "pca_explained_variance": args.part1_run_dir / "results" / "pca_explained_variance.csv",
        "part2_manifest": args.part2_run_dir / "run_manifest.json",
        "part2_input_validation": args.part2_run_dir / "results" / "input_validation_summary.json",
        "part2_output_validation": args.part2_run_dir / "results" / "output_validation_summary.json",
        "part2_baseline_portfolio_weights": args.part2_run_dir / "results" / "baseline_portfolio_weights.csv",
    }
    for model_id in MODEL_IDS:
        paths[f"{model_id}_state_labels"] = args.part1_run_dir / "results" / f"{model_id}_state_labels.csv"
        paths[f"{model_id}_state_profiles"] = args.part1_run_dir / "results" / f"{model_id}_state_profiles.csv"
        if model_id.startswith("hmm"):
            paths[f"{model_id}_transition_matrix"] = args.part1_run_dir / "results" / f"{model_id}_transition_matrix.csv"

    for name, path in paths.items():
        require(path.exists(), f"Missing required input file: {name} at {path}")

    payload: dict[str, Any] = {
        "paths": paths,
        "state": pd.read_csv(paths["state_model_panel_weekly"], parse_dates=["date"]),
        "part1_manifest": read_json(paths["part1_manifest"]),
        "part1_validation": read_json(paths["part1_validation"]),
        "model_robustness": pd.read_csv(paths["model_robustness_summary"]),
        "pca_explained_variance": pd.read_csv(paths["pca_explained_variance"]),
        "part2_manifest": read_json(paths["part2_manifest"]),
        "part2_input_validation": read_json(paths["part2_input_validation"]),
        "part2_output_validation": read_json(paths["part2_output_validation"]),
        "part2_baseline_weights": pd.read_csv(paths["part2_baseline_portfolio_weights"]),
        "labels": {},
        "profiles": {},
        "transitions": {},
        "input_hashes": {name: file_sha256(path) for name, path in paths.items()},
    }
    for model_id in MODEL_IDS:
        payload["labels"][model_id] = pd.read_csv(paths[f"{model_id}_state_labels"], parse_dates=["date"])
        payload["profiles"][model_id] = pd.read_csv(paths[f"{model_id}_state_profiles"])
        if model_id.startswith("hmm"):
            payload["transitions"][model_id] = read_transition_matrix(paths[f"{model_id}_transition_matrix"])
    return payload


def read_transition_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = [str(x).replace("from_", "") for x in frame.index]
    frame.columns = [str(x).replace("to_", "") for x in frame.columns]
    return frame.astype(float)


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
    require(inputs["part1_validation"].get("status") == "passed", "Part 1 validation did not pass")
    require(inputs["part2_input_validation"].get("status") == "passed", "Part 2 input validation did not pass")
    require(inputs["part2_output_validation"].get("status") == "passed", "Part 2 output validation did not pass")
    require(len(state) == EXPECTED_STATE_ROWS, f"Unexpected state panel rows: {len(state)}")
    require(date_string(state["date"], "min") == EXPECTED_STATE_START, "Unexpected state start date")
    require(date_string(state["date"], "max") == EXPECTED_STATE_END, "Unexpected state end date")
    require(all(col in state.columns for col in ["date"] + PORTFOLIO_ASSETS), "Missing required return columns")

    for model_id in MODEL_IDS:
        labels = inputs["labels"][model_id]
        state_col = f"{model_id}_state"
        id_col = f"{model_id}_state_id"
        require(len(labels) == EXPECTED_STATE_ROWS, f"Unexpected label rows for {model_id}")
        require(date_string(labels["date"], "min") == EXPECTED_STATE_START, f"Unexpected label start for {model_id}")
        require(date_string(labels["date"], "max") == EXPECTED_STATE_END, f"Unexpected label end for {model_id}")
        require(all(col in labels.columns for col in ["date", state_col, id_col]), f"Missing label columns for {model_id}")
        profiles = inputs["profiles"][model_id]
        require(len(profiles) == int(inputs["model_robustness"].loc[inputs["model_robustness"]["model"].eq(model_id), "n_states"].iloc[0]), f"Unexpected profile count for {model_id}")

    base_weights = build_base_weights(inputs["part2_baseline_weights"])
    summary = {
        "status": "passed",
        "sample_frozen": True,
        "sample_rows": EXPECTED_STATE_ROWS,
        "sample_start": EXPECTED_STATE_START,
        "sample_end": EXPECTED_STATE_END,
        "model_ids": MODEL_IDS,
        "base_weights": base_weights,
        "risk_budget_cap": RISK_BUDGET_CAP,
        "btc_grid": ORIGINAL_BTC_GRID,
        "input_hashes": inputs["input_hashes"],
        "part1_run_id": inputs["part1_manifest"].get("run_id"),
        "part2_run_id": inputs["part2_manifest"].get("run_id"),
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


def parse_state_counts(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        return {str(k): int(v) for k, v in value.items()}
    parsed = ast.literal_eval(str(value))
    return {str(k): int(v) for k, v in parsed.items()}


def empirical_transition(labels: pd.Series) -> pd.DataFrame:
    states = sorted(labels.unique(), key=lambda x: int(str(x).split("_")[-1]))
    counts = pd.DataFrame(0.0, index=states, columns=states)
    prev = labels.iloc[:-1].to_numpy()
    nxt = labels.iloc[1:].to_numpy()
    for a, b in zip(prev, nxt):
        counts.loc[a, b] += 1.0
    row_sums = counts.sum(axis=1).replace(0.0, np.nan)
    probs = counts.div(row_sums, axis=0).fillna(0.0)
    return probs


def build_model_selection_summary(inputs: dict[str, Any], dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    robustness = inputs["model_robustness"].set_index("model")
    for model_id in MODEL_IDS:
        r = robustness.loc[model_id]
        labels = inputs["labels"][model_id][f"{model_id}_state"]
        state_counts = parse_state_counts(r["state_counts"])
        transition = inputs["transitions"].get(model_id, empirical_transition(labels))
        diag = np.diag(transition.to_numpy(dtype=float))
        min_state_count = min(state_counts.values())
        row = {
            "model_id": model_id,
            "model_family": r["method"],
            "n_states": int(r["n_states"]),
            "n_pcs": int(r["n_pca_components"]),
            "covariance_type": "" if pd.isna(r["covariance_type"]) else str(r["covariance_type"]),
            "converged": bool(r["converged"]),
            "n_iter": int(r["n_iter"]),
            "log_likelihood": float(r["log_likelihood"]) if pd.notna(r["log_likelihood"]) else np.nan,
            "aic": float(r["aic"]) if pd.notna(r["aic"]) else np.nan,
            "bic": float(r["bic"]) if pd.notna(r["bic"]) else np.nan,
            "min_state_count": int(min_state_count),
            "min_state_share": float(r["min_state_share"]),
            "max_state_share": float(r["max_state_share"]),
            "state_count_imbalance": float(max(state_counts.values()) / min_state_count),
            "mean_diagonal_transition_probability": float(np.nanmean(diag)),
            "min_diagonal_transition_probability": float(np.nanmin(diag)),
            "interpretability_flag": interpretability_flag(model_id, min_state_count, r),
            "main_model_candidate": bool(model_id == "hmm4"),
        }
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["model_family", "n_states"]).reset_index(drop=True)
    out.to_csv(dirs["results"] / "part11_model_selection_summary.csv", index=False)
    return out


def interpretability_flag(model_id: str, min_state_count: int, row: pd.Series) -> str:
    if model_id == "hmm4":
        return "main_descriptive_specification; compact but HMM5 has better information criteria"
    if model_id == "hmm5":
        return "best_hmm_aic_bic; adds split states and model-complexity risk"
    if model_id == "hmm3":
        return "coarser_state_map; lower fit than HMM4/HMM5"
    if model_id == "kmeans4":
        return "auxiliary_clustering; very small highest-stress cluster"
    return "auxiliary"


def build_state_profile_summary(inputs: dict[str, Any], dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for model_id in MODEL_IDS:
        profiles = inputs["profiles"][model_id].copy()
        for _, row in profiles.iterrows():
            rows.append(
                {
                    "model_id": model_id,
                    "state_label": row["state"],
                    "state_rank": int(row["state_id"]),
                    "state_count": int(row["n_weeks"]),
                    "state_share": float(row["sample_share"]),
                    "macro_stress_composite_mean": float(row["macro_stress_composite_mean"]),
                    "macro_vix_z_mean": safe_float(row, "macro_vix_z_mean"),
                    "macro_credit_spread_baa10y_z_mean": safe_float(row, "macro_credit_spread_baa10y_z_mean"),
                    "macro_adjusted_financial_conditions_z_mean": safe_float(row, "macro_adjusted_financial_conditions_z_mean"),
                    "macro_real_yield_10y_z_mean": safe_float(row, "macro_real_yield_10y_z_mean"),
                    "macro_dollar_chg_4w_z_mean": safe_float(row, "macro_dollar_chg_4w_z_mean"),
                    "macro_net_liquidity_chg_4w_z_mean": safe_float(row, "macro_net_liquidity_chg_4w_z_mean"),
                    "macro_yield_curve_10y_2y_z_mean": safe_float(row, "macro_yield_curve_10y_2y_z_mean"),
                    "profile_description": row.get("candidate_profile", ""),
                }
            )
    out = pd.DataFrame(rows).sort_values(["model_id", "state_rank"]).reset_index(drop=True)
    out.to_csv(dirs["results"] / "part11_state_profile_summary.csv", index=False)
    return out


def safe_float(row: pd.Series, col: str) -> float:
    return float(row[col]) if col in row and pd.notna(row[col]) else np.nan


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


def normalized_mutual_information(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    table = contingency(labels_a, labels_b).astype(float)
    n = table.sum()
    if n <= 0:
        return np.nan
    pxy = table / n
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = 0.0
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            if pxy[i, j] > 0:
                mi += pxy[i, j] * math.log(pxy[i, j] / (px[i] * py[j]))
    hx = -sum(p * math.log(p) for p in px if p > 0)
    hy = -sum(p * math.log(p) for p in py if p > 0)
    denom = math.sqrt(hx * hy)
    return float(mi / denom) if denom > FLOAT_TOL else 0.0


def greedy_overlap_agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> tuple[float, str]:
    table = contingency(labels_a, labels_b)
    remaining_rows = set(range(table.shape[0]))
    remaining_cols = set(range(table.shape[1]))
    matched = 0
    pairs = []
    while remaining_rows and remaining_cols:
        best = None
        for i in remaining_rows:
            for j in remaining_cols:
                value = table[i, j]
                if best is None or value > best[0]:
                    best = (value, i, j)
        if best is None:
            break
        value, i, j = best
        matched += int(value)
        pairs.append(f"{i}->{j}:{int(value)}")
        remaining_rows.remove(i)
        remaining_cols.remove(j)
    return float(matched / len(labels_a)), ";".join(pairs)


def build_state_mapping_stability(inputs: dict[str, Any], dirs: dict[str, Path]) -> pd.DataFrame:
    pairs = [("hmm3", "hmm4"), ("hmm4", "hmm5"), ("hmm4", "kmeans4")]
    rows = []
    for left, right in pairs:
        left_labels = inputs["labels"][left][f"{left}_state_id"].to_numpy()
        right_labels = inputs["labels"][right][f"{right}_state_id"].to_numpy()
        agreement, mapping_note = greedy_overlap_agreement(left_labels, right_labels)
        rows.append(
            {
                "left_model_id": left,
                "right_model_id": right,
                "agreement_metric": "greedy_max_overlap_share",
                "agreement_value": agreement,
                "adjusted_rand_index": adjusted_rand_index(left_labels, right_labels),
                "normalized_mutual_information": normalized_mutual_information(left_labels, right_labels),
                "mapping_method": "greedy_max_overlap",
                "notes": mapping_note,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(dirs["results"] / "part11_state_mapping_stability.csv", index=False)
    return out


def state_raw_weights(n_states: int) -> dict[str, float]:
    weights = {f"state_{i}": 0.0 for i in range(n_states)}
    weights["state_0"] = 0.03
    if n_states > 1:
        weights["state_1"] = 0.01
    return weights


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


def build_rule_results(inputs: dict[str, Any], validation: dict[str, Any], dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    state_returns = inputs["state"][["date"] + PORTFOLIO_ASSETS].copy()
    base_weights_all = validation["base_weights"]
    weight_rows = []
    perf_rows = []

    for model_id in MODEL_IDS:
        labels = inputs["labels"][model_id].copy()
        state_col = f"{model_id}_state"
        id_col = f"{model_id}_state_id"
        panel = state_returns.merge(labels[["date", state_col, id_col]], on="date", how="inner", validate="one_to_one")
        n_states = int(inputs["model_robustness"].loc[inputs["model_robustness"]["model"].eq(model_id), "n_states"].iloc[0])
        raw_map = state_raw_weights(n_states)
        for family in PORTFOLIO_FAMILIES:
            base_weights = base_weights_all[family]
            executed_map: dict[str, float] = {}
            for state, raw_weight in raw_map.items():
                selected, reason, audit = select_state_weight(panel, base_weights, state_col, state, raw_weight)
                executed_map[state] = selected
                weight_rows.append(
                    {
                        "model_id": model_id,
                        "rule_id": "common_stress_ranked_rule_cap_10pct",
                        "portfolio_family": family,
                        "risk_budget_cap": RISK_BUDGET_CAP,
                        "state_label": state,
                        "raw_btc_weight": raw_weight,
                        "selected_btc_weight": selected,
                        "adjustment_reason": reason,
                        "state_n_weeks": int((panel[state_col] == state).sum()),
                        **audit,
                    }
                )

            component_rows = []
            btc_weights = []
            for _, obs in panel.sort_values("date").iterrows():
                btc_weight = executed_map[str(obs[state_col])]
                weights = weights_from_btc(base_weights, btc_weight)
                btc_weights.append(btc_weight)
                component_rows.append([weights[asset] * float(obs[asset]) for asset in PORTFOLIO_ASSETS])
            component = np.asarray(component_rows, dtype=float)
            returns = pd.Series(component.sum(axis=1))
            perf = performance_metrics(returns)
            risk = risk_shares(component)
            perf_rows.append(
                {
                    "model_id": model_id,
                    "rule_id": "common_stress_ranked_rule_cap_10pct",
                    "portfolio_family": family,
                    "risk_budget_cap": RISK_BUDGET_CAP,
                    "average_btc_weight": float(np.mean(btc_weights)),
                    "max_btc_weight": float(np.max(btc_weights)),
                    "active_week_share": float(np.mean(np.asarray(btc_weights) > FLOAT_TOL)),
                    **perf,
                    "btc_share_vol": risk["btc_share_vol"],
                    "btc_share_cvar": risk["btc_share_cvar"],
                    "state_mapping_warning": model_warning(model_id),
                }
            )

    weights_out = pd.DataFrame(weight_rows).sort_values(["model_id", "portfolio_family", "state_label"]).reset_index(drop=True)
    perf_out = pd.DataFrame(perf_rows).sort_values(["portfolio_family", "model_id"]).reset_index(drop=True)
    weights_out.to_csv(dirs["results"] / "part11_rule_weights_by_model.csv", index=False)
    perf_out.to_csv(dirs["results"] / "part11_rule_performance_by_model.csv", index=False)
    return {"weights": weights_out, "performance": perf_out}


def model_warning(model_id: str) -> str:
    if model_id == "hmm5":
        return "HMM5 has better AIC/BIC but splits the HMM4 low/elevated stress structure."
    if model_id == "hmm3":
        return "HMM3 is coarser and merges some HMM4 profiles."
    if model_id == "kmeans4":
        return "KMeans4 has a very small highest-stress cluster and is auxiliary only."
    return "Main descriptive HMM4 specification."


def build_key_findings(selection: pd.DataFrame, stability: pd.DataFrame, rule_results: dict[str, pd.DataFrame], dirs: dict[str, Path]) -> dict[str, Any]:
    hmm_rows = selection[selection["model_id"].str.startswith("hmm")].copy()
    best_aic = hmm_rows.loc[hmm_rows["aic"].idxmin()]
    best_bic = hmm_rows.loc[hmm_rows["bic"].idxmin()]
    h4 = selection[selection["model_id"].eq("hmm4")].iloc[0]
    h5 = selection[selection["model_id"].eq("hmm5")].iloc[0]
    kmeans = selection[selection["model_id"].eq("kmeans4")].iloc[0]
    perf = rule_results["performance"]
    payload = {
        "best_hmm_by_aic": best_aic["model_id"],
        "best_hmm_by_bic": best_bic["model_id"],
        "hmm4_bic_minus_hmm5_bic": float(h4["bic"] - h5["bic"]),
        "hmm4_min_state_count": int(h4["min_state_count"]),
        "hmm5_min_state_count": int(h5["min_state_count"]),
        "kmeans4_min_state_count": int(kmeans["min_state_count"]),
        "hmm4_vs_hmm5_stability": stability[(stability["left_model_id"].eq("hmm4")) & (stability["right_model_id"].eq("hmm5"))].iloc[0].to_dict(),
        "rule_performance_snapshot": perf.to_dict(orient="records"),
        "recommended_main_model_statement": (
            "HMM5 is favored by AIC/BIC, so HMM4 should be defended as a compact interpretability choice rather than as the statistical best fit. "
            "The paper should report HMM5 as model-risk evidence and avoid saying HMM4 was selected by information criteria."
        ),
    }
    write_json(dirs["results"] / "part11_model_comparison_key_findings.json", normalize_for_json(payload))
    return payload


def make_figures(selection: pd.DataFrame, inputs: dict[str, Any], rule_results: dict[str, pd.DataFrame], dirs: dict[str, Path]) -> None:
    hmm = selection[selection["model_id"].str.startswith("hmm")].sort_values("n_states")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(hmm["n_states"], hmm["aic"], marker="o", label="AIC")
    ax.plot(hmm["n_states"], hmm["bic"], marker="s", label="BIC")
    ax.set_xlabel("Number of HMM states")
    ax.set_ylabel("Information criterion")
    ax.set_title("Part 11 HMM information criteria")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "part11_aic_bic_by_model.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(selection["model_id"], selection["min_state_share"] * 100)
    ax.set_ylabel("Minimum state share (%)")
    ax.set_title("Part 11 minimum state share by model")
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "part11_min_state_share_by_model.png", dpi=160)
    plt.close(fig)

    labels4 = inputs["labels"]["hmm4"][["date", "hmm4_state_id"]].copy()
    labels5 = inputs["labels"]["hmm5"][["date", "hmm5_state_id"]].copy()
    timeline = labels4.merge(labels5, on="date", how="inner")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.step(timeline["date"], timeline["hmm4_state_id"], where="post", label="HMM4", linewidth=1.2)
    ax.step(timeline["date"], timeline["hmm5_state_id"], where="post", label="HMM5", linewidth=1.2, alpha=0.75)
    ax.set_ylabel("Stress-ranked state id")
    ax.set_title("Part 11 HMM4 vs HMM5 state timeline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "part11_hmm4_vs_hmm5_state_timeline.png", dpi=160)
    plt.close(fig)

    perf = rule_results["performance"].copy()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(perf))
    ax.barh(x, perf["btc_share_vol"] * 100)
    ax.set_yticks(x)
    ax.set_yticklabels(perf["portfolio_family"] + " | " + perf["model_id"], fontsize=8)
    ax.set_xlabel("BTC volatility contribution (%)")
    ax.set_title("Part 11 rule result by model")
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "part11_rule_result_by_model.png", dpi=160)
    plt.close(fig)


def write_audits(args: argparse.Namespace, inputs: dict[str, Any], dirs: dict[str, Path]) -> None:
    lineage_rows = []
    for name, path in inputs["paths"].items():
        lineage_rows.append({"source_name": name, "path": str(path), "sha256": inputs["input_hashes"][name], "role": "frozen_input"})
    pd.DataFrame(lineage_rows).to_csv(dirs["results"] / "data_lineage.csv", index=False)

    assumption = {
        "status": "passed",
        "sample_frozen": True,
        "reestimates_models": False,
        "models_compared": MODEL_IDS,
        "rule_mapping": "state_0=3%, state_1=1%, all higher stress-ranked states=0%, then original-grid 10% BTC risk cap",
        "interpretation_boundary": "Part 11 audits frozen Part 1 model variants; it does not replace Part 1 or reselect a new main model.",
    }
    write_json(dirs["results"] / "model_assumption_audit.json", normalize_for_json(assumption))

    methodology = f"""# Part 11 Methodology Audit

Part 11 compares model variants already generated by Part 1: HMM-3, HMM-4, HMM-5, and KMeans-4.
It does not re-estimate PCA, HMM, or KMeans models. This keeps the audit tied to the frozen Part 1 evidence chain.

The allocation comparison applies one common stress-ranked raw rule to each model:
- state_0 = 3% BTC
- state_1 = 1% BTC
- all higher stress-ranked states = 0% BTC

The executed rule uses the original BTC grid {ORIGINAL_BTC_GRID} and a {RISK_BUDGET_CAP:.0%} BTC
volatility/CVaR contribution cap. The purpose is to test model dependence, not to optimize a rule for each model.

Interpretation boundary:
If HMM-5 has lower AIC/BIC than HMM-4, the thesis should not claim that HMM-4 is selected by information criteria.
HMM-4 can only be defended as a compact descriptive and interpretability choice, with HMM-5 reported as model-risk evidence.
"""
    (dirs["results"] / "methodology_audit.md").write_text(methodology, encoding="utf-8")


def validate_outputs(selection: pd.DataFrame, profiles: pd.DataFrame, stability: pd.DataFrame, rule_results: dict[str, pd.DataFrame], dirs: dict[str, Path]) -> dict[str, Any]:
    required_paths = [dirs["results"] / name for name in REQUIRED_RESULTS if name != "output_validation_summary.json"] + [
        dirs["figures"] / name for name in REQUIRED_FIGURES
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    require(not missing, f"Missing required output files: {missing}")
    require(set(selection["model_id"]) == set(MODEL_IDS), "Missing model selection rows")
    require(len(profiles) == sum(selection["n_states"]), "Unexpected profile row count")
    require(len(stability) == 3, "Unexpected stability row count")
    require(len(rule_results["performance"]) == len(MODEL_IDS) * len(PORTFOLIO_FAMILIES), "Unexpected rule performance row count")
    require(selection.loc[selection["model_id"].eq("hmm5"), "bic"].iloc[0] < selection.loc[selection["model_id"].eq("hmm4"), "bic"].iloc[0], "Expected HMM5 BIC to be lower than HMM4")

    summary = {
        "status": "passed",
        "model_rows": int(len(selection)),
        "profile_rows": int(len(profiles)),
        "stability_rows": int(len(stability)),
        "rule_performance_rows": int(len(rule_results["performance"])),
        "hmm5_bic_lower_than_hmm4": True,
        "required_files_present": True,
    }
    write_json(dirs["results"] / "output_validation_summary.json", normalize_for_json(summary))
    logging.info("Output validation passed")
    return summary


def write_manifest(args: argparse.Namespace, inputs: dict[str, Any], output_validation: dict[str, Any], dirs: dict[str, Path]) -> None:
    manifest = {
        "part_id": "part11_hmm_model_comparison",
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_frozen": True,
        "sample_end": EXPECTED_STATE_END,
        "seed": args.seed,
        "inputs": {name: str(path) for name, path in inputs["paths"].items()},
        "input_hashes": inputs["input_hashes"],
        "parameters": {
            "models_compared": MODEL_IDS,
            "risk_budget_cap": RISK_BUDGET_CAP,
            "btc_grid": ORIGINAL_BTC_GRID,
            "rule_mapping": "state_0=3%, state_1=1%, all higher states=0%",
            "reestimates_models": False,
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

    logging.info("Starting Part 11 run in %s", run_dir)
    inputs = read_inputs(args)
    enforce_resume_input_hashes(args, dirs, inputs["input_hashes"])
    validation = load_or_run(dirs, "01_input_validation", args.resume, lambda: validate_inputs(inputs, dirs))
    selection = load_or_run(dirs, "02_model_selection", args.resume, lambda: build_model_selection_summary(inputs, dirs))
    profiles = load_or_run(dirs, "03_state_profiles", args.resume, lambda: build_state_profile_summary(inputs, dirs))
    stability = load_or_run(dirs, "04_state_mapping_stability", args.resume, lambda: build_state_mapping_stability(inputs, dirs))
    rule_results = load_or_run(dirs, "05_rule_results", args.resume, lambda: build_rule_results(inputs, validation, dirs))
    build_key_findings(selection, stability, rule_results, dirs)
    make_figures(selection, inputs, rule_results, dirs)
    write_audits(args, inputs, dirs)
    output_validation = validate_outputs(selection, profiles, stability, rule_results, dirs)
    write_manifest(args, inputs, output_validation, dirs)
    logging.info("Part 11 completed successfully")


if __name__ == "__main__":
    main()
