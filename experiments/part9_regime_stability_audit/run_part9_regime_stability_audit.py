#!/usr/bin/env python3
"""Part 9 supplementary regime stability audit.

This runner does not estimate a new regime model and does not change Parts 1-8.
It explains where Part 7 pseudo real-time regime labels differ from Part 1
full-sample HMM labels, and whether those differences materially change BTC
allocation signals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_RESULTS = [
    "input_validation_summary.json",
    "regime_stability_overview.csv",
    "state_confusion_matrix_counts.csv",
    "state_confusion_matrix_row_shares.csv",
    "state_stability_by_part1_state.csv",
    "state_instability_timeline.csv",
    "state_instability_episodes.csv",
    "state_instability_by_year.csv",
    "rule_signal_instability_impact.csv",
    "rule_weight_delta_by_state_pair.csv",
    "risk_overlay_instability_impact.csv",
    "ensemble_stability_summary.csv",
    "ensemble_rule_sensitivity_summary.csv",
    "regime_stability_decision_matrix.csv",
    "methodology_audit.md",
    "data_lineage.csv",
    "model_assumption_audit.json",
    "output_validation_summary.json",
]

REQUIRED_FIGURES = [
    "state_confusion_heatmap.png",
    "state_agreement_timeline.png",
    "btc_weight_delta_by_state_pair.png",
    "ensemble_agreement_summary.png",
    "rule_signal_sensitivity.png",
    "instability_episode_lengths.png",
]

STATE_ORDER = ["state_0", "state_1", "state_2", "state_3"]
PROB_COLS = [f"realtime_prob_state_{i}" for i in range(4)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 9 regime stability audit.")
    parser.add_argument("--input-dir", default="data_2026/cleaned", type=Path)
    parser.add_argument("--part1-run-dir", default="outputs/part1_btc_macro_state/colab_part1_seed42", type=Path)
    parser.add_argument("--part2-run-dir", default="outputs/part2_portfolio_risk_budget/colab_part2_seed42", type=Path)
    parser.add_argument("--part3-run-dir", default="outputs/part3_btc_state_dependence/colab_part3_seed42", type=Path)
    parser.add_argument("--part4-run-dir", default="outputs/part4_conditional_btc_allocation/colab_part4_seed42", type=Path)
    parser.add_argument("--part5-run-dir", default="outputs/part5_implementability_rebalancing/colab_part5_seed42", type=Path)
    parser.add_argument("--part6-run-dir", default="outputs/part6_robustness_analysis/colab_part6_seed42", type=Path)
    parser.add_argument("--part7-run-dir", default="outputs/part7_realtime_probabilistic_regime_robustness/colab_part7_seed42", type=Path)
    parser.add_argument("--part8-run-dir", default="outputs/part8_uncertainty_quantification/colab_part8_seed42", type=Path)
    parser.add_argument("--output-dir", default="outputs/part9_regime_stability_audit", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def now_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def ensure_dirs(run_dir: Path) -> dict[str, Path]:
    dirs = {
        "run": run_dir,
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
            logging.FileHandler(log_dir / "part9_regime_stability_audit.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def save_checkpoint(dirs: dict[str, Path], name: str, payload: dict[str, Any]) -> None:
    pd.to_pickle(payload, dirs["checkpoints"] / f"{name}.pkl")
    logging.info("Saved checkpoint: %s", dirs["checkpoints"] / f"{name}.pkl")


def load_validation_status(part_dir: Path, filename: str) -> str:
    path = part_dir / "results" / filename
    payload = read_json(path)
    return str(payload.get("status", "missing"))


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    required = {
        "asset_returns_main_weekly": args.input_dir / "asset_returns_main_weekly.csv",
        "cleaning_report": args.input_dir / "cleaning_report.json",
        "part1_manifest": args.part1_run_dir / "run_manifest.json",
        "part1_validation": args.part1_run_dir / "results" / "validation_summary.json",
        "part1_labels": args.part1_run_dir / "results" / "hmm4_state_labels.csv",
        "part4_manifest": args.part4_run_dir / "run_manifest.json",
        "part4_output_validation": args.part4_run_dir / "results" / "output_validation_summary.json",
        "part4_rule_definition": args.part4_run_dir / "results" / "allocation_rule_definition.csv",
        "part7_manifest": args.part7_run_dir / "run_manifest.json",
        "part7_input_validation": args.part7_run_dir / "results" / "input_validation_summary.json",
        "part7_output_validation": args.part7_run_dir / "results" / "output_validation_summary.json",
        "part7_probabilities": args.part7_run_dir / "results" / "realtime_state_probabilities.csv",
        "part7_rule_signals": args.part7_run_dir / "results" / "realtime_rule_signal_series.csv",
        "part7_overlay": args.part7_run_dir / "results" / "risk_budget_overlay_audit.csv",
        "part8_manifest": args.part8_run_dir / "run_manifest.json",
        "part8_output_validation": args.part8_run_dir / "results" / "output_validation_summary.json",
        "part8_ensemble_agreement": args.part8_run_dir / "results" / "hmm_ensemble_state_agreement.csv",
        "part8_ensemble_signal": args.part8_run_dir / "results" / "hmm_ensemble_rule_signal_sensitivity.csv",
        "part8_ensemble_performance": args.part8_run_dir / "results" / "hmm_ensemble_rule_performance_sensitivity.csv",
    }
    optional_upstream = {
        "part2_manifest": args.part2_run_dir / "run_manifest.json",
        "part2_output_validation": args.part2_run_dir / "results" / "output_validation_summary.json",
        "part3_manifest": args.part3_run_dir / "run_manifest.json",
        "part3_output_validation": args.part3_run_dir / "results" / "output_validation_summary.json",
        "part5_manifest": args.part5_run_dir / "run_manifest.json",
        "part5_output_validation": args.part5_run_dir / "results" / "output_validation_summary.json",
        "part6_manifest": args.part6_run_dir / "run_manifest.json",
        "part6_output_validation": args.part6_run_dir / "results" / "output_validation_summary.json",
    }

    missing = [name for name, path in {**required, **optional_upstream}.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")

    statuses = {
        "part1_validation": load_validation_status(args.part1_run_dir, "validation_summary.json"),
        "part4_output_validation": load_validation_status(args.part4_run_dir, "output_validation_summary.json"),
        "part7_input_validation": load_validation_status(args.part7_run_dir, "input_validation_summary.json"),
        "part7_output_validation": load_validation_status(args.part7_run_dir, "output_validation_summary.json"),
        "part8_output_validation": load_validation_status(args.part8_run_dir, "output_validation_summary.json"),
    }
    for label, path in [
        ("part2_output_validation", optional_upstream["part2_output_validation"]),
        ("part3_output_validation", optional_upstream["part3_output_validation"]),
        ("part5_output_validation", optional_upstream["part5_output_validation"]),
        ("part6_output_validation", optional_upstream["part6_output_validation"]),
    ]:
        statuses[label] = read_json(path).get("status", "missing")

    failed = {name: status for name, status in statuses.items() if status != "passed"}
    if failed:
        raise ValueError(f"Upstream validation did not pass: {failed}")

    part7_manifest = read_json(args.part7_run_dir / "run_manifest.json")
    part8_manifest = read_json(args.part8_run_dir / "run_manifest.json")
    sample = part7_manifest.get("sample", {})
    if sample.get("probability_rows") != 269 or sample.get("lagged_return_rows") != 268:
        raise ValueError(f"Unexpected Part 7 sample sizes: {sample}")

    input_hashes = {name: sha256_file(path) for name, path in {**required, **optional_upstream}.items()}
    validation = {
        "status": "passed",
        "role": "supplementary_regime_stability_audit",
        "input_dir": str(args.input_dir),
        "part7_probability_sample": {
            "rows": sample.get("probability_rows"),
            "start": sample.get("probability_start"),
            "end": sample.get("probability_end"),
        },
        "part7_lagged_return_sample": {
            "rows": sample.get("lagged_return_rows"),
            "start": sample.get("lagged_return_start"),
            "end": sample.get("lagged_return_end"),
        },
        "upstream_validation_status": statuses,
        "part8_bootstrap_reps": part8_manifest.get("parameters", {}).get("bootstrap_reps"),
        "part8_hmm_ensemble_variants": part8_manifest.get("parameters", {}).get("hmm_ensemble_variants"),
        "input_hashes": input_hashes,
    }
    return validation


def normalize_state_matrix(df: pd.DataFrame, index_col: str, col_col: str, value_col: str = "rows") -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = (
        df.pivot_table(index=index_col, columns=col_col, values=value_col, aggfunc="sum", fill_value=0, observed=False)
        .reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)
        .astype(int)
    )
    row_sums = counts.sum(axis=1).replace(0, np.nan)
    row_shares = counts.div(row_sums, axis=0).fillna(0.0)
    return counts, row_shares


def build_state_stability(prob: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = prob.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["year"] = work["date"].dt.year
    work["state_match"] = work["part1_hmm4_state"] == work["realtime_map_state"]
    work["probability_margin"] = work[PROB_COLS].apply(lambda row: np.sort(row.values)[-1] - np.sort(row.values)[-2], axis=1)
    work["part1_hmm4_state"] = pd.Categorical(work["part1_hmm4_state"], categories=STATE_ORDER, ordered=True)
    work["realtime_map_state"] = pd.Categorical(work["realtime_map_state"], categories=STATE_ORDER, ordered=True)

    pair_counts = (
        work.groupby(["part1_hmm4_state", "realtime_map_state"], observed=False)
        .size()
        .reset_index(name="rows")
    )
    counts, row_shares = normalize_state_matrix(pair_counts, "part1_hmm4_state", "realtime_map_state")

    stability_rows = []
    for state in STATE_ORDER:
        sub = work[work["part1_hmm4_state"] == state]
        if sub.empty:
            stability_rows.append(
                {
                    "part1_state": state,
                    "rows": 0,
                    "agreement_rate": np.nan,
                    "dominant_realtime_state": None,
                    "dominant_realtime_share": np.nan,
                    "average_max_probability": np.nan,
                    "average_normalized_entropy": np.nan,
                    "average_probability_margin": np.nan,
                    "low_confidence_share": np.nan,
                }
            )
            continue
        dominant = sub["realtime_map_state"].value_counts().sort_values(ascending=False)
        stability_rows.append(
            {
                "part1_state": state,
                "rows": int(len(sub)),
                "agreement_rate": float(sub["state_match"].mean()),
                "dominant_realtime_state": str(dominant.index[0]),
                "dominant_realtime_share": float(dominant.iloc[0] / len(sub)),
                "average_max_probability": float(sub["realtime_max_probability"].mean()),
                "average_normalized_entropy": float(sub["realtime_normalized_entropy"].mean()),
                "average_probability_margin": float(sub["probability_margin"].mean()),
                "low_confidence_share": float(sub["low_confidence_flag"].mean()),
            }
        )
    by_state = pd.DataFrame(stability_rows)

    by_year = (
        work.groupby("year")
        .agg(
            rows=("date", "count"),
            agreement_rate=("state_match", "mean"),
            average_max_probability=("realtime_max_probability", "mean"),
            average_normalized_entropy=("realtime_normalized_entropy", "mean"),
            low_confidence_share=("low_confidence_flag", "mean"),
        )
        .reset_index()
    )

    timeline = work[
        [
            "date",
            "training_end_date",
            "refit_index",
            "part1_hmm4_state",
            "realtime_map_state",
            "state_match",
            "realtime_max_probability",
            "realtime_normalized_entropy",
            "probability_margin",
            "low_confidence_flag",
            *PROB_COLS,
        ]
    ].copy()
    timeline["part1_hmm4_state"] = timeline["part1_hmm4_state"].astype(str)
    timeline["realtime_map_state"] = timeline["realtime_map_state"].astype(str)
    timeline["date"] = timeline["date"].dt.strftime("%Y-%m-%d")

    episodes = build_disagreement_episodes(work)
    overview = pd.DataFrame(
        [
            {
                "metric": "probability_rows",
                "value": len(work),
                "interpretation": "Part 7 decision-date probability rows used for state stability audit.",
            },
            {
                "metric": "overall_agreement_rate",
                "value": float(work["state_match"].mean()),
                "interpretation": "Share of Part 7 real-time MAP states matching Part 1 full-sample states.",
            },
            {
                "metric": "disagreement_rate",
                "value": float(1.0 - work["state_match"].mean()),
                "interpretation": "Share of decision dates where pseudo real-time and ex-post labels differ.",
            },
            {
                "metric": "average_max_probability",
                "value": float(work["realtime_max_probability"].mean()),
                "interpretation": "Average posterior confidence of the real-time model.",
            },
            {
                "metric": "low_confidence_share",
                "value": float(work["low_confidence_flag"].mean()),
                "interpretation": "Share flagged as low posterior confidence by Part 7.",
            },
            {
                "metric": "disagreement_episode_count",
                "value": int(len(episodes)),
                "interpretation": "Number of contiguous disagreement episodes.",
            },
            {
                "metric": "max_disagreement_episode_weeks",
                "value": int(episodes["episode_weeks"].max()) if not episodes.empty else 0,
                "interpretation": "Longest contiguous disagreement episode in weeks.",
            },
        ]
    )

    return {
        "overview": overview,
        "counts": counts.reset_index().rename(columns={"part1_hmm4_state": "part1_state"}),
        "row_shares": row_shares.reset_index().rename(columns={"part1_hmm4_state": "part1_state"}),
        "by_state": by_state,
        "timeline": timeline,
        "episodes": episodes,
        "by_year": by_year,
        "work": work,
    }


def build_disagreement_episodes(work: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = work.sort_values("date").reset_index(drop=True)
    active = None
    for _, row in sub.iterrows():
        is_disagree = not bool(row["state_match"])
        if is_disagree and active is None:
            active = {
                "start_date": row["date"],
                "end_date": row["date"],
                "dates": [row["date"]],
                "part1_states": [str(row["part1_hmm4_state"])],
                "realtime_states": [str(row["realtime_map_state"])],
                "max_probs": [float(row["realtime_max_probability"])],
                "entropies": [float(row["realtime_normalized_entropy"])],
            }
        elif is_disagree and active is not None:
            active["end_date"] = row["date"]
            active["dates"].append(row["date"])
            active["part1_states"].append(str(row["part1_hmm4_state"]))
            active["realtime_states"].append(str(row["realtime_map_state"]))
            active["max_probs"].append(float(row["realtime_max_probability"]))
            active["entropies"].append(float(row["realtime_normalized_entropy"]))
        elif not is_disagree and active is not None:
            rows.append(summarize_episode(active))
            active = None
    if active is not None:
        rows.append(summarize_episode(active))
    return pd.DataFrame(rows)


def summarize_episode(active: dict[str, Any]) -> dict[str, Any]:
    pair_series = pd.Series(list(zip(active["part1_states"], active["realtime_states"])))
    dominant_pair = pair_series.value_counts().index[0]
    return {
        "episode_id": None,
        "start_date": active["start_date"].strftime("%Y-%m-%d"),
        "end_date": active["end_date"].strftime("%Y-%m-%d"),
        "episode_weeks": len(active["dates"]),
        "dominant_part1_state": dominant_pair[0],
        "dominant_realtime_state": dominant_pair[1],
        "average_max_probability": float(np.mean(active["max_probs"])),
        "average_normalized_entropy": float(np.mean(active["entropies"])),
    }


def add_episode_ids(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame(
            columns=[
                "episode_id",
                "start_date",
                "end_date",
                "episode_weeks",
                "dominant_part1_state",
                "dominant_realtime_state",
                "average_max_probability",
                "average_normalized_entropy",
            ]
        )
    out = episodes.copy()
    out["episode_id"] = np.arange(1, len(out) + 1)
    return out[
        [
            "episode_id",
            "start_date",
            "end_date",
            "episode_weeks",
            "dominant_part1_state",
            "dominant_realtime_state",
            "average_max_probability",
            "average_normalized_entropy",
        ]
    ]


def build_rule_weight_lookup(rule_definition: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    executed = rule_definition[rule_definition["constraint_stage"] == "risk_budget_executed"].copy()
    lookup = {}
    for _, row in executed.iterrows():
        lookup[(row["rule_id"], row["portfolio_family"], row["hmm4_state"])] = float(row["selected_btc_weight"])
    return lookup


def build_rule_impact(rule_signals: pd.DataFrame, rule_definition: pd.DataFrame) -> dict[str, pd.DataFrame]:
    signals = rule_signals.copy()
    signals["decision_date"] = pd.to_datetime(signals["decision_date"])
    signals["return_date"] = pd.to_datetime(signals["return_date"])
    signals["state_match"] = signals["part1_hmm4_state"] == signals["realtime_map_state"]

    lookup = build_rule_weight_lookup(rule_definition)
    signals["part1_hard_label_btc_weight"] = [
        lookup.get((row.rule_id, row.portfolio_family, row.part1_hmm4_state), np.nan)
        for row in signals.itertuples(index=False)
    ]
    signals["candidate_minus_part1_hard_weight"] = signals["candidate_btc_weight"] - signals["part1_hard_label_btc_weight"]
    signals["final_minus_part1_hard_weight"] = signals["final_btc_weight"] - signals["part1_hard_label_btc_weight"]
    signals["absolute_final_weight_delta"] = signals["final_minus_part1_hard_weight"].abs()
    signals["absolute_candidate_weight_delta"] = signals["candidate_minus_part1_hard_weight"].abs()

    group_cols = ["rule_type", "rule_id", "portfolio_family", "funding_convention"]
    impact = (
        signals.groupby(group_cols)
        .agg(
            rows=("return_date", "count"),
            disagreement_share=("state_match", lambda s: float(1.0 - s.mean())),
            average_candidate_btc_weight=("candidate_btc_weight", "mean"),
            average_final_btc_weight=("final_btc_weight", "mean"),
            average_part1_hard_label_btc_weight=("part1_hard_label_btc_weight", "mean"),
            average_candidate_minus_part1_hard_weight=("candidate_minus_part1_hard_weight", "mean"),
            average_final_minus_part1_hard_weight=("final_minus_part1_hard_weight", "mean"),
            average_absolute_final_weight_delta=("absolute_final_weight_delta", "mean"),
            max_absolute_final_weight_delta=("absolute_final_weight_delta", "max"),
            active_final_weight_share=("final_btc_weight", lambda s: float((s > 1e-12).mean())),
        )
        .reset_index()
    )

    pair = (
        signals.groupby(group_cols + ["part1_hmm4_state", "realtime_map_state"])
        .agg(
            rows=("return_date", "count"),
            average_candidate_btc_weight=("candidate_btc_weight", "mean"),
            average_final_btc_weight=("final_btc_weight", "mean"),
            average_part1_hard_label_btc_weight=("part1_hard_label_btc_weight", "mean"),
            average_final_minus_part1_hard_weight=("final_minus_part1_hard_weight", "mean"),
            average_absolute_final_weight_delta=("absolute_final_weight_delta", "mean"),
            max_absolute_final_weight_delta=("absolute_final_weight_delta", "max"),
        )
        .reset_index()
        .sort_values(group_cols + ["part1_hmm4_state", "realtime_map_state"])
    )

    return {"impact": impact, "pair": pair, "signals": signals}


def build_overlay_impact(overlay: pd.DataFrame) -> pd.DataFrame:
    work = overlay.copy()
    work["state_match"] = work["part1_hmm4_state"] == work["realtime_map_state"]
    work["weight_reduction"] = work["candidate_btc_weight"] - work["final_btc_weight"]
    work["material_weight_reduction"] = work["weight_reduction"] > 1e-6
    work["vol_cap_margin"] = work["risk_budget_cap"] - work["final_btc_component_share_vol"]
    work["cvar_cap_margin"] = work["risk_budget_cap"] - work["final_btc_component_share_cvar"]
    group_cols = ["rule_type", "rule_id", "portfolio_family", "funding_convention", "state_match"]
    out = (
        work.groupby(group_cols)
        .agg(
            rows=("return_date", "count"),
            average_candidate_btc_weight=("candidate_btc_weight", "mean"),
            average_final_btc_weight=("final_btc_weight", "mean"),
            max_final_btc_weight=("final_btc_weight", "max"),
            average_weight_reduction=("weight_reduction", "mean"),
            max_weight_reduction=("weight_reduction", "max"),
            material_overlay_reduced_share=("material_weight_reduction", "mean"),
            average_final_vol_share=("final_btc_component_share_vol", "mean"),
            max_final_vol_share=("final_btc_component_share_vol", "max"),
            average_final_cvar_share=("final_btc_component_share_cvar", "mean"),
            max_final_cvar_share=("final_btc_component_share_cvar", "max"),
            min_vol_cap_margin=("vol_cap_margin", "min"),
            min_cvar_cap_margin=("cvar_cap_margin", "min"),
        )
        .reset_index()
    )
    out["state_match_group"] = np.where(out["state_match"], "matched_state_dates", "mismatched_state_dates")
    return out.drop(columns=["state_match"])


def build_ensemble_summary(agreement: pd.DataFrame, signal: pd.DataFrame, performance: pd.DataFrame) -> dict[str, pd.DataFrame]:
    stability = pd.DataFrame(
        [
            {
                "metric": "variant_count",
                "value": int(len(agreement)),
                "interpretation": "Number of Part 8 HMM ensemble variants.",
            },
            {
                "metric": "agreement_min",
                "value": float(agreement["overall_agreement_with_part1"].min()),
                "interpretation": "Lowest agreement with Part 1 full-sample labels across variants.",
            },
            {
                "metric": "agreement_max",
                "value": float(agreement["overall_agreement_with_part1"].max()),
                "interpretation": "Highest agreement with Part 1 full-sample labels across variants.",
            },
            {
                "metric": "agreement_range",
                "value": float(agreement["overall_agreement_with_part1"].max() - agreement["overall_agreement_with_part1"].min()),
                "interpretation": "Dispersion in label agreement across seed/window/refit variants.",
            },
            {
                "metric": "state_2_count_max",
                "value": int(agreement["state_2_count"].max()),
                "interpretation": "State 2 remains rare in pseudo real-time variants.",
            },
            {
                "metric": "average_max_probability_min",
                "value": float(agreement["average_max_probability"].min()),
                "interpretation": "Posterior confidence remains high even when ex-post label agreement is low.",
            },
        ]
    )

    signal_summary = (
        signal.groupby("rule_id")
        .agg(
            variants=("variant_id", "count"),
            average_candidate_weight_min=("average_candidate_btc_weight", "min"),
            average_candidate_weight_max=("average_candidate_btc_weight", "max"),
            average_candidate_weight_mean=("average_candidate_btc_weight", "mean"),
            active_candidate_share_min=("active_candidate_share", "min"),
            active_candidate_share_max=("active_candidate_share", "max"),
            active_candidate_share_mean=("active_candidate_share", "mean"),
            max_candidate_weight_max=("max_candidate_btc_weight", "max"),
        )
        .reset_index()
    )
    perf_summary = (
        performance.groupby(["rule_id", "portfolio_family"])
        .agg(
            variants=("variant_id", "count"),
            annualized_mean_min=("annualized_mean_arithmetic", "min"),
            annualized_mean_max=("annualized_mean_arithmetic", "max"),
            annualized_mean_range=("annualized_mean_arithmetic", lambda s: float(s.max() - s.min())),
            annualized_volatility_min=("annualized_volatility", "min"),
            annualized_volatility_max=("annualized_volatility", "max"),
            max_drawdown_min=("max_drawdown", "min"),
            max_drawdown_max=("max_drawdown", "max"),
            sharpe_min=("sharpe_annualized_zero_rf", "min"),
            sharpe_max=("sharpe_annualized_zero_rf", "max"),
        )
        .reset_index()
    )
    return {"stability": stability, "signal": signal_summary, "performance": perf_summary}


def build_decision_matrix(
    overview: pd.DataFrame,
    by_state: pd.DataFrame,
    rule_impact: pd.DataFrame,
    overlay_impact: pd.DataFrame,
    ensemble_stability: pd.DataFrame,
) -> pd.DataFrame:
    agreement = float(overview.loc[overview["metric"] == "overall_agreement_rate", "value"].iloc[0])
    longest_episode = float(overview.loc[overview["metric"] == "max_disagreement_episode_weeks", "value"].iloc[0])
    max_weight_delta = float(rule_impact["max_absolute_final_weight_delta"].max())
    max_overlay_share = float(overlay_impact["material_overlay_reduced_share"].max())
    agreement_min = float(ensemble_stability.loc[ensemble_stability["metric"] == "agreement_min", "value"].iloc[0])
    rows = [
        {
            "evidence_area": "state_label_stability",
            "diagnostic": "part7_vs_part1_agreement",
            "status": "caution" if agreement < 0.75 else "pass",
            "key_value": agreement,
            "note": "Pseudo real-time labels differ materially from full-sample labels; do not present full-sample labels as stable forecasts.",
        },
        {
            "evidence_area": "state_drift_duration",
            "diagnostic": "longest_disagreement_episode_weeks",
            "status": "caution" if longest_episode >= 8 else "pass",
            "key_value": longest_episode,
            "note": "Contiguous disagreement episodes are documented to show when label drift persists.",
        },
        {
            "evidence_area": "allocation_signal_impact",
            "diagnostic": "max_absolute_btc_weight_delta_vs_part1_hard_label",
            "status": "pass" if max_weight_delta <= 0.03 else "caution",
            "key_value": max_weight_delta,
            "note": "State drift affects BTC signal timing, but the small allocation grid keeps absolute BTC weight changes bounded.",
        },
        {
            "evidence_area": "risk_overlay_dependency",
            "diagnostic": "material_overlay_reduced_share",
            "status": "pass" if max_overlay_share < 0.05 else "caution",
            "key_value": max_overlay_share,
            "note": "Material overlay reductions use a 1e-6 BTC-weight threshold to avoid counting numerical zeroing as economic deleveraging.",
        },
        {
            "evidence_area": "ensemble_state_drift",
            "diagnostic": "minimum_ensemble_agreement",
            "status": "caution" if agreement_min < 0.70 else "pass",
            "key_value": agreement_min,
            "note": "Ensemble variants confirm that label drift is not a single-seed artifact.",
        },
    ]
    return pd.DataFrame(rows)


def plot_confusion(row_shares: pd.DataFrame, fig_path: Path) -> None:
    matrix = row_shares.set_index("part1_state")[STATE_ORDER].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xticks(range(4), STATE_ORDER)
    ax.set_yticks(range(4), STATE_ORDER)
    ax.set_xlabel("Pseudo real-time MAP state")
    ax.set_ylabel("Part 1 full-sample HMM state")
    ax.set_title("State confusion matrix row shares")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def plot_agreement_timeline(timeline: pd.DataFrame, fig_path: Path) -> None:
    work = timeline.copy()
    work["date"] = pd.to_datetime(work["date"])
    match = work["state_match"].astype(int)
    rolling = match.rolling(26, min_periods=4).mean()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(work["date"], rolling, color="#1f77b4", linewidth=2, label="26-week rolling agreement")
    ax.scatter(work.loc[~work["state_match"], "date"], np.zeros((~work["state_match"]).sum()), s=8, color="#d62728", alpha=0.55, label="disagreement dates")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Agreement rate")
    ax.set_title("Pseudo real-time vs full-sample state agreement")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def plot_weight_delta(pair: pd.DataFrame, fig_path: Path) -> None:
    filt = (
        (pair["rule_type"] == "realtime_probability_weighted_overlay")
        & (pair["rule_id"] == "main_executed")
        & (pair["portfolio_family"] == "all_weather")
        & (pair["funding_convention"] == "pro_rata_base")
    )
    data = pair.loc[filt].copy()
    data["state_pair"] = data["part1_hmm4_state"] + " -> " + data["realtime_map_state"]
    data = data.sort_values("average_absolute_final_weight_delta", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(data["state_pair"], data["average_absolute_final_weight_delta"], color="#9467bd")
    ax.invert_yaxis()
    ax.set_xlabel("Average absolute BTC weight delta")
    ax.set_title("BTC weight impact of state label differences")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def plot_ensemble_agreement(agreement: pd.DataFrame, fig_path: Path) -> None:
    data = agreement.sort_values("overall_agreement_with_part1")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(data["variant_id"], data["overall_agreement_with_part1"], color="#2ca02c")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Agreement with Part 1 full-sample labels")
    ax.set_title("HMM ensemble state agreement")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def plot_rule_signal_sensitivity(signal_summary: pd.DataFrame, fig_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(signal_summary))
    lower = signal_summary["average_candidate_weight_mean"] - signal_summary["average_candidate_weight_min"]
    upper = signal_summary["average_candidate_weight_max"] - signal_summary["average_candidate_weight_mean"]
    ax.errorbar(
        x,
        signal_summary["average_candidate_weight_mean"],
        yerr=[lower, upper],
        fmt="o",
        color="#ff7f0e",
        capsize=5,
    )
    ax.set_xticks(x, signal_summary["rule_id"], rotation=20, ha="right")
    ax.set_ylabel("Average candidate BTC weight")
    ax.set_title("Ensemble sensitivity of BTC rule signals")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def plot_episode_lengths(episodes: pd.DataFrame, fig_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    if episodes.empty:
        ax.text(0.5, 0.5, "No disagreement episodes", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(episodes["episode_weeks"], bins=range(1, int(episodes["episode_weeks"].max()) + 3), color="#8c564b", edgecolor="white")
        ax.set_xlabel("Episode length in weeks")
        ax.set_ylabel("Episode count")
        ax.set_title("Distribution of state disagreement episode lengths")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def save_outputs(
    dirs: dict[str, Path],
    validation: dict[str, Any],
    state_outputs: dict[str, pd.DataFrame],
    rule_outputs: dict[str, pd.DataFrame],
    overlay_impact: pd.DataFrame,
    ensemble_outputs: dict[str, pd.DataFrame],
    ensemble_agreement: pd.DataFrame,
    decision_matrix: pd.DataFrame,
) -> None:
    results = dirs["results"]
    write_json(results / "input_validation_summary.json", validation)
    state_outputs["overview"].to_csv(results / "regime_stability_overview.csv", index=False)
    state_outputs["counts"].to_csv(results / "state_confusion_matrix_counts.csv", index=False)
    state_outputs["row_shares"].to_csv(results / "state_confusion_matrix_row_shares.csv", index=False)
    state_outputs["by_state"].to_csv(results / "state_stability_by_part1_state.csv", index=False)
    state_outputs["timeline"].to_csv(results / "state_instability_timeline.csv", index=False)
    add_episode_ids(state_outputs["episodes"]).to_csv(results / "state_instability_episodes.csv", index=False)
    state_outputs["by_year"].to_csv(results / "state_instability_by_year.csv", index=False)
    rule_outputs["impact"].to_csv(results / "rule_signal_instability_impact.csv", index=False)
    rule_outputs["pair"].to_csv(results / "rule_weight_delta_by_state_pair.csv", index=False)
    overlay_impact.to_csv(results / "risk_overlay_instability_impact.csv", index=False)
    ensemble_outputs["stability"].to_csv(results / "ensemble_stability_summary.csv", index=False)
    ensemble_outputs["signal"].to_csv(results / "ensemble_rule_sensitivity_summary.csv", index=False)
    decision_matrix.to_csv(results / "regime_stability_decision_matrix.csv", index=False)

    methodology = (
        "# Part 9 Methodology Audit\n\n"
        "Part 9 is a supplementary audit of regime stability. It does not re-estimate PCA/HMM models, "
        "does not search for new BTC allocation rules, and does not modify Parts 1-8.\n\n"
        "The audit compares Part 7 pseudo real-time MAP states with Part 1 full-sample HMM labels, "
        "documents disagreement episodes, measures how state disagreement changes BTC allocation signals, "
        "and summarizes Part 8 HMM ensemble sensitivity.\n\n"
        "The outputs are intended for thesis caveats and supplementary tables, not as a new strategy result.\n"
    )
    (results / "methodology_audit.md").write_text(methodology, encoding="utf-8")

    lineage = []
    for name, digest in validation["input_hashes"].items():
        lineage.append({"source": name, "sha256": digest, "role": "input_or_upstream_lineage"})
    pd.DataFrame(lineage).to_csv(results / "data_lineage.csv", index=False)

    assumptions = {
        "experiment_role": "supplementary_regime_stability_audit",
        "new_model_estimated": False,
        "new_allocation_rule_created": False,
        "upstream_outputs_modified": False,
        "primary_state_comparison": "part7_pseudo_realtime_map_state_vs_part1_full_sample_hmm_state",
        "interpretation_limit": "Documents instability and bounded allocation impact; does not prove real-time point-in-time validity.",
    }
    write_json(results / "model_assumption_audit.json", assumptions)

    plot_confusion(state_outputs["row_shares"], dirs["figures"] / "state_confusion_heatmap.png")
    plot_agreement_timeline(state_outputs["timeline"], dirs["figures"] / "state_agreement_timeline.png")
    plot_weight_delta(rule_outputs["pair"], dirs["figures"] / "btc_weight_delta_by_state_pair.png")
    plot_ensemble_agreement(ensemble_agreement, dirs["figures"] / "ensemble_agreement_summary.png")
    plot_rule_signal_sensitivity(ensemble_outputs["signal"], dirs["figures"] / "rule_signal_sensitivity.png")
    plot_episode_lengths(add_episode_ids(state_outputs["episodes"]), dirs["figures"] / "instability_episode_lengths.png")


def validate_outputs(dirs: dict[str, Path], state_outputs: dict[str, pd.DataFrame], decision_matrix: pd.DataFrame) -> dict[str, Any]:
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

    timeline = state_outputs["timeline"]
    overview = state_outputs["overview"]
    agreement = float(overview.loc[overview["metric"] == "overall_agreement_rate", "value"].iloc[0])
    checks = {
        "required_results_ok": all(row["exists"] and row["nonempty"] for row in result_checks if row["file"] != "output_validation_summary.json"),
        "required_figures_ok": all(row["exists"] and row["nonempty"] and row["readable"] for row in figure_checks),
        "probability_rows_ok": int(len(timeline)) == 269,
        "state_match_values_ok": set(timeline["state_match"].dropna().unique()).issubset({True, False}),
        "agreement_rate_bounded": 0.0 <= agreement <= 1.0,
        "decision_matrix_nonempty": len(decision_matrix) >= 5,
    }
    status = "passed" if all(checks.values()) else "failed"
    return {
        "status": status,
        "checks": checks,
        "required_result_checks": result_checks,
        "required_figure_checks": figure_checks,
    }


def write_manifest(
    dirs: dict[str, Path],
    args: argparse.Namespace,
    run_id: str,
    validation: dict[str, Any],
    output_validation: dict[str, Any],
) -> None:
    manifest = {
        "created_at": datetime.now().replace(microsecond=0).isoformat(),
        "run_id": run_id,
        "objective": "Part 9 supplementary regime stability audit",
        "input_dir": str(args.input_dir),
        "part1_run_dir": str(args.part1_run_dir),
        "part2_run_dir": str(args.part2_run_dir),
        "part3_run_dir": str(args.part3_run_dir),
        "part4_run_dir": str(args.part4_run_dir),
        "part5_run_dir": str(args.part5_run_dir),
        "part6_run_dir": str(args.part6_run_dir),
        "part7_run_dir": str(args.part7_run_dir),
        "part8_run_dir": str(args.part8_run_dir),
        "output_dir": str(args.output_dir),
        "run_dir": str(dirs["run"]),
        "random_seed": args.seed,
        "package_versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "input_hashes": validation["input_hashes"],
        "sample": {
            "part7_probability_rows": validation["part7_probability_sample"]["rows"],
            "part7_probability_start": validation["part7_probability_sample"]["start"],
            "part7_probability_end": validation["part7_probability_sample"]["end"],
            "part7_lagged_return_rows": validation["part7_lagged_return_sample"]["rows"],
            "part7_lagged_return_start": validation["part7_lagged_return_sample"]["start"],
            "part7_lagged_return_end": validation["part7_lagged_return_sample"]["end"],
        },
        "output_validation": output_validation,
        "outputs": {"results": REQUIRED_RESULTS, "figures": REQUIRED_FIGURES},
        "scope_notes": [
            "Supplementary audit only; no new allocation rule.",
            "Uses Part 7/8 outputs to explain state drift and allocation impact.",
            "Does not claim point-in-time macro vintage validity.",
        ],
    }
    write_json(dirs["run"] / "run_manifest.json", manifest)


def main() -> None:
    args = parse_args()
    run_id = args.run_id or now_run_id()
    dirs = ensure_dirs(args.output_dir / run_id)
    setup_logging(dirs["logs"])
    logging.info("Starting Part 9 run_id=%s", run_id)

    validation = validate_inputs(args)
    save_checkpoint(dirs, "01_input_validation", validation)

    prob = pd.read_csv(args.part7_run_dir / "results" / "realtime_state_probabilities.csv")
    rule_signals = pd.read_csv(args.part7_run_dir / "results" / "realtime_rule_signal_series.csv")
    overlay = pd.read_csv(args.part7_run_dir / "results" / "risk_budget_overlay_audit.csv")
    rule_definition = pd.read_csv(args.part4_run_dir / "results" / "allocation_rule_definition.csv")
    ensemble_agreement = pd.read_csv(args.part8_run_dir / "results" / "hmm_ensemble_state_agreement.csv")
    ensemble_signal = pd.read_csv(args.part8_run_dir / "results" / "hmm_ensemble_rule_signal_sensitivity.csv")
    ensemble_perf = pd.read_csv(args.part8_run_dir / "results" / "hmm_ensemble_rule_performance_sensitivity.csv")

    state_outputs = build_state_stability(prob)
    save_checkpoint(dirs, "02_state_stability", {k: v for k, v in state_outputs.items() if k != "work"})

    rule_outputs = build_rule_impact(rule_signals, rule_definition)
    overlay_impact = build_overlay_impact(overlay)
    save_checkpoint(dirs, "03_rule_and_overlay_impact", {"rule": rule_outputs["impact"], "pair": rule_outputs["pair"], "overlay": overlay_impact})

    ensemble_outputs = build_ensemble_summary(ensemble_agreement, ensemble_signal, ensemble_perf)
    decision_matrix = build_decision_matrix(
        state_outputs["overview"],
        state_outputs["by_state"],
        rule_outputs["impact"],
        overlay_impact,
        ensemble_outputs["stability"],
    )
    save_checkpoint(dirs, "04_ensemble_and_decision_matrix", {"ensemble": ensemble_outputs, "decision_matrix": decision_matrix})

    save_outputs(dirs, validation, state_outputs, rule_outputs, overlay_impact, ensemble_outputs, ensemble_agreement, decision_matrix)
    output_validation = validate_outputs(dirs, state_outputs, decision_matrix)
    write_json(dirs["results"] / "output_validation_summary.json", output_validation)
    write_manifest(dirs, args, run_id, validation, output_validation)
    if output_validation["status"] != "passed":
        raise RuntimeError(f"Part 9 output validation failed: {output_validation}")
    logging.info("Part 9 completed successfully: %s", dirs["run"])


if __name__ == "__main__":
    main()
