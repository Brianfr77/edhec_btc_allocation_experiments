#!/usr/bin/env python3
"""Part 7 runner: pseudo real-time probabilistic regime robustness.

The runner extends the Part 1-6 evidence chain without mutating upstream
outputs. It estimates expanding-window PCA/HMM state probabilities using only
historical macro predictors, translates state uncertainty into BTC allocation
signals, applies an ex-ante risk-budget overlay, and evaluates implementability
with the same broad conventions used in Part 5.
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
from dataclasses import dataclass
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
EXPECTED_INITIAL_TRAIN_ROWS = 156
EXPECTED_INITIAL_TRAIN_END = "2021-01-29"
EXPECTED_FIRST_PROBABILITY_DATE = "2021-02-05"
EXPECTED_FIRST_RETURN_DATE = "2021-02-12"
EXPECTED_STATE_COUNTS = {"state_0": 149, "state_1": 167, "state_2": 30, "state_3": 79}

N_STATES = 4
N_COMPONENTS = 5
HMM_N_INIT = 10
HMM_MAX_ITER = 300
HMM_TOL = 1e-5
TAIL_ALPHA = 0.05
TRADING_WEEKS_PER_YEAR = 52
RISK_BUDGET_CAP = 0.10
RISK_BUDGET_TOL = 1e-6
LOW_CONFIDENCE_THRESHOLD = 0.60
FLOAT_TOL = 1e-10

BASE_ASSETS = ["ret_spy", "ret_tlt", "ret_ief", "ret_gld", "ret_dbc"]
ASSETS = ["ret_btc"] + BASE_ASSETS + ["ret_bil"]
IMPLEMENTED_RULE_IDS = ["main_executed", "sensitivity_state2_low_executed"]
RULE_TYPES = ["realtime_probability_weighted_overlay", "realtime_hard_map_overlay"]
PORTFOLIO_FAMILIES = ["all_weather", "erc"]
FUNDING_CONVENTIONS = ["pro_rata_base", "bil_sleeve"]
REBALANCE_FREQUENCIES = ["monthly", "quarterly"]
MAIN_RULE_TYPE = "realtime_probability_weighted_overlay"
MAIN_COST_SCENARIO = "moderate_cost"
COST_SCENARIOS = {"moderate_cost": {"ret_btc": 0.0025, "etf_and_bil": 0.0005}}

REQUIRED_RESULTS = [
    "input_validation_summary.json",
    "realtime_model_refit_calendar.csv",
    "realtime_model_refit_diagnostics.csv",
    "realtime_state_probabilities.csv",
    "realtime_state_uncertainty_summary.csv",
    "state_label_agreement_summary.csv",
    "state_profile_drift_summary.csv",
    "realtime_rule_signal_series.csv",
    "risk_budget_overlay_audit.csv",
    "realtime_weekly_target_weights.csv",
    "realtime_target_weight_return_series.csv",
    "realtime_target_weight_performance_summary.csv",
    "realtime_rebalance_calendar.csv",
    "realtime_rebalanced_weight_series.csv",
    "realtime_rebalanced_return_series.csv",
    "realtime_rebalanced_performance_summary.csv",
    "realtime_turnover_cost_summary.csv",
    "realtime_vs_expost_part5_comparison.csv",
    "full_sample_zscore_bridge_summary.csv",
    "methodology_audit.md",
    "data_lineage.csv",
    "model_assumption_audit.json",
    "output_validation_summary.json",
]
REQUIRED_FIGURES = [
    "state_probability_timeline.png",
    "state_uncertainty.png",
    "state_label_agreement_heatmap.png",
    "realtime_btc_weight_timeline.png",
    "risk_budget_overlay_audit.png",
    "realtime_drawdown_performance_comparison.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 7 pseudo real-time probabilistic regime robustness.")
    parser.add_argument("--input-dir", default="data_2026/cleaned", type=Path)
    parser.add_argument("--part1-run-dir", default="outputs/part1_btc_macro_state/colab_part1_seed42", type=Path)
    parser.add_argument("--part2-run-dir", default="outputs/part2_portfolio_risk_budget/colab_part2_seed42", type=Path)
    parser.add_argument("--part3-run-dir", default="outputs/part3_btc_state_dependence/colab_part3_seed42", type=Path)
    parser.add_argument("--part4-run-dir", default="outputs/part4_conditional_btc_allocation/colab_part4_seed42", type=Path)
    parser.add_argument("--part5-run-dir", default="outputs/part5_implementability_rebalancing/colab_part5_seed42", type=Path)
    parser.add_argument("--part6-run-dir", default="outputs/part6_robustness_analysis/colab_part6_seed42", type=Path)
    parser.add_argument("--output-dir", default="outputs/part7_realtime_probabilistic_regime_robustness", type=Path)
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
    var_value = float(clean.quantile(alpha))
    tail = clean[clean <= var_value]
    cvar_value = float(tail.mean()) if len(tail) else var_value
    return var_value, cvar_value, int(len(tail))


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1.0


def performance_metrics(returns: pd.Series, prefix: str = "", periods_per_year: int = TRADING_WEEKS_PER_YEAR) -> dict[str, Any]:
    clean = returns.dropna()
    var_value, cvar_value, tail_count = var_cvar(clean)
    mean = float(clean.mean()) if len(clean) else float("nan")
    vol = float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")
    sharpe = mean / vol * math.sqrt(periods_per_year) if vol and vol > 0 else float("nan")
    return {
        f"{prefix}count": int(len(clean)),
        f"{prefix}mean_weekly": mean,
        f"{prefix}median_weekly": float(clean.median()) if len(clean) else float("nan"),
        f"{prefix}volatility_weekly": vol,
        f"{prefix}annualized_mean_arithmetic": mean * periods_per_year if math.isfinite(mean) else float("nan"),
        f"{prefix}annualized_volatility": vol * math.sqrt(periods_per_year) if math.isfinite(vol) else float("nan"),
        f"{prefix}var_95_weekly": var_value,
        f"{prefix}cvar_95_weekly": cvar_value,
        f"{prefix}tail_scenario_count": tail_count,
        f"{prefix}max_drawdown": float(drawdown_series(clean).min()) if len(clean) else float("nan"),
        f"{prefix}positive_week_share": float((clean > 0).mean()) if len(clean) else float("nan"),
        f"{prefix}sharpe_annualized_zero_rf": sharpe,
    }


def cost_rates_for_scenario(cost_scenario: str) -> dict[str, float]:
    spec = COST_SCENARIOS[cost_scenario]
    return {asset: (spec["ret_btc"] if asset == "ret_btc" else spec["etf_and_bil"]) for asset in ASSETS}


@dataclass
class KMeansResult:
    n_clusters: int
    centers: np.ndarray
    labels: np.ndarray
    inertia: float
    n_iter: int
    seed: int


def squared_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)


def kmeans_plus_plus_init(x: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    centers = np.empty((n_clusters, x.shape[1]))
    first = int(rng.integers(0, x.shape[0]))
    centers[0] = x[first]
    closest_dist = squared_distances(x, centers[[0]]).reshape(-1)
    for i in range(1, n_clusters):
        total = float(closest_dist.sum())
        if total <= 0:
            centers[i] = x[int(rng.integers(0, x.shape[0]))]
            continue
        probabilities = closest_dist / total
        index = int(rng.choice(x.shape[0], p=probabilities))
        centers[i] = x[index]
        closest_dist = np.minimum(closest_dist, squared_distances(x, centers[[i]]).reshape(-1))
    return centers


def kmeans_fit(x: np.ndarray, n_clusters: int, seed: int, n_init: int = 10, max_iter: int = 200, tol: float = 1e-8) -> KMeansResult:
    best: KMeansResult | None = None
    for init_idx in range(n_init):
        rng = np.random.default_rng(seed + init_idx * 1297)
        centers = kmeans_plus_plus_init(x, n_clusters, rng)
        labels = np.zeros(x.shape[0], dtype=int)
        for iteration in range(1, max_iter + 1):
            distances = squared_distances(x, centers)
            new_labels = distances.argmin(axis=1)
            new_centers = centers.copy()
            for cluster in range(n_clusters):
                members = x[new_labels == cluster]
                if len(members):
                    new_centers[cluster] = members.mean(axis=0)
            shift = float(np.sqrt(((new_centers - centers) ** 2).sum()))
            centers = new_centers
            labels = new_labels
            if shift < tol:
                break
        inertia = float(squared_distances(x, centers).min(axis=1).sum())
        result = KMeansResult(n_clusters, centers, labels, inertia, iteration, seed + init_idx * 1297)
        if best is None or result.inertia < best.inertia:
            best = result
    require(best is not None, "KMeans failed to initialize")
    return best


@dataclass
class HMMResult:
    n_states: int
    n_features: int
    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    covars: np.ndarray
    log_likelihood: float
    converged: bool
    n_iter: int
    seed: int
    gamma: np.ndarray
    labels: np.ndarray
    aic: float
    bic: float
    init_history: list[dict[str, Any]]


def logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    stable = np.exp(values - max_value)
    summed = np.sum(stable, axis=axis, keepdims=True)
    result = max_value + np.log(summed)
    if axis is not None:
        result = np.squeeze(result, axis=axis)
    return result


def gaussian_log_prob_diag(x: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    covars = np.maximum(covars, 1e-6)
    diff = x[:, None, :] - means[None, :, :]
    log_det = np.log(covars).sum(axis=1)
    quadratic = ((diff**2) / covars[None, :, :]).sum(axis=2)
    return -0.5 * (x.shape[1] * np.log(2.0 * np.pi) + log_det[None, :] + quadratic)


def forward_backward(
    x: np.ndarray,
    startprob: np.ndarray,
    transmat: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    n_obs = x.shape[0]
    n_states = startprob.shape[0]
    log_start = np.log(np.maximum(startprob, 1e-300))
    log_trans = np.log(np.maximum(transmat, 1e-300))
    log_emission = gaussian_log_prob_diag(x, means, covars)
    log_alpha = np.empty((n_obs, n_states))
    log_alpha[0] = log_start + log_emission[0]
    for t in range(1, n_obs):
        log_alpha[t] = log_emission[t] + logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)
    log_likelihood = float(logsumexp(log_alpha[-1], axis=0))
    log_beta = np.zeros((n_obs, n_states))
    for t in range(n_obs - 2, -1, -1):
        log_beta[t] = logsumexp(log_trans + log_emission[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)
    log_gamma = log_alpha + log_beta - log_likelihood
    gamma = np.exp(log_gamma)
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi_sum = np.zeros((n_states, n_states))
    for t in range(n_obs - 1):
        log_xi = (
            log_alpha[t][:, None]
            + log_trans
            + log_emission[t + 1][None, :]
            + log_beta[t + 1][None, :]
            - log_likelihood
        )
        xi_sum += np.exp(log_xi)
    return log_likelihood, gamma, xi_sum


def viterbi(x: np.ndarray, startprob: np.ndarray, transmat: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    n_obs = x.shape[0]
    n_states = startprob.shape[0]
    log_start = np.log(np.maximum(startprob, 1e-300))
    log_trans = np.log(np.maximum(transmat, 1e-300))
    log_emission = gaussian_log_prob_diag(x, means, covars)
    delta = np.empty((n_obs, n_states))
    psi = np.zeros((n_obs, n_states), dtype=int)
    delta[0] = log_start + log_emission[0]
    for t in range(1, n_obs):
        scores = delta[t - 1][:, None] + log_trans
        psi[t] = scores.argmax(axis=0)
        delta[t] = scores.max(axis=0) + log_emission[t]
    labels = np.empty(n_obs, dtype=int)
    labels[-1] = int(delta[-1].argmax())
    for t in range(n_obs - 2, -1, -1):
        labels[t] = psi[t + 1, labels[t + 1]]
    return labels


def hmm_parameter_count(n_states: int, n_features: int) -> int:
    return (n_states - 1) + n_states * (n_states - 1) + 2 * n_states * n_features


def initialize_hmm_from_kmeans(x: np.ndarray, n_states: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    km = kmeans_fit(x, n_states, seed=seed, n_init=5, max_iter=100)
    labels = km.labels
    startprob = np.full(n_states, 1.0 / n_states)
    transmat = np.full((n_states, n_states), 1e-2)
    for left, right in zip(labels[:-1], labels[1:]):
        transmat[left, right] += 1.0
    transmat /= transmat.sum(axis=1, keepdims=True)
    means = np.zeros((n_states, x.shape[1]))
    covars = np.zeros((n_states, x.shape[1]))
    global_var = np.var(x, axis=0) + 1e-3
    for state in range(n_states):
        members = x[labels == state]
        if len(members) == 0:
            means[state] = x.mean(axis=0)
            covars[state] = global_var
        else:
            means[state] = members.mean(axis=0)
            covars[state] = np.maximum(members.var(axis=0), 1e-3)
    return startprob, transmat, means, covars


def fit_diag_gaussian_hmm(
    x: np.ndarray,
    n_states: int,
    seed: int,
    n_init: int = HMM_N_INIT,
    max_iter: int = HMM_MAX_ITER,
    tol: float = HMM_TOL,
) -> HMMResult:
    best: HMMResult | None = None
    init_history: list[dict[str, Any]] = []
    n_features = x.shape[1]
    for init_idx in range(n_init):
        init_seed = seed + init_idx * 7919 + n_states * 101
        startprob, transmat, means, covars = initialize_hmm_from_kmeans(x, n_states, init_seed)
        previous_ll = -np.inf
        converged = False
        gamma = np.empty((x.shape[0], n_states))
        iteration = 0
        for iteration in range(1, max_iter + 1):
            log_likelihood, gamma, xi_sum = forward_backward(x, startprob, transmat, means, covars)
            weights = gamma.sum(axis=0) + 1e-12
            startprob = gamma[0] + 1e-3
            startprob /= startprob.sum()
            transmat = xi_sum + 1e-3
            transmat /= transmat.sum(axis=1, keepdims=True)
            means = (gamma.T @ x) / weights[:, None]
            for state in range(n_states):
                diff = x - means[state]
                covars[state] = (gamma[:, state][:, None] * diff**2).sum(axis=0) / weights[state]
            covars = np.maximum(covars, 1e-5)
            if abs(log_likelihood - previous_ll) < tol:
                converged = True
                break
            previous_ll = log_likelihood
        log_likelihood, gamma, _ = forward_backward(x, startprob, transmat, means, covars)
        labels = viterbi(x, startprob, transmat, means, covars)
        param_count = hmm_parameter_count(n_states, n_features)
        aic = -2.0 * log_likelihood + 2.0 * param_count
        bic = -2.0 * log_likelihood + math.log(x.shape[0]) * param_count
        result = HMMResult(
            n_states=n_states,
            n_features=n_features,
            startprob=startprob,
            transmat=transmat,
            means=means,
            covars=covars,
            log_likelihood=log_likelihood,
            converged=converged,
            n_iter=iteration,
            seed=init_seed,
            gamma=gamma,
            labels=labels,
            aic=float(aic),
            bic=float(bic),
            init_history=[],
        )
        init_history.append(
            {
                "init_index": init_idx,
                "seed": init_seed,
                "log_likelihood": float(log_likelihood),
                "aic": float(aic),
                "bic": float(bic),
                "converged": bool(converged),
                "n_iter": int(iteration),
            }
        )
        if best is None or result.log_likelihood > best.log_likelihood:
            best = result
    require(best is not None, "HMM failed to initialize")
    best.init_history = [{**row, "selected_best": bool(row["seed"] == best.seed)} for row in init_history]
    return best


def stress_composite_from_z(z_frame: pd.DataFrame) -> pd.Series:
    additive = [
        "macro_vix_z",
        "macro_credit_spread_baa10y_z",
        "macro_adjusted_financial_conditions_z",
        "macro_real_yield_10y_z",
        "macro_dollar_chg_4w_z",
    ]
    subtractive = ["macro_net_liquidity_chg_4w_z", "macro_yield_curve_10y_2y_z"]
    components = pd.Series(0.0, index=z_frame.index)
    for col in additive:
        components += z_frame[col]
    for col in subtractive:
        components -= z_frame[col]
    return components / float(len(additive) + len(subtractive))


def reorder_hmm_by_stress(hmm: HMMResult, stress: pd.Series) -> tuple[HMMResult, dict[int, int]]:
    old_state_stress = pd.DataFrame({"old_state": hmm.labels, "stress": stress}).groupby("old_state")["stress"].mean()
    order = list(old_state_stress.sort_values().index)
    mapping = {old_state: new_state for new_state, old_state in enumerate(order)}
    inverse_order = np.array(order, dtype=int)
    new_labels = np.array([mapping[int(label)] for label in hmm.labels], dtype=int)
    reordered = HMMResult(
        n_states=hmm.n_states,
        n_features=hmm.n_features,
        startprob=hmm.startprob[inverse_order],
        transmat=hmm.transmat[np.ix_(inverse_order, inverse_order)],
        means=hmm.means[inverse_order],
        covars=hmm.covars[inverse_order],
        log_likelihood=hmm.log_likelihood,
        converged=hmm.converged,
        n_iter=hmm.n_iter,
        seed=hmm.seed,
        gamma=hmm.gamma[:, inverse_order],
        labels=new_labels,
        aic=hmm.aic,
        bic=hmm.bic,
        init_history=hmm.init_history,
    )
    return reordered, mapping


def fit_pca(z_values: np.ndarray, n_components: int = N_COMPONENTS) -> dict[str, Any]:
    mean = z_values.mean(axis=0)
    centered = z_values - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    scores = centered @ components.T
    explained_variance = (singular_values**2) / (z_values.shape[0] - 1)
    explained_ratio = explained_variance / explained_variance.sum()
    return {
        "mean": mean,
        "components": components,
        "scores": scores,
        "explained_variance_ratio": explained_ratio[:n_components],
        "cumulative_explained_variance_ratio": np.cumsum(explained_ratio[:n_components]),
    }


def pca_transform(z_values: np.ndarray, pca: dict[str, Any]) -> np.ndarray:
    return (z_values - pca["mean"]) @ pca["components"].T


def normalize_probabilities(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0:
        return np.full_like(values, 1.0 / len(values), dtype=float)
    return values / total


def filtered_update(previous_posterior: np.ndarray, observation_score: np.ndarray, hmm: HMMResult) -> np.ndarray:
    prior = previous_posterior @ hmm.transmat
    log_emission = gaussian_log_prob_diag(observation_score.reshape(1, -1), hmm.means, hmm.covars).reshape(-1)
    unnormalized = prior * np.exp(log_emission - np.max(log_emission))
    return normalize_probabilities(unnormalized)


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "asset": args.input_dir / "asset_returns_main_weekly.csv",
        "state": args.input_dir / "state_model_panel_weekly.csv",
        "cleaning_report": args.input_dir / "cleaning_report.json",
        "part1_manifest": args.part1_run_dir / "run_manifest.json",
        "part1_validation": args.part1_run_dir / "results" / "validation_summary.json",
        "part1_hmm_labels": args.part1_run_dir / "results" / "hmm4_state_labels.csv",
        "part1_hmm_profiles": args.part1_run_dir / "results" / "hmm4_state_profiles.csv",
        "part2_manifest": args.part2_run_dir / "run_manifest.json",
        "part2_input_validation": args.part2_run_dir / "results" / "input_validation_summary.json",
        "part2_output_validation": args.part2_run_dir / "results" / "output_validation_summary.json",
        "part2_baseline_weights": args.part2_run_dir / "results" / "baseline_portfolio_weights.csv",
        "part3_manifest": args.part3_run_dir / "run_manifest.json",
        "part3_output_validation": args.part3_run_dir / "results" / "output_validation_summary.json",
        "part4_manifest": args.part4_run_dir / "run_manifest.json",
        "part4_input_validation": args.part4_run_dir / "results" / "input_validation_summary.json",
        "part4_output_validation": args.part4_run_dir / "results" / "output_validation_summary.json",
        "part4_rule_definition": args.part4_run_dir / "results" / "allocation_rule_definition.csv",
        "part5_manifest": args.part5_run_dir / "run_manifest.json",
        "part5_input_validation": args.part5_run_dir / "results" / "input_validation_summary.json",
        "part5_output_validation": args.part5_run_dir / "results" / "output_validation_summary.json",
        "part5_rebalanced_returns": args.part5_run_dir / "results" / "rebalanced_portfolio_return_series.csv",
        "part6_manifest": args.part6_run_dir / "run_manifest.json",
        "part6_input_validation": args.part6_run_dir / "results" / "input_validation_summary.json",
        "part6_output_validation": args.part6_run_dir / "results" / "output_validation_summary.json",
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
        "part1_hmm_labels": pd.read_csv(paths["part1_hmm_labels"], parse_dates=["date"]),
        "part1_hmm_profiles": pd.read_csv(paths["part1_hmm_profiles"]),
        "part2_manifest": read_json(paths["part2_manifest"]),
        "part2_input_validation": read_json(paths["part2_input_validation"]),
        "part2_output_validation": read_json(paths["part2_output_validation"]),
        "part2_baseline_weights": pd.read_csv(paths["part2_baseline_weights"]),
        "part3_manifest": read_json(paths["part3_manifest"]),
        "part3_output_validation": read_json(paths["part3_output_validation"]),
        "part4_manifest": read_json(paths["part4_manifest"]),
        "part4_input_validation": read_json(paths["part4_input_validation"]),
        "part4_output_validation": read_json(paths["part4_output_validation"]),
        "part4_rule_definition": pd.read_csv(paths["part4_rule_definition"]),
        "part5_manifest": read_json(paths["part5_manifest"]),
        "part5_input_validation": read_json(paths["part5_input_validation"]),
        "part5_output_validation": read_json(paths["part5_output_validation"]),
        "part5_rebalanced_returns": pd.read_csv(paths["part5_rebalanced_returns"], parse_dates=["date"]),
        "part6_manifest": read_json(paths["part6_manifest"]),
        "part6_input_validation": read_json(paths["part6_input_validation"]),
        "part6_output_validation": read_json(paths["part6_output_validation"]),
    }


def validation_status(payload: dict[str, Any]) -> str:
    if "status" in payload:
        return str(payload["status"])
    if payload.get("validation", {}).get("status"):
        return str(payload["validation"]["status"])
    return ""


def build_input_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {name: file_sha256(path) for name, path in paths.items()}


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
    asset = inputs["asset"].copy()
    state = inputs["state"].copy()
    labels = inputs["part1_hmm_labels"].copy()
    rules = inputs["part4_rule_definition"].copy()
    report = inputs["cleaning_report"]
    paths = inputs["paths"]

    require(len(asset) == EXPECTED_ASSET_ROWS, "Unexpected asset row count")
    require(date_string(asset["date"], "min") == EXPECTED_ASSET_START, "Unexpected asset start date")
    require(date_string(asset["date"], "max") == EXPECTED_ASSET_END, "Unexpected asset end date")
    require(len(state) == EXPECTED_STATE_ROWS, "Unexpected state panel row count")
    require(date_string(state["date"], "min") == EXPECTED_STATE_START, "Unexpected state panel start date")
    require(date_string(state["date"], "max") == EXPECTED_STATE_END, "Unexpected state panel end date")
    require(len(labels) == EXPECTED_STATE_ROWS, "Unexpected Part 1 HMM label row count")
    require(date_string(labels["date"], "min") == EXPECTED_STATE_START, "Unexpected HMM label start date")
    require(date_string(labels["date"], "max") == EXPECTED_STATE_END, "Unexpected HMM label end date")
    state_counts = labels["hmm4_state"].value_counts().sort_index().to_dict()
    require(state_counts == EXPECTED_STATE_COUNTS, f"Unexpected HMM state counts: {state_counts}")

    state_predictor_map = report["column_mapping"]["state_predictors"]
    raw_predictor_cols = list(state_predictor_map.values())
    z_predictor_cols = [f"{col}_z" for col in raw_predictor_cols]
    for col in raw_predictor_cols + z_predictor_cols + ASSETS:
        require(col in state.columns, f"Missing state panel column: {col}")
        require(not state[col].isna().any(), f"Missing values in state panel column: {col}")

    cleaned_hashes = {
        "asset_returns_main_weekly": file_sha256(paths["asset"]),
        "state_model_panel_weekly": file_sha256(paths["state"]),
        "cleaning_report": file_sha256(paths["cleaning_report"]),
    }
    part1_hashes = inputs["part1_manifest"].get("input_hashes", {})
    for key, value in cleaned_hashes.items():
        require(part1_hashes.get(key) == value, f"Cleaned input hash mismatch versus Part 1 manifest: {key}")

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
    }
    for name, payload in validation_payloads.items():
        require(validation_status(payload) == "passed", f"Upstream validation did not pass: {name}")

    executed = rules[rules["rule_id"].isin(IMPLEMENTED_RULE_IDS)].copy()
    require(len(executed) == 16, "Part 4 executed rule table must contain 16 rows")
    require(set(executed["portfolio_family"]) == set(PORTFOLIO_FAMILIES), "Unexpected Part 4 portfolio families")
    require(set(executed["hmm4_state"]) == set(EXPECTED_STATE_COUNTS), "Unexpected Part 4 rule states")

    panel = state[["date"] + ASSETS + raw_predictor_cols + z_predictor_cols].merge(
        labels[
            [
                "date",
                "hmm4_state",
                "hmm4_state_id",
                "hmm4_state_posterior_probability",
                "hmm4_prob_state_0",
                "hmm4_prob_state_1",
                "hmm4_prob_state_2",
                "hmm4_prob_state_3",
            ]
        ],
        on="date",
        how="inner",
        validate="one_to_one",
    )
    require(len(panel) == EXPECTED_STATE_ROWS, "State/label inner join lost rows")

    summary = {
        "status": "passed",
        "asset_sample": {"rows": int(len(asset)), "start": date_string(asset["date"], "min"), "end": date_string(asset["date"], "max")},
        "state_sample": {"rows": int(len(state)), "start": date_string(state["date"], "min"), "end": date_string(state["date"], "max")},
        "pseudo_realtime_sample": {
            "initial_train_rows": EXPECTED_INITIAL_TRAIN_ROWS,
            "initial_train_start": EXPECTED_STATE_START,
            "initial_train_end": EXPECTED_INITIAL_TRAIN_END,
            "first_probability_date": EXPECTED_FIRST_PROBABILITY_DATE,
            "first_lagged_return_date": EXPECTED_FIRST_RETURN_DATE,
        },
        "state_counts": state_counts,
        "raw_predictor_columns": raw_predictor_cols,
        "z_predictor_columns_bridge_only": z_predictor_cols,
        "cleaned_hashes": cleaned_hashes,
        "upstream_validation_status": {name: validation_status(payload) for name, payload in validation_payloads.items()},
        "upstream_hashes": {name: file_sha256(path) for name, path in paths.items() if name.startswith("part")},
    }
    write_json(dirs["results"] / "input_validation_summary.json", summary)
    logging.info("Input validation passed")
    return {"summary": summary, "analysis_panel": panel, "raw_predictor_cols": raw_predictor_cols, "z_predictor_cols": z_predictor_cols}


def build_base_weights(inputs: dict[str, Any]) -> dict[str, dict[str, float]]:
    baseline = inputs["part2_baseline_weights"].copy()
    result: dict[str, dict[str, float]] = {}
    for family, frame in baseline.groupby("portfolio_family"):
        result[family] = {str(row["asset"]): float(row["weight"]) for _, row in frame.iterrows() if row["asset"] in BASE_ASSETS}
    require(set(result) == set(PORTFOLIO_FAMILIES), "Missing baseline portfolio family weights")
    for family, weights in result.items():
        require(set(weights) == set(BASE_ASSETS), f"Missing base assets for {family}")
        require(abs(sum(weights.values()) - 1.0) <= FLOAT_TOL, f"Base weights do not sum to 1 for {family}")
    return result


def build_refit_calendar(panel: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    initial_train_end = pd.Timestamp(EXPECTED_INITIAL_TRAIN_END)
    require(panel.iloc[EXPECTED_INITIAL_TRAIN_ROWS - 1]["date"] == initial_train_end, "Initial training window end mismatch")
    month_end = panel.groupby(panel["date"].dt.to_period("M")).tail(1)["date"].reset_index(drop=True)
    training_ends = [initial_train_end] + [pd.Timestamp(date) for date in month_end if pd.Timestamp(date) > initial_train_end]
    rows = []
    for idx, train_end in enumerate(training_ends):
        next_train_end = training_ends[idx + 1] if idx + 1 < len(training_ends) else pd.NaT
        probability_dates = panel[(panel["date"] > train_end) & (panel["date"] <= (next_train_end if pd.notna(next_train_end) else panel["date"].max()))][
            "date"
        ]
        rows.append(
            {
                "refit_index": idx,
                "training_end_date": train_end.strftime("%Y-%m-%d"),
                "next_training_end_date": "" if pd.isna(next_train_end) else next_train_end.strftime("%Y-%m-%d"),
                "training_rows": int((panel["date"] <= train_end).sum()),
                "first_probability_date": "" if probability_dates.empty else pd.Timestamp(probability_dates.min()).strftime("%Y-%m-%d"),
                "last_probability_date": "" if probability_dates.empty else pd.Timestamp(probability_dates.max()).strftime("%Y-%m-%d"),
                "probability_rows": int(len(probability_dates)),
            }
        )
    calendar = pd.DataFrame(rows)
    calendar.to_csv(dirs["results"] / "realtime_model_refit_calendar.csv", index=False)
    return calendar


def realtime_standardize(train_raw: np.ndarray, current_raw: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    means = train_raw.mean(axis=0)
    stds = train_raw.std(axis=0, ddof=0)
    stds = np.where(stds <= 1e-12, 1.0, stds)
    train_z = (train_raw - means) / stds
    current_z = None if current_raw is None else (current_raw - means) / stds
    return train_z, current_z, means, stds


def run_realtime_state_model(inputs: dict[str, Any], validation: dict[str, Any], args: argparse.Namespace, dirs: dict[str, Path]) -> dict[str, Any]:
    panel = validation["analysis_panel"].copy()
    raw_cols = validation["raw_predictor_cols"]
    z_cols = validation["z_predictor_cols"]
    refit_calendar = build_refit_calendar(panel, dirs)
    probability_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    realtime_z_rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    full_profiles = inputs["part1_hmm_profiles"].set_index("state")

    for _, calendar_row in refit_calendar.iterrows():
        train_end = pd.Timestamp(calendar_row["training_end_date"])
        train = panel[panel["date"] <= train_end].copy().reset_index(drop=True)
        if int(calendar_row["probability_rows"]) == 0:
            continue
        train_raw = train[raw_cols].to_numpy(dtype=float)
        train_z, _, scaler_mean, scaler_std = realtime_standardize(train_raw)
        train_z_frame = pd.DataFrame(train_z, columns=z_cols)
        stress = stress_composite_from_z(train_z_frame)
        pca = fit_pca(train_z, N_COMPONENTS)
        hmm_raw = fit_diag_gaussian_hmm(pca["scores"], N_STATES, seed=args.seed + int(calendar_row["refit_index"]) * 104729)
        hmm, mapping = reorder_hmm_by_stress(hmm_raw, stress)
        state_masses = hmm.gamma.sum(axis=0)
        state_counts = pd.Series(hmm.labels).value_counts().to_dict()
        posterior = hmm.gamma[-1].copy()

        diagnostic_rows.append(
            {
                "refit_index": int(calendar_row["refit_index"]),
                "training_end_date": train_end.strftime("%Y-%m-%d"),
                "training_rows": int(len(train)),
                "hmm_seed": int(hmm.seed),
                "converged": bool(hmm.converged),
                "n_iter": int(hmm.n_iter),
                "log_likelihood": float(hmm.log_likelihood),
                "aic": float(hmm.aic),
                "bic": float(hmm.bic),
                "pca_cumulative_explained_variance_5": float(pca["cumulative_explained_variance_ratio"][-1]),
                "state_0_posterior_mass": float(state_masses[0]),
                "state_1_posterior_mass": float(state_masses[1]),
                "state_2_posterior_mass": float(state_masses[2]),
                "state_3_posterior_mass": float(state_masses[3]),
                "state_0_viterbi_count": int(state_counts.get(0, 0)),
                "state_1_viterbi_count": int(state_counts.get(1, 0)),
                "state_2_viterbi_count": int(state_counts.get(2, 0)),
                "state_3_viterbi_count": int(state_counts.get(3, 0)),
                "warning": "" if hmm.converged else "finite_nonconverged_hmm_refit_recorded",
            }
        )

        profile_frame = pd.DataFrame({"state_id": hmm.labels, "stress": stress})
        for state_id, group in profile_frame.groupby("state_id"):
            state_name = f"state_{int(state_id)}"
            full_stress = float(full_profiles.loc[state_name, "macro_stress_composite_mean"]) if not full_profiles.empty else float("nan")
            drift_rows.append(
                {
                    "refit_index": int(calendar_row["refit_index"]),
                    "training_end_date": train_end.strftime("%Y-%m-%d"),
                    "state": state_name,
                    "realtime_training_stress_mean": float(group["stress"].mean()),
                    "part1_full_sample_stress_mean": full_stress,
                    "stress_mean_difference": float(group["stress"].mean() - full_stress) if math.isfinite(full_stress) else float("nan"),
                    "training_viterbi_count": int(len(group)),
                }
            )

        start = train_end
        next_train_end = pd.Timestamp(calendar_row["next_training_end_date"]) if str(calendar_row["next_training_end_date"]) else pd.NaT
        end = next_train_end if pd.notna(next_train_end) else panel["date"].max()
        future = panel[(panel["date"] > start) & (panel["date"] <= end)].copy().reset_index(drop=True)
        for _, obs in future.iterrows():
            current_raw = obs[raw_cols].to_numpy(dtype=float).reshape(1, -1)
            current_z = (current_raw - scaler_mean) / scaler_std
            current_score = pca_transform(current_z, pca).reshape(-1)
            posterior = filtered_update(posterior, current_score, hmm)
            max_prob = float(posterior.max())
            map_state_id = int(posterior.argmax())
            entropy = -float(np.sum(np.where(posterior > 0, posterior * np.log(posterior), 0.0))) / math.log(N_STATES)
            row = {
                "date": pd.Timestamp(obs["date"]),
                "training_end_date": train_end,
                "refit_index": int(calendar_row["refit_index"]),
                "realtime_map_state": f"state_{map_state_id}",
                "realtime_map_state_id": map_state_id,
                "realtime_max_probability": max_prob,
                "realtime_normalized_entropy": entropy,
                "low_confidence_flag": bool(max_prob < LOW_CONFIDENCE_THRESHOLD),
                "part1_hmm4_state": obs["hmm4_state"],
                "part1_hmm4_state_id": int(obs["hmm4_state_id"]),
            }
            for state_id in range(N_STATES):
                row[f"realtime_prob_state_{state_id}"] = float(posterior[state_id])
            probability_rows.append(row)
            z_row = {"date": pd.Timestamp(obs["date"]), "training_end_date": train_end}
            for col, value in zip(z_cols, current_z.reshape(-1)):
                z_row[f"realtime_{col}"] = float(value)
                z_row[f"full_sample_{col}"] = float(obs[col])
            realtime_z_rows.append(z_row)

        snapshots.append(
            {
                "refit_index": int(calendar_row["refit_index"]),
                "training_end_date": train_end.strftime("%Y-%m-%d"),
                "raw_predictor_cols": raw_cols,
                "z_predictor_cols": z_cols,
                "scaler_mean": scaler_mean,
                "scaler_std": scaler_std,
                "pca_mean": pca["mean"],
                "pca_components": pca["components"],
                "pca_explained_variance_ratio": pca["explained_variance_ratio"],
                "hmm_startprob": hmm.startprob,
                "hmm_transmat": hmm.transmat,
                "hmm_means": hmm.means,
                "hmm_covars": hmm.covars,
                "hmm_mapping_by_stress": mapping,
            }
        )

    probabilities = pd.DataFrame(probability_rows).sort_values("date").reset_index(drop=True)
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values("refit_index").reset_index(drop=True)
    profile_drift = pd.DataFrame(drift_rows).sort_values(["refit_index", "state"]).reset_index(drop=True)
    realtime_z = pd.DataFrame(realtime_z_rows).sort_values("date").reset_index(drop=True)
    require(date_string(probabilities["date"], "min") == EXPECTED_FIRST_PROBABILITY_DATE, "Unexpected first probability date")
    require(np.allclose(probabilities[[f"realtime_prob_state_{i}" for i in range(N_STATES)]].sum(axis=1), 1.0, atol=1e-10), "Posterior probabilities do not sum to 1")

    probabilities.to_csv(dirs["results"] / "realtime_state_probabilities.csv", index=False)
    diagnostics.to_csv(dirs["results"] / "realtime_model_refit_diagnostics.csv", index=False)
    profile_drift.to_csv(dirs["results"] / "state_profile_drift_summary.csv", index=False)
    save_pickle(dirs["models"] / "realtime_hmm_model_snapshots.pkl", snapshots)
    bridge = build_zscore_bridge(realtime_z, validation["z_predictor_cols"], dirs)
    uncertainty = build_state_uncertainty_summary(probabilities, dirs)
    agreement = build_state_label_agreement(probabilities, dirs)
    logging.info("Pseudo real-time state model completed with %d probability rows", len(probabilities))
    return {
        "refit_calendar": refit_calendar,
        "probabilities": probabilities,
        "diagnostics": diagnostics,
        "profile_drift": profile_drift,
        "realtime_z": realtime_z,
        "bridge": bridge,
        "uncertainty": uncertainty,
        "agreement": agreement,
        "snapshots": snapshots,
    }


def build_zscore_bridge(realtime_z: pd.DataFrame, z_cols: list[str], dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for col in z_cols:
        rt = realtime_z[f"realtime_{col}"].astype(float)
        fs = realtime_z[f"full_sample_{col}"].astype(float)
        rows.append(
            {
                "predictor": col,
                "rows": int(len(realtime_z)),
                "realtime_mean": float(rt.mean()),
                "realtime_std": float(rt.std(ddof=1)),
                "full_sample_z_mean": float(fs.mean()),
                "full_sample_z_std": float(fs.std(ddof=1)),
                "correlation": float(rt.corr(fs)),
                "mean_absolute_difference": float((rt - fs).abs().mean()),
                "max_absolute_difference": float((rt - fs).abs().max()),
                "usage": "bridge_only_not_main_model",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(dirs["results"] / "full_sample_zscore_bridge_summary.csv", index=False)
    return out


def build_state_uncertainty_summary(probabilities: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    rows = [
        {
            "group": "full_oos_probability_sample",
            "value": "all",
            "rows": int(len(probabilities)),
            "average_max_probability": float(probabilities["realtime_max_probability"].mean()),
            "median_max_probability": float(probabilities["realtime_max_probability"].median()),
            "average_normalized_entropy": float(probabilities["realtime_normalized_entropy"].mean()),
            "low_confidence_share": float(probabilities["low_confidence_flag"].mean()),
            "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        }
    ]
    for state, frame in probabilities.groupby("realtime_map_state", sort=True):
        rows.append(
            {
                "group": "realtime_map_state",
                "value": state,
                "rows": int(len(frame)),
                "average_max_probability": float(frame["realtime_max_probability"].mean()),
                "median_max_probability": float(frame["realtime_max_probability"].median()),
                "average_normalized_entropy": float(frame["realtime_normalized_entropy"].mean()),
                "low_confidence_share": float(frame["low_confidence_flag"].mean()),
                "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(dirs["results"] / "realtime_state_uncertainty_summary.csv", index=False)
    return out


def build_state_label_agreement(probabilities: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    match = probabilities["realtime_map_state"].eq(probabilities["part1_hmm4_state"])
    rows.append({"metric_type": "overall", "part1_state": "all", "realtime_state": "all", "rows": int(len(probabilities)), "value": float(match.mean())})
    confusion = pd.crosstab(probabilities["part1_hmm4_state"], probabilities["realtime_map_state"])
    for part1_state in EXPECTED_STATE_COUNTS:
        for realtime_state in EXPECTED_STATE_COUNTS:
            rows.append(
                {
                    "metric_type": "confusion_count",
                    "part1_state": part1_state,
                    "realtime_state": realtime_state,
                    "rows": int(confusion.loc[part1_state, realtime_state]) if part1_state in confusion.index and realtime_state in confusion.columns else 0,
                    "value": float(confusion.loc[part1_state, realtime_state]) if part1_state in confusion.index and realtime_state in confusion.columns else 0.0,
                }
            )
    for state in EXPECTED_STATE_COUNTS:
        frame = probabilities[probabilities["part1_hmm4_state"] == state]
        rows.append(
            {
                "metric_type": "part1_state_recall",
                "part1_state": state,
                "realtime_state": state,
                "rows": int(len(frame)),
                "value": float(frame["realtime_map_state"].eq(state).mean()) if len(frame) else float("nan"),
            }
        )
        pred = probabilities[probabilities["realtime_map_state"] == state]
        rows.append(
            {
                "metric_type": "realtime_state_precision",
                "part1_state": state,
                "realtime_state": state,
                "rows": int(len(pred)),
                "value": float(pred["part1_hmm4_state"].eq(state).mean()) if len(pred) else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(dirs["results"] / "state_label_agreement_summary.csv", index=False)
    return out


def target_weights_from_btc_weight(base_weights: dict[str, float], btc_weight: float, funding_convention: str, max_sleeve: float) -> dict[str, float]:
    btc_weight = float(max(0.0, btc_weight))
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
    require(abs(sum(weights.values()) - 1.0) <= 1e-8, f"Target weights do not sum to 1: {weights}")
    require(min(weights.values()) >= -1e-8, f"Negative target weight: {weights}")
    return weights


def compute_btc_risk_shares(weights: dict[str, float], returns: pd.DataFrame) -> dict[str, Any]:
    component = returns[ASSETS].astype(float).mul(pd.Series(weights), axis=1)
    portfolio = component.sum(axis=1)
    vol = float(portfolio.std(ddof=1))
    if vol <= 0 or not math.isfinite(vol):
        vol_share = float("nan")
    elif abs(weights["ret_btc"]) <= FLOAT_TOL:
        vol_share = 0.0
    else:
        cov = float(np.cov(component["ret_btc"], portfolio, ddof=1)[0, 1])
        vol_share = cov / (vol * vol)
    var_value = float(portfolio.quantile(TAIL_ALPHA))
    tail_mask = portfolio <= var_value
    tail_count = int(tail_mask.sum())
    cvar_loss = float((-portfolio.loc[tail_mask]).mean()) if tail_count else float("nan")
    if not math.isfinite(cvar_loss) or abs(cvar_loss) <= FLOAT_TOL:
        cvar_share = float("nan")
    elif abs(weights["ret_btc"]) <= FLOAT_TOL:
        cvar_share = 0.0
    else:
        cvar_contribution = float((-component.loc[tail_mask, "ret_btc"]).mean())
        cvar_share = cvar_contribution / cvar_loss
    return {
        "portfolio_volatility_weekly": vol,
        "btc_component_share_vol": float(vol_share),
        "portfolio_var_95_weekly": var_value,
        "portfolio_cvar_loss": cvar_loss,
        "btc_component_share_cvar": float(cvar_share),
        "tail_scenario_count": tail_count,
    }


def risk_cap_ok(risk: dict[str, Any]) -> bool:
    vol_share = risk["btc_component_share_vol"]
    cvar_share = risk["btc_component_share_cvar"]
    return (
        (not math.isfinite(vol_share) or vol_share <= RISK_BUDGET_CAP + RISK_BUDGET_TOL)
        and (not math.isfinite(cvar_share) or cvar_share <= RISK_BUDGET_CAP + RISK_BUDGET_TOL)
    )


def apply_risk_budget_overlay(
    candidate_btc_weight: float,
    base_weights: dict[str, float],
    funding_convention: str,
    max_sleeve: float,
    history: pd.DataFrame,
) -> tuple[float, dict[str, Any], dict[str, Any], str]:
    candidate_weights = target_weights_from_btc_weight(base_weights, candidate_btc_weight, funding_convention, max_sleeve)
    candidate_risk = compute_btc_risk_shares(candidate_weights, history)
    if candidate_btc_weight <= FLOAT_TOL:
        return 0.0, candidate_risk, candidate_risk, "zero_candidate_weight"
    if risk_cap_ok(candidate_risk):
        return float(candidate_btc_weight), candidate_risk, candidate_risk, "candidate_within_realtime_risk_budget_cap"
    low = 0.0
    high = float(candidate_btc_weight)
    final_risk = candidate_risk
    for _ in range(60):
        mid = (low + high) / 2.0
        mid_weights = target_weights_from_btc_weight(base_weights, mid, funding_convention, max_sleeve)
        mid_risk = compute_btc_risk_shares(mid_weights, history)
        if risk_cap_ok(mid_risk):
            low = mid
            final_risk = mid_risk
        else:
            high = mid
    return float(low), candidate_risk, final_risk, "reduced_by_realtime_vol_or_cvar_risk_budget_cap"


def build_rule_weights(inputs: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    rules = inputs["part4_rule_definition"]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for rule_id in IMPLEMENTED_RULE_IDS:
        out[rule_id] = {}
        for family in PORTFOLIO_FAMILIES:
            frame = rules[(rules["rule_id"] == rule_id) & (rules["portfolio_family"] == family)]
            require(len(frame) == N_STATES, f"Missing rule rows for {rule_id}/{family}")
            out[rule_id][family] = {str(row["hmm4_state"]): float(row["selected_btc_weight"]) for _, row in frame.iterrows()}
    return out


def candidate_btc_weight(rule_type: str, probs: pd.Series, state_weights: dict[str, float]) -> float:
    if rule_type == "realtime_hard_map_overlay":
        return float(state_weights[str(probs["realtime_map_state"])])
    if rule_type == "realtime_probability_weighted_overlay":
        return float(sum(float(probs[f"realtime_prob_state_{i}"]) * state_weights[f"state_{i}"] for i in range(N_STATES)))
    raise ValueError(f"Unknown rule type: {rule_type}")


def build_realtime_rules(inputs: dict[str, Any], validation: dict[str, Any], realtime: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    panel = validation["analysis_panel"].copy()
    probabilities = realtime["probabilities"].copy()
    base_weights = build_base_weights(inputs)
    state_weights = build_rule_weights(inputs)
    signal_rows: list[dict[str, Any]] = []
    overlay_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    target_return_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []

    panel_by_date = panel.set_index("date")
    next_dates = {date: panel["date"].iloc[idx + 1] for idx, date in enumerate(panel["date"].iloc[:-1])}

    for _, prob in probabilities.iterrows():
        decision_date = pd.Timestamp(prob["date"])
        if decision_date not in next_dates:
            continue
        return_date = pd.Timestamp(next_dates[decision_date])
        return_row = panel_by_date.loc[return_date]
        history = panel[panel["date"] <= decision_date].copy()
        for rule_type in RULE_TYPES:
            for rule_id in IMPLEMENTED_RULE_IDS:
                for family in PORTFOLIO_FAMILIES:
                    max_sleeve = max(state_weights[rule_id][family].values())
                    cand = candidate_btc_weight(rule_type, prob, state_weights[rule_id][family])
                    for funding in FUNDING_CONVENTIONS:
                        final_weight, candidate_risk, final_risk, reason = apply_risk_budget_overlay(
                            cand, base_weights[family], funding, max_sleeve, history
                        )
                        target = target_weights_from_btc_weight(base_weights[family], final_weight, funding, max_sleeve)
                        common = {
                            "decision_date": decision_date,
                            "return_date": return_date,
                            "rule_type": rule_type,
                            "rule_id": rule_id,
                            "portfolio_family": family,
                            "funding_convention": funding,
                            "realtime_map_state": prob["realtime_map_state"],
                            "realtime_map_state_id": int(prob["realtime_map_state_id"]),
                            "part1_hmm4_state": prob["part1_hmm4_state"],
                            "part1_hmm4_state_id": int(prob["part1_hmm4_state_id"]),
                            "candidate_btc_weight": cand,
                            "final_btc_weight": final_weight,
                            "max_btc_sleeve": max_sleeve,
                            "risk_budget_cap": RISK_BUDGET_CAP,
                            "overlay_adjustment_reason": reason,
                        }
                        for state_id in range(N_STATES):
                            common[f"realtime_prob_state_{state_id}"] = float(prob[f"realtime_prob_state_{state_id}"])
                        signal_rows.append(
                            {
                                **common,
                                "realtime_max_probability": float(prob["realtime_max_probability"]),
                                "realtime_normalized_entropy": float(prob["realtime_normalized_entropy"]),
                                "low_confidence_flag": bool(prob["low_confidence_flag"]),
                            }
                        )
                        overlay_rows.append(
                            {
                                **common,
                                "history_start_date": date_string(history["date"], "min"),
                                "history_end_date": date_string(history["date"], "max"),
                                "history_rows": int(len(history)),
                                "candidate_btc_component_share_vol": candidate_risk["btc_component_share_vol"],
                                "candidate_btc_component_share_cvar": candidate_risk["btc_component_share_cvar"],
                                "candidate_tail_scenario_count": candidate_risk["tail_scenario_count"],
                                "final_btc_component_share_vol": final_risk["btc_component_share_vol"],
                                "final_btc_component_share_cvar": final_risk["btc_component_share_cvar"],
                                "final_tail_scenario_count": final_risk["tail_scenario_count"],
                            }
                        )
                        portfolio_return = float(sum(target[asset] * float(return_row[asset]) for asset in ASSETS))
                        target_return_rows.append(
                            {
                                **common,
                                "portfolio_return": portfolio_return,
                                "btc_return": float(return_row["ret_btc"]),
                            }
                        )
                        for asset, weight in target.items():
                            component_return = weight * float(return_row[asset])
                            weight_rows.append({**common, "asset": asset, "weight": weight, "asset_return": float(return_row[asset])})
                            component_rows.append({**common, "asset": asset, "weight": weight, "component_return": component_return})

    signals = pd.DataFrame(signal_rows).sort_values(["rule_type", "rule_id", "portfolio_family", "funding_convention", "decision_date"])
    overlays = pd.DataFrame(overlay_rows).sort_values(["rule_type", "rule_id", "portfolio_family", "funding_convention", "decision_date"])
    weights = pd.DataFrame(weight_rows).sort_values(["rule_type", "rule_id", "portfolio_family", "funding_convention", "return_date", "asset"])
    returns = pd.DataFrame(target_return_rows).sort_values(["rule_type", "rule_id", "portfolio_family", "funding_convention", "return_date"])
    components = pd.DataFrame(component_rows).sort_values(["rule_type", "rule_id", "portfolio_family", "funding_convention", "return_date", "asset"])
    require(date_string(returns["return_date"], "min") == EXPECTED_FIRST_RETURN_DATE, "Unexpected first target return date")
    signals.to_csv(dirs["results"] / "realtime_rule_signal_series.csv", index=False)
    overlays.to_csv(dirs["results"] / "risk_budget_overlay_audit.csv", index=False)
    weights.to_csv(dirs["results"] / "realtime_weekly_target_weights.csv", index=False)
    returns.to_csv(dirs["results"] / "realtime_target_weight_return_series.csv", index=False)
    target_perf = build_target_weight_performance(returns, dirs)
    logging.info("Realtime rule signals completed")
    return {"signals": signals, "overlays": overlays, "weights": weights, "returns": returns, "components": components, "target_performance": target_perf}


def build_target_weight_performance(returns: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for keys, frame in returns.groupby(["rule_type", "rule_id", "portfolio_family", "funding_convention"], sort=True):
        rule_type, rule_id, family, funding = keys
        rows.append(
            {
                "rule_type": rule_type,
                "rule_id": rule_id,
                "portfolio_family": family,
                "funding_convention": funding,
                "start_date": date_string(frame["return_date"], "min"),
                "end_date": date_string(frame["return_date"], "max"),
                "average_candidate_btc_weight": float(frame["candidate_btc_weight"].mean()),
                "average_final_btc_weight": float(frame["final_btc_weight"].mean()),
                "max_final_btc_weight": float(frame["final_btc_weight"].max()),
                "adjusted_week_share": float((frame["final_btc_weight"] < frame["candidate_btc_weight"] - 1e-12).mean()),
                **performance_metrics(frame["portfolio_return"]),
            }
        )
    out = pd.DataFrame(rows).sort_values(["rule_type", "rule_id", "portfolio_family", "funding_convention"])
    out.to_csv(dirs["results"] / "realtime_target_weight_performance_summary.csv", index=False)
    return out


def build_rebalance_calendar(returns: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    dates = returns[["return_date", "decision_date"]].drop_duplicates().sort_values("return_date").reset_index(drop=True)
    dates["year_month"] = dates["return_date"].dt.to_period("M").astype(str)
    dates["year_quarter"] = dates["return_date"].dt.to_period("Q").astype(str)
    dates["is_month_end_rebalance"] = dates["return_date"].eq(dates.groupby("year_month")["return_date"].transform("max"))
    dates["is_quarter_end_rebalance"] = dates["return_date"].eq(dates.groupby("year_quarter")["return_date"].transform("max"))
    dates["is_formation_date"] = dates.index == 0
    dates.to_csv(dirs["results"] / "realtime_rebalance_calendar.csv", index=False)
    return dates


def drift_weights(start_weights: dict[str, float], returns: pd.Series, gross_return: float) -> dict[str, float]:
    denominator = 1.0 + gross_return
    require(abs(denominator) > FLOAT_TOL, "Portfolio gross return denominator too close to zero")
    return {asset: float(start_weights[asset] * (1.0 + float(returns[asset])) / denominator) for asset in ASSETS}


def build_rebalance_scenarios() -> pd.DataFrame:
    rows = []
    for rule_type in RULE_TYPES:
        for rule_id in IMPLEMENTED_RULE_IDS:
            for family in PORTFOLIO_FAMILIES:
                for funding in FUNDING_CONVENTIONS:
                    for frequency in REBALANCE_FREQUENCIES:
                        for cost_scenario in COST_SCENARIOS:
                            scenario_id = "__".join([rule_type, rule_id, family, funding, frequency, cost_scenario])
                            rows.append(
                                {
                                    "scenario_id": scenario_id,
                                    "rule_type": rule_type,
                                    "rule_id": rule_id,
                                    "portfolio_family": family,
                                    "funding_convention": funding,
                                    "rebalance_frequency": frequency,
                                    "cost_scenario": cost_scenario,
                                    "is_main_specification": bool(rule_type == MAIN_RULE_TYPE and cost_scenario == MAIN_COST_SCENARIO),
                                }
                            )
    return pd.DataFrame(rows).sort_values("scenario_id").reset_index(drop=True)


def simulate_rebalanced_scenario(
    scenario: pd.Series,
    targets: pd.DataFrame,
    panel: pd.DataFrame,
    calendar: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    subset = targets[
        (targets["rule_type"] == scenario["rule_type"])
        & (targets["rule_id"] == scenario["rule_id"])
        & (targets["portfolio_family"] == scenario["portfolio_family"])
        & (targets["funding_convention"] == scenario["funding_convention"])
    ].copy()
    pivot = subset.pivot_table(index=["return_date", "decision_date"], columns="asset", values="weight", aggfunc="first").reset_index()
    pivot = pivot.rename(columns={asset: f"target_{asset}" for asset in ASSETS})
    data = pivot.merge(panel[["date"] + ASSETS], left_on="return_date", right_on="date", how="left", validate="one_to_one")
    data = data.merge(calendar, on=["return_date", "decision_date"], how="left", validate="one_to_one", suffixes=("", "_calendar"))
    data = data.sort_values("return_date").reset_index(drop=True)
    frequency_col = "is_month_end_rebalance" if scenario["rebalance_frequency"] == "monthly" else "is_quarter_end_rebalance"
    cost_rates = cost_rates_for_scenario(str(scenario["cost_scenario"]))
    previous_end_weights: dict[str, float] | None = None
    cumulative_gross = 1.0
    cumulative_net = 1.0
    return_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []

    for _, row in data.iterrows():
        target = {asset: float(row[f"target_{asset}"]) for asset in ASSETS}
        is_formation = bool(row["is_formation_date"])
        is_scheduled = bool(row[frequency_col])
        is_rebalance = is_formation or is_scheduled
        if previous_end_weights is None:
            beginning = target.copy()
            pre_trade = {asset: 0.0 for asset in ASSETS}
            turnover = float(sum(abs(beginning[asset] - pre_trade[asset]) for asset in ASSETS))
            setup_cost = float(sum(abs(beginning[asset]) * cost_rates[asset] for asset in ASSETS))
            transaction_cost = 0.0
            event_type = "formation"
        elif is_rebalance:
            pre_trade = previous_end_weights.copy()
            beginning = target.copy()
            turnover = float(sum(abs(beginning[asset] - pre_trade[asset]) for asset in ASSETS))
            setup_cost = 0.0
            transaction_cost = float(sum(abs(beginning[asset] - pre_trade[asset]) * cost_rates[asset] for asset in ASSETS))
            event_type = "scheduled_rebalance"
        else:
            pre_trade = previous_end_weights.copy()
            beginning = previous_end_weights.copy()
            turnover = 0.0
            setup_cost = 0.0
            transaction_cost = 0.0
            event_type = "hold"
        asset_returns = row[ASSETS]
        gross_return = float(sum(beginning[asset] * float(asset_returns[asset]) for asset in ASSETS))
        net_return = gross_return - transaction_cost
        cumulative_gross *= 1.0 + gross_return
        cumulative_net *= 1.0 + net_return
        ending = drift_weights(beginning, asset_returns, gross_return)
        common = {
            "scenario_id": scenario["scenario_id"],
            "rule_type": scenario["rule_type"],
            "rule_id": scenario["rule_id"],
            "portfolio_family": scenario["portfolio_family"],
            "funding_convention": scenario["funding_convention"],
            "rebalance_frequency": scenario["rebalance_frequency"],
            "cost_scenario": scenario["cost_scenario"],
            "decision_date": row["decision_date"],
            "return_date": row["return_date"],
            "is_rebalance_date": bool(is_rebalance),
            "event_type": event_type,
        }
        return_rows.append(
            {
                **common,
                "btc_target_weight": target["ret_btc"],
                "btc_beginning_weight": beginning["ret_btc"],
                "btc_ending_weight": ending["ret_btc"],
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "turnover": turnover if is_rebalance else 0.0,
                "setup_cost_estimate": setup_cost,
                "cumulative_gross_value": cumulative_gross,
                "cumulative_net_value": cumulative_net,
            }
        )
        for asset in ASSETS:
            weight_rows.append(
                {
                    **common,
                    "asset": asset,
                    "asset_return": float(asset_returns[asset]),
                    "target_weight": target[asset],
                    "pre_trade_weight": pre_trade[asset],
                    "beginning_weight": beginning[asset],
                    "component_return": beginning[asset] * float(asset_returns[asset]),
                    "ending_weight_before_next_trade": ending[asset],
                }
            )
        previous_end_weights = ending
    return return_rows, weight_rows


def run_rebalanced_implementability(validation: dict[str, Any], realtime_rules: dict[str, Any], dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    calendar = build_rebalance_calendar(realtime_rules["returns"], dirs)
    scenarios = build_rebalance_scenarios()
    panel = validation["analysis_panel"].copy()
    all_returns: list[dict[str, Any]] = []
    all_weights: list[dict[str, Any]] = []
    for _, scenario in scenarios.iterrows():
        rows, weights = simulate_rebalanced_scenario(scenario, realtime_rules["weights"], panel, calendar)
        all_returns.extend(rows)
        all_weights.extend(weights)
    returns = pd.DataFrame(all_returns).sort_values(["scenario_id", "return_date"]).reset_index(drop=True)
    weights = pd.DataFrame(all_weights).sort_values(["scenario_id", "return_date", "asset"]).reset_index(drop=True)
    returns.to_csv(dirs["results"] / "realtime_rebalanced_return_series.csv", index=False)
    weights.to_csv(dirs["results"] / "realtime_rebalanced_weight_series.csv", index=False)
    performance = build_rebalanced_performance(returns, dirs)
    turnover = build_turnover_cost_summary(returns, dirs)
    logging.info("Realtime rebalanced implementability completed")
    return {"calendar": calendar, "scenarios": scenarios, "returns": returns, "weights": weights, "performance": performance, "turnover": turnover}


def build_rebalanced_performance(returns: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for keys, frame in returns.groupby(["scenario_id", "rule_type", "rule_id", "portfolio_family", "funding_convention", "rebalance_frequency", "cost_scenario"], sort=True):
        scenario_id, rule_type, rule_id, family, funding, frequency, cost_scenario = keys
        rows.append(
            {
                "scenario_id": scenario_id,
                "rule_type": rule_type,
                "rule_id": rule_id,
                "portfolio_family": family,
                "funding_convention": funding,
                "rebalance_frequency": frequency,
                "cost_scenario": cost_scenario,
                "start_date": date_string(frame["return_date"], "min"),
                "end_date": date_string(frame["return_date"], "max"),
                "average_btc_beginning_weight": float(frame["btc_beginning_weight"].mean()),
                "max_btc_beginning_weight": float(frame["btc_beginning_weight"].max()),
                "average_btc_target_weight": float(frame["btc_target_weight"].mean()),
                "total_transaction_cost": float(frame["transaction_cost"].sum()),
                "total_turnover_including_formation": float(frame["turnover"].sum()),
                "final_cumulative_gross_value": float(frame["cumulative_gross_value"].iloc[-1]),
                "final_cumulative_net_value": float(frame["cumulative_net_value"].iloc[-1]),
                **performance_metrics(frame["gross_return"], prefix="gross_"),
                **performance_metrics(frame["net_return"], prefix="net_"),
            }
        )
    out = pd.DataFrame(rows).sort_values("scenario_id").reset_index(drop=True)
    out.to_csv(dirs["results"] / "realtime_rebalanced_performance_summary.csv", index=False)
    return out


def build_turnover_cost_summary(returns: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for keys, frame in returns.groupby(["rule_type", "rule_id", "portfolio_family", "funding_convention", "rebalance_frequency", "cost_scenario"], sort=True):
        rule_type, rule_id, family, funding, frequency, cost_scenario = keys
        rebalances = frame[frame["is_rebalance_date"]]
        rows.append(
            {
                "rule_type": rule_type,
                "rule_id": rule_id,
                "portfolio_family": family,
                "funding_convention": funding,
                "rebalance_frequency": frequency,
                "cost_scenario": cost_scenario,
                "rows": int(len(frame)),
                "rebalance_events": int(len(rebalances)),
                "total_turnover_including_formation": float(frame["turnover"].sum()),
                "average_turnover_on_rebalance": float(rebalances["turnover"].mean()) if len(rebalances) else float("nan"),
                "total_transaction_cost": float(frame["transaction_cost"].sum()),
                "average_weekly_transaction_cost": float(frame["transaction_cost"].mean()),
                "transaction_cost_only_on_rebalance_weeks": bool((frame.loc[~frame["is_rebalance_date"], "transaction_cost"].abs() <= FLOAT_TOL).all()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["rule_type", "rule_id", "portfolio_family", "funding_convention", "rebalance_frequency"])
    out.to_csv(dirs["results"] / "realtime_turnover_cost_summary.csv", index=False)
    return out


def build_part5_comparison(inputs: dict[str, Any], simulations: dict[str, pd.DataFrame], dirs: dict[str, Path]) -> pd.DataFrame:
    realtime_perf = simulations["performance"].copy()
    part5 = inputs["part5_rebalanced_returns"].copy()
    part5 = part5[
        (part5["rule_id"].isin(IMPLEMENTED_RULE_IDS))
        & (part5["cost_scenario"] == MAIN_COST_SCENARIO)
        & (part5["signal_timing"] == "lagged_one_week")
        & (part5["date"] >= pd.Timestamp(EXPECTED_FIRST_RETURN_DATE))
    ].copy()
    rows = []
    for keys, frame in part5.groupby(["rule_id", "portfolio_family", "funding_convention", "rebalance_frequency", "cost_scenario"], sort=True):
        rule_id, family, funding, frequency, cost_scenario = keys
        metrics = performance_metrics(frame["net_return"], prefix="part5_expost_net_")
        rt = realtime_perf[
            (realtime_perf["rule_type"] == MAIN_RULE_TYPE)
            & (realtime_perf["rule_id"] == rule_id)
            & (realtime_perf["portfolio_family"] == family)
            & (realtime_perf["funding_convention"] == funding)
            & (realtime_perf["rebalance_frequency"] == frequency)
            & (realtime_perf["cost_scenario"] == cost_scenario)
        ]
        if rt.empty:
            continue
        rt_row = rt.iloc[0]
        rows.append(
            {
                "rule_id": rule_id,
                "portfolio_family": family,
                "funding_convention": funding,
                "rebalance_frequency": frequency,
                "cost_scenario": cost_scenario,
                "comparison_sample_start": EXPECTED_FIRST_RETURN_DATE,
                "comparison_sample_end": EXPECTED_STATE_END,
                **metrics,
                "realtime_net_count": int(rt_row["net_count"]),
                "realtime_net_annualized_mean_arithmetic": float(rt_row["net_annualized_mean_arithmetic"]),
                "realtime_net_annualized_volatility": float(rt_row["net_annualized_volatility"]),
                "realtime_net_max_drawdown": float(rt_row["net_max_drawdown"]),
                "realtime_net_sharpe_annualized_zero_rf": float(rt_row["net_sharpe_annualized_zero_rf"]),
                "delta_realtime_minus_part5_expost_net_annualized_mean": float(rt_row["net_annualized_mean_arithmetic"] - metrics["part5_expost_net_annualized_mean_arithmetic"]),
                "delta_realtime_minus_part5_expost_net_volatility": float(rt_row["net_annualized_volatility"] - metrics["part5_expost_net_annualized_volatility"]),
                "delta_realtime_minus_part5_expost_net_max_drawdown": float(rt_row["net_max_drawdown"] - metrics["part5_expost_net_max_drawdown"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(dirs["results"] / "realtime_vs_expost_part5_comparison.csv", index=False)
    return out


def write_audit_files(
    args: argparse.Namespace,
    inputs: dict[str, Any],
    validation: dict[str, Any],
    realtime: dict[str, Any],
    realtime_rules: dict[str, Any],
    simulations: dict[str, pd.DataFrame],
    comparison: pd.DataFrame,
    dirs: dict[str, Path],
) -> None:
    audit = f"""# Part 7 Methodology Audit

This experiment is a pseudo real-time robustness diagnostic. It does not mutate Part 1-6 outputs and does not search for new BTC state weights.

## State model
- Initial training window: {EXPECTED_STATE_START} to {EXPECTED_INITIAL_TRAIN_END}, {EXPECTED_INITIAL_TRAIN_ROWS} weeks.
- Macro predictors are the lagged raw state predictors from the cleaned panel.
- Each refit recomputes expanding-window standardization, PCA with {N_COMPONENTS} components, and a {N_STATES}-state diagonal Gaussian HMM.
- HMM states are ordered by ascending macro stress composite, matching Part 1.
- Full-sample z-score columns are used only for the bridge audit.

## Timing
- The state probability on decision date `t` may use macro predictors observed at `t`, which were already lagged during data cleaning.
- Portfolio returns use the signal from the previous weekly decision date.
- The experiment does not use ALFRED/FRED vintage data and therefore is not a complete point-in-time macro-vintage backtest.

## Allocation and overlay
- Candidate BTC weights come only from Part 4 executed rules.
- The main rule type is continuous probability-weighted BTC allocation.
- The risk-budget overlay uses expanding historical returns available at each decision date and caps BTC volatility and CVaR contribution at {RISK_BUDGET_CAP:.0%}.
- This is a robustness diagnostic, not a final trading rule or paper conclusion.
"""
    (dirs["results"] / "methodology_audit.md").write_text(audit, encoding="utf-8")

    assumptions = {
        "experiment_role": "pseudo_real_time_probabilistic_regime_robustness",
        "not_a_strategy_search": True,
        "initial_training_rows": EXPECTED_INITIAL_TRAIN_ROWS,
        "state_probability_start": EXPECTED_FIRST_PROBABILITY_DATE,
        "lagged_return_start": EXPECTED_FIRST_RETURN_DATE,
        "macro_vintage_limitation": "Cleaned macro predictors are lagged, but ALFRED/FRED point-in-time vintages are not used.",
        "state_model": {
            "n_states": N_STATES,
            "pca_components": N_COMPONENTS,
            "covariance_type": "diag",
            "refit_frequency": "monthly",
            "standardization": "expanding_window",
            "state_ordering": "ascending_macro_stress_composite",
        },
        "allocation": {
            "candidate_weights_source": "Part 4 executed rules",
            "main_rule_type": MAIN_RULE_TYPE,
            "risk_budget_cap": RISK_BUDGET_CAP,
            "risk_budget_estimation_window": "expanding_history_available_at_decision_date",
        },
        "upstream_outputs_not_modified": True,
    }
    write_json(dirs["results"] / "model_assumption_audit.json", assumptions)

    lineage_rows = []
    for name, path in inputs["paths"].items():
        lineage_rows.append({"input_name": name, "path": str(path), "sha256": file_sha256(path), "usage": lineage_usage(name)})
    pd.DataFrame(lineage_rows).to_csv(dirs["results"] / "data_lineage.csv", index=False)


def lineage_usage(name: str) -> str:
    if name in {"asset", "state", "cleaning_report"}:
        return "cleaned_input"
    if "part1" in name:
        return "full_sample_hmm_baseline_and_lineage"
    if "part2" in name:
        return "baseline_weights_and_lineage"
    if "part4" in name:
        return "executed_rule_weights"
    if "part5" in name:
        return "implementability_baseline_comparison"
    if "part6" in name:
        return "robustness_lineage"
    return "upstream_context"


def plot_outputs(realtime: dict[str, Any], realtime_rules: dict[str, Any], simulations: dict[str, pd.DataFrame], dirs: dict[str, Path]) -> None:
    probabilities = realtime["probabilities"]
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(N_STATES):
        ax.plot(probabilities["date"], probabilities[f"realtime_prob_state_{i}"], label=f"state_{i}", linewidth=1.2)
    ax.set_title("Pseudo Real-Time Posterior State Probabilities")
    ax.set_ylabel("Probability")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "state_probability_timeline.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(probabilities["date"], probabilities["realtime_max_probability"], label="max posterior probability")
    ax.plot(probabilities["date"], probabilities["realtime_normalized_entropy"], label="normalized entropy")
    ax.axhline(LOW_CONFIDENCE_THRESHOLD, color="red", linestyle="--", linewidth=1, label="low-confidence threshold")
    ax.set_title("Regime Uncertainty Diagnostics")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "state_uncertainty.png", dpi=160)
    plt.close(fig)

    confusion = pd.crosstab(probabilities["part1_hmm4_state"], probabilities["realtime_map_state"]).reindex(
        index=list(EXPECTED_STATE_COUNTS), columns=list(EXPECTED_STATE_COUNTS), fill_value=0
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion.to_numpy(), cmap="Blues")
    ax.set_xticks(range(N_STATES), labels=confusion.columns)
    ax.set_yticks(range(N_STATES), labels=confusion.index)
    for row in range(N_STATES):
        for col in range(N_STATES):
            ax.text(col, row, int(confusion.iloc[row, col]), ha="center", va="center", color="black", fontsize=9)
    ax.set_xlabel("Pseudo real-time MAP state")
    ax.set_ylabel("Part 1 full-sample HMM state")
    ax.set_title("State Label Agreement")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "state_label_agreement_heatmap.png", dpi=160)
    plt.close(fig)

    signals = realtime_rules["signals"]
    focus = signals[
        (signals["rule_type"] == MAIN_RULE_TYPE)
        & (signals["rule_id"] == "main_executed")
        & (signals["portfolio_family"] == "all_weather")
        & (signals["funding_convention"] == "pro_rata_base")
    ]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(focus["decision_date"], focus["candidate_btc_weight"], label="candidate BTC weight", linewidth=1.2)
    ax.plot(focus["decision_date"], focus["final_btc_weight"], label="overlay final BTC weight", linewidth=1.2)
    ax.set_title("Realtime Probability-Weighted BTC Signal")
    ax.set_ylabel("BTC weight")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "realtime_btc_weight_timeline.png", dpi=160)
    plt.close(fig)

    overlay = realtime_rules["overlays"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(overlay["candidate_btc_component_share_vol"], overlay["final_btc_component_share_vol"], s=8, alpha=0.45, label="volatility")
    ax.scatter(overlay["candidate_btc_component_share_cvar"], overlay["final_btc_component_share_cvar"], s=8, alpha=0.45, label="CVaR")
    ax.axhline(RISK_BUDGET_CAP, color="red", linestyle="--", linewidth=1)
    ax.axvline(RISK_BUDGET_CAP, color="red", linestyle="--", linewidth=1)
    ax.set_title("Risk-Budget Overlay Audit")
    ax.set_xlabel("Candidate BTC risk contribution share")
    ax.set_ylabel("Final BTC risk contribution share")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "risk_budget_overlay_audit.png", dpi=160)
    plt.close(fig)

    returns = simulations["returns"]
    focus_returns = returns[
        (returns["rule_type"] == MAIN_RULE_TYPE)
        & (returns["rule_id"] == "main_executed")
        & (returns["funding_convention"] == "pro_rata_base")
        & (returns["rebalance_frequency"] == "monthly")
    ]
    fig, ax = plt.subplots(figsize=(12, 4))
    for family, frame in focus_returns.groupby("portfolio_family", sort=True):
        ax.plot(frame["return_date"], drawdown_series(frame["net_return"]), label=family)
    ax.set_title("Realtime Main Rule Net Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "realtime_drawdown_performance_comparison.png", dpi=160)
    plt.close(fig)


def validate_outputs(
    dirs: dict[str, Path],
    validation: dict[str, Any],
    realtime: dict[str, Any],
    realtime_rules: dict[str, Any],
    simulations: dict[str, pd.DataFrame],
) -> dict[str, Any]:
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

    probabilities = realtime["probabilities"]
    prob_sum_ok = bool(np.allclose(probabilities[[f"realtime_prob_state_{i}" for i in range(N_STATES)]].sum(axis=1), 1.0, atol=1e-10))
    returns = realtime_rules["returns"]
    time_ok = bool((pd.to_datetime(returns["decision_date"]) < pd.to_datetime(returns["return_date"])).all())
    overlay = realtime_rules["overlays"]
    overlay_weight_ok = bool((overlay["final_btc_weight"] <= overlay["candidate_btc_weight"] + 1e-12).all())
    overlay_cap_ok = bool(
        (overlay["final_btc_component_share_vol"] <= RISK_BUDGET_CAP + RISK_BUDGET_TOL).all()
        and (overlay["final_btc_component_share_cvar"] <= RISK_BUDGET_CAP + RISK_BUDGET_TOL).all()
    )
    target_weight_sum = realtime_rules["weights"].groupby(["rule_type", "rule_id", "portfolio_family", "funding_convention", "return_date"])["weight"].sum()
    target_weight_sum_ok = bool(np.allclose(target_weight_sum.to_numpy(), 1.0, atol=1e-8))
    bil_rows = realtime_rules["weights"][realtime_rules["weights"]["funding_convention"] == "bil_sleeve"]
    bil_pivot = bil_rows.pivot_table(
        index=["rule_type", "rule_id", "portfolio_family", "funding_convention", "return_date"],
        columns="asset",
        values="weight",
        aggfunc="first",
    )
    bil_sleeve_ok = bool(np.allclose((bil_pivot["ret_btc"] + bil_pivot["ret_bil"]).round(12), 0.02, atol=1e-8))
    nonrebalance_cost_ok = bool((simulations["returns"].loc[~simulations["returns"]["is_rebalance_date"], "transaction_cost"].abs() <= FLOAT_TOL).all())

    checks = {
        "required_results_ok": all(item["exists"] and item["nonempty"] for item in result_checks if item["file"] != "output_validation_summary.json"),
        "required_figures_ok": all(item["exists"] and item["nonempty"] and item["readable"] for item in figure_checks),
        "probability_rows": int(len(probabilities)),
        "target_return_rows_per_series": int(returns.groupby(["rule_type", "rule_id", "portfolio_family", "funding_convention"]).size().iloc[0]),
        "rebalanced_rows_per_scenario": int(simulations["returns"].groupby("scenario_id").size().iloc[0]),
        "posterior_probabilities_sum_to_one": prob_sum_ok,
        "decision_date_before_return_date": time_ok,
        "overlay_final_weight_not_above_candidate": overlay_weight_ok,
        "overlay_risk_cap_ok": overlay_cap_ok,
        "target_weight_sums_ok": target_weight_sum_ok,
        "bil_sleeve_identity_ok": bil_sleeve_ok,
        "nonrebalance_transaction_cost_zero": nonrebalance_cost_ok,
    }
    status = "passed" if all(v for k, v in checks.items() if isinstance(v, bool)) else "failed"
    payload = {
        "status": status,
        "checks": checks,
        "required_result_checks": result_checks,
        "required_figure_checks": figure_checks,
    }
    write_json(dirs["results"] / "output_validation_summary.json", payload)
    require(status == "passed", "Output validation failed")
    return payload


def write_manifest(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    input_hashes: dict[str, str],
    validation: dict[str, Any],
    realtime: dict[str, Any],
    realtime_rules: dict[str, Any],
    simulations: dict[str, pd.DataFrame],
    output_validation: dict[str, Any],
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": args.run_id or dirs["root"].name,
        "objective": "Part 7 pseudo real-time probabilistic regime robustness",
        "input_dir": str(args.input_dir),
        "part1_run_dir": str(args.part1_run_dir),
        "part2_run_dir": str(args.part2_run_dir),
        "part3_run_dir": str(args.part3_run_dir),
        "part4_run_dir": str(args.part4_run_dir),
        "part5_run_dir": str(args.part5_run_dir),
        "part6_run_dir": str(args.part6_run_dir),
        "output_dir": str(args.output_dir),
        "run_dir": str(dirs["root"]),
        "random_seed": args.seed,
        "sample": {
            "state_rows": EXPECTED_STATE_ROWS,
            "state_start": EXPECTED_STATE_START,
            "state_end": EXPECTED_STATE_END,
            "initial_train_rows": EXPECTED_INITIAL_TRAIN_ROWS,
            "initial_train_end": EXPECTED_INITIAL_TRAIN_END,
            "probability_rows": int(len(realtime["probabilities"])),
            "probability_start": date_string(realtime["probabilities"]["date"], "min"),
            "probability_end": date_string(realtime["probabilities"]["date"], "max"),
            "lagged_return_rows": int(realtime_rules["returns"].groupby(["rule_type", "rule_id", "portfolio_family", "funding_convention"]).size().iloc[0]),
            "lagged_return_start": date_string(realtime_rules["returns"]["return_date"], "min"),
            "lagged_return_end": date_string(realtime_rules["returns"]["return_date"], "max"),
        },
        "input_hashes": input_hashes,
        "package_versions": package_versions(),
        "parameters": {
            "pca_components": N_COMPONENTS,
            "hmm_states": N_STATES,
            "hmm_covariance_type": "diag",
            "hmm_n_init": HMM_N_INIT,
            "hmm_max_iter": HMM_MAX_ITER,
            "hmm_tol": HMM_TOL,
            "refit_frequency": "monthly",
            "standardization": "expanding_window",
            "risk_budget_cap": RISK_BUDGET_CAP,
            "risk_budget_overlay_estimation_window": "expanding_history",
            "main_rule_type": MAIN_RULE_TYPE,
            "cost_scenarios": COST_SCENARIOS,
        },
        "model_diagnostics": {
            "refit_count": int(len(realtime["diagnostics"])),
            "nonconverged_refits": int((~realtime["diagnostics"]["converged"]).sum()),
            "average_max_probability": float(realtime["probabilities"]["realtime_max_probability"].mean()),
            "low_confidence_share": float(realtime["probabilities"]["low_confidence_flag"].mean()),
        },
        "output_validation": output_validation,
        "outputs": {
            "results": REQUIRED_RESULTS,
            "figures": REQUIRED_FIGURES,
            "models": ["realtime_hmm_model_snapshots.pkl"],
        },
        "scope_notes": [
            "Pseudo real-time robustness diagnostic; not a final trading rule.",
            "No upstream Part 1-6 outputs are modified.",
            "No ALFRED/FRED point-in-time vintage data are used.",
            "Part 4 executed state weights remain fixed; Part 7 changes state identification and uncertainty transmission.",
        ],
    }
    write_json(dirs["root"] / "run_manifest.json", manifest)


def main() -> None:
    args = parse_args()
    run_id = args.run_id or now_run_id()
    run_dir = args.output_dir / run_id
    dirs = ensure_dirs(run_dir)
    setup_logging(dirs["logs"])
    logging.info("Starting Part 7 run_id=%s", run_id)

    inputs = load_inputs(args)
    input_hashes = build_input_hashes(inputs["paths"])
    enforce_resume_input_hashes(args, dirs, input_hashes)
    validation = load_or_run(dirs, "01_input_validation", args.resume, lambda: validate_inputs(inputs, dirs))
    realtime = load_or_run(dirs, "02_realtime_state_model", args.resume, lambda: run_realtime_state_model(inputs, validation, args, dirs))
    realtime_rules = load_or_run(dirs, "03_realtime_rules", args.resume, lambda: build_realtime_rules(inputs, validation, realtime, dirs))
    simulations = load_or_run(dirs, "04_rebalanced_implementability", args.resume, lambda: run_rebalanced_implementability(validation, realtime_rules, dirs))
    comparison = load_or_run(dirs, "05_part5_comparison", args.resume, lambda: build_part5_comparison(inputs, simulations, dirs))
    write_audit_files(args, inputs, validation, realtime, realtime_rules, simulations, comparison, dirs)
    plot_outputs(realtime, realtime_rules, simulations, dirs)
    output_validation = validate_outputs(dirs, validation, realtime, realtime_rules, simulations)
    write_manifest(args, dirs, input_hashes, validation, realtime, realtime_rules, simulations, output_validation)
    logging.info("Part 7 completed successfully: %s", run_dir)


if __name__ == "__main__":
    main()
