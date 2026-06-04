#!/usr/bin/env python3
"""Part 1 experiment runner: BTC asset diagnostics and macro state identification.

This script is intentionally self-contained. PCA, K-means, and a diagonal
Gaussian HMM are implemented with NumPy so the experiment can run in Colab even
when optional ML packages are unavailable.
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
from dataclasses import asdict, dataclass
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
TRADING_WEEKS_PER_YEAR = 52
TAIL_ALPHA = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BTC asset role diagnostics and macro-state identification."
    )
    parser.add_argument("--input-dir", default="data_2026/cleaned", type=Path)
    parser.add_argument("--output-dir", default="outputs/part1_btc_macro_state", type=Path)
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
    log_path = log_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a", encoding="utf-8"), logging.StreamHandler()],
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_pickle(path: Path, payload: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def checkpoint_path(dirs: dict[str, Path], name: str) -> Path:
    return dirs["checkpoints"] / f"{name}.pkl"


def load_or_run(
    dirs: dict[str, Path],
    name: str,
    resume: bool,
    compute_fn,
) -> Any:
    path = checkpoint_path(dirs, name)
    if resume and path.exists():
        logging.info("Loading checkpoint: %s", path)
        return load_pickle(path)
    payload = compute_fn()
    save_pickle(path, payload)
    logging.info("Saved checkpoint: %s", path)
    return payload


def package_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "matplotlib"]
    versions: dict[str, str] = {"python": sys.version.replace("\n", " "), "platform": platform.platform()}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def date_string(series: pd.Series, fn: str) -> str:
    value = series.min() if fn == "min" else series.max()
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def read_inputs(input_dir: Path) -> dict[str, Any]:
    paths = {
        "asset_returns_main_weekly": input_dir / "asset_returns_main_weekly.csv",
        "state_model_panel_weekly": input_dir / "state_model_panel_weekly.csv",
        "robustness_weekly_panel": input_dir / "robustness_weekly_panel.csv",
        "cleaning_report": input_dir / "cleaning_report.json",
    }
    for name, path in paths.items():
        require(path.exists(), f"Missing required input file: {name} at {path}")
    asset = pd.read_csv(paths["asset_returns_main_weekly"], parse_dates=["date"])
    state = pd.read_csv(paths["state_model_panel_weekly"], parse_dates=["date"])
    robustness = pd.read_csv(paths["robustness_weekly_panel"], parse_dates=["date"])
    report = json.loads(paths["cleaning_report"].read_text(encoding="utf-8"))
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    return {
        "paths": paths,
        "asset": asset,
        "state": state,
        "robustness": robustness,
        "cleaning_report": report,
        "input_hashes": hashes,
    }


def validate_inputs(inputs: dict[str, Any], dirs: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    asset = inputs["asset"]
    state = inputs["state"]
    robustness = inputs["robustness"]
    report = inputs["cleaning_report"]
    input_hashes = inputs["input_hashes"]

    main_cols = list(report["column_mapping"]["main_assets"].values())
    raw_predictor_cols = list(report["column_mapping"]["state_predictors"].values())
    z_predictor_cols = [f"{col}_z" for col in raw_predictor_cols]

    require(len(asset) == EXPECTED_ASSET_ROWS, f"Unexpected asset row count: {len(asset)}")
    require(date_string(asset["date"], "min") == EXPECTED_ASSET_START, "Unexpected asset start date")
    require(date_string(asset["date"], "max") == EXPECTED_ASSET_END, "Unexpected asset end date")
    require(len(state) == EXPECTED_STATE_ROWS, f"Unexpected state panel row count: {len(state)}")
    require(date_string(state["date"], "min") == EXPECTED_STATE_START, "Unexpected state start date")
    require(date_string(state["date"], "max") == EXPECTED_STATE_END, "Unexpected state end date")
    require(asset["date"].dt.dayofweek.eq(4).all(), "Asset dates are not all Fridays")
    require(state["date"].dt.dayofweek.eq(4).all(), "State dates are not all Fridays")
    require(robustness["date"].dt.dayofweek.eq(4).all(), "Robustness dates are not all Fridays")
    require(list(asset.columns) == ["date"] + main_cols, "Unexpected asset column layout")

    missing_main = asset[main_cols].isna().sum()
    missing_state = state[main_cols + raw_predictor_cols + z_predictor_cols].isna().sum()
    require(int(missing_main.sum()) == 0, f"Missing values in main asset returns: {missing_main.to_dict()}")
    require(int(missing_state.sum()) == 0, f"Missing values in state panel: {missing_state.to_dict()}")
    require(all(col in state.columns for col in raw_predictor_cols), "Missing raw predictor columns")
    require(all(col in state.columns for col in z_predictor_cols), "Missing z-score predictor columns")

    manifest_path = dirs["root"] / "run_manifest.json"
    if args.resume and manifest_path.exists():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_hashes = old_manifest.get("input_hashes", {})
        require(old_hashes == input_hashes, "Input hashes changed since the previous run manifest")

    z_checks = {}
    for col in z_predictor_cols:
        z_checks[col] = {
            "mean": float(state[col].mean()),
            "std_population": float(state[col].std(ddof=0)),
        }
        require(abs(z_checks[col]["mean"]) < 1e-10, f"Z-score mean check failed for {col}")
        require(abs(z_checks[col]["std_population"] - 1.0) < 1e-10, f"Z-score std check failed for {col}")

    summary = {
        "status": "passed",
        "asset_sample": {
            "rows": int(len(asset)),
            "start": date_string(asset["date"], "min"),
            "end": date_string(asset["date"], "max"),
        },
        "state_sample": {
            "rows": int(len(state)),
            "start": date_string(state["date"], "min"),
            "end": date_string(state["date"], "max"),
        },
        "robustness_sample": {
            "rows": int(len(robustness)),
            "start": date_string(robustness["date"], "min"),
            "end": date_string(robustness["date"], "max"),
        },
        "main_return_columns": main_cols,
        "raw_predictor_columns": raw_predictor_cols,
        "z_predictor_columns": z_predictor_cols,
        "input_hashes": input_hashes,
        "zscore_checks": z_checks,
    }
    write_json(dirs["results"] / "validation_summary.json", summary)
    logging.info("Input validation passed")
    return summary


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


def drawdown_from_returns(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    return float(drawdown.min())


def var_cvar(returns: pd.Series, alpha: float = TAIL_ALPHA) -> tuple[float, float]:
    clean = returns.dropna()
    if clean.empty:
        return float("nan"), float("nan")
    var_value = float(clean.quantile(alpha))
    tail = clean[clean <= var_value]
    cvar_value = float(tail.mean()) if not tail.empty else var_value
    return var_value, cvar_value


def return_metrics(name: str, returns: pd.Series) -> dict[str, Any]:
    clean = returns.dropna()
    var_95, cvar_95 = var_cvar(clean)
    mean_weekly = float(clean.mean())
    vol_weekly = float(clean.std(ddof=1))
    sharpe = (
        float(mean_weekly / vol_weekly * math.sqrt(TRADING_WEEKS_PER_YEAR))
        if vol_weekly > 0
        else float("nan")
    )
    return {
        "asset": name,
        "count": int(clean.shape[0]),
        "mean_weekly": mean_weekly,
        "median_weekly": float(clean.median()),
        "volatility_weekly": vol_weekly,
        "annualized_mean_arithmetic": mean_weekly * TRADING_WEEKS_PER_YEAR,
        "annualized_volatility": vol_weekly * math.sqrt(TRADING_WEEKS_PER_YEAR),
        "min_weekly": float(clean.min()),
        "max_weekly": float(clean.max()),
        "var_95_weekly": var_95,
        "cvar_95_weekly": cvar_95,
        "max_drawdown": drawdown_from_returns(clean),
        "positive_week_share": float((clean > 0).mean()),
        "sharpe_annualized_zero_rf": sharpe,
    }


def beta(y: pd.Series, x: pd.Series) -> float:
    aligned = pd.concat([y, x], axis=1).dropna()
    if aligned.shape[0] < 3:
        return float("nan")
    x_values = aligned.iloc[:, 1]
    variance = float(x_values.var(ddof=1))
    if variance <= 0:
        return float("nan")
    return float(aligned.iloc[:, 0].cov(x_values) / variance)


def compute_asset_diagnostics(inputs: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    asset = inputs["asset"].copy()
    robustness = inputs["robustness"].copy()
    report = inputs["cleaning_report"]
    main_cols = list(report["column_mapping"]["main_assets"].values())
    other_assets = [col for col in main_cols if col != "ret_btc"]

    diagnostics = pd.DataFrame([return_metrics(col, asset[col]) for col in main_cols])
    diagnostics.to_csv(dirs["results"] / "btc_asset_diagnostics.csv", index=False)

    corr = asset[main_cols].corr()
    corr.to_csv(dirs["results"] / "return_correlation_matrix.csv")

    rolling_frames = []
    for col in other_assets:
        rolling = asset["ret_btc"].rolling(52).corr(asset[col])
        rolling_frames.append(
            pd.DataFrame(
                {
                    "date": asset["date"],
                    "asset": col,
                    "rolling_correlation_52w": rolling,
                }
            )
        )
    rolling_corr = pd.concat(rolling_frames, ignore_index=True)
    rolling_corr.to_csv(dirs["results"] / "rolling_correlations_52w.csv", index=False)

    pairwise = []
    for col in other_assets:
        aligned = asset[["ret_btc", col]].dropna()
        pairwise.append(
            {
                "asset": col,
                "count": int(aligned.shape[0]),
                "correlation_with_btc": float(aligned["ret_btc"].corr(aligned[col])),
                "beta_btc_to_asset": beta(aligned["ret_btc"], aligned[col]),
                "beta_asset_to_btc": beta(aligned[col], aligned["ret_btc"]),
            }
        )
    pairwise_df = pd.DataFrame(pairwise)
    pairwise_df.to_csv(dirs["results"] / "btc_pairwise_diagnostics.csv", index=False)

    source_overlap = robustness[["date", "ret_btc", "ret_btc_coinmetrics_usd"]].dropna().copy()
    source_overlap["ret_diff_btcusdt_minus_coinmetrics"] = (
        source_overlap["ret_btc"] - source_overlap["ret_btc_coinmetrics_usd"]
    )
    source_overlap.to_csv(dirs["results"] / "btc_source_weekly_diffs.csv", index=False)
    abs_diff = source_overlap["ret_diff_btcusdt_minus_coinmetrics"].abs()
    max_idx = abs_diff.idxmax()
    source_validation = pd.DataFrame(
        [
            {
                "metric": "BTCUSDT_vs_CoinMetrics_weekly_returns",
                "overlap_count": int(source_overlap.shape[0]),
                "overlap_start": source_overlap["date"].min().strftime("%Y-%m-%d"),
                "overlap_end": source_overlap["date"].max().strftime("%Y-%m-%d"),
                "correlation": float(source_overlap["ret_btc"].corr(source_overlap["ret_btc_coinmetrics_usd"])),
                "mean_difference": float(source_overlap["ret_diff_btcusdt_minus_coinmetrics"].mean()),
                "mean_absolute_difference": float(abs_diff.mean()),
                "p95_absolute_difference": float(abs_diff.quantile(0.95)),
                "max_absolute_difference": float(abs_diff.loc[max_idx]),
                "max_absolute_difference_date": source_overlap.loc[max_idx, "date"].strftime("%Y-%m-%d"),
            }
        ]
    )
    source_validation.to_csv(dirs["results"] / "btc_source_validation.csv", index=False)

    plot_correlation_matrix(corr, dirs["figures"] / "return_correlation_matrix.png")
    plot_rolling_correlations(rolling_corr, dirs["figures"] / "btc_rolling_correlations_52w.png")

    logging.info("Asset diagnostics completed")
    return {
        "diagnostics": diagnostics,
        "correlation_matrix": corr,
        "rolling_correlations": rolling_corr,
        "pairwise": pairwise_df,
        "btc_source_validation": source_validation,
    }


def plot_correlation_matrix(corr: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Weekly Return Correlation Matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_rolling_correlations(rolling_corr: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for asset, frame in rolling_corr.groupby("asset"):
        ax.plot(frame["date"], frame["rolling_correlation_52w"], label=asset, linewidth=1.4)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("BTC 52-Week Rolling Correlations")
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation")
    ax.legend(loc="best", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def orient_components(components: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    components = components.copy()
    scores = scores.copy()
    for i in range(components.shape[0]):
        largest = int(np.argmax(np.abs(components[i])))
        if components[i, largest] < 0:
            components[i] *= -1
            scores[:, i] *= -1
    return components, scores


def compute_pca(inputs: dict[str, Any], dirs: dict[str, Path], n_components: int = 5) -> dict[str, Any]:
    state = inputs["state"]
    report = inputs["cleaning_report"]
    raw_predictor_cols = list(report["column_mapping"]["state_predictors"].values())
    z_cols = [f"{col}_z" for col in raw_predictor_cols]
    x = state[z_cols].to_numpy(dtype=float)
    mean = x.mean(axis=0)
    x_centered = x - mean
    _, singular_values, vt = np.linalg.svd(x_centered, full_matrices=False)
    components = vt[:n_components, :]
    scores = x_centered @ components.T
    components, scores = orient_components(components, scores)

    explained_variance = (singular_values**2) / (x.shape[0] - 1)
    explained_ratio = explained_variance / explained_variance.sum()
    pca_variance = pd.DataFrame(
        {
            "component": [f"PC{i}" for i in range(1, len(explained_ratio) + 1)],
            "explained_variance": explained_variance,
            "explained_variance_ratio": explained_ratio,
            "cumulative_explained_variance_ratio": np.cumsum(explained_ratio),
        }
    )
    pca_variance.to_csv(dirs["results"] / "pca_explained_variance.csv", index=False)

    loadings = pd.DataFrame(
        components.T,
        columns=[f"PC{i}" for i in range(1, n_components + 1)],
    )
    loadings.insert(0, "predictor", z_cols)
    loadings.to_csv(dirs["results"] / "pca_loadings.csv", index=False)

    score_df = pd.DataFrame(scores, columns=[f"PC{i}" for i in range(1, n_components + 1)])
    score_df.insert(0, "date", state["date"])
    score_df.to_csv(dirs["results"] / "pca_scores.csv", index=False)

    pca_model = {
        "feature_names": z_cols,
        "n_components": n_components,
        "mean": mean,
        "components": components,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_ratio,
    }
    save_pickle(dirs["models"] / "pca_model.pkl", pca_model)
    plot_pca_variance(pca_variance, dirs["figures"] / "pca_explained_variance.png")

    logging.info("PCA completed with %s components", n_components)
    return {
        "z_cols": z_cols,
        "raw_predictor_cols": raw_predictor_cols,
        "scores": scores,
        "score_df": score_df,
        "loadings": loadings,
        "variance": pca_variance,
        "model": pca_model,
    }


def plot_pca_variance(pca_variance: pd.DataFrame, output_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    first_five = pca_variance.iloc[:10]
    ax1.bar(first_five["component"], first_five["explained_variance_ratio"], color="#4C78A8")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_xlabel("Component")
    ax2 = ax1.twinx()
    ax2.plot(
        first_five["component"],
        first_five["cumulative_explained_variance_ratio"],
        color="#F58518",
        marker="o",
    )
    ax2.set_ylabel("Cumulative Ratio")
    ax1.set_title("PCA Explained Variance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@dataclass
class KMeansResult:
    n_clusters: int
    centers: np.ndarray
    labels: np.ndarray
    inertia: float
    n_iter: int
    seed: int


def kmeans_fit(
    x: np.ndarray,
    n_clusters: int,
    seed: int,
    n_init: int = 20,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> KMeansResult:
    best: KMeansResult | None = None
    for init_idx in range(n_init):
        rng = np.random.default_rng(seed + init_idx * 9973)
        centers = initialize_kmeans_pp(x, n_clusters, rng)
        labels = np.zeros(x.shape[0], dtype=int)
        for iteration in range(1, max_iter + 1):
            distances = squared_distances(x, centers)
            new_labels = distances.argmin(axis=1)
            new_centers = centers.copy()
            for cluster in range(n_clusters):
                members = x[new_labels == cluster]
                if len(members) == 0:
                    farthest = int(np.argmax(distances.min(axis=1)))
                    new_centers[cluster] = x[farthest]
                else:
                    new_centers[cluster] = members.mean(axis=0)
            shift = float(np.sqrt(((new_centers - centers) ** 2).sum()))
            centers = new_centers
            labels = new_labels
            if shift < tol:
                break
        inertia = float(squared_distances(x, centers).min(axis=1).sum())
        result = KMeansResult(n_clusters, centers, labels, inertia, iteration, seed + init_idx * 9973)
        if best is None or result.inertia < best.inertia:
            best = result
    require(best is not None, "KMeans failed to initialize")
    return best


def initialize_kmeans_pp(x: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    centers = np.empty((n_clusters, x.shape[1]), dtype=float)
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


def squared_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)


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


def viterbi(
    x: np.ndarray,
    startprob: np.ndarray,
    transmat: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
) -> np.ndarray:
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
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-5,
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
    best.init_history = [
        {**row, "selected_best": bool(row["seed"] == best.seed)}
        for row in init_history
    ]
    return best


def reorder_hmm_by_stress(
    hmm: HMMResult,
    state_df: pd.DataFrame,
    z_cols: list[str],
) -> tuple[HMMResult, dict[int, int], pd.Series]:
    stress = stress_composite(state_df)
    old_state_stress = pd.DataFrame({"old_state": hmm.labels, "stress": stress}).groupby("old_state")["stress"].mean()
    order = list(old_state_stress.sort_values().index)
    mapping = {old_state: new_state for new_state, old_state in enumerate(order)}
    inverse_order = np.array(order, dtype=int)
    new_labels = np.array([mapping[int(label)] for label in hmm.labels], dtype=int)
    new_gamma = hmm.gamma[:, inverse_order]
    new_startprob = hmm.startprob[inverse_order]
    new_transmat = hmm.transmat[np.ix_(inverse_order, inverse_order)]
    new_means = hmm.means[inverse_order]
    new_covars = hmm.covars[inverse_order]
    reordered = HMMResult(
        n_states=hmm.n_states,
        n_features=hmm.n_features,
        startprob=new_startprob,
        transmat=new_transmat,
        means=new_means,
        covars=new_covars,
        log_likelihood=hmm.log_likelihood,
        converged=hmm.converged,
        n_iter=hmm.n_iter,
        seed=hmm.seed,
        gamma=new_gamma,
        labels=new_labels,
        aic=hmm.aic,
        bic=hmm.bic,
        init_history=hmm.init_history,
    )
    return reordered, mapping, stress


def stress_composite(state_df: pd.DataFrame) -> pd.Series:
    components = pd.Series(0.0, index=state_df.index)
    additive = [
        "macro_vix_z",
        "macro_credit_spread_baa10y_z",
        "macro_adjusted_financial_conditions_z",
        "macro_real_yield_10y_z",
        "macro_dollar_chg_4w_z",
    ]
    subtractive = [
        "macro_net_liquidity_chg_4w_z",
        "macro_yield_curve_10y_2y_z",
    ]
    for col in additive:
        components += state_df[col]
    for col in subtractive:
        components -= state_df[col]
    return components / float(len(additive) + len(subtractive))


def state_profile_label(state_id: int, n_states: int) -> str:
    if n_states == 4:
        labels = [
            "candidate_lower_stress_profile",
            "candidate_moderate_stress_profile",
            "candidate_elevated_stress_profile",
            "candidate_highest_stress_profile",
        ]
        return labels[state_id]
    return f"candidate_stress_rank_{state_id}_of_{n_states - 1}"


def average_duration(transmat: np.ndarray) -> np.ndarray:
    diag = np.diag(transmat)
    durations = np.empty_like(diag)
    for i, value in enumerate(diag):
        durations[i] = np.inf if value >= 1.0 else 1.0 / max(1e-12, 1.0 - value)
    return durations


def make_hmm_state_outputs(
    hmm: HMMResult,
    state: pd.DataFrame,
    raw_predictor_cols: list[str],
    z_cols: list[str],
    stress: pd.Series,
    prefix: str,
    dirs: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = pd.DataFrame(
        {
            "date": state["date"],
            f"{prefix}_state": [f"state_{value}" for value in hmm.labels],
            f"{prefix}_state_id": hmm.labels,
            f"{prefix}_state_posterior_probability": hmm.gamma.max(axis=1),
            "macro_stress_composite": stress,
        }
    )
    for i in range(hmm.n_states):
        labels[f"{prefix}_prob_state_{i}"] = hmm.gamma[:, i]
    labels.to_csv(dirs["results"] / f"{prefix}_state_labels.csv", index=False)

    profile_rows = []
    durations = average_duration(hmm.transmat)
    for state_id in range(hmm.n_states):
        mask = hmm.labels == state_id
        frame = state.loc[mask]
        row: dict[str, Any] = {
            "state": f"state_{state_id}",
            "state_id": state_id,
            "candidate_profile": state_profile_label(state_id, hmm.n_states),
            "n_weeks": int(mask.sum()),
            "sample_share": float(mask.mean()),
            "average_duration_weeks": float(durations[state_id]),
            "macro_stress_composite_mean": float(stress.loc[mask].mean()),
        }
        for col in raw_predictor_cols + z_cols:
            row[f"{col}_mean"] = float(frame[col].mean())
        profile_rows.append(row)
    profiles = pd.DataFrame(profile_rows)
    profiles.to_csv(dirs["results"] / f"{prefix}_state_profiles.csv", index=False)

    trans = pd.DataFrame(
        hmm.transmat,
        index=[f"from_state_{i}" for i in range(hmm.n_states)],
        columns=[f"to_state_{i}" for i in range(hmm.n_states)],
    )
    trans.to_csv(dirs["results"] / f"{prefix}_transition_matrix.csv")
    return labels, profiles, trans


def compute_state_models(
    inputs: dict[str, Any],
    pca: dict[str, Any],
    dirs: dict[str, Path],
    seed: int,
) -> dict[str, Any]:
    state = inputs["state"].copy()
    x = pca["scores"]
    raw_predictor_cols = pca["raw_predictor_cols"]
    z_cols = pca["z_cols"]

    hmm_results: dict[int, HMMResult] = {}
    hmm_profiles: dict[int, pd.DataFrame] = {}
    hmm_labels: dict[int, pd.DataFrame] = {}
    robustness_rows = []
    hmm_init_audit_rows = []
    for n_states in [3, 4, 5]:
        hmm = fit_diag_gaussian_hmm(x, n_states=n_states, seed=seed, n_init=10)
        reordered, mapping, stress = reorder_hmm_by_stress(hmm, state, z_cols)
        hmm_results[n_states] = reordered
        prefix = f"hmm{n_states}"
        labels, profiles, _ = make_hmm_state_outputs(
            reordered, state, raw_predictor_cols, z_cols, stress, prefix, dirs
        )
        hmm_labels[n_states] = labels
        hmm_profiles[n_states] = profiles
        save_pickle(dirs["models"] / f"{prefix}_model.pkl", reordered)
        for row in reordered.init_history:
            hmm_init_audit_rows.append({"model": prefix, "n_states": n_states, **row})
        counts = pd.Series(reordered.labels).value_counts().sort_index()
        robustness_rows.append(
            {
                "model": prefix,
                "method": "diagonal_gaussian_hmm",
                "n_states": n_states,
                "n_pca_components": x.shape[1],
                "covariance_type": "diag",
                "log_likelihood": reordered.log_likelihood,
                "aic": reordered.aic,
                "bic": reordered.bic,
                "converged": reordered.converged,
                "n_iter": reordered.n_iter,
                "seed": reordered.seed,
                "inertia": np.nan,
                "min_state_share": float(counts.min() / len(reordered.labels)),
                "max_state_share": float(counts.max() / len(reordered.labels)),
                "state_counts": json.dumps({f"state_{k}": int(v) for k, v in counts.items()}),
                "mean_average_duration_weeks": float(np.nanmean(average_duration(reordered.transmat))),
                "state_order_mapping": json.dumps({str(k): int(v) for k, v in mapping.items()}),
            }
        )

    kmeans = kmeans_fit(x, n_clusters=4, seed=seed, n_init=30)
    kmeans_labels, kmeans_mapping, kmeans_stress = reorder_labels_by_stress(kmeans.labels, state)
    kmeans_out = KMeansResult(
        n_clusters=kmeans.n_clusters,
        centers=reorder_array_by_mapping(kmeans.centers, kmeans_mapping),
        labels=kmeans_labels,
        inertia=kmeans.inertia,
        n_iter=kmeans.n_iter,
        seed=kmeans.seed,
    )
    save_pickle(dirs["models"] / "kmeans4_model.pkl", kmeans_out)
    kmeans_label_df, kmeans_profile_df = make_kmeans_outputs(
        kmeans_out, state, raw_predictor_cols, z_cols, kmeans_stress, dirs
    )
    counts = pd.Series(kmeans_out.labels).value_counts().sort_index()
    robustness_rows.append(
        {
            "model": "kmeans4",
            "method": "kmeans",
            "n_states": 4,
            "n_pca_components": x.shape[1],
            "covariance_type": "",
            "log_likelihood": np.nan,
            "aic": np.nan,
            "bic": np.nan,
            "converged": True,
            "n_iter": kmeans_out.n_iter,
            "seed": kmeans_out.seed,
            "inertia": kmeans_out.inertia,
            "min_state_share": float(counts.min() / len(kmeans_out.labels)),
            "max_state_share": float(counts.max() / len(kmeans_out.labels)),
            "state_counts": json.dumps({f"state_{k}": int(v) for k, v in counts.items()}),
            "mean_average_duration_weeks": np.nan,
            "state_order_mapping": json.dumps({str(k): int(v) for k, v in kmeans_mapping.items()}),
        }
    )

    robustness_summary = pd.DataFrame(robustness_rows)
    robustness_summary.to_csv(dirs["results"] / "model_robustness_summary.csv", index=False)
    hmm_init_audit = pd.DataFrame(hmm_init_audit_rows)
    hmm_init_audit.to_csv(dirs["results"] / "hmm_initialization_audit.csv", index=False)
    plot_state_timeline(hmm_labels[4], dirs["figures"] / "hmm4_state_timeline.png")
    plot_transition_matrix(hmm_results[4].transmat, dirs["figures"] / "hmm4_transition_matrix.png")

    logging.info("State models completed")
    return {
        "hmm4": hmm_results[4],
        "hmm3": hmm_results[3],
        "hmm5": hmm_results[5],
        "hmm4_labels": hmm_labels[4],
        "hmm4_profiles": hmm_profiles[4],
        "kmeans4": kmeans_out,
        "kmeans4_labels": kmeans_label_df,
        "kmeans4_profiles": kmeans_profile_df,
        "robustness_summary": robustness_summary,
        "hmm_initialization_audit": hmm_init_audit,
    }


def reorder_labels_by_stress(labels: np.ndarray, state_df: pd.DataFrame) -> tuple[np.ndarray, dict[int, int], pd.Series]:
    stress = stress_composite(state_df)
    state_stress = pd.DataFrame({"old_state": labels, "stress": stress}).groupby("old_state")["stress"].mean()
    order = list(state_stress.sort_values().index)
    mapping = {old_state: new_state for new_state, old_state in enumerate(order)}
    new_labels = np.array([mapping[int(label)] for label in labels], dtype=int)
    return new_labels, mapping, stress


def reorder_array_by_mapping(array: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    order = [old for old, _ in sorted(mapping.items(), key=lambda item: item[1])]
    return array[np.array(order, dtype=int)]


def make_kmeans_outputs(
    kmeans: KMeansResult,
    state: pd.DataFrame,
    raw_predictor_cols: list[str],
    z_cols: list[str],
    stress: pd.Series,
    dirs: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.DataFrame(
        {
            "date": state["date"],
            "kmeans4_state": [f"state_{value}" for value in kmeans.labels],
            "kmeans4_state_id": kmeans.labels,
            "macro_stress_composite": stress,
        }
    )
    labels.to_csv(dirs["results"] / "kmeans4_state_labels.csv", index=False)

    profile_rows = []
    for state_id in range(kmeans.n_clusters):
        mask = kmeans.labels == state_id
        frame = state.loc[mask]
        row: dict[str, Any] = {
            "state": f"state_{state_id}",
            "state_id": state_id,
            "candidate_profile": state_profile_label(state_id, kmeans.n_clusters),
            "n_weeks": int(mask.sum()),
            "sample_share": float(mask.mean()),
            "macro_stress_composite_mean": float(stress.loc[mask].mean()),
        }
        for col in raw_predictor_cols + z_cols:
            row[f"{col}_mean"] = float(frame[col].mean())
        profile_rows.append(row)
    profiles = pd.DataFrame(profile_rows)
    profiles.to_csv(dirs["results"] / "kmeans4_state_profiles.csv", index=False)
    return labels, profiles


def plot_state_timeline(labels: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.scatter(labels["date"], labels["hmm4_state_id"], c=labels["hmm4_state_id"], cmap="viridis", s=12)
    ax.set_yticks(sorted(labels["hmm4_state_id"].unique()))
    ax.set_yticklabels([f"state_{i}" for i in sorted(labels["hmm4_state_id"].unique())])
    ax.set_title("HMM-4 Macro State Timeline")
    ax.set_xlabel("Date")
    ax.set_ylabel("State")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_transition_matrix(transmat: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(transmat, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(transmat.shape[1]))
    ax.set_yticks(range(transmat.shape[0]))
    ax.set_xticklabels([f"to_{i}" for i in range(transmat.shape[1])])
    ax.set_yticklabels([f"from_{i}" for i in range(transmat.shape[0])])
    for i in range(transmat.shape[0]):
        for j in range(transmat.shape[1]):
            ax.text(j, i, f"{transmat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("HMM-4 Transition Matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def compute_state_conditioned_diagnostics(
    inputs: dict[str, Any],
    models: dict[str, Any],
    dirs: dict[str, Path],
) -> dict[str, Any]:
    state = inputs["state"].copy()
    report = inputs["cleaning_report"]
    main_cols = list(report["column_mapping"]["main_assets"].values())
    labels = models["hmm4_labels"][["date", "hmm4_state", "hmm4_state_id"]]
    panel = state.merge(labels, on="date", how="inner")

    summary_rows = []
    for state_id, frame in panel.groupby("hmm4_state_id"):
        btc = frame["ret_btc"]
        var_95, cvar_95 = var_cvar(btc)
        vol = float(btc.std(ddof=1))
        mean = float(btc.mean())
        summary_rows.append(
            {
                "state": f"state_{int(state_id)}",
                "state_id": int(state_id),
                "n_weeks": int(frame.shape[0]),
                "sample_share": float(frame.shape[0] / panel.shape[0]),
                "btc_mean_weekly": mean,
                "btc_median_weekly": float(btc.median()),
                "btc_volatility_weekly": vol,
                "btc_annualized_mean_arithmetic": mean * TRADING_WEEKS_PER_YEAR,
                "btc_annualized_volatility": vol * math.sqrt(TRADING_WEEKS_PER_YEAR),
                "btc_min_weekly": float(btc.min()),
                "btc_max_weekly": float(btc.max()),
                "btc_var_95_weekly": var_95,
                "btc_cvar_95_weekly": cvar_95,
                "btc_positive_week_share": float((btc > 0).mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("state_id")
    summary.to_csv(dirs["results"] / "state_conditioned_btc_summary.csv", index=False)

    corr_rows = []
    for state_id, frame in panel.groupby("hmm4_state_id"):
        for col in [asset for asset in main_cols if asset != "ret_btc"]:
            aligned = frame[["ret_btc", col]].dropna()
            corr_rows.append(
                {
                    "state": f"state_{int(state_id)}",
                    "state_id": int(state_id),
                    "asset": col,
                    "n_weeks": int(aligned.shape[0]),
                    "correlation_with_btc": float(aligned["ret_btc"].corr(aligned[col]))
                    if aligned.shape[0] >= 3
                    else np.nan,
                }
            )
    correlations = pd.DataFrame(corr_rows).sort_values(["state_id", "asset"])
    correlations.to_csv(dirs["results"] / "state_conditioned_correlations.csv", index=False)
    plot_state_conditioned_btc(summary, dirs["figures"] / "state_conditioned_btc_summary.png")

    logging.info("State-conditioned BTC diagnostics completed")
    return {"summary": summary, "correlations": correlations, "panel": panel}


def write_explainability_artifacts(
    args: argparse.Namespace,
    inputs: dict[str, Any],
    validation: dict[str, Any],
    pca: dict[str, Any],
    models: dict[str, Any],
    dirs: dict[str, Path],
) -> dict[str, str]:
    report = inputs["cleaning_report"]
    paths = inputs["paths"]
    asset = inputs["asset"]
    state = inputs["state"]
    robustness = inputs["robustness"]

    data_lineage_rows = [
        {
            "artifact": "asset_returns_main_weekly",
            "path": str(paths["asset_returns_main_weekly"]),
            "role": "Main weekly asset returns for unconditional BTC diagnostics.",
            "rows": int(len(asset)),
            "start_date": date_string(asset["date"], "min"),
            "end_date": date_string(asset["date"], "max"),
            "sha256": inputs["input_hashes"]["asset_returns_main_weekly"],
            "modified_by_part1_runner": False,
        },
        {
            "artifact": "state_model_panel_weekly",
            "path": str(paths["state_model_panel_weekly"]),
            "role": "Main macro-state panel with raw lagged predictors and full-sample z-scores.",
            "rows": int(len(state)),
            "start_date": date_string(state["date"], "min"),
            "end_date": date_string(state["date"], "max"),
            "sha256": inputs["input_hashes"]["state_model_panel_weekly"],
            "modified_by_part1_runner": False,
        },
        {
            "artifact": "robustness_weekly_panel",
            "path": str(paths["robustness_weekly_panel"]),
            "role": "BTC source validation and extended variables not used in the main state model.",
            "rows": int(len(robustness)),
            "start_date": date_string(robustness["date"], "min"),
            "end_date": date_string(robustness["date"], "max"),
            "sha256": inputs["input_hashes"]["robustness_weekly_panel"],
            "modified_by_part1_runner": False,
        },
        {
            "artifact": "cleaning_report",
            "path": str(paths["cleaning_report"]),
            "role": "Frozen data-cleaning audit source used to derive variable mappings.",
            "rows": np.nan,
            "start_date": "",
            "end_date": "",
            "sha256": inputs["input_hashes"]["cleaning_report"],
            "modified_by_part1_runner": False,
        },
    ]
    data_lineage = pd.DataFrame(data_lineage_rows)
    data_lineage_path = dirs["results"] / "data_lineage.csv"
    data_lineage.to_csv(data_lineage_path, index=False)

    variable_rows = []
    for source_col, output_col in report["column_mapping"]["main_assets"].items():
        variable_rows.append(
            {
                "panel": "asset_returns_main_weekly",
                "source_column": source_col,
                "output_column": output_col,
                "variable_type": "weekly_simple_return",
                "timing": "week-ending Friday",
                "used_in_part1": True,
                "used_for_state_model": False,
                "transformation": "Renamed from cleaned weekly simple return.",
                "interpretation_note": "Asset return used for BTC role diagnostics; no winsorization.",
            }
        )
    for source_col, output_col in report["column_mapping"]["state_predictors"].items():
        variable_rows.append(
            {
                "panel": "state_model_panel_weekly",
                "source_column": source_col,
                "output_column": output_col,
                "variable_type": "raw_lagged_macro_predictor",
                "timing": "one-week lagged macro predictor",
                "used_in_part1": True,
                "used_for_state_model": False,
                "transformation": "Renamed from the cleaning-report mapping.",
                "interpretation_note": "Retained for macro-state profile interpretation.",
            }
        )
        variable_rows.append(
            {
                "panel": "state_model_panel_weekly",
                "source_column": source_col,
                "output_column": f"{output_col}_z",
                "variable_type": "full_sample_zscore_macro_predictor",
                "timing": "one-week lagged macro predictor, standardized on the full state sample",
                "used_in_part1": True,
                "used_for_state_model": True,
                "transformation": "Population z-score computed during data cleaning.",
                "interpretation_note": "Input to PCA; raw column remains available for interpretation.",
            }
        )
    for source_col, output_col in report["column_mapping"]["robustness"].items():
        variable_rows.append(
            {
                "panel": "robustness_weekly_panel",
                "source_column": source_col,
                "output_column": output_col,
                "variable_type": "robustness_or_extended_variable",
                "timing": "week-ending Friday where available",
                "used_in_part1": output_col in {"ret_btc", "ret_btc_coinmetrics_usd"},
                "used_for_state_model": False,
                "transformation": "Renamed from cleaned robustness mapping.",
                "interpretation_note": "Not part of the main macro-state model.",
            }
        )
    variable_dictionary = pd.DataFrame(variable_rows)
    variable_dictionary_path = dirs["results"] / "variable_dictionary.csv"
    variable_dictionary.to_csv(variable_dictionary_path, index=False)

    assumption_audit = {
        "purpose": "Make Part 1 modeling choices auditable before writing thesis discussion.",
        "assumptions": [
            {
                "choice": "Full-sample descriptive state identification",
                "rationale": "Part 1 identifies macro-state structure and BTC state dependence; it is not a trading or allocation backtest.",
                "implication": "State labels are ex-post descriptive and must not be presented as real-time tradable signals.",
                "evidence_artifacts": ["hmm4_state_labels.csv", "hmm4_state_profiles.csv", "run_manifest.json"],
            },
            {
                "choice": "PCA with five components",
                "rationale": "Five components retain about 84% of macro predictor variance in the cleaned sample.",
                "implication": "Macro dimensions are compressed before HMM estimation, reducing noise and parameter burden.",
                "evidence_artifacts": ["pca_explained_variance.csv", "pca_loadings.csv"],
            },
            {
                "choice": "Four-state diagonal Gaussian HMM as the main model",
                "rationale": "Four states balance interpretability and regime-switching flexibility for a 425-week sample.",
                "implication": "State profiles require economic interpretation using macro means; state numbers alone are not labels.",
                "evidence_artifacts": [
                    "hmm4_state_profiles.csv",
                    "hmm4_transition_matrix.csv",
                    "hmm_initialization_audit.csv",
                ],
            },
            {
                "choice": "KMeans-4 and HMM-3/HMM-5 as auxiliary checks",
                "rationale": "Auxiliary models show whether the state structure is sensitive to one clustering method or one state count.",
                "implication": "Auxiliary outputs support discussion but do not replace the pre-specified main HMM-4 model.",
                "evidence_artifacts": ["model_robustness_summary.csv", "kmeans4_state_profiles.csv"],
            },
            {
                "choice": "No portfolio, risk-budget, or allocation rule in Part 1",
                "rationale": "This stage establishes BTC role diagnostics and macro-state labels before portfolio experiments.",
                "implication": "No result in Part 1 should be described as a recommended allocation or trading rule.",
                "evidence_artifacts": ["methodology_audit.md", "run_manifest.json"],
            },
        ],
        "limitations_to_discuss": [
            "Full-sample z-scores and full-sample HMM labels are descriptive rather than real-time.",
            "State candidate profiles are based on macro averages and require careful thesis discussion.",
            "BTC state-conditioned statistics are descriptive and may reflect small state sample sizes.",
            "KMeans-4 is auxiliary and can produce small clusters; it is not the main state model.",
        ],
    }
    assumption_audit_path = dirs["results"] / "model_assumption_audit.json"
    write_json(assumption_audit_path, assumption_audit)

    state_rationale = models["hmm4_profiles"][
        ["state", "state_id", "candidate_profile", "n_weeks", "sample_share", "macro_stress_composite_mean"]
    ].copy()
    state_rationale["state_ordering_rule"] = "ascending macro stress composite"
    state_rationale["interpretation_boundary"] = (
        "Candidate profile only; thesis discussion must interpret macro means before assigning economic names."
    )
    state_rationale_path = dirs["results"] / "state_labeling_rationale.csv"
    state_rationale.to_csv(state_rationale_path, index=False)

    methodology_text = f"""# Part 1 Methodology Audit

## Purpose
This run documents BTC asset role diagnostics and full-sample descriptive macro-state identification. It does not estimate portfolio weights, risk budgets, allocation rules, trading rules, or thesis conclusions.

## Frozen Inputs
- Main asset panel: `{paths["asset_returns_main_weekly"]}`.
- Macro-state panel: `{paths["state_model_panel_weekly"]}`.
- Robustness panel: `{paths["robustness_weekly_panel"]}`.
- Cleaning report: `{paths["cleaning_report"]}`.

Input hashes and sample windows are recorded in `run_manifest.json`, `validation_summary.json`, and `data_lineage.csv`.

## BTC Asset Role Diagnostics
The runner computes weekly descriptive statistics, weekly 5% VaR/CVaR, maximum drawdown, pairwise beta, the full return correlation matrix, and 52-week rolling correlations. Sharpe is saved only as a descriptive statistic and is not treated as a primary conclusion metric.

## Macro-State Identification
The state model uses the ten cleaning-report macro predictor mappings. Only the `_z` columns enter PCA. Raw predictor columns are retained for interpreting state profiles. PCA is fixed at five components. The main model is a four-state diagonal Gaussian HMM with multiple initializations and fixed seed `{args.seed}`. State ordering is deterministic: ascending macro stress composite.

## Robustness and Audit Trail
The runner saves HMM-3, HMM-5, and KMeans-4 auxiliary outputs. The file `hmm_initialization_audit.csv` records every HMM initialization, log-likelihood, convergence flag, iteration count, and whether it was selected as the best run.

## Discussion Boundaries
- State labels are ex-post descriptive labels, not real-time tradable signals.
- Candidate state profiles are not final economic names.
- State-conditioned BTC tables are descriptive diagnostics, not allocation recommendations.
- Portfolio construction, risk contribution, conditional allocation discipline, transaction costs, Autoencoder/LSTM, FRED-MD, CFTC, and ETF flow are outside Part 1.
"""
    methodology_path = dirs["results"] / "methodology_audit.md"
    methodology_path.write_text(methodology_text, encoding="utf-8")

    logging.info("Explainability artifacts completed")
    return {
        "data_lineage": str(data_lineage_path),
        "variable_dictionary": str(variable_dictionary_path),
        "model_assumption_audit": str(assumption_audit_path),
        "state_labeling_rationale": str(state_rationale_path),
        "hmm_initialization_audit": str(dirs["results"] / "hmm_initialization_audit.csv"),
        "methodology_audit": str(methodology_path),
    }


def plot_state_conditioned_btc(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(summary["state"], summary["btc_mean_weekly"], color="#4C78A8")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("BTC Mean Weekly Return by HMM-4 State")
    axes[0].set_ylabel("Mean Weekly Return")
    axes[1].bar(summary["state"], summary["btc_cvar_95_weekly"], color="#E45756")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("BTC Weekly CVaR 95 by HMM-4 State")
    axes[1].set_ylabel("CVaR 95")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_manifest(
    args: argparse.Namespace,
    run_id: str,
    dirs: dict[str, Path],
    inputs: dict[str, Any],
    validation: dict[str, Any],
    pca: dict[str, Any],
    models: dict[str, Any],
    explainability: dict[str, str],
) -> dict[str, Any]:
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "objective": "Part 1 BTC asset role diagnostics and macro-state identification",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "run_dir": str(dirs["root"]),
        "input_hashes": inputs["input_hashes"],
        "package_versions": package_versions(),
        "random_seed": args.seed,
        "sample": {
            "asset": validation["asset_sample"],
            "state": validation["state_sample"],
            "robustness": validation["robustness_sample"],
        },
        "parameters": {
            "pca": {
                "input": "10 full-sample z-score lagged macro predictors",
                "n_components": int(pca["model"]["n_components"]),
            },
            "hmm": {
                "main_model": "hmm4",
                "n_states_main": 4,
                "robustness_states": [3, 5],
                "covariance_type": "diag",
                "n_init": 10,
                "max_iter": 300,
                "tol": 1e-5,
                "state_ordering": "ascending macro stress composite",
                "timing": "full-sample ex-post descriptive",
            },
            "kmeans": {
                "n_states": 4,
                "n_init": 30,
                "state_ordering": "ascending macro stress composite",
            },
            "tail_alpha": TAIL_ALPHA,
            "rolling_correlation_window_weeks": 52,
        },
        "model_diagnostics": {
            "hmm4_log_likelihood": models["hmm4"].log_likelihood,
            "hmm4_aic": models["hmm4"].aic,
            "hmm4_bic": models["hmm4"].bic,
            "hmm4_converged": models["hmm4"].converged,
            "hmm4_n_iter": models["hmm4"].n_iter,
            "kmeans4_inertia": models["kmeans4"].inertia,
        },
        "outputs": {
            "checkpoints": str(dirs["checkpoints"]),
            "results": str(dirs["results"]),
            "figures": str(dirs["figures"]),
            "models": str(dirs["models"]),
            "logs": str(dirs["logs"]),
            "explainability_artifacts": explainability,
        },
        "scope_notes": [
            "No portfolio construction in Part 1.",
            "No risk-budget experiment in Part 1.",
            "No conditional allocation rule or trading strategy in Part 1.",
            "No Autoencoder, LSTM, FRED-MD, CFTC, or ETF flow in Part 1.",
            "Notebook and code avoid writing empirical conclusions.",
        ],
    }
    write_json(dirs["root"] / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    run_id = args.run_id or now_run_id()
    run_dir = args.output_dir / run_id
    dirs = ensure_dirs(run_dir)
    setup_logging(dirs["logs"])
    logging.info("Starting Part 1 run: %s", run_id)

    inputs = read_inputs(args.input_dir)
    enforce_resume_input_hashes(args, dirs, inputs["input_hashes"])
    validation = load_or_run(
        dirs,
        "01_validation",
        args.resume,
        lambda: validate_inputs(inputs, dirs, args),
    )
    asset_diagnostics = load_or_run(
        dirs,
        "02_asset_diagnostics",
        args.resume,
        lambda: compute_asset_diagnostics(inputs, dirs),
    )
    pca = load_or_run(
        dirs,
        "03_pca",
        args.resume,
        lambda: compute_pca(inputs, dirs, n_components=5),
    )
    models = load_or_run(
        dirs,
        "04_state_models",
        args.resume,
        lambda: compute_state_models(inputs, pca, dirs, seed=args.seed),
    )
    state_conditioned = load_or_run(
        dirs,
        "05_state_conditioned_diagnostics",
        args.resume,
        lambda: compute_state_conditioned_diagnostics(inputs, models, dirs),
    )
    explainability = load_or_run(
        dirs,
        "06_explainability_artifacts",
        args.resume,
        lambda: write_explainability_artifacts(args, inputs, validation, pca, models, dirs),
    )
    manifest = build_manifest(args, run_id, dirs, inputs, validation, pca, models, explainability)
    logging.info("Completed Part 1 run: %s", run_id)
    logging.info("Results directory: %s", dirs["results"])
    _ = asset_diagnostics, state_conditioned, explainability, manifest


if __name__ == "__main__":
    main()
