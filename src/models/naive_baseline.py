"""Nivel 0: Auditoría de datos y líneas base ingenuas registradas en MLflow."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.utils.eda_zero_inflation import run_eda
from src.utils.mlflow_utils import init_mlflow


def compute_count_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Calcula métricas robustas para conteos epidemiológicos con exceso de ceros."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), a_min=0.0, a_max=None)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rmsle = float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))
    denom = float(np.sum(np.abs(y_true)))
    wape = float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else np.nan

    zero_mask = y_true == 0
    pos_mask = y_true > 0
    mae_zeros = float(np.mean(np.abs(y_true[zero_mask] - y_pred[zero_mask]))) if zero_mask.any() else np.nan
    mae_pos = float(np.mean(np.abs(y_true[pos_mask] - y_pred[pos_mask]))) if pos_mask.any() else np.nan

    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}mae": round(mae, 4),
        f"{p}rmse": round(rmse, 4),
        f"{p}rmsle": round(rmsle, 4),
        f"{p}wape": round(wape, 4),
        f"{p}mae_zeros": round(mae_zeros, 4),
        f"{p}mae_positives": round(mae_pos, 4),
    }


def evaluate_naive_models(df: pd.DataFrame) -> None:
    """Registra la auditoría EDA y dos baselines ingenuos (Persistencia y Estacional t-52) en MLflow."""
    df = df.sort_values(["canton", "time_idx"]).reset_index(drop=True)
    df["pred_persistence_t1"] = df.groupby("canton")["casos"].shift(1)
    df["pred_seasonal_t52"] = df.groupby("canton")["casos"].shift(52)

    val_mask = df["year"] == 2024
    test_mask = df["year"] == 2025

    tracking_uri = init_mlflow()
    print("MLflow tracking URI:", tracking_uri)

    # 1. Run de auditoría de datos y diagnóstico de ceros
    eda_summary = run_eda()
    with mlflow.start_run(run_name="00_eda_zero_inflation_audit"):
        mlflow.set_tags(
            {
                "stage": "eda",
                "dataset": "dengue_panel_clean",
                "split_policy": "train<=2023|val=2024|test=2025",
            }
        )
        mlflow.log_params(
            {
                "n_obs": eda_summary["n_obs"],
                "n_cantons": eda_summary["n_cantons"],
                "n_weeks": eda_summary["n_weeks"],
                "dropped_lag_cols": 232,
                "dropped_leakage_cols": "rr,clasi_rr,clasi_rr_no_0",
            }
        )
        mlflow.log_metrics(
            {
                "global_pct_zeros": eda_summary["global_pct_zeros"],
                "global_mean": eda_summary["global_mean"],
                "global_dispersion_index": eda_summary["global_dispersion_index"],
                "global_poisson_p0_pct": eda_summary["global_poisson_p0_pct"],
                "global_negbin_p0_pct": eda_summary["global_negbin_p0_pct"],
                "cantons_excess_zero_vs_poisson": eda_summary["cantons_excess_zero_vs_poisson_gt_5pct"],
                "cantons_excess_zero_vs_negbin": eda_summary["cantons_excess_zero_vs_negbin_gt_5pct"],
            }
        )
        for fig_path in eda_summary["figures"]:
            mlflow.log_artifact(fig_path, artifact_path="eda_figures")
        mlflow.log_artifact("data/processed/eda/canton_zero_inflation_diagnostics.csv", artifact_path="eda_tables")
        mlflow.log_artifact("data/processed/eda/eda_summary.json", artifact_path="eda_tables")

    # 2. Runs de referencia ingenua (benchmark mínimo)
    baselines = [
        ("00_naive_persistence_t1", "pred_persistence_t1", {"lag_weeks": 1, "model_family": "naive_persistence"}),
        ("00_naive_seasonal_t52", "pred_seasonal_t52", {"lag_weeks": 52, "model_family": "naive_seasonal"}),
    ]
    for run_name, pred_col, params in baselines:
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({"stage": "baseline_naive", "horizon": "1_week", "split_policy": "train<=2023|val=2024|test=2025"})
            mlflow.log_params(params)
            val_metrics = compute_count_metrics(df.loc[val_mask, "casos"], df.loc[val_mask, pred_col], prefix="val")
            test_metrics = compute_count_metrics(df.loc[test_mask, "casos"], df.loc[test_mask, pred_col], prefix="test")
            mlflow.log_metrics({**val_metrics, **test_metrics})
            print(f"{run_name}: val_mae={val_metrics['val_mae']} | val_rmsle={val_metrics['val_rmsle']} | test_mae={test_metrics['test_mae']} | test_rmsle={test_metrics['test_rmsle']}")


def main() -> None:
    clean_csv = Path("data/processed/dengue_panel_clean.csv")
    df = pd.read_csv(clean_csv)
    evaluate_naive_models(df)


if __name__ == "__main__":
    main()
