"""Diagnóstico estadístico de inflación de ceros, sobredispersión y estacionalidad."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def compute_zero_inflation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas de conteo, dispersión y ceros esperados bajo Poisson vs Binomial Negativa por cantón."""
    rows = []
    for canton, grp in df.groupby("canton"):
        y = grp["casos"].to_numpy(dtype=float)
        mean = float(np.mean(y))
        var = float(np.var(y, ddof=1))
        p0_obs = float(np.mean(y == 0))
        p0_poisson = float(np.exp(-mean))

        if var > mean and mean > 0:
            r_nb = (mean ** 2) / (var - mean)
            p0_nb = float((r_nb / (r_nb + mean)) ** r_nb)
        else:
            r_nb = np.inf
            p0_nb = p0_poisson

        rows.append(
            {
                "canton": int(canton),
                "province": int(grp["province"].iloc[0]),
                "urb": int(grp["urb"].iloc[0]),
                "n_weeks": int(len(y)),
                "total_casos": int(np.sum(y)),
                "mean": mean,
                "var": var,
                "dispersion_idx": var / mean if mean > 0 else np.nan,
                "p0_obs": p0_obs,
                "p0_poisson": p0_poisson,
                "p0_negbin": p0_nb,
                "excess_zero_vs_poisson": p0_obs - p0_poisson,
                "excess_zero_vs_negbin": p0_obs - p0_nb,
                "nb_r": r_nb,
            }
        )
    return pd.DataFrame(rows).sort_values("canton").reset_index(drop=True)


def generate_eda_figures(df: pd.DataFrame, canton_diag: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Genera las figuras principales de auditoría de ceros, sobredispersión y dinámica temporal."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    paths: list[Path] = []

    # 1. Serie nacional semanal + divisiones propuestas (Train / Val / Test)
    weekly = df.groupby(["time_idx", "year", "week"], as_index=False)["casos"].sum()
    val_start = int(weekly.loc[weekly["year"] == 2024, "time_idx"].min())
    test_start = int(weekly.loc[weekly["year"] == 2025, "time_idx"].min())

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(weekly["time_idx"], weekly["casos"], color="#1f77b4", lw=1.8, label="Casos semanales (82 cantones)")
    ax.axvline(val_start, color="#ff7f0e", ls="--", lw=1.6, label="Inicio Val (2024-W01)")
    ax.axvline(test_start, color="#d62728", ls="--", lw=1.6, label="Inicio Test (2025-W01)")
    ax.axvspan(0, val_start, color="#1f77b4", alpha=0.05)
    ax.axvspan(val_start, test_start, color="#ff7f0e", alpha=0.07)
    ax.axvspan(test_start, weekly["time_idx"].max(), color="#d62728", alpha=0.06)

    year_ticks = weekly.groupby("year")["time_idx"].first()
    ax.set_xticks(year_ticks.values)
    ax.set_xticklabels([str(y) for y in year_ticks.index])
    ax.set_title("Dinámica Epidémica Nacional del Dengue en Costa Rica (2020–2025)")
    ax.set_xlabel("Año epidemiológico")
    ax.set_ylabel("Total de casos semanales")
    ax.legend(loc="upper left")
    fig.tight_layout()
    p1 = out_dir / "01_serie_nacional_splits.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(p1)

    # 2. Diagnóstico de inflación de ceros y sobredispersión por cantón
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel izquierdo: P(Y=0) observado vs esperado bajo Poisson y NegBin
    ax = axes[0]
    ax.scatter(canton_diag["p0_poisson"], canton_diag["p0_obs"], alpha=0.75, label="vs Poisson", color="#d62728", s=38)
    ax.scatter(canton_diag["p0_negbin"], canton_diag["p0_obs"], alpha=0.8, label="vs Binomial Negativa", color="#2ca02c", s=38, marker="^")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Ajuste exacto (y = x)")
    ax.set_title("Proporción de Ceros: Observada vs Esperada por Cantón")
    ax.set_xlabel("P(Y = 0) esperada por el modelo")
    ax.set_ylabel("P(Y = 0) observada")
    ax.legend()

    # Panel derecho: Relación Media - Varianza en escala log-log
    ax = axes[1]
    pos = canton_diag[canton_diag["mean"] > 0]
    ax.scatter(pos["mean"], pos["var"], c=pos["p0_obs"], cmap="viridis", s=45, alpha=0.85)
    m_grid = np.logspace(np.log10(pos["mean"].min()), np.log10(pos["mean"].max()), 100)
    ax.plot(m_grid, m_grid, "k--", lw=1.2, label="Poisson (Var = Media)")
    ax.plot(m_grid, m_grid + 2.0 * (m_grid ** 2), color="#ff7f0e", ls="-.", lw=1.4, label="Sobredispersión NB (r=0.5)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Sobredispersión Cantonal: Media vs Varianza (log-log)")
    ax.set_xlabel("Media semanal de casos por cantón")
    ax.set_ylabel("Varianza semanal de casos")
    ax.legend()

    fig.tight_layout()
    p2 = out_dir / "02_zero_inflation_y_sobredispersion.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(p2)

    # 3. Estacionalidad por semana epidemiológica + top/bottom heterogeneidad espacial
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    seas = df.groupby("week", as_index=False).agg(
        mean_casos=("casos", "mean"),
        pct_zeros=("casos", lambda s: float((s == 0).mean() * 100)),
    )
    ax = axes[0]
    ax.bar(seas["week"], seas["mean_casos"], color="#1f77b4", alpha=0.85)
    ax.set_title("Perfil Estacional Promedio (Semana Epidemiológica 1–52)")
    ax.set_xlabel("Semana epidemiológica")
    ax.set_ylabel("Promedio de casos por cantón")

    ax2 = ax.twinx()
    ax2.plot(seas["week"], seas["pct_zeros"], color="#d62728", lw=1.8, label="% Ceros")
    ax2.set_ylabel("% de cantones con 0 casos", color="#d62728")
    ax2.grid(False)

    ax = axes[1]
    ax.hist(canton_diag["p0_obs"] * 100, bins=16, color="#4c72b0", edgecolor="white", alpha=0.9)
    ax.axvline(float((df["casos"] == 0).mean() * 100), color="#d62728", ls="--", lw=1.6, label="Media global (59.9%)")
    ax.set_title("Distribución del % de Ceros entre los 82 Cantones")
    ax.set_xlabel("% de semanas con 0 casos en el cantón")
    ax.set_ylabel("Número de cantones")
    ax.legend()

    fig.tight_layout()
    p3 = out_dir / "03_estacionalidad_y_heterogeneidad.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    paths.append(p3)

    return paths


def run_eda(clean_csv: Path = Path("data/processed/dengue_panel_clean.csv"), out_dir: Path = Path("data/processed/eda")) -> dict:
    """Ejecuta el análisis de ceros/sobredispersión y guarda reportes y gráficas."""
    df = pd.read_csv(clean_csv)
    canton_diag = compute_zero_inflation_table(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    canton_diag.to_csv(out_dir / "canton_zero_inflation_diagnostics.csv", index=False)

    fig_paths = generate_eda_figures(df, canton_diag, out_dir)

    y = df["casos"].to_numpy(dtype=float)
    mean_g = float(np.mean(y))
    var_g = float(np.var(y, ddof=1))
    r_g = (mean_g ** 2) / (var_g - mean_g)
    summary = {
        "n_obs": int(len(df)),
        "n_cantons": int(df["canton"].nunique()),
        "n_weeks": int(df["time_idx"].nunique()),
        "global_pct_zeros": round(float(np.mean(y == 0) * 100), 2),
        "global_mean": round(mean_g, 4),
        "global_var": round(var_g, 4),
        "global_dispersion_index": round(var_g / mean_g, 4),
        "global_poisson_p0_pct": round(float(np.exp(-mean_g) * 100), 2),
        "global_negbin_p0_pct": round(float(((r_g / (r_g + mean_g)) ** r_g) * 100), 2),
        "cantons_excess_zero_vs_poisson_gt_5pct": int((canton_diag["excess_zero_vs_poisson"] > 0.05).sum()),
        "cantons_excess_zero_vs_negbin_gt_5pct": int((canton_diag["excess_zero_vs_negbin"] > 0.05).sum()),
        "median_excess_zero_vs_negbin": round(float(canton_diag["excess_zero_vs_negbin"].median()), 4),
        "figures": [str(p) for p in fig_paths],
    }
    (out_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run_eda()
    print(json.dumps(result, indent=2, ensure_ascii=False))
