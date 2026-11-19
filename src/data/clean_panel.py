"""Módulo de limpieza y preparación del panel base de dengue sin rezagos precalculados."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

TARGET_COL = "casos"
LEAKAGE_COLS = ("rr", "clasi_rr", "clasi_rr_no_0")
ID_COLS = ("week_canton", "year", "week", "time_idx", "canton", "province")
STATIC_CANTON_COLS = ("urb",)
GLOBAL_WEEKLY_COLS = ("nino34ssta",)
CLIMATE_COLS = (
    "tmin_mean",
    "tmax_mean",
    "temp_prom",
    "precip_total",
    "precip_max",
    "precip_min",
    "precip_mean",
    "precip_median",
    "ndvi",
)
GOOGLE_TRENDS_COLS = (
    "aedes_aegypti",
    "criaderos_de_mosquito",
    "dengue",
    "fiebre",
    "plaquetas",
    "buscar_trabajo",
    "desempleo",
    "empleo",
    "comprar_carro",
    "préstamo",
    "tiquetes_avion",
    "vacaciones",
    "corte_agua",
    "corte_de_agua",
    "suspension_de_agua",
)


def parse_week_canton(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae year, week, canton y province desde week_canton y crea time_idx continuo."""
    out = df.copy()
    parts = out["week_canton"].str.split("-", expand=True).astype(int)
    out["year"] = parts[0]
    out["week"] = parts[1]
    out["canton"] = parts[2]
    out["province"] = out["canton"] // 100

    yw_pairs = (
        out[["year", "week"]]
        .drop_duplicates()
        .sort_values(["year", "week"])
        .reset_index(drop=True)
    )
    yw_pairs["time_idx"] = np.arange(len(yw_pairs), dtype=int)
    out = out.merge(yw_pairs, on=["year", "week"], how="left")

    angle = 2.0 * np.pi * (out["week"] - 1) / 52.0
    out["week_sin"] = np.sin(angle)
    out["week_cos"] = np.cos(angle)
    return out


def validate_panel(df: pd.DataFrame) -> dict:
    """Valida balance espacio-temporal, ausencia de nulos y duplicados en el panel."""
    if df["week_canton"].duplicated().any():
        raise ValueError("Existen filas duplicadas en 'week_canton'.")
    if df.isna().any().any():
        null_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Se encontraron valores nulos en columnas: {null_cols}")

    n_cantons = df["canton"].nunique()
    n_weeks = df["time_idx"].nunique()
    expected_rows = n_cantons * n_weeks
    if len(df) != expected_rows:
        raise ValueError(
            f"Panel desbalanceado: {len(df)} filas vs {expected_rows} esperadas "
            f"({n_cantons} cantones x {n_weeks} semanas)."
        )

    return {
        "n_rows": int(len(df)),
        "n_cantons": int(n_cantons),
        "n_weeks": int(n_weeks),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "yw_start": f"{int(df.sort_values('time_idx').iloc[0]['year'])}-W{int(df.sort_values('time_idx').iloc[0]['week']):02d}",
        "yw_end": f"{int(df.sort_values('time_idx').iloc[-1]['year'])}-W{int(df.sort_values('time_idx').iloc[-1]['week']):02d}",
    }


def load_and_clean_raw(
    raw_path: Path,
    drop_leakage: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Carga full_data.csv, elimina columnas *_lag_* y devuelve el panel limpio y metadatos."""
    df_raw = pd.read_csv(raw_path)
    lag_cols = [c for c in df_raw.columns if "_lag_" in c]
    non_lag_cols = [c for c in df_raw.columns if "_lag_" not in c]

    df = parse_week_canton(df_raw[non_lag_cols])

    if drop_leakage:
        cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
        df = df.drop(columns=cols_to_drop)

    df = df.sort_values(["canton", "time_idx"]).reset_index(drop=True)

    ordered_prefix = [
        c
        for c in ("week_canton", "time_idx", "year", "week", "week_sin", "week_cos", "canton", "province", TARGET_COL)
        if c in df.columns
    ]
    remaining = [c for c in df.columns if c not in ordered_prefix]
    df = df[ordered_prefix + remaining]

    summary = validate_panel(df)
    summary["raw_cols"] = int(df_raw.shape[1])
    summary["dropped_lag_cols"] = len(lag_cols)
    summary["dropped_leakage_cols"] = list(LEAKAGE_COLS) if drop_leakage else []
    summary["clean_cols"] = int(df.shape[1])
    return df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara el panel limpio de dengue sin lags precalculados.")
    parser.add_argument("--raw-path", type=Path, default=Path("data/raw/full_data.csv"))
    parser.add_argument("--out-path", type=Path, default=Path("data/processed/dengue_panel_clean.csv"))
    parser.add_argument("--keep-leakage", action="store_true", help="Mantiene rr, clasi_rr y clasi_rr_no_0.")
    args = parser.parse_args()

    df_clean, summary = load_and_clean_raw(args.raw_path, drop_leakage=not args.keep_leakage)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(args.out_path, index=False)

    print("Panel limpio guardado en:", args.out_path)
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
