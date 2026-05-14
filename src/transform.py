"""
src/transform.py
Requisito 3 - Transformaciones
Requisito 4 - Mapeo
Requisito 5 - Ordenacion
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_CSV = os.path.join(DATA_DIR, "airquality_processed.csv")

FEATURE_COLS = [
    "PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)",
    "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)",
    "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
    "hour", "month", "is_rush_hour", "temp_cat_num",
    "NO2_NOx_ratio", "sensor_mean"
]
TARGET_COL = "CO(GT)"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[transform] Filas originales: {len(df)}")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.dropna(subset=["Date", "Time"])
    df = df.replace(-200, np.nan)
    df = df.dropna(subset=[TARGET_COL])
    sensor_cols = [c for c in df.columns if c not in ["Date", "Time"]]
    df[sensor_cols] = df[sensor_cols].fillna(df[sensor_cols].median())
    print(f"[transform] Filas tras limpieza: {len(df)}\n")
    return df.reset_index(drop=True)


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H.%M.%S",
        errors="coerce"
    )
    df = df.dropna(subset=["datetime"])
    df["hour"]    = df["datetime"].dt.hour
    df["month"]   = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday
    print("[transform] Variables temporales creadas: hour, month, weekday\n")
    return df


def create_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["NO2_NOx_ratio"] = df["NO2(GT)"] / df["NOx(GT)"].replace(0, np.nan)
    sensor_cols = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]
    df["sensor_mean"] = df[sensor_cols].mean(axis=1)
    df["log_CO"] = np.log1p(df[TARGET_COL].clip(lower=0))
    print("[transform] Variables derivadas: NO2_NOx_ratio, sensor_mean, log_CO\n")
    return df


def aggregate_stats(df: pd.DataFrame) -> None:
    agg = df.groupby("hour")[TARGET_COL].agg(["mean", "median", "std"]).round(3)
    print("[transform] CO medio por hora del dia:")
    print(agg.to_string(), "\n")


def apply_mappings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_rush_hour"] = df["hour"].map(
        lambda h: 1 if h in range(7, 10) or h in range(17, 21) else 0
    )

    def temp_cat(t):
        if t < 5:    return "frio"
        elif t < 15: return "templado"
        elif t < 25: return "calido"
        else:        return "caluroso"

    df["temp_cat"]     = df["T"].map(temp_cat)
    df["temp_cat_num"] = df["temp_cat"].map({"frio": 0, "templado": 1, "calido": 2, "caluroso": 3})

    def pollution_level(row):
        score = 0
        if row[TARGET_COL] > 2:      score += 2
        if row["NOx(GT)"] > 200:     score += 1
        if row["NO2(GT)"] > 100:     score += 1
        if row["is_rush_hour"] == 1: score += 1
        return score

    df["pollution_score"] = df.apply(pollution_level, axis=1)
    print("[transform] Mapeos aplicados: is_rush_hour, temp_cat, pollution_score\n")
    return df


def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=[TARGET_COL, "NOx(GT)"], ascending=[False, False]).reset_index(drop=True)
    df["rank_co"] = df[TARGET_COL].rank(ascending=False, method="dense").astype(int)
    print("[transform] Dataset ordenado por CO(GT) desc, NOx(GT) desc")
    print(df[["datetime", TARGET_COL, "NOx(GT)", "temp_cat"]].head().to_string(), "\n")
    return df


def run(df: pd.DataFrame) -> pd.DataFrame:
    df = clean(df)
    df = parse_datetime(df)
    df = create_derived(df)
    aggregate_stats(df)
    df = apply_mappings(df)
    df = sort_data(df)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)
    print(f"[transform] Dataset procesado guardado -> {PROCESSED_CSV}\n")
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from ingest import run as ingest_run
    df = run(ingest_run())
    print(df.dtypes)
