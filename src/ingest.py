"""
src/ingest.py
Requisito 2 - Importacion de datos desde multiples fuentes.
"""

import os
import json
import pandas as pd

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_CSV   = os.path.join(DATA_DIR, "airquality_raw.csv")
META_JSON = os.path.join(DATA_DIR, "metadata.json")


def load_from_csv_raw() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "AirQuality.csv")
    print("[ingest] Leyendo AirQuality.csv...")
    df = pd.read_csv(path, sep=";", decimal=",")
    print(f"[ingest] Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    return df


def save_clean_csv(df: pd.DataFrame) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(RAW_CSV, index=False)
    print(f"[ingest] CSV limpio guardado -> {RAW_CSV}")


def load_from_clean_csv() -> pd.DataFrame:
    print(f"[ingest] Leyendo CSV limpio: {RAW_CSV}")
    return pd.read_csv(RAW_CSV)


def save_metadata(df: pd.DataFrame) -> None:
    meta = {
        "nombre":    "Air Quality UCI Dataset",
        "origen":    "UCI Machine Learning Repository / Kaggle",
        "filas":     int(df.shape[0]),
        "columnas":  int(df.shape[1]),
        "variables": list(df.columns),
        "target":    "CO(GT)",
        "descripcion": "Mediciones horarias de contaminantes del aire en Italia (2004-2005)"
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[ingest] Metadatos JSON guardados -> {META_JSON}")


def run() -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    df = load_from_csv_raw()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    save_clean_csv(df)
    df = load_from_clean_csv()
    save_metadata(df)
    print("[ingest] Ingesta completa.\n")
    return df


if __name__ == "__main__":
    df = run()
    print(df.head())
