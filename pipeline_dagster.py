import os
import sys
import importlib.util
import pandas as pd

from dagster import (
    asset, AssetExecutionContext, Definitions,
    define_asset_job, AssetSelection,
    MaterializeResult, MetadataValue
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, "src")


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SRC_DIR, f"{name}.py"))
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@asset(group_name="air_quality", description="Carga el CSV original de Air Quality UCI")
def raw_data(context: AssetExecutionContext) -> pd.DataFrame:
    ingest = load_module("ingest")
    df = ingest.run()
    context.log.info(f"Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    return df


@asset(group_name="air_quality", description="Limpieza, mapeo, variables derivadas y ordenacion")
def processed_data(context: AssetExecutionContext, raw_data: pd.DataFrame) -> pd.DataFrame:
    transform = load_module("transform")
    df = transform.run(raw_data)
    context.log.info(f"Dataset procesado: {df.shape[0]} filas")
    return df


@asset(group_name="air_quality", description="Valida las figuras Plotly interactivas (sin PNGs)")
def visualizations(context: AssetExecutionContext, processed_data: pd.DataFrame) -> MaterializeResult:
    visualize = load_module("visualize")
    nombres = visualize.run(processed_data)   # devuelve lista de nombres, no rutas
    context.log.info(f"{len(nombres)} figuras Plotly validadas")
    return MaterializeResult(
        metadata={"num_graficos": MetadataValue.int(len(nombres)),
                  "graficos": MetadataValue.text(", ".join(nombres))}
    )


@asset(group_name="air_quality", description="Entrena modelos ML con scikit-learn")
def ml_models(context: AssetExecutionContext, processed_data: pd.DataFrame) -> MaterializeResult:
    model_ml = load_module("model_ml")
    results  = model_ml.run(processed_data)
    best = max(results["results"], key=lambda r: r["R2"])
    context.log.info(f"Mejor modelo: {best['model']} R2={best['R2']}")
    return MaterializeResult(
        metadata={
            "mejor_modelo": MetadataValue.text(best["model"]),
            "R2":           MetadataValue.float(best["R2"]),
            "MAE":          MetadataValue.float(best["MAE"]),
        }
    )


@asset(group_name="air_quality", description="Entrena red neuronal con Keras", deps=["ml_models"])
def dl_model(context: AssetExecutionContext, processed_data: pd.DataFrame) -> MaterializeResult:
    model_dl = load_module("model_dl")
    results  = model_dl.run(processed_data)
    m = results["metrics"]
    context.log.info(f"DL entrenado: R2={m['R2']:.4f}")
    return MaterializeResult(
        metadata={
            "R2":   MetadataValue.float(float(m["R2"])),
            "MAE":  MetadataValue.float(float(m["MAE"])),
            "RMSE": MetadataValue.float(float(m["RMSE"])),
        }
    )


defs = Definitions(
    assets=[raw_data, processed_data, visualizations, ml_models, dl_model],
    jobs=[define_asset_job("air_quality_pipeline", AssetSelection.all())]
)
