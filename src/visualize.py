"""
src/visualize.py
Requisito 6 - Visualizaciones interactivas con Plotly
"""

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

TARGET = "CO(GT)"

PALETTE = {
    "red":    "#E53935",
    "blue":   "#1565C0",
    "teal":   "#26A69A",
    "orange": "#FB8C00",
    "amber":  "#FFA726",
    "purple": "#7B1FA2",
    "light":  "#42A5F5",
    "white":  "#FFFFFF"
}


def plot_distribution(df,var) -> go.Figure:
    """Distribuciones original y log-transformada de las variables"""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Distribución original", "Log-transformada"])
    vals = df[var].dropna()
    fig.add_trace(go.Histogram(x=vals, nbinsx=50, marker_color=PALETTE["red"],
                               opacity=0.85, name=f"{var} original"), row=1, col=1)
    fig.add_vline(x=float(vals.median()), line_dash="dash", line_color="black",
                  annotation_text="Mediana", row=1, col=1)
    fig.update_traces(marker_line_color = "white",
                    marker_line_width = 1,
                    opacity = 0.95)
    log_vals = np.log1p(vals)
    fig.add_trace(go.Histogram(x=log_vals, nbinsx=50, marker_color=PALETTE["orange"],
                               opacity=0.85, name=f"log(1+{var})"), row=1, col=2)
    fig.update_layout(title_text=f"Distribuciones de la variable {var}",
                      title_font_size=16, showlegend=False, height=420)
    fig.update_xaxes(title_text=var, row=1, col=1)
    fig.update_xaxes(title_text=f"log(1+{var})", row=1, col=2)
    fig.update_traces(marker_line_color = "white",
                    marker_line_width = 1,
                    opacity = 0.95)
    return fig


def plot_by_hour(df,var) -> go.Figure:
    """Niveles medios por hora del día con barras de error."""
    hourly = df.groupby("hour")[var].agg(["mean", "std"]).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly["hour"], y=hourly["mean"],
        error_y=dict(type="data", array=hourly["std"], visible=True,
                     color=PALETTE["white"]),
        marker_color=PALETTE["red"], opacity=0.85, name=f"{var} medio",
    ))
    for x0, x1, color, label in [
        (7, 9,  PALETTE["light"],  "Hora punta mañana"),
        (17, 20, PALETTE["light"], "Hora punta tarde"),
    ]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, line_width=0,
                      annotation_text=label, annotation_position="top", opacity = 0.25)
    fig.update_layout(
        title=f"Nivel medio de {var} por hora del día",
        xaxis_title="Hora", yaxis_title="cantidad",
        xaxis=dict(tickmode="linear", dtick=1), height=450,
    )
    return fig


def plot_correlation_heatmap(df) -> go.Figure:
    """Mapa de calor de correlaciones entre contaminantes."""
    cols = [TARGET, "PT08.S1(CO)", "C6H6(GT)", "NOx(GT)", "NO2(GT)",
            "T", "RH", "AH", "sensor_mean"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdYlGn_r", zmid=0,
        text=corr.values, texttemplate="%{text}", textfont_size=10,
        colorbar_title="r",
    ))
    fig.update_layout(title="Matriz de correlación entre contaminantes",
                      height=520, xaxis=dict(tickangle=-30))
    return fig


def plot_by_month(df,var) -> go.Figure:
    """Niveles medianos por mes con diferenciación de grupos."""
    monthly = df.groupby("month")[var].median()
    months = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    colors = [PALETTE["red"] if v > monthly.median() else PALETTE["light"]
              for v in monthly.values]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[months[i - 1] for i in monthly.index],
        y=monthly.values, marker_color=colors, name=f"Mediana de {var}",
    ))
    fig.add_hline(y=float(monthly.median()), line_dash="dash", line_color="white",
                  annotation_text="Mediana global")
    fig.update_layout(title=f"Nivel de {var} mediano por mes",
                      xaxis_title="Mes", yaxis_title="cantidad", height=420)
    return fig


def plot_temp_vs_co(df) -> go.Figure:
    """Temperatura vs CO coloreado por categoría térmica."""
    sample = df.sample(n=min(2000, len(df)), random_state=42)
    color_map = {"frio": PALETTE["blue"], "templado": PALETTE["teal"],
                 "calido": PALETTE["amber"], "caluroso": PALETTE["red"]}
    fig = go.Figure()
    for cat, color in color_map.items():
        sub = sample[sample["temp_cat"] == cat]
        if len(sub):
            fig.add_trace(go.Scatter(
                x=sub["T"], y=sub[TARGET], mode="markers",
                marker=dict(color=color, size=5, opacity=0.5), name=cat,
            ))
    fig.update_layout(title="Temperatura vs CO por categoría térmica",
                      xaxis_title="Temperatura (°C)", yaxis_title="CO (mg/m³)", height=480)
    return fig


def plot_rush_hour_boxplot(df) -> go.Figure:
    """Boxplot CO en hora punta vs fuera de punta."""
    df = df.copy()
    df["hora_punta"] = df["is_rush_hour"].map({1: "Hora punta", 0: "Fuera de punta"})
    fig = go.Figure()
    for label, color in [("Hora punta", PALETTE["red"]), ("Fuera de punta", PALETTE["light"])]:
        vals = df[df["hora_punta"] == label][TARGET].dropna()
        fig.add_trace(go.Box(y=vals, name=label, marker_color=color, boxmean=True))
    fig.update_layout(title="CO en hora punta vs fuera de punta",
                      yaxis_title="CO (mg/m³)", height=450)
    return fig


def plot_dashboard(df) -> go.Figure:
    """Dashboard resumen con 6 subplots."""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=["Distribución CO", "Temperatura", "Humedad relativa",
                        "CO por hora", "Temp vs CO", "Pollution score"],
        vertical_spacing=0.14, horizontal_spacing=0.10,
    )
    fig.add_trace(go.Histogram(x=df[TARGET].dropna(), nbinsx=40,
                               marker_color=PALETTE["red"], showlegend=False), row=1, col=1)
    fig.add_trace(go.Histogram(x=df["T"].dropna(), nbinsx=40,
                               marker_color=PALETTE["light"], showlegend=False), row=1, col=2)
    fig.add_trace(go.Histogram(x=df["RH"].dropna(), nbinsx=40,
                               marker_color=PALETTE["teal"], showlegend=False), row=1, col=3)
    hourly = df.groupby("hour")[TARGET].mean()
    fig.add_trace(go.Scatter(x=hourly.index, y=hourly.values, mode="lines+markers",
                             marker=dict(size=4), line_color=PALETTE["red"],
                             showlegend=False), row=2, col=1)
    s = df.sample(min(800, len(df)), random_state=0)
    fig.add_trace(go.Scatter(x=s["T"], y=s[TARGET], mode="markers",
                             marker=dict(size=4, opacity=0.4, color=PALETTE["orange"]),
                             showlegend=False), row=2, col=2)
    if "pollution_score" in df.columns:
        vc = df["pollution_score"].value_counts().sort_index()
        fig.add_trace(go.Bar(x=vc.index.astype(str), y=vc.values,
                             marker_color=PALETTE["purple"], showlegend=False), row=2, col=3)
    fig.update_layout(title_text="Dashboard — Air Quality UCI Dataset",
                      title_font_size=16, height=640)
    return fig


def plot_ml_predictions(y_test, y_pred, model_name="Best Model") -> go.Figure:
    """Predicho vs real para el mejor modelo ML."""
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                             marker=dict(color=PALETTE["blue"], size=5, opacity=0.4),
                             name="Predicciones"))
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                             line=dict(color="red", dash="dash", width=2),
                             name="Predicción perfecta"))
    fig.update_layout(title=f"Predicho vs Real — {model_name}",
                      xaxis_title="CO real", yaxis_title="CO predicho", height=480)
    return fig


def plot_feature_importance(feature_cols, importances, model_name="Model") -> go.Figure:
    """Importancia de variables."""
    idx = np.argsort(importances)[::-1]
    fig = go.Figure(go.Bar(
        x=[feature_cols[i] for i in idx], y=importances[idx],
        marker_color=PALETTE["light"],
    ))
    fig.update_layout(title=f"Importancia de variables — {model_name}",
                      xaxis_tickangle=-40, yaxis_title="Importancia", height=450)
    return fig


def plot_dl_training(history_dict: dict) -> go.Figure:
    """Curvas de entrenamiento del modelo DL."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Loss (Huber)", "MAE"])
    for col_i, (metric, label) in enumerate([("loss", "Loss"), ("mae", "MAE")], start=1):
        fig.add_trace(go.Scatter(y=history_dict[metric], mode="lines",
                                 name=f"Train {label}", line=dict(color=PALETTE["blue"])),
                      row=1, col=col_i)
        val_key = f"val_{metric}"
        if val_key in history_dict:
            fig.add_trace(go.Scatter(y=history_dict[val_key], mode="lines",
                                     name=f"Val {label}",
                                     line=dict(color=PALETTE["orange"], dash="dash")),
                          row=1, col=col_i)
    fig.update_layout(title="Curvas de entrenamiento — Red Neuronal", height=400)
    return fig


def plot_dl_predictions(y_test, y_pred) -> go.Figure:
    return plot_ml_predictions(y_test, y_pred, model_name="Red Neuronal (DL)")


# ── Compatibilidad con pipeline_dagster.py y run_pipeline.py ────
# run() ya NO guarda PNGs; devuelve una lista con los nombres lógicos
# para que el asset de Dagster pueda registrar el recuento.
def run(df: pd.DataFrame) -> list:
    print("[viz] Generando figuras Plotly (sin guardar en disco)...")
    nombres = [
        "distribucion_co", "co_por_hora", "correlaciones",
        "co_por_mes", "temp_vs_co", "hora_punta", "dashboard",
    ]
    # Validamos que las figuras se construyen correctamente
    funcs = [
        plot_distribution, plot_by_hour, plot_correlation_heatmap,
        plot_by_month, plot_temp_vs_co, plot_rush_hour_boxplot, plot_dashboard,
    ]
    ok = []
    for nombre, fn in zip(nombres, funcs):
        try:
            fn(df)
            ok.append(nombre)
        except Exception as e:
            print(f"[viz] AVISO: {nombre} -> {e}")
    print(f"[viz] {len(ok)} figuras validadas.\n")
    return ok
