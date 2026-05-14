"""
app.py - Aplicación web Streamlit con gráficas Plotly interactivas
Uso: streamlit run app.py
"""

import os, sys, warnings, pickle
import numpy as np
import pandas as pd
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
PLOTS_DIR  = os.path.join(DATA_DIR, "plots")   # se mantiene para los PNGs de R
TARGET     = "CO(GT)"

st.set_page_config(page_title="Air Quality", page_icon="🌫️", layout="wide")

# ── Importar funciones de visualización Plotly ──────────────────
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from visualize import (
    plot_distribution, plot_by_hour, plot_correlation_heatmap,
    plot_by_month, plot_temp_vs_co, plot_rush_hour_boxplot, plot_dashboard,
    plot_ml_predictions, plot_feature_importance, plot_dl_training, plot_dl_predictions,
)


@st.cache_data
def load_data():
    p = os.path.join(DATA_DIR, "airquality_processed.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_resource
def load_models():
    ml, sc, fc = None, None, None
    try:
        ml = joblib.load(os.path.join(MODELS_DIR, "best_ml_model.pkl"))
        sc = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        fc = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
    except Exception:
        pass
    dl = None
    try:
        import tensorflow as tf
        dl_path = os.path.join(MODELS_DIR, "dl_model.keras")
        if os.path.exists(dl_path):
            dl = tf.keras.models.load_model(dl_path)
    except Exception:
        pass
    return ml, dl, sc, fc


@st.cache_data
def load_results():
    ml_path = os.path.join(DATA_DIR, "ml_results.csv")
    dl_path = os.path.join(DATA_DIR, "dl_results.csv")
    ml = pd.read_csv(ml_path) if os.path.exists(ml_path) else None
    dl = pd.read_csv(dl_path) if os.path.exists(dl_path) else None
    return ml, dl


df = load_data()
ml_model, dl_model, scaler, feature_cols = load_models()
ml_res, dl_res = load_results()

if df is None:
    st.error("Ejecuta primero:  python run_pipeline.py")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.title("🌫️ Air Quality UCI")
st.sidebar.markdown("*Proyecto Final - IA & Estadistica 2025-26*")
st.sidebar.divider()
page = st.sidebar.radio("Sección", ["Exploración", "Modelos", "Predicción", "Metodología"])


# ── EXPLORACIÓN ──────────────────────────────────────────────────
if page == "Exploración":
    st.title("Exploración del Dataset")
    st.markdown("**Air Quality UCI** — Mediciones horarias de contaminantes en Italia (2004-2005)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros",     f"{len(df):,}")
    c2.metric("CO medio",      f"{df[TARGET].mean():.2f} mg/m³")
    c3.metric("Temperatura media",   f"{df['T'].mean():.1f} °C")
    c4.metric("Humedad media", f"{df['RH'].mean():.1f} %")

    st.divider()
    st.subheader("Muestra del dataset")
    cols_show = [TARGET, "NOx(GT)", "NO2(GT)", "T", "RH", "hour", "temp_cat", "is_rush_hour"]
    cols_show = [c for c in cols_show if c in df.columns]
    st.dataframe(df[cols_show].head(100), use_container_width=True)

    st.divider()
    st.subheader("Visualizaciones interactivas")

    # Gráficas Plotly generadas al vuelo
    opciones_plotly = {
        "Distribuciones":    plot_distribution,
        "Niveles por hora":  plot_by_hour,
        "Niveles por mes":   plot_by_month,
        "Correlaciones":     plot_correlation_heatmap,
        "Temperatura vs CO": plot_temp_vs_co,
        "Hora punta":        plot_rush_hour_boxplot,
        "Dashboard":         plot_dashboard,
    }

    # PNGs de R (se muestran si existen; no son obligatorios)
    opciones_r = {
        "R - Temp boxplot": "12_r_temp_boxplot.png",
        "R - CO por hora":  "13_r_co_hora.png",
    }

    todas = list(opciones_plotly.keys()) + list(opciones_r.keys())
    sel = st.selectbox("Selecciona gráfico:", todas)

    if sel == "Distribuciones":
        sel_var = st.selectbox("Selecciona variable:", ["CO(GT)", "NOx(GT)", "NO2(GT)", "T", "RH"])
        with st.spinner("Generando gráfico..."):
            st.plotly_chart(opciones_plotly[sel](df,sel_var), use_container_width=True)

    elif sel == 'Niveles por hora':
        sel_var = st.selectbox("Selecciona variable:", ["CO(GT)", "NOx(GT)", "NO2(GT)", "T", "RH"])
        with st.spinner("Generando gráfico..."):
            st.plotly_chart(opciones_plotly[sel](df,sel_var), use_container_width=True)

    elif sel == 'Niveles por mes':
        sel_var = st.selectbox("Selecciona variable:", ["CO(GT)", "NOx(GT)", "NO2(GT)", "T", "RH"])
        with st.spinner("Generando gráfico..."):
            st.plotly_chart(opciones_plotly[sel](df,sel_var), use_container_width=True)
    
    elif sel == 'Correlaciones':
        with st.spinner("Generando gráfico..."):
            st.plotly_chart(opciones_plotly[sel](df), use_container_width=True)

    elif sel == 'Temperatura vs CO':
        with st.spinner("Generando gráfico..."):
            st.plotly_chart(opciones_plotly[sel](df), use_container_width=True)

    elif sel == 'Hora punta':
        with st.spinner("Generando gráfico..."):
            st.plotly_chart(opciones_plotly[sel](df), use_container_width=True)

    elif sel == 'Dashboard':
        with st.spinner("Generando gráfico..."):
            st.plotly_chart(opciones_plotly[sel](df), use_container_width=True)

    else:
        img = os.path.join(PLOTS_DIR, opciones_r[sel])
        if os.path.exists(img):
            st.image(img, use_container_width=True)
        else:
            st.info("Gráfico R no generado aún. Ejecuta:  Rscript src/r_analysis.R")

# ── MODELOS ──────────────────────────────────────────────────────
elif page == "Modelos":
    st.title("Resultados de Modelos")

    if ml_res is not None:
        st.subheader("Machine Learning (scikit-learn)")
        st.dataframe(ml_res.sort_values("R2", ascending=False), use_container_width=True)

    if dl_res is not None:
        st.subheader("Deep Learning (Keras)")
        st.dataframe(dl_res, use_container_width=True)

    st.divider()
    st.subheader("Gráficas de modelos")

    opciones_mod = [
        "Predicho vs Real (ML)",
        "Importancia de variables",
        "Curvas entrenamiento DL",
        "Predicho vs Real (DL)",
    ]
    sel_mod = st.radio("Selecciona gráfica:", opciones_mod, horizontal=True)

    if ml_model is not None and scaler is not None and feature_cols is not None:
        from sklearn.model_selection import train_test_split

        available = [c for c in feature_cols if c in df.columns]
        df_model  = df[available + [TARGET]].dropna()
        X_sc = scaler.transform(df_model[available].values)
        y    = df_model[TARGET].values
        _, X_test, _, y_test = train_test_split(X_sc, y, test_size=0.2, random_state=42)
        y_pred_ml = ml_model.predict(X_test)
        best_name = type(ml_model).__name__

        with st.spinner("Generando gráfica..."):
            if sel_mod == "Predicho vs Real (ML)":
                st.plotly_chart(plot_ml_predictions(y_test, y_pred_ml, best_name),
                                use_container_width=True)

            elif sel_mod == "Importancia de variables":
                if hasattr(ml_model, "feature_importances_"):
                    st.plotly_chart(
                        plot_feature_importance(available, ml_model.feature_importances_, best_name),
                        use_container_width=True)
                else:
                    st.info("El modelo no expone importancia de variables.")

            elif sel_mod == "Curvas entrenamiento DL":
                hist_path = os.path.join(DATA_DIR, "dl_history.pkl")
                if os.path.exists(hist_path):
                    with open(hist_path, "rb") as f:
                        hist = pickle.load(f)
                    st.plotly_chart(plot_dl_training(hist), use_container_width=True)
                else:
                    st.info("Historial DL no encontrado. Vuelve a ejecutar run_pipeline.py.")

            elif sel_mod == "Predicho vs Real (DL)":
                if dl_model is not None:
                    y_pred_dl = dl_model.predict(X_sc, verbose=0).flatten()
                    _, _, _, y_test_dl = train_test_split(X_sc, y, test_size=0.2, random_state=42)
                    st.plotly_chart(plot_dl_predictions(y_test_dl, y_pred_dl),
                                    use_container_width=True)
                else:
                    st.info("Modelo DL no encontrado.")
    else:
        st.warning("Modelos no encontrados. Ejecuta `run_pipeline.py` primero.")


# ── PREDICCIÓN ───────────────────────────────────────────────────
elif page == "Predicción":
    st.title("Predictor de Calidad del Aire")
    st.markdown("Ajusta los parámetros y el modelo estimará la concentración de CO.")

    if ml_model is None:
        st.error("Modelos no encontrados. Ejecuta run_pipeline.py primero.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Condiciones ambientales")
        T     = st.slider("Temperatura (°C)",      -5.0, 45.0, 18.0, 0.5)
        RH    = st.slider("Humedad relativa (%)",   0.0, 100.0, 50.0, 1.0)
        AH    = st.slider("Humedad absoluta",       0.0, 2.5, 1.0, 0.05)
        hour  = st.slider("Hora del día",           0, 23, 8)
        month = st.slider("Mes",                    1, 12, 6)

    with col2:
        st.subheader("Lecturas de sensores")
        s1   = st.slider("PT08.S1 (CO sensor)",   500, 2500, 1200, 10)
        c6h6 = st.slider("C6H6 Benceno",          0.0, 60.0, 10.0, 0.5)
        nmhc = st.slider("PT08.S2 (NMHC sensor)", 500, 2500, 1000, 10)
        nox  = st.slider("NOx (gt)",              0.0, 1000.0, 200.0, 5.0)
        s3   = st.slider("PT08.S3 (NOx sensor)",  500, 2500, 1000, 10)
        no2  = st.slider("NO2 (gt)",              0.0, 400.0, 100.0, 5.0)
        s4   = st.slider("PT08.S4 (NO2 sensor)",  500, 3000, 1500, 10)
        s5   = st.slider("PT08.S5 (O3 sensor)",   500, 3000, 1000, 10)

    if st.button("Predecir CO", type="primary"):
        is_rush    = 1 if hour in list(range(7, 10)) + list(range(17, 21)) else 0
        temp_num   = 0 if T < 5 else (1 if T < 15 else (2 if T < 25 else 3))
        no2_nox    = no2 / nox if nox > 0 else 0
        sensor_avg = np.mean([s1, nmhc, s3, s4, s5])

        row = {
            "PT08.S1(CO)": s1, "C6H6(GT)": c6h6, "PT08.S2(NMHC)": nmhc,
            "NOx(GT)": nox, "PT08.S3(NOx)": s3, "NO2(GT)": no2,
            "PT08.S4(NO2)": s4, "PT08.S5(O3)": s5, "T": T, "RH": RH, "AH": AH,
            "hour": hour, "month": month, "is_rush_hour": is_rush,
            "temp_cat_num": temp_num, "NO2_NOx_ratio": no2_nox, "sensor_mean": sensor_avg,
        }

        X_in = np.array([[row.get(c, 0) for c in feature_cols]])
        X_sc = scaler.transform(X_in)
        pred = ml_model.predict(X_sc)[0]

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("CO predicho (ML)", f"{pred:.2f} mg/m³")
        percentil = int((df[TARGET] < pred).mean() * 100)
        r2.metric("Percentil", f"{percentil}%",
                  help="Porcentaje de mediciones históricas superadas")
        if pred < 1:
            r3.success("🟢 Calidad del aire: BUENA")
        elif pred < 2:
            r3.warning("🟡 Calidad del aire: MODERADA")
        else:
            r3.error("🔴 Calidad del aire: MALA")

        if dl_model is not None:
            pred_dl = dl_model.predict(X_sc, verbose=0)[0][0]
            st.metric("CO predicho (Red Neuronal)", f"{pred_dl:.2f} mg/m³")

        # Gráfica interactiva: predicción en contexto histórico
        import plotly.graph_objects as go
        vals = df[TARGET].dropna()
        fig_ctx = go.Figure()
        fig_ctx.add_trace(go.Histogram(x=vals, nbinsx=50,
                                       marker_color="#1565C0", opacity=0.6,
                                       name="Distribución histórica"))
        fig_ctx.add_vline(x=float(pred), line_dash="dash", line_color="#E53935", line_width=2,
                          annotation_text=f"Tu predicción: {pred:.2f}",
                          annotation_position="top right")
        fig_ctx.update_layout(
            title="Tu predicción en el contexto histórico",
            xaxis_title="CO (mg/m³)", yaxis_title="Frecuencia", height=350,
        )
        st.plotly_chart(fig_ctx, use_container_width=True)


# ── METODOLOGÍA ──────────────────────────────────────────────────
elif page == "Metodología":
    st.title("Metodología y Requisitos")

    with st.expander("Requisitos mínimos cubiertos", expanded=True):
        st.markdown("""
| Requisito | Archivo |
|---|---|
| 1. Definición del problema | README.md — predicción de CO en aire urbano |
| 2. Importación de datos | src/ingest.py — CSV con sep=; decimal=, y JSON |
| 3. Transformaciones | src/transform.py — limpieza -200, fechas, derivadas |
| 4. Mapeo | src/transform.py — map() y apply() |
| 5. Ordenación | src/transform.py — sort_values() multicritero |
| 6. Visualización | src/visualize.py — **Plotly interactivo** (7 gráficos) |
| 7. Modelización ML | src/model_ml.py — LinearReg, Ridge, RF, GBM |
| 8. Comunicación | app.py — Streamlit con predictor interactivo |
        """)

    with st.expander("Orquestación con Dagster"):
        st.markdown("""
**`pipeline_dagster.py`** define los assets del pipeline como un grafo de dependencias:

- `raw_data` → carga el CSV original
- `processed_data` → limpieza y transformaciones
- `visualizations` → valida las 7 figuras Plotly (sin guardar PNGs)
- `ml_models` → entrena y guarda los modelos scikit-learn
- `dl_model` → entrena la red neuronal y guarda `dl_history.pkl`

Para lanzar la UI de Dagster:
```bash
dagster dev -f pipeline_dagster.py
```

**`run_pipeline.py`** ejecuta el mismo pipeline en modo script secuencial:
```bash
python run_pipeline.py
```
        """)

    with st.expander("Temas avanzados"):
        st.markdown("""
**Deep Learning** — `src/model_dl.py`
Red neuronal densa (256-128-64-32) con BatchNormalization, Dropout y EarlyStopping. Guarda el historial de entrenamiento en `data/dl_history.pkl` para visualizarlo con Plotly.

**Scripts en R** — `src/r_analysis.R`
Análisis estadístico con tidyverse, regresión lineal con lm() y visualizaciones con ggplot2. Los PNGs generados se muestran en la pestaña Exploración.
        """)

    with st.expander("Estructura del proyecto"):
        st.code("""
proyecto_air/
├── data/
│   ├── AirQuality.csv
│   ├── airquality_raw.csv
│   ├── airquality_processed.csv
│   ├── dl_history.pkl           <- historial DL para Plotly
│   ├── ml_results.csv
│   ├── dl_results.csv
│   ├── metadata.json
│   ├── models/
│   └── plots/                   <- solo PNGs de R (opcionales)
├── src/
│   ├── ingest.py
│   ├── transform.py
│   ├── visualize.py             <- Plotly (sin guardar PNGs)
│   ├── model_ml.py
│   ├── model_dl.py
│   └── r_analysis.R
├── pipeline_dagster.py          <- orquestación con Dagster
├── run_pipeline.py
└── app.py
        """)
