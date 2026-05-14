# Air Quality UCI - Proyecto Final IA & Estadistica 2025-26

## Descripcion
Prediccion de la concentracion de monoxido de carbono CO(GT) a partir de
mediciones horarias de sensores de calidad del aire en una ciudad italiana (2004-2005).

## Dataset
**Air Quality UCI** — 9357 mediciones horarias, 13 variables de sensores quimicos
y condiciones ambientales (temperatura, humedad).

## Instalacion
```
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow streamlit joblib
```

## Uso
```
python run_pipeline.py   # genera datos, modelos y graficos
streamlit run app.py     # lanza la app web
```

## Requisitos cubiertos
| # | Requisito | Archivo |
|---|-----------|---------|
| 1 | Definicion del problema | README.md |
| 2 | Importacion de datos | src/ingest.py |
| 3 | Transformaciones | src/transform.py |
| 4 | Mapeo | src/transform.py |
| 5 | Ordenacion | src/transform.py |
| 6 | Visualizacion | src/visualize.py |
| 7 | Modelizacion ML | src/model_ml.py |
| 8 | Comunicacion web | app.py |
| * | Deep Learning | src/model_dl.py |
| * | Scripts en R | src/r_analysis.R |
