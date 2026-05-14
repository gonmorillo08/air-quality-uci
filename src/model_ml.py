"""
src/model_ml.py
Requisito 7 - Modelizacion con scikit-learn
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing   import StandardScaler

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")

FEATURE_COLS = [
    "PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)",
    "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)",
    "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
    "hour", "month", "is_rush_hour", "temp_cat_num",
    "NO2_NOx_ratio", "sensor_mean"
]
TARGET_COL = "CO(GT)"


def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  {name:25s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
    return {"model": name, "MAE": round(mae,4), "RMSE": round(rmse,4), "R2": round(r2,4)}


def run(df: pd.DataFrame) -> dict:
    os.makedirs(MODELS_DIR, exist_ok=True)

    available = [c for c in FEATURE_COLS if c in df.columns]
    df_model  = df[available + [TARGET_COL]].dropna()
    X = df_model[available].values
    y = df_model[TARGET_COL].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(f"[model_ml] Train: {len(X_train)}  Test: {len(X_test)}\n")

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge":            Ridge(alpha=1.0),
        "RandomForest":     RandomForestRegressor(n_estimators=150, max_depth=12, n_jobs=-1, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42),
    }

    results, preds, trained = [], {}, {}

    print("[model_ml] Resultados en test set:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred        = model.predict(X_test)
        preds[name]   = y_pred
        trained[name] = model
        results.append(evaluate(name, y_test, y_pred))

    best_name  = max(results, key=lambda r: r["R2"])["model"]
    best_model = trained[best_name]

    cv = cross_val_score(best_model, X_scaled, y, cv=5, scoring="r2", n_jobs=-1)
    print(f"\n[model_ml] Cross-val R2 ({best_name}): {cv.mean():.4f} +/- {cv.std():.4f}")

    # Las gráficas se generan dinámicamente en app.py con Plotly
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_ml_model.pkl"))
    joblib.dump(scaler,     os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(available,  os.path.join(MODELS_DIR, "feature_cols.pkl"))
    pd.DataFrame(results).to_csv(os.path.join(DATA_DIR, "ml_results.csv"), index=False)
    print(f"\n[model_ml] Mejor modelo: {best_name} guardado.\n")

    return {"results": results, "best_model": best_model,
            "scaler": scaler, "feature_cols": available}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from ingest import run as ingest_run
    from transform import run as transform_run
    df = transform_run(ingest_run())
    run(df)
