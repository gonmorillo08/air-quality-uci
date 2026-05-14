"""
src/model_dl.py
Tema Avanzado - Deep Learning con Keras
"""

import os
import pickle
import numpy as np
import pandas as pd
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
TARGET_COL = "CO(GT)"


def build_model(input_dim):
    inp = keras.Input(shape=(input_dim,))
    x   = layers.Dense(256, activation="relu")(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(0.1)(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="huber", metrics=["mae"])
    return model


def run(df: pd.DataFrame) -> dict:
    os.makedirs(MODELS_DIR, exist_ok=True)

    scaler    = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    available = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))

    df_model = df[available + [TARGET_COL]].dropna()
    X = scaler.transform(df_model[available].values)
    y = df_model[TARGET_COL].values.astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    print(f"[model_dl] Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

    model = build_model(X_train.shape[1])
    model.summary()

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6)
    ]

    print("\n[model_dl] Entrenando red neuronal...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150, batch_size=128,
        callbacks=cb, verbose=0
    )
    print(f"[model_dl] Completado en {len(history.history['loss'])} epocas.")

    y_pred = model.predict(X_test, verbose=0).flatten()
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"\n[model_dl] MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}\n")

    # Guardar historial para visualizarlo en la app con Plotly
    hist_path = os.path.join(DATA_DIR, "dl_history.pkl")
    with open(hist_path, "wb") as f:
        pickle.dump(history.history, f)
    print(f"[model_dl] Historial de entrenamiento guardado -> {hist_path}")

    # Las gráficas se generan dinámicamente en app.py con Plotly
    model.save(os.path.join(MODELS_DIR, "dl_model.keras"))
    pd.DataFrame([{"model": "DeepLearning", "MAE": round(mae,4), "RMSE": round(rmse,4), "R2": round(r2,4)}]).to_csv(
        os.path.join(DATA_DIR, "dl_results.csv"), index=False)

    return {"model": model, "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2}}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from ingest import run as ingest_run
    from transform import run as transform_run
    df = transform_run(ingest_run())
    run(df)
