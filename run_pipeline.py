"""
run_pipeline.py - Ejecuta el pipeline completo
Uso: python run_pipeline.py
"""

import os, sys, time

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_DIR)


def run_r():
    r_script = os.path.join(SRC_DIR, "r_analysis.R")
    try:
        import rpy2.robjects as ro
        ro.r.source(r_script)
        print("[pipeline] R completado via rpy2.\n")
        return
    except Exception:
        pass
    import subprocess
    res = subprocess.run(["Rscript", r_script], capture_output=True, text=True)
    if res.returncode == 0:
        print(res.stdout)
    else:
        print(f"[pipeline] R no disponible. Ejecuta manualmente:\n  Rscript src/r_analysis.R\n")


def sep(title):
    print(f"\n{'='*55}\n  {title}\n{'='*55}\n")


def main():
    start = time.time()
    print("\n  AIR QUALITY - Pipeline completo")
    print("="*55)

    sep("1/5 - Ingesta")
    from ingest import run as ingest_run
    df = ingest_run()

    sep("2/5 - Transformaciones")
    from transform import run as transform_run
    df = transform_run(df)

    sep("3/5 - Visualizaciones")
    from visualize import run as viz_run
    viz_run(df)

    sep("4/5 - Modelos ML")
    from model_ml import run as ml_run
    ml_run(df)

    sep("5/5 - Deep Learning")
    from model_dl import run as dl_run
    dl_run(df)

    sep("Extra - Analisis R")
    run_r()

    print(f"\nPipeline completado en {time.time()-start:.1f}s")
    print("Lanza la app con:  streamlit run app.py\n")


if __name__ == "__main__":
    main()
