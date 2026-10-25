"""Descarga desde MLflow las gráficas de los runs y arma las figuras del README en docs/img.

    python -m bach.report
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from mlflow import MlflowClient

from .plots import comparison_curves

IMG = Path(__file__).resolve().parents[2] / "docs" / "img"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(find_dotenv(usecwd=True))
    client = MlflowClient()
    exp = client.get_experiment_by_name(os.getenv("MLFLOW_EXPERIMENT", "nlp-bach-equipoF"))
    runs = client.search_runs([exp.experiment_id], filter_string="attributes.status = 'FINISHED'")
    runs = sorted((r for r in runs if "best_val_loss" in r.data.metrics), key=lambda r: r.info.run_name)

    IMG.mkdir(parents=True, exist_ok=True)
    histories, best = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for r in runs:
            name, run_id = r.info.run_name, r.info.run_id
            d = Path(tmp) / name
            d.mkdir()
            histories[name] = json.loads(Path(client.download_artifacts(run_id, "history.json", d))
                                         .read_text(encoding="utf-8"))
            best[name] = int(r.data.metrics["best_epoch"])
            shutil.copy(client.download_artifacts(run_id, "loss_curve.png", d), IMG / f"loss_{name}.png")
            if r.data.params.get("attention") == "True":
                shutil.copy(client.download_artifacts(run_id, "attention_map.png", d), IMG / f"attention_{name}.png")
            print(f"{name}: gráficas descargadas")
    comparison_curves(histories, best, IMG / "comparacion_perdidas.png")
    print(f"Figuras en {IMG}")


if __name__ == "__main__":
    main()
