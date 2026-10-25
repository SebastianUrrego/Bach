"""Selección del modelo por pérdida de validación y evaluación en test.

    python -m bach.select

Lee todos los runs terminados del experimento en MLflow (E1-E4 y cualquier run adicional),
elige el de menor `best_val_loss` y evalúa solo ese en `test`, con el mismo protocolo de validación
y con el checkpoint descargado del propio run. Las métricas de test se registran en ese run.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import mlflow
import torch
from dotenv import find_dotenv, load_dotenv
from mlflow import MlflowClient

from .data import load_windows, make_loaders
from .metrics import evaluate
from .models import Seq2Seq

RESULTS = Path(__file__).resolve().parents[2] / "outputs" / "resultados"


def comparison_table(runs) -> str:
    rows = ["| Run | Arquitectura | Hidden | lr | Parámetros | Mejor época / épocas | Val loss | Exactitud nota "
            "| Exactitud paso | Tiempo (s) |",
            "|-----|--------------|-------:|----:|-----------:|:----:|:----:|:----:|:----:|----:|"]
    for r in runs:
        p, m = r.data.params, r.data.metrics
        rows.append(f"| {r.info.run_name} | {p['arquitectura']} | {p['hidden']} | {p['lr']} "
                    f"| {int(m['trainable_params']):,} | {int(m['best_epoch'])} / {int(m['epochs_run'])} "
                    f"| {m['best_val_loss']:.4f} | {m['val_note_acc']:.4f} | {m['val_step_acc']:.4f} "
                    f"| {m['train_time_s']:.0f} |")
    return "\n".join(rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(find_dotenv(usecwd=True))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default=os.getenv("MLFLOW_EXPERIMENT", "nlp-bach-equipoF"))
    p.add_argument("--force", action="store_true", help="Repetir la evaluación en test aunque ya exista")
    args = p.parse_args()

    client = MlflowClient()
    exp = client.get_experiment_by_name(args.experiment)
    runs = client.search_runs([exp.experiment_id], filter_string="attributes.status = 'FINISHED'",
                              order_by=["metrics.best_val_loss ASC"])
    runs = [r for r in runs if "best_val_loss" in r.data.metrics]
    table = comparison_table(sorted(runs, key=lambda r: r.info.run_name))
    print(table)

    chosen = runs[0]
    name, run_id = chosen.info.run_name, chosen.info.run_id
    print(f"\nElegido por menor pérdida de validación: {name} ({run_id}) "
          f"val_loss={chosen.data.metrics['best_val_loss']:.4f}")

    if "test_loss" in chosen.data.metrics and not args.force:
        print("Este run ya tiene métricas de test; no se repite la evaluación (use --force).")
        test = {k: chosen.data.metrics[f"test_{k}"] for k in ("loss", "note_acc", "step_acc")}
    else:
        with tempfile.TemporaryDirectory() as tmp:
            path = client.download_artifacts(run_id, f"checkpoint/{name}.pt", tmp)
            ckpt = torch.load(path, map_location="cpu")
        model = Seq2Seq(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        test = evaluate(model, make_loaders(load_windows())["test"])
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({"test_loss": test["loss"], "test_note_acc": test["note_acc"],
                                "test_step_acc": test["step_acc"]})
            mlflow.set_tag("seleccionado", "true")
    print(f"Test ({name}, época {int(chosen.data.metrics['best_epoch'])}): loss={test['loss']:.4f}  "
          f"exactitud nota={test['note_acc']:.4f}  paso={test['step_acc']:.4f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "tabla_comparacion.md").write_text(table + "\n", encoding="utf-8")
    seleccion = {"run_name": name, "run_id": run_id, "experiment": args.experiment,
                 "best_epoch": int(chosen.data.metrics["best_epoch"]),
                 "best_val_loss": chosen.data.metrics["best_val_loss"],
                 "test_loss": test["loss"], "test_note_acc": test["note_acc"], "test_step_acc": test["step_acc"],
                 "candidatos": [r.info.run_name for r in runs]}
    (RESULTS / "seleccion.json").write_text(json.dumps(seleccion, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
