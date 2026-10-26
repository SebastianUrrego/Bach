"""Registro del modelo elegido en MLflow Model Registry con el alias `champion`.

    python -m bach.register

Restaura los pesos del checkpoint del run elegido, guarda el objeto PyTorch completo (con sus métodos de
generación) con `mlflow.pytorch.log_model` dentro de ese mismo run y lo registra como MODEL_NAME.
Después comprueba la carga en un proceso aparte, como lo haría la API.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from dotenv import find_dotenv, load_dotenv
from mlflow import MlflowClient

from .models import Seq2Seq

ROOT = Path(__file__).resolve().parents[2]
SELECTION = ROOT / "outputs" / "resultados" / "seleccion.json"
REGISTRO = ROOT / "outputs" / "resultados" / "registro.json"

# Carga y generación en otro proceso, sin archivos locales: solo MLflow
CHECK = """
import sys, torch, mlflow.pytorch
from mlflow import MlflowClient
name = sys.argv[1]
version = MlflowClient().get_model_version_by_alias(name, "champion").version
model = mlflow.pytorch.load_model(f"models:/{name}/{version}", map_location="cpu")
model.eval()
out = model.generate(torch.full((1, 32, 4), 60), steps=33, generator=torch.Generator().manual_seed(0))
print(f"proceso aparte: {name} v{version} -> {type(model).__name__}, generate(33) {tuple(out.shape)}")
"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(find_dotenv(usecwd=True))
    name = os.getenv("MODEL_NAME", "bach-chorales-equipoF")
    sel = json.loads(SELECTION.read_text(encoding="utf-8"))
    client = MlflowClient()

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = torch.load(client.download_artifacts(sel["run_id"], f"checkpoint/{sel['run_name']}.pt", tmp),
                          map_location="cpu")
    model = Seq2Seq(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Checkpoint {sel['run_name']} época {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    with mlflow.start_run(run_id=sel["run_id"]):
        info = mlflow.pytorch.log_model(
            model, name="model", registered_model_name=name, serialization_format="pickle",
            code_paths=[str(ROOT / "src" / "bach")],
            pip_requirements=[f"torch=={torch.__version__.split('+')[0]}", f"mlflow=={mlflow.__version__}",
                              "numpy"],
            metadata={"run_name": sel["run_name"], "best_epoch": ckpt["epoch"], "val_loss": ckpt["val_loss"]},
        )
    version = str(info.registered_model_version)
    client.set_registered_model_alias(name, "champion", version)
    client.update_model_version(name, version, description=(
        f"{sel['run_name']}: encoder-decoder {ckpt['config']['cell'].upper()} hidden {ckpt['config']['hidden']}, "
        f"época {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f}, test_loss {sel['test_loss']:.4f}"))
    print(f"Registrado {name} versión {version} (run {sel['run_id']}) con alias champion")

    REGISTRO.write_text(json.dumps({"model_name": name, "version": version, "alias": "champion",
                                    "run_name": sel["run_name"], "run_id": sel["run_id"],
                                    "model_uri": info.model_uri}, indent=2), encoding="utf-8")

    subprocess.run([sys.executable, "-c", CHECK, name], check=True, cwd=tempfile.gettempdir(),
                   env={**os.environ, "PYTHONUTF8": "1"})


if __name__ == "__main__":
    main()
