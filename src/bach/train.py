"""Entrenamiento de E1-E4 con registro en MLflow.

    python -m bach.train E1                 # LSTM
    python -m bach.train E2                 # GRU
    python -m bach.train E3                 # GRU + atención
    python -m bach.train E4 --cell gru --attention --hidden 256   # un solo cambio sobre el mejor

Cada run guarda el checkpoint de menor pérdida de validación, lo recarga para calcular las
exactitudes y registra parámetros, métricas por época y artefactos en el experimento de MLFLOW_EXPERIMENT.
`test` no se usa aquí: solo se evalúa el modelo elegido (ver bach.select).
"""

import argparse
import json
import os
import platform
import random
import sys
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv

from .data import STRIDES, load_split, load_windows, make_loaders
from .metrics import evaluate, loss_fn
from .models import CONTEXT, HORIZON, Seq2Seq, count_parameters
from .plots import attention_map, loss_curve

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "outputs" / "resultados"

PRESETS = {
    "E1": dict(cell="lstm", attention=False),
    "E2": dict(cell="gru", attention=False),
    "E3": dict(cell="gru", attention=True),
}
DEFAULTS = dict(hidden=128, emb_dim=16, layers=1, lr=1e-3, batch_size=64, max_epochs=20, patience=4, seed=42)

# Fragmento fijo de valid para el mapa de atención de los modelos con atención
ATTN_CHORALE_INDEX = 0
ATTN_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def hardware() -> dict:
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }


def train(run_name: str, cfg: dict) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg["seed"])
    loaders = make_loaders(load_windows(), batch_size=cfg["batch_size"], seed=cfg["seed"])
    model = Seq2Seq(cell=cfg["cell"], attention=cfg["attention"], hidden=cfg["hidden"],
                    emb_dim=cfg["emb_dim"], layers=cfg["layers"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    n_params = count_parameters(model)
    hw = hardware()

    params = {
        "arquitectura": f"encoder-decoder {cfg['cell'].upper()}" + (" + atención aditiva" if cfg["attention"] else ""),
        "cell": cfg["cell"], "attention": cfg["attention"], "layers": cfg["layers"],
        "hidden": cfg["hidden"], "emb_dim": cfg["emb_dim"], "lr": cfg["lr"], "batch_size": cfg["batch_size"],
        "max_epochs": cfg["max_epochs"], "patience": cfg["patience"], "seed": cfg["seed"],
        "context": CONTEXT, "horizon": HORIZON, "optimizer": "Adam",
        "stride_train": STRIDES["train"], "stride_valid": STRIDES["valid"],
        "n_train": len(loaders["train"].dataset), "n_valid": len(loaders["valid"].dataset),
        "device": hw["device"],
    }
    if cfg.get("change"):
        params["cambio_e4"] = cfg["change"]

    CHECKPOINTS.mkdir(exist_ok=True)
    ckpt_path = CHECKPOINTS / f"{run_name}.pt"
    history, best_loss, best_epoch, bad_epochs = [], float("inf"), 0, 0

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("experimento", run_name)
        mlflow.log_params(params)
        mlflow.log_metric("trainable_params", n_params)
        print(f"[{run_name}] {params['arquitectura']}  parámetros entrenables={n_params:,}  device={device}")

        t0 = time.perf_counter()
        for epoch in range(1, cfg["max_epochs"] + 1):
            model.train()
            loss_sum = n = 0
            for x, y in loaders["train"]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(x, y), y)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * y.numel()
                n += y.numel()
            train_loss = loss_sum / n
            val = evaluate(model, loaders["valid"], device)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
                            "val_note_acc": val["note_acc"], "val_step_acc": val["step_acc"]})
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val["loss"],
                                "val_note_acc_epoch": val["note_acc"], "val_step_acc_epoch": val["step_acc"]},
                               step=epoch)

            improved = val["loss"] < best_loss
            if improved:
                best_loss, best_epoch, bad_epochs = val["loss"], epoch, 0
                torch.save({"state_dict": model.state_dict(), "config": model.config, "epoch": epoch,
                            "val_loss": val["loss"], "run_name": run_name}, ckpt_path)
            else:
                bad_epochs += 1
            print(f"  época {epoch:2d}  train={train_loss:.4f}  val={val['loss']:.4f}  "
                  f"nota={val['note_acc']:.3f}  paso={val['step_acc']:.3f}" + ("  *" if improved else ""))
            if bad_epochs >= cfg["patience"]:
                print(f"  parada temprana: {cfg['patience']} épocas sin mejorar")
                break
        train_time = time.perf_counter() - t0

        # Recargar el mejor checkpoint: sus exactitudes, no las de la última época
        ckpt = torch.load(ckpt_path, map_location=device)
        best = Seq2Seq(**ckpt["config"]).to(device)
        best.load_state_dict(ckpt["state_dict"])
        val = evaluate(best, loaders["valid"], device)
        summary = {
            "run_name": run_name, "run_id": run.info.run_id, **params, "trainable_params": n_params,
            "best_epoch": best_epoch, "epochs_run": len(history), "best_val_loss": val["loss"],
            "val_note_acc": val["note_acc"], "val_step_acc": val["step_acc"],
            "train_time_s": round(train_time, 1), "hardware": hw,
        }
        mlflow.log_metrics({"best_epoch": best_epoch, "best_val_loss": val["loss"], "val_note_acc": val["note_acc"],
                            "val_step_acc": val["step_acc"], "train_time_s": train_time,
                            "epochs_run": len(history)})
        mlflow.log_dict(hardware(), "hardware.json")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "config.json").write_text(json.dumps({**cfg, **params}, indent=2, ensure_ascii=False), encoding="utf-8")
            (tmp / "history.json").write_text(json.dumps(history, indent=2))
            loss_curve(history, best_epoch, f"{run_name}: {params['arquitectura']}", tmp / "loss_curve.png")
            for f in ("config.json", "history.json", "loss_curve.png"):
                mlflow.log_artifact(str(tmp / f))
            mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoint")

            if cfg["attention"]:
                chorale = load_split("valid")[ATTN_CHORALE_INDEX]
                context = torch.tensor(chorale[["note0", "note1", "note2", "note3"]].to_numpy()[:CONTEXT]).unsqueeze(0)
                gen = torch.Generator().manual_seed(ATTN_SEED)
                best.eval()
                _, attn = best.generate(context.to(device), steps=HORIZON, temperature=1.0, generator=gen,
                                        return_attention=True)
                name = chorale.attrs["name"]
                attention_map(attn[0].cpu().numpy(),
                              f"{run_name}: atención en generación libre ({name}, T=1.0, semilla {ATTN_SEED})",
                              tmp / "attention_map.png")
                mlflow.log_artifact(str(tmp / "attention_map.png"))
                summary["attention_chorale"] = name

        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"{run_name}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{run_name}] mejor época={best_epoch}  val_loss={val['loss']:.4f}  "
              f"exactitud nota={val['note_acc']:.4f}  paso={val['step_acc']:.4f}  tiempo={train_time:.0f}s")
    return summary


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # MLflow imprime emojis; la consola de Windows usa cp1252
    load_dotenv(find_dotenv(usecwd=True))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_name", help="E1, E2, E3, E4 u otro nombre de run")
    p.add_argument("--cell", choices=["lstm", "gru"])
    p.add_argument("--attention", action=argparse.BooleanOptionalAction, default=None)
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=None)
    p.add_argument("--change", help="Descripción del cambio de E4, p. ej. 'hidden 128 -> 256'")
    p.add_argument("--experiment", default=os.getenv("MLFLOW_EXPERIMENT", "nlp-bach-equipoF"))
    args = p.parse_args()

    cfg = {**DEFAULTS, **PRESETS.get(args.run_name, {})}
    for k in ("cell", "attention", *DEFAULTS):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    cfg["change"] = args.change
    if "cell" not in cfg or "attention" not in cfg:
        p.error(f"{args.run_name} no es un preset: indique --cell y --attention/--no-attention")

    mlflow.set_experiment(args.experiment)
    print(f"MLflow: {mlflow.get_tracking_uri()}  experimento: {args.experiment}")
    train(args.run_name, cfg)


if __name__ == "__main__":
    main()
