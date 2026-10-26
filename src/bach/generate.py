"""Generación libre con el modelo elegido: 2 corales de valid -> MIDI, piano roll y estadísticas.

    python -m bach.generate

Toma los primeros 32 pasos de cada coral, genera `--steps` pasos nuevos por bloques de 16 (sin teacher
forcing) con muestreo `softmax(logits / T)` y escribe en outputs/generacion/ un MIDI por coral
(contexto + continuación). También calcula estadísticas de repetición, variedad y silencios para varias
temperaturas, y lo registra todo en el run del modelo elegido.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from mlflow import MlflowClient

from .data import COLUMNS, load_split
from .midi import step_seconds, to_midi
from .models import CONTEXT, Seq2Seq
from .plots import piano_roll

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "generacion"
SELECTION = ROOT / "outputs" / "resultados" / "seleccion.json"
IMG = ROOT / "docs" / "img"


def load_selected(client: MlflowClient) -> tuple[Seq2Seq, dict]:
    """Modelo elegido con el checkpoint descargado de su run en MLflow."""
    sel = json.loads(SELECTION.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = torch.load(client.download_artifacts(sel["run_id"], f"checkpoint/{sel['run_name']}.pt", tmp),
                          map_location="cpu")
    model = Seq2Seq(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), sel


def voice_ranges() -> list[tuple[int, int]]:
    """Rango (mín, máx) de notas no silenciosas por voz en train."""
    notes = np.concatenate([df[COLUMNS].to_numpy() for df in load_split("train")])
    return [(int(notes[notes[:, v] > 0, v].min()), int(notes[notes[:, v] > 0, v].max())) for v in range(4)]


def stats(cont: np.ndarray, last_context_step: np.ndarray, ranges, real: np.ndarray | None = None) -> dict:
    """Repeticiones, variedad y silencios de una continuación (pasos, 4)."""
    prev = np.vstack([last_context_step, cont[:-1]])
    out_of_range = sum(int(((cont[:, v] > 0) & ((cont[:, v] < lo) | (cont[:, v] > hi))).sum())
                       for v, (lo, hi) in enumerate(ranges))
    s = {
        "repite_paso_anterior": float((cont == prev).all(axis=1).mean()),
        "repite_nota_por_voz": float((cont == prev).mean()),
        "notas_distintas_por_voz": [int(len(np.unique(cont[cont[:, v] > 0, v]))) for v in range(4)],
        "silencios": float((cont == 0).mean()),
        "notas_tras_fusionar": int(sum(((np.r_[True, np.diff(cont[:, v]) != 0]) & (cont[:, v] > 0)).sum()
                                       for v in range(4))),
        "notas_fuera_de_rango": out_of_range,
    }
    if real is not None:
        s["coincidencia_con_real"] = float((cont == real).mean())
    return s


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(find_dotenv(usecwd=True))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chorales", nargs=2, type=int, default=[0, 1], help="Índices de los corales de valid")
    p.add_argument("--steps", type=int, default=64, help="Pasos nuevos por fragmento (>= 32)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--explore", type=float, nargs="*", default=[0.7, 1.3],
                   help="Temperaturas extra, solo para las estadísticas")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria del muestreo")
    p.add_argument("--tempo", type=float, default=80.0, help="Negras por minuto")
    args = p.parse_args()

    client = MlflowClient()
    model, sel = load_selected(client)
    ranges = voice_ranges()
    valid = load_split("valid")
    OUT.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)

    info = {"modelo": sel["run_name"], "run_id": sel["run_id"], "pasos_nuevos": args.steps,
            "temperatura": args.temperature, "semilla_muestreo": args.seed, "tempo_negras_por_min": args.tempo,
            "segundos_por_paso": step_seconds(args.tempo), "rangos_train_por_voz": ranges, "fragmentos": []}
    temps = [args.temperature, *[t for t in args.explore if t != args.temperature]]
    for idx in args.chorales:
        chorale = valid[idx]
        name = chorale.attrs["name"]
        notes = chorale[COLUMNS].to_numpy()
        context = notes[:CONTEXT]
        real = notes[CONTEXT:CONTEXT + args.steps]
        real = real if len(real) == args.steps else None
        frag = {"coral": name, "indice_valid": idx, "pasos_coral": len(notes),
                "real": stats(real, context[-1], ranges) if real is not None else None, "por_temperatura": {}}

        for t in temps:
            gen = torch.Generator().manual_seed(args.seed)
            new = model.generate(torch.tensor(context).unsqueeze(0), steps=args.steps, temperature=t,
                                 generator=gen)[0].numpy()
            frag["por_temperatura"][str(t)] = stats(new, context[-1], ranges, real)
            if t == args.temperature:
                full = np.vstack([context, new])
                stem = f"{name}_{sel['run_name']}_T{t}_s{args.seed}"
                frag["midi"] = f"{stem}.mid"
                frag["eventos_midi"] = to_midi(full, OUT / f"{stem}.mid", tempo=args.tempo, marker_step=CONTEXT)
                np.savetxt(OUT / f"{stem}.csv", full, fmt="%d", delimiter=",", header=",".join(COLUMNS),
                           comments="")
                piano_roll(full, CONTEXT, f"{name}: 32 pasos de contexto + {args.steps} generados por "
                           f"{sel['run_name']} (T={t}, semilla {args.seed})", IMG / f"pianoroll_{name}.png", real)
        info["fragmentos"].append(frag)
        g = frag["por_temperatura"][str(args.temperature)]
        print(f"{name}: {frag['midi']}  repite paso={g['repite_paso_anterior']:.2f}  "
              f"silencios={g['silencios']:.3f}  distintas={g['notas_distintas_por_voz']}  "
              f"fuera de rango={g['notas_fuera_de_rango']}")

    (OUT / "generacion.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    with mlflow.start_run(run_id=sel["run_id"]):
        mlflow.set_tags({"gen_corales": ",".join(f["coral"] for f in info["fragmentos"]),
                         "gen_semilla": args.seed, "gen_temperatura": args.temperature, "gen_tempo": args.tempo})
        mlflow.log_artifacts(str(OUT), artifact_path="generacion")
        for f in info["fragmentos"]:
            mlflow.log_artifact(str(IMG / f"pianoroll_{f['coral']}.png"), artifact_path="generacion")
    print(f"Resultados en {OUT} y en el run {sel['run_name']} ({sel['run_id']})")


if __name__ == "__main__":
    main()
