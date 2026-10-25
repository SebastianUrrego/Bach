"""Carga de JSB Chorales y construcción de ventanas 32 -> 16.

Cada coral es una matriz (pasos, 4) con enteros 0-127 (0 = silencio).
Las ventanas se cortan dentro de cada coral, nunca entre corales ni particiones.

Uso: `python -m bach.data` descarga los datos (si faltan) y muestra la verificación.
"""

import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import CONTEXT, HORIZON, VOCAB

URL = "https://github.com/ageron/handson-ml2/raw/master/datasets/jsb_chorales/jsb_chorales.tgz"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

COLUMNS = ["note0", "note1", "note2", "note3"]
WINDOW = CONTEXT + HORIZON
STRIDES = {"train": 8, "valid": 16, "test": 16}
SPLITS = ("train", "valid", "test")


def download(data_dir: Path = DATA_DIR) -> Path:
    """Descarga y descomprime jsb_chorales.tgz; devuelve la carpeta con train/valid/test."""
    root = data_dir / "jsb_chorales"
    if root.exists():
        return root
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "jsb_chorales.tgz"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    with tarfile.open(archive) as tar:  # contiene train/, valid/ y test/ sin carpeta raíz
        tar.extractall(root, filter="data")
    return root


def load_split(split: str, data_dir: Path = DATA_DIR) -> list[pd.DataFrame]:
    """Lee los CSV de una partición en orden de nombre de archivo."""
    files = sorted((download(data_dir) / split).glob("*.csv"))
    chorales = []
    for f in files:
        df = pd.read_csv(f)
        df.attrs["name"] = f.stem
        chorales.append(df)
    return chorales


def make_windows(chorales: list[pd.DataFrame], stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Ventanas completas de 48 pasos por coral: X (n, 32, 4) y Y (n, 16, 4)."""
    xs, ys = [], []
    for df in chorales:
        notes = df[COLUMNS].to_numpy(dtype=np.int64)
        for start in range(0, len(notes) - WINDOW + 1, stride):
            xs.append(notes[start:start + CONTEXT])
            ys.append(notes[start + CONTEXT:start + WINDOW])
    return np.stack(xs), np.stack(ys)


def load_windows(data_dir: Path = DATA_DIR) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Ventanas de las tres particiones con su stride (8 train, 16 valid/test)."""
    return {s: make_windows(load_split(s, data_dir), STRIDES[s]) for s in SPLITS}


def make_loaders(windows: dict, batch_size: int = 64, seed: int = 42) -> dict[str, DataLoader]:
    """DataLoaders de (batch, 32, 4) y (batch, 16, 4). Solo train se baraja."""
    loaders = {}
    for split, (x, y) in windows.items():
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        gen = torch.Generator().manual_seed(seed) if split == "train" else None
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=split == "train", generator=gen)
    return loaders


def verify(data_dir: Path = DATA_DIR) -> None:
    """Imprime las comprobaciones de la guía: columnas, corales, longitudes, rangos y ventanas."""
    for split in SPLITS:
        chorales = load_split(split, data_dir)
        cols = {tuple(df.columns) for df in chorales}
        assert cols == {tuple(COLUMNS)}, f"{split}: columnas inesperadas {cols}"
        lengths = np.array([len(df) for df in chorales])
        allnotes = pd.concat(chorales)
        assert allnotes.isna().sum().sum() == 0, f"{split}: hay valores faltantes"
        assert allnotes.min().min() >= 0 and allnotes.max().max() < VOCAB, f"{split}: valores fuera de 0-127"

        print(f"\n[{split}] corales={len(chorales)}  pasos: min={lengths.min()} max={lengths.max()} "
              f"media={lengths.mean():.1f} total={lengths.sum()}")
        for c in COLUMNS:
            v = allnotes[c]
            nz = v[v > 0]
            print(f"  {c}: silencios={(v == 0).mean():6.2%}  rango notas={nz.min()}-{nz.max()}  "
                  f"distintas={nz.nunique()}")

        x, y = make_windows(chorales, STRIDES[split])
        expected = sum((n - WINDOW) // STRIDES[split] + 1 for n in lengths if n >= WINDOW)
        assert len(x) == expected
        print(f"  ventanas (stride {STRIDES[split]}): X={x.shape} Y={y.shape}")


if __name__ == "__main__":
    verify()
