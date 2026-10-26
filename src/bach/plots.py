"""Gráficas que se guardan como artefactos: curvas de pérdida, piano roll y mapa de atención."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


def loss_curve(history: list[dict], best_epoch: int, title: str, path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [h["train_loss"] for h in history], "o-", label="train")
    ax.plot(epochs, [h["val_loss"] for h in history], "o-", label="valid")
    ax.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"mejor época ({best_epoch})")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Época")
    ax.set_ylabel("Entropía cruzada")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def comparison_curves(histories: dict[str, list[dict]], best_epochs: dict[str, int], path) -> None:
    """Pérdida de entrenamiento (punteada) y validación (continua) de varios runs; marca la mejor época."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, history in histories.items():
        epochs = [h["epoch"] for h in history]
        (line,) = ax.plot(epochs, [h["val_loss"] for h in history], "o-", ms=3, label=f"{name} valid")
        ax.plot(epochs, [h["train_loss"] for h in history], ":", color=line.get_color(), label=f"{name} train")
        best = next(h for h in history if h["epoch"] == best_epochs[name])
        ax.plot(best["epoch"], best["val_loss"], "*", ms=14, color=line.get_color(), mec="black")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Época")
    ax.set_ylabel("Entropía cruzada")
    ax.set_ylim(top=1.6)
    ax.set_title("E1–E4: pérdida por época (★ = mejor checkpoint)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def piano_roll(steps, context_len: int, title: str, path, real=None) -> None:
    """Notas por voz en el tiempo; la línea vertical separa contexto y continuación.
    Si se pasa `real`, se dibuja en gris la continuación verdadera del coral."""
    names = ["Soprano", "Alto", "Tenor", "Bajo"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    t = np.arange(len(steps))
    if real is not None:
        tr = np.arange(context_len, context_len + len(real))
        for v in range(4):
            y = np.where(real[:, v] > 0, real[:, v], np.nan)
            ax.step(tr, y, where="post", color="gray", alpha=0.45, lw=3, label="coral real" if v == 0 else None)
    for v in range(4):
        y = np.where(steps[:, v] > 0, steps[:, v], np.nan).astype(float)
        ax.step(t, y, where="post", lw=1.6, label=names[v])
    ax.axvline(context_len, color="black", ls="--", lw=1)
    ax.text(context_len + 0.5, ax.get_ylim()[1] - 1, "continuación generada →", va="top", fontsize=9)
    ax.set_xlabel("Paso (semicorchea)")
    ax.set_ylabel("Nota MIDI")
    ax.set_title(title)
    ax.legend(ncol=5, fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def attention_map(weights, title: str, path) -> None:
    """weights (16, 32): fila = paso generado, columna = posición del contexto."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(weights, aspect="auto", cmap="viridis", origin="upper")
    ax.set_xticks(range(0, weights.shape[1], 4), [str(i + 1) for i in range(0, weights.shape[1], 4)])
    ax.set_yticks(range(weights.shape[0]), [str(i + 1) for i in range(weights.shape[0])])
    ax.set_xlabel("Posición en el contexto (paso 1-32 del encoder)")
    ax.set_ylabel("Paso generado (1-16)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Peso de atención")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
