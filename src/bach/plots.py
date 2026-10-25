"""Gráficas que se guardan como artefactos: curva de pérdidas y mapa de atención."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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
