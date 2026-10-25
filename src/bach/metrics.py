"""Pérdida y exactitudes de la guía, acumuladas sobre toda una partición."""

import torch
from torch import nn

from .models import VOCAB

criterion = nn.CrossEntropyLoss()


def loss_fn(logits, y):
    """CE promedio sobre pasos, voces y ejemplos: logits (B*16*4, 128), objetivos (B*16*4)."""
    return criterion(logits.reshape(-1, VOCAB), y.reshape(-1))


@torch.no_grad()
def evaluate(model, loader, device="cpu") -> dict:
    """Modo evaluación, sin gradientes y con teacher forcing. Exactitudes con argmax."""
    model.eval()
    loss_sum = notes_ok = steps_ok = n_notes = n_steps = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, y)
        n = y.numel()
        loss_sum += loss_fn(logits, y).item() * n
        hit = logits.argmax(-1) == y  # (B, 16, 4)
        notes_ok += hit.sum().item()
        steps_ok += hit.all(-1).sum().item()
        n_notes += n
        n_steps += hit.shape[0] * hit.shape[1]
    return {"loss": loss_sum / n_notes, "note_acc": notes_ok / n_notes, "step_acc": steps_ok / n_steps}
