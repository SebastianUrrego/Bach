"""Encoder-decoder para continuar corales: LSTM (E1), GRU (E2) y GRU con atención aditiva (E3).

Cada paso musical son 4 notas (0-127). Una tabla de embeddings compartida convierte cada nota
en un vector de `emb_dim` y los 4 vectores se concatenan en el orden note0..note3.
La salida son logits (batch, pasos, 4, 128), sin softmax.

Este módulo se registra con el modelo en MLflow, así que la API debe poder importarlo.
"""

import torch
import torch.nn.functional as F
from torch import nn

VOICES = 4
VOCAB = 128
CONTEXT = 32
HORIZON = 16


class AdditiveAttention(nn.Module):
    """Atención de Bahdanau: score(q, k) = v^T tanh(W_q q + W_k k)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.w_q = nn.Linear(hidden, hidden, bias=False)
        self.w_k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, query, keys):
        # query (B, H), keys (B, 32, H) -> contexto (B, H), pesos (B, 32)
        scores = self.v(torch.tanh(self.w_q(query).unsqueeze(1) + self.w_k(keys))).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights


class Seq2Seq(nn.Module):
    def __init__(self, cell: str = "gru", attention: bool = False, hidden: int = 128,
                 emb_dim: int = 16, layers: int = 1):
        super().__init__()
        if cell not in ("lstm", "gru"):
            raise ValueError(f"cell debe ser 'lstm' o 'gru', no {cell!r}")
        self.config = dict(cell=cell, attention=attention, hidden=hidden, emb_dim=emb_dim, layers=layers)
        self.cell = cell
        self.attention = attention
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        step_dim = VOICES * emb_dim

        self.embedding = nn.Embedding(VOCAB, emb_dim)
        self.encoder = rnn(step_dim, hidden, num_layers=layers, batch_first=True)
        dec_in = step_dim + (hidden if attention else 0)
        self.decoder = rnn(dec_in, hidden, num_layers=layers, batch_first=True)
        self.attn = AdditiveAttention(hidden) if attention else None
        self.out = nn.Linear(hidden, VOICES * VOCAB)

    def embed(self, notes):
        # (B, T, 4) -> (B, T, 4 * emb_dim), voces concatenadas en orden
        e = self.embedding(notes)
        return e.reshape(*notes.shape[:2], -1)

    def encode(self, x):
        enc_out, state = self.encoder(self.embed(x))
        return enc_out, state

    def _top_hidden(self, state):
        h = state[0] if self.cell == "lstm" else state
        return h[-1]

    def decode_step(self, inp, state, enc_out):
        """Un paso del decoder. inp (B, 1, 4) -> logits (B, 1, 4, 128), estado, pesos (B, 32) o None."""
        e = self.embed(inp)
        weights = None
        if self.attention:
            context, weights = self.attn(self._top_hidden(state), enc_out)
            e = torch.cat([e, context.unsqueeze(1)], dim=-1)
        out, state = self.decoder(e, state)
        logits = self.out(out).reshape(out.shape[0], 1, VOICES, VOCAB)
        return logits, state, weights

    def forward(self, x, y, return_attention: bool = False):
        """Teacher forcing: el decoder recibe el último paso de X y luego Y[:, :-1]."""
        dec_in = torch.cat([x[:, -1:], y[:, :-1]], dim=1)
        enc_out, state = self.encode(x)
        if not self.attention:
            out, _ = self.decoder(self.embed(dec_in), state)
            logits = self.out(out).reshape(*dec_in.shape[:2], VOICES, VOCAB)
            return (logits, None) if return_attention else logits

        steps, attn = [], []
        for t in range(dec_in.shape[1]):
            logits_t, state, w = self.decode_step(dec_in[:, t:t + 1], state, enc_out)
            steps.append(logits_t)
            attn.append(w)
        logits = torch.cat(steps, dim=1)
        return (logits, torch.stack(attn, dim=1)) if return_attention else logits

    @torch.no_grad()
    def generate(self, context, steps: int, temperature: float = 1.0, generator=None,
                 return_attention: bool = False):
        """Generación libre por bloques de hasta 16 pasos, sin teacher forcing.

        context (B, 32, 4) -> nuevos pasos (B, steps, 4). Al terminar cada bloque se toman los
        últimos 32 pasos como nuevo contexto y el decoder se reinicia con el estado del encoder.
        Con return_attention devuelve también los pesos del primer bloque (B, 16, 32).
        """
        seq = context.clone()
        new, first_attn = [], None
        while len(new) < steps:
            ctx = seq[:, -CONTEXT:]
            enc_out, state = self.encode(ctx)
            inp = ctx[:, -1:]
            block_attn = []
            for _ in range(min(HORIZON, steps - len(new))):
                logits, state, w = self.decode_step(inp, state, enc_out)
                probs = F.softmax(logits[:, 0] / temperature, dim=-1)  # (B, 4, 128)
                sample = torch.multinomial(probs.reshape(-1, VOCAB), 1, generator=generator)
                inp = sample.reshape(-1, 1, VOICES)
                new.append(inp)
                block_attn.append(w)
            seq = torch.cat([seq, torch.cat(new[-len(block_attn):], dim=1)], dim=1)
            if first_attn is None and self.attention:
                first_attn = torch.stack(block_attn, dim=1)
        generated = torch.cat(new, dim=1)
        return (generated, first_attn) if return_attention else generated


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
