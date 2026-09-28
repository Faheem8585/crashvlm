"""Temporal convolution head over frozen frame embeddings (the classic-CV temporal reference).

One small dilated 1-D CNN predicts, for every frame and every event, "has the event happened
yet?" from a window of neighbouring frames (receptive field about 15 frames = 120 ms at 8 ms).
Events that are undefined for a clip's view/test type are masked out of the loss. Training
uses random start crops (the same augmentation idea as the LoRA data), so the model cannot
learn a fixed position of the impact.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TCN(nn.Module):
    def __init__(self, d_in: int, n_events: int, width: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, width, 5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 5, padding=4, dilation=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 3, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv1d(width, n_events, 1),
        )

    def forward(self, x):  # x: (B, T, D) -> logits (B, T, E)
        return self.net(x.transpose(1, 2)).transpose(1, 2)


def train_tcn(
    seqs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    val: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_events: int,
    epochs: int = 60,
    lr: float = 1e-3,
    min_len: int = 24,
    seed: int = 0,
) -> TCN:
    """seqs: (emb (T,D), targets (T,E) in {0,1}, mask (E,) in {0,1}). Early stopping on val."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TCN(seqs[0][0].shape[1], n_events)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def loss_on(batch, train: bool):
        total, count = 0.0, 0.0
        for emb, y, m in batch:
            if train:  # random start crop
                s = int(rng.integers(0, max(1, len(emb) - min_len)))
                emb, y = emb[s:], y[s:]
            x = torch.from_numpy(emb.astype(np.float32))[None]
            yt = torch.from_numpy(y.astype(np.float32))[None]
            mt = torch.from_numpy(m.astype(np.float32))[None, None].expand_as(yt)
            ll = bce(model(x), yt) * mt
            total = total + ll.sum()
            count += float(mt.sum())
        return total / max(count, 1.0)

    best, best_state, patience = np.inf, None, 0
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(seqs))
        for i in range(0, len(order), 8):
            opt.zero_grad()
            loss = loss_on([seqs[j] for j in order[i : i + 8]], True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_on(val, False))
        if vl < best - 1e-4:
            best, best_state, patience = (
                vl,
                {k: v.clone() for k, v in model.state_dict().items()},
                0,
            )
        else:
            patience += 1
            if patience >= 10:
                break
    model.load_state_dict(best_state)
    return model.eval()


def predict_curves(model: TCN, emb: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(emb.astype(np.float32))[None]))[0].numpy()
