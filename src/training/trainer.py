"""Training loop for the GRU / LSTM joint embedding models. Requires PyTorch.

Model selection and early stopping use ONLY the validation split. The test split is never touched here.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from src.data.pairs import SplitData
from src.evaluation.metrics import evaluate_embeddings, higher_is_better
from src.utils.common import get_logger, progress, save_json, set_seed

log = get_logger("ncir.train")


class EarlyStopper:
    """Stops when the monitored metric has not improved for ``patience`` evaluations."""

    def __init__(self, patience: int, higher_is_better_: bool = True):
        self.patience = patience
        self.higher = higher_is_better_
        self.best: Optional[float] = None
        self.bad_epochs = 0

    def update(self, value: float) -> bool:
        """Record ``value``; return True if it is a new best."""
        improved = self.best is None or (value > self.best if self.higher else value < self.best)
        if improved:
            self.best, self.bad_epochs = float(value), 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.bad_epochs >= self.patience


def resolve_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("train.device is 'cuda' but CUDA is not available; use 'auto' or 'cpu'")
    return torch.device(name)


def encode_images(model, feats: np.ndarray, device, batch_size: int = 1024) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.inference_mode():
        for s in range(0, feats.shape[0], batch_size):
            x = torch.from_numpy(np.ascontiguousarray(feats[s:s + batch_size], dtype=np.float32)).to(device)
            out.append(model.encode_image(x).cpu().numpy())
    return np.concatenate(out, axis=0)


def encode_captions(model, cap_ids: np.ndarray, cap_lengths: np.ndarray, device, batch_size: int = 1024) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.inference_mode():
        for s in range(0, cap_ids.shape[0], batch_size):
            lens = cap_lengths[s:s + batch_size]
            width = int(lens.max())
            ids = torch.from_numpy(cap_ids[s:s + batch_size, :width].astype(np.int64)).to(device)
            out.append(model.encode_text(ids, torch.from_numpy(lens.astype(np.int64))).cpu().numpy())
    return np.concatenate(out, axis=0)


def validation_subset(val: SplitData, n_images: int) -> SplitData:
    """First ``n_images`` validation images (and their captions) for fast per-epoch evaluation."""
    if n_images <= 0 or n_images >= val.num_images:
        return val
    mask = val.caption_img_idx < n_images
    return SplitData(val.split, val.image_ids[:n_images], val.features[:n_images],
                     [c for c, m in zip(val.captions, mask) if m], val.caption_img_idx[mask],
                     val.cap_ids[mask], val.cap_lengths[mask])


def evaluate_model(model, data: SplitData, device) -> dict[str, float]:
    img = encode_images(model, data.features, device)
    txt = encode_captions(model, data.cap_ids, data.cap_lengths, device)
    return evaluate_embeddings(img, txt, data.caption_img_idx)


def make_optimizer(params, train_cfg: Mapping[str, Any]):
    import torch

    name = str(train_cfg.get("optimizer", "adam")).lower()
    lr = float(train_cfg.get("lr", 2e-4))
    wd = float(train_cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"unknown optimizer '{name}' (adam | adamw | sgd)")


def train_neural(cfg: Mapping[str, Any], train: SplitData, val: SplitData, vocab_size: int,
                 embedding_matrix: Optional[np.ndarray], run_dir: str | Path) -> dict[str, Any]:
    """Train a GRU / LSTM joint model; returns a summary dict.

    Writes ``last.pt`` every epoch, ``best.pt`` when the validation metric improves, and
    ``history.json``. With ``train.resume: true`` training continues from ``last.pt``.
    """
    import torch

    from src.models.joint_model import build_joint_model, load_checkpoint, save_checkpoint
    from src.training.loss import ranking_loss

    tcfg = cfg["train"]
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = resolve_device(str(tcfg.get("device", "auto")))
    log.info("training %s on %s (%d train images, %d captions)", cfg["name"], device, train.num_images,
             train.num_captions)

    model = build_joint_model(cfg["model"], train.features.shape[1], vocab_size, embedding_matrix).to(device)
    optimizer = make_optimizer([p for p in model.parameters() if p.requires_grad], tcfg)
    metric = str(tcfg.get("selection_metric", "t2i_rsum"))
    stopper = EarlyStopper(int(tcfg.get("patience", 5)), higher_is_better(metric))
    val_small = validation_subset(val, int(tcfg.get("val_subset_images", 0)))

    history: list[dict[str, Any]] = []
    start_epoch = 0
    last_path, best_path = run / "last.pt", run / "best.pt"
    if bool(tcfg.get("resume", False)) and last_path.is_file():
        ckpt = load_checkpoint(last_path, map_location=str(device))
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        history = list(ckpt["history"])
        start_epoch = int(ckpt["epoch"]) + 1
        stopper.best = ckpt.get("best_metric")
        stopper.bad_epochs = int(ckpt.get("extra", {}).get("bad_epochs", 0))
        log.info("resumed from %s at epoch %d", last_path, start_epoch)

    batch_size = int(tcfg.get("batch_size", 128))
    margin = float(tcfg.get("margin", 0.2))
    max_violation = bool(tcfg.get("max_violation", False))
    clip = float(tcfg.get("grad_clip", 0.0))
    epochs = int(tcfg.get("epochs", 30))
    rng = np.random.RandomState(seed)
    for _ in range(start_epoch):  # keep the shuffling stream aligned when resuming
        rng.permutation(train.num_captions)

    t_start = time.time()
    best_epoch = None
    for epoch in range(start_epoch, epochs):
        model.train()
        order = rng.permutation(train.num_captions)
        losses: list[float] = []
        n_batches = (len(order) + batch_size - 1) // batch_size
        for b in progress(range(n_batches), total=n_batches, desc=f"epoch {epoch + 1}/{epochs}"):
            idx = order[b * batch_size:(b + 1) * batch_size]
            if len(idx) < 2:
                continue
            img_rows = train.caption_img_idx[idx]
            width = int(train.cap_lengths[idx].max())
            feats = torch.from_numpy(np.ascontiguousarray(train.features[img_rows], dtype=np.float32)).to(device)
            ids = torch.from_numpy(train.cap_ids[idx, :width].astype(np.int64)).to(device)
            lens = torch.from_numpy(train.cap_lengths[idx].astype(np.int64))
            img_emb, txt_emb = model(feats, ids, lens)
            loss = ranking_loss(img_emb, txt_emb, margin, torch.from_numpy(img_rows), max_violation)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses)) if losses else float("nan")

        val_metrics = evaluate_model(model, val_small, device)
        # validation loss on (at most) the first 5 000 validation captions, no gradient
        val_loss = _validation_loss(model, val_small, device, batch_size, margin, max_violation)
        improved = stopper.update(val_metrics[metric])
        entry = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                 "elapsed_s": round(time.time() - t_start, 1), **{k: float(v) for k, v in val_metrics.items()}}
        history.append(entry)
        log.info("epoch %d | train_loss %.4f | val_loss %.4f | %s %.2f%s", epoch + 1, train_loss, val_loss,
                 metric, val_metrics[metric], "  *best*" if improved else "")
        extra = {"bad_epochs": stopper.bad_epochs}
        save_checkpoint(last_path, model, config=cfg, vocab_size=vocab_size, epoch=epoch,
                        best_metric=stopper.best, history=history, optimizer=optimizer, extra=extra)
        if improved:
            best_epoch = epoch + 1
            save_checkpoint(best_path, model, config=cfg, vocab_size=vocab_size, epoch=epoch,
                            best_metric=stopper.best, history=history, extra=extra)
        save_json(history, run / "history.json")
        if stopper.should_stop:
            log.info("early stopping: no improvement in %d epochs", stopper.patience)
            break
    if not best_path.is_file():
        raise RuntimeError("training finished without producing a best checkpoint (0 epochs run?)")
    return {"epochs_run": len(history), "best_epoch": best_epoch, "best_val_metric": {metric: stopper.best},
            "train_seconds": round(time.time() - t_start, 1), "device": str(device), "selection_metric": metric,
            "stopped_early": stopper.should_stop}


def _validation_loss(model, data: SplitData, device, batch_size: int, margin: float, max_violation: bool,
                     max_captions: int = 5000) -> float:
    import torch

    from src.training.loss import ranking_loss

    model.eval()
    n = min(data.num_captions, max_captions)
    losses = []
    with torch.inference_mode():
        for s in range(0, n, batch_size):
            sl = slice(s, min(n, s + batch_size))
            rows = data.caption_img_idx[sl]
            if len(rows) < 2:
                continue
            width = int(data.cap_lengths[sl].max())
            feats = torch.from_numpy(np.ascontiguousarray(data.features[rows], dtype=np.float32)).to(device)
            ids = torch.from_numpy(data.cap_ids[sl, :width].astype(np.int64)).to(device)
            lens = torch.from_numpy(data.cap_lengths[sl].astype(np.int64))
            img_emb, txt_emb = model(feats, ids, lens)
            losses.append(float(ranking_loss(img_emb, txt_emb, margin, torch.from_numpy(rows), max_violation)))
    return float(np.mean(losses)) if losses else float("nan")
