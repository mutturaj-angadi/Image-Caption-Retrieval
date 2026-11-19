"""Baseline, ranking loss, early stopping and (when PyTorch is installed) the neural models."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.baseline import LinearRegressionBaseline
from src.training.loss import ranking_loss_numpy
from src.training.trainer import EarlyStopper


# ------------------------------------------------------------------ baseline
def test_baseline_recovers_a_known_linear_map():
    rng = np.random.RandomState(0)
    X = rng.randn(500, 6).astype(np.float32)
    W, b = rng.randn(6, 9), rng.randn(9)
    Y = (X @ W + b).astype(np.float32)
    model = LinearRegressionBaseline(l2=1e-6).fit(X, Y)
    assert np.allclose(model.predict(X), Y, atol=1e-3)
    assert model.predict(X[0]).shape == (1, 9)  # a single 1-D row is accepted


def test_baseline_image_index_matches_materialised_targets():
    rng = np.random.RandomState(1)
    X = rng.randn(40, 4).astype(np.float32)
    imgs = rng.randn(8, 5).astype(np.float32)
    idx = rng.randint(0, 8, size=40)
    a = LinearRegressionBaseline(0.5).accumulate(X, imgs, idx, chunk=7).solve()
    b = LinearRegressionBaseline(0.5).accumulate(X, imgs[idx]).solve()
    assert np.allclose(a.W, b.W, atol=1e-4)


def test_baseline_ridge_grid_reuses_statistics_and_shrinks_weights():
    rng = np.random.RandomState(2)
    X, Y = rng.randn(100, 5).astype(np.float32), rng.randn(100, 3).astype(np.float32)
    m = LinearRegressionBaseline().accumulate(X, Y)
    small = np.linalg.norm(m.solve(0.01).W[:-1])
    large = np.linalg.norm(m.solve(1000.0).W[:-1])
    assert large < small


def test_baseline_errors_and_persistence(tmp_path):
    m = LinearRegressionBaseline()
    with pytest.raises(RuntimeError):
        m.predict(np.zeros((1, 3)))
    with pytest.raises(RuntimeError):
        m.solve()
    X, Y = np.random.randn(30, 3).astype(np.float32), np.random.randn(30, 4).astype(np.float32)
    m.fit(X, Y)
    with pytest.raises(ValueError):
        m.predict(np.zeros((1, 5)))
    wv = np.random.randn(7, 3).astype(np.float32)
    m.save(tmp_path / "b.npz", wv)
    m2, wv2 = LinearRegressionBaseline.load(tmp_path / "b.npz")
    assert np.allclose(m2.predict(X), m.predict(X)) and np.array_equal(wv, wv2)


# ------------------------------------------------------------------ ranking loss (reference implementation)
def test_ranking_loss_hand_computed_cases():
    perfect = np.eye(2, dtype=np.float32)
    assert ranking_loss_numpy(perfect, perfect, margin=0.2) == 0.0
    same = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)  # identical embeddings: every pair scores 1
    # per sample: 1 wrong caption + 1 wrong image, each costing margin 0.2 -> (0.4 + 0.4) / 2
    assert ranking_loss_numpy(same, same, margin=0.2) == pytest.approx(0.4)
    # the two rows describe the SAME image -> they are not negatives of each other -> zero loss
    assert ranking_loss_numpy(same, same, 0.2, image_index=np.array([5, 5])) == 0.0
    assert ranking_loss_numpy(same, same, 0.2, max_violation=True) == pytest.approx(0.4)


# ------------------------------------------------------------------ early stopping
def test_early_stopper_higher_and_lower_is_better():
    s = EarlyStopper(patience=2, higher_is_better_=True)
    assert [s.update(v) for v in (1.0, 2.0, 1.5, 1.4)] == [True, True, False, False]
    assert s.should_stop and s.best == 2.0
    lo = EarlyStopper(patience=2, higher_is_better_=False)
    assert [lo.update(v) for v in (5.0, 4.0, 4.5)] == [True, True, False] and not lo.should_stop
    assert not EarlyStopper(patience=0).should_stop


# ------------------------------------------------------------------ neural models (need PyTorch)
torch = pytest.importorskip("torch", reason="PyTorch is not installed; neural-model tests skipped")


def _vocab_matrix(v=20, d=8):
    m = np.random.RandomState(0).randn(v, d).astype(np.float32)
    m[0] = 0
    return m


@pytest.mark.parametrize("rnn", ["gru", "lstm"])
def test_text_encoder_shape_unit_norm_and_padding_invariance(rnn):
    from src.models.text_encoder import TextEncoder

    enc = TextEncoder(20, 8, 16, 12, rnn_type=rnn, embedding_matrix=_vocab_matrix()).eval()
    ids = torch.tensor([[3, 4, 5, 0, 0], [3, 4, 5, 6, 7]])
    out = enc(ids, torch.tensor([3, 5]))
    assert out.shape == (2, 12)
    assert torch.allclose(out.norm(dim=1), torch.ones(2), atol=1e-5)
    short = enc(torch.tensor([[3, 4, 5]]), torch.tensor([3]))
    assert torch.allclose(out[0], short[0], atol=1e-5)  # padding must not change the embedding
    with pytest.raises(ValueError):
        TextEncoder(20, 8, 16, 12, rnn_type="rnn")


def test_frozen_embeddings_are_not_trained():
    from src.models.text_encoder import TextEncoder

    enc = TextEncoder(20, 8, 16, 12, embedding_matrix=_vocab_matrix(), freeze_embeddings=True)
    assert not enc.embed.weight.requires_grad
    assert TextEncoder(20, 8, 16, 12).embed.weight.requires_grad


def test_torch_ranking_loss_matches_numpy_reference():
    from src.training.loss import ranking_loss

    rng = np.random.RandomState(3)
    img = rng.randn(6, 5).astype(np.float32)
    txt = rng.randn(6, 5).astype(np.float32)
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    idx = np.array([0, 1, 1, 2, 3, 0])
    for index in (None, idx):
        for mv in (False, True):
            ref = ranking_loss_numpy(img, txt, 0.3, index, mv)
            got = float(ranking_loss(torch.from_numpy(img), torch.from_numpy(txt), 0.3,
                                     None if index is None else torch.from_numpy(index), mv))
            assert got == pytest.approx(ref, rel=1e-5, abs=1e-6)


def test_joint_model_forward_and_checkpoint_roundtrip(tmp_path):
    from src.models.joint_model import (build_joint_model, load_checkpoint, model_from_checkpoint, save_checkpoint)

    cfg = {"model": {"type": "gru", "embedding_dim": 12, "hidden_dim": 16, "word_dim": 8}}
    model = build_joint_model(cfg["model"], 10, 20, _vocab_matrix()).eval()
    feats = torch.randn(4, 10)
    ids, lens = torch.tensor([[3, 4, 0], [5, 6, 7], [2, 0, 0], [9, 9, 9]]), torch.tensor([2, 3, 1, 3])
    img, txt = model(feats, ids, lens)
    assert img.shape == txt.shape == (4, 12)
    save_checkpoint(tmp_path / "c.pt", model, config=cfg, vocab_size=20, epoch=3, best_metric=1.5,
                    history=[{"epoch": 1, "train_loss": 0.5}])
    ckpt = load_checkpoint(tmp_path / "c.pt")
    assert ckpt["epoch"] == 3 and ckpt["history"][0]["train_loss"] == 0.5
    img2, txt2 = model_from_checkpoint(ckpt)(feats, ids, lens)
    assert torch.allclose(img, img2, atol=1e-6) and torch.allclose(txt, txt2, atol=1e-6)
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")


def test_neural_models_train_end_to_end_on_smoke_data(tmp_path):
    """GRU and LSTM train, checkpoint, evaluate, index and serve through the shared pipeline."""
    from src.smoke import build_smoke_workspace, smoke_service

    info = build_smoke_workspace(tmp_path, with_neural=True)
    assert set(info["models"]) == {"smoke_baseline", "smoke_gru", "smoke_lstm"}
    for name in ("smoke_gru", "smoke_lstm"):
        run = info["artifacts"] / name
        assert (run / "best.pt").is_file() and (run / "last.pt").is_file() and (run / "history.json").is_file()
        t2i = info["models"][name]["test"]["metrics"]
        assert t2i["t2i_R@10"] > 100 * 10 / 30  # better than chance on 30 test images (chance R@10 = 33%)
    svc = smoke_service(info)
    assert svc.search("a red circle", 5, model="smoke_gru")["num_results"] == 5
