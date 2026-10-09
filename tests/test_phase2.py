"""Phase 2 invariants (6) — metrics + confusion."""
from pathlib import Path

import pandas as pd
import pytest

CHECKPOINTS = Path(__file__).resolve().parent.parent / "checkpoints"


def _load(name):
    p = CHECKPOINTS / name
    if not p.exists():
        pytest.skip(f"{name} missing — run `python src/phase2_evaluate.py` first")
    return pd.read_parquet(p)


def test_metrics_has_ten_classes():
    m = _load("metrics.parquet")
    assert len(m) == 10


def test_metrics_accuracy_in_range():
    m = _load("metrics.parquet")
    assert ((m["accuracy"] >= 0) & (m["accuracy"] <= 1)).all()


def test_metrics_overall_reasonable():
    """A sane CNN trained for 2 epochs should hit at least 70% overall accuracy."""
    m = _load("metrics.parquet")
    overall = (m["accuracy"] * m["support"]).sum() / m["support"].sum()
    assert overall > 0.7, f"overall accuracy {overall:.3f} too low — model probably broken"


def test_confusion_shape():
    c = _load("confusion.parquet")
    assert len(c) == 100  # 10×10


def test_confusion_totals_match_support():
    c = _load("confusion.parquet")
    m = _load("metrics.parquet")
    for cls_id, row in m.iterrows():
        n_in_conf = c[c["true_id"] == row["class_id"]]["n"].sum()
        assert n_in_conf == row["support"], (
            f"support mismatch for class {row['class_name']}: "
            f"{row['support']} vs confusion sum {n_in_conf}"
        )


def test_confusion_diagonal_dominant_on_easy_classes():
    """Trouser and Sneaker should be the easiest; their diagonal should dominate."""
    c = _load("confusion.parquet")
    for easy_cls in ("Trouser", "Sneaker"):
        row = c[c["true"] == easy_cls]
        diag = row[row["pred"] == easy_cls]["n"].iloc[0]
        total = row["n"].sum()
        if total > 0:
            assert diag / total > 0.7, (
                f"{easy_cls} diagonal only {diag}/{total} — model confused on easy class?"
            )
