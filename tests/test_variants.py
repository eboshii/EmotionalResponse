import numpy as np
import torch

from emosteer import data as D
from emosteer import variants as V
from emosteer.encoding import decompose, eta_squared, quality_from_calib, spearman


def _sets(d):
    return {e: s["sentences"] for e, s in d["emotions"].items()}


def test_splits_disjoint_and_cover():
    d = D.load_emotions()
    sets = _sets(d)
    names = list(sets)
    a, _ = V.build_sets(d, "split0of2", names, sets)
    b, _ = V.build_sets(d, "split1of2", names, sets)
    for n in names:
        assert not set(a[n]) & set(b[n])
        assert len(a[n]) + len(b[n]) == len(sets[n])


def test_boot_suffix_and_set():
    d = D.load_emotions()
    sets = _sets(d)
    names = list(sets)
    bt, neu = V.build_sets(d, "boot0", names, sets)
    assert all(len(bt[n]) == len(sets[n]) for n in names) and len(neu) == len(d["neutral"]["sentences"])
    bt2, _ = V.build_sets(d, "boot0", names, sets)
    assert bt == bt2  # deterministic
    sf, neu = V.build_sets(d, "suffix= Right now I am", names, sets)
    assert all(t.endswith(" Right now I am") for t in sf["pain"] + neu)
    ex, neu = V.build_sets(d, "set=explicit", names, sets)
    assert "Right now I am in pain. I feel:" in ex["pain"]
    assert any("ordinary" in t for t in neu)
    assert V.tag_of("suffix= Right now I am") == "suffix_right_now_i_am"
    assert V.key_for("pain", "base") == "pain" and V.base_of("pain@boot0") == "pain"
    assert V.variant_of("pain@boot0") == "boot0" and V.variant_of("pain") == "base"


def test_consistency():
    e1, e2 = torch.zeros(3, 4), torch.zeros(3, 4)
    e1[:, 0] = 1
    e2[:, 1] = 1
    dirs = {"a": e1, "a@x": e1, "b": e2, "b@x": e2}
    within, between = V.consistency(dirs, 0)
    assert within["a"] == 1.0 and between["a"] == 0.0


def test_decompose_emotion_vs_encoding():
    rng = np.random.default_rng(0)
    emos = ["pain", "fear", "anger", "calm"]
    true = {"pain": 0.3, "fear": 0.2, "anger": 0.1, "calm": -0.1}
    tags = ["base", "split0of2", "split1of2", "set_explicit"]
    key = lambda e, t: V.key_for(e, t)
    # emotion-driven: small encoding noise
    eff = {key(e, t): true[e] + rng.normal(0, 0.01) for e in emos for t in tags}
    md, rec = decompose(eff, n_perm=500)
    assert rec["eta2"] > 0.9 and rec["perm_p"] < 0.01
    assert min(rec["rank_stability"].values()) > 0.7
    # encoding-driven: effect is pure noise across encodings
    eff = {key(e, t): rng.normal(0, 0.2) for e in emos for t in tags}
    _, rec = decompose(eff, n_perm=500)
    assert rec["perm_p"] > 0.01
    # quality-driven: effect = AUC, emotion identity irrelevant after regression
    q = {key(e, t): {"auc": rng.uniform(0.6, 1.0), "specificity": rng.uniform(0, 1)} for e in emos for t in tags}
    eff = {k: q[k]["auc"] for k in q}
    md, rec = decompose(eff, q, n_perm=500)
    assert rec["corr_auc"] > 0.99
    assert "η² after regressing out" in "\n".join(md)
    assert eta_squared([1, 1, 2, 2], ["a", "a", "b", "b"]) == 1.0
    assert spearman([1, 2, 3], [3, 2, 1]) == -1.0


def test_quality_from_calib():
    calib = {"meta": {"auc_read": {"pain": 0.9}},
             "results": {"+pain": {"levels": {"0.3": {"alpha": 1.0, "readout": {"pain": 2.0},
                                                       "shift_norm": 4.0}}},
                         "-pain": {"levels": {}}}}
    q = quality_from_calib(calib, 0.3)
    assert q == {"pain": {"auc": 0.9, "specificity": 0.5}}
