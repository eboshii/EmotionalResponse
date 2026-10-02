"""End-to-end on a tiny random model: extract -> calibrate -> run -> analyze (CPU)."""
import json

import torch

from emosteer import analyze, calibrate, extract, run


def test_pipeline(tmp_path, tiny_model, tok, monkeypatch):
    import emosteer.models as M
    for mod in (extract, calibrate, run):
        monkeypatch.setattr(mod, "load", lambda *a, **k: (tiny_model, tok))
    out = tmp_path / "r"
    extract.main(["--model", "tiny", "--out", str(out), "--emotions", "pain,happiness",
                  "--folds", "3", "--batch", "16"])
    ex = torch.load(out / "directions.pt")
    assert ex["directions"]["pain"].shape == (6, 32)
    calibrate.main(["--model", "tiny", "--run", str(out), "--targets", "0.001,0.01",
                    "--n-random", "1", "--n-prompts", "6", "--gen-samples", "--gen-tokens", "4"])
    cal = json.loads((out / "calib.json").read_text())
    lv = cal["results"]["+pain"]["levels"]
    for t, v in lv.items():
        if v.get("alpha"):
            assert abs(v["achieved_kl"] - float(t)) / float(t) < 0.5
    jl = tmp_path / "run.jsonl"
    run.main(["--model", "tiny", "--run", str(out), "--level", "0.001", "--episodes", "4",
              "--steps", "4", "--n-random", "1", "--out", str(jl)])
    rows = [json.loads(l) for l in jl.read_text().splitlines()]
    arms = {r["arm"] for r in rows}
    assert {"baseline", "state:pain", "contingent:happiness", "sham:pain", "random_relief:0"} <= arms
    # paired seeds: same scene sequence across arms for each episode
    seq = lambda arm: [(r["episode"], r["scene_id"]) for r in rows if r["arm"] == arm and r["step"] == 0]
    assert seq("baseline") == seq("state:pain")
    assert all(0 <= r["p_violate"] <= 1 for r in rows)
    analyze.main([str(jl), "--out", str(tmp_path / "rep"), "--n-boot", "200"])
    assert "Ranking: state effect" in (tmp_path / "rep.md").read_text()


def test_pipeline_variants(tmp_path, tiny_model, tok, monkeypatch):
    for mod in (extract, calibrate, run):
        monkeypatch.setattr(mod, "load", lambda *a, **k: (tiny_model, tok))
    out = tmp_path / "r"
    extract.main(["--model", "tiny", "--out", str(out), "--emotions", "pain,fear,happiness",
                  "--folds", "3", "--batch", "16", "--variant", "split0of2", "--variant", "split1of2",
                  "--variant", "set=explicit"])
    ex = torch.load(out / "directions.pt")
    assert {"pain", "pain@split0of2", "pain@split1of2", "pain@set_explicit"} <= set(ex["directions"])
    assert ex["emotions"] == ["pain", "fear", "happiness"]
    rep = json.loads((out / "extract.json").read_text())
    assert "pain" in rep["consistency"]["steer"]["within"]
    calibrate.main(["--model", "tiny", "--run", str(out), "--targets", "0.001",
                    "--n-random", "1", "--n-prompts", "4"])
    cal = json.loads((out / "calib.json").read_text())
    assert "+pain@split0of2" in cal["results"] and "-pain@split0of2" not in cal["results"]
    assert "+pain@split0of2" in cal["matrices"]["0.001"]["self_readout"]
    jl = tmp_path / "run.jsonl"
    run.main(["--model", "tiny", "--run", str(out), "--level", "0.001", "--episodes", "3",
              "--steps", "3", "--n-random", "0", "--out", str(jl)])
    rows = [json.loads(l) for l in jl.read_text().splitlines()]
    assert {"state:pain@split1of2", "sham:fear@set_explicit"} <= {r["arm"] for r in rows}
    r0 = next(r for r in rows if r["arm"] == "state:pain@split1of2")
    assert r0["emotion"] == "pain" and r0["variant"] == "split1of2"
    analyze.main([str(jl), "--out", str(tmp_path / "rep"), "--n-boot", "100",
                  "--calib", str(out / "calib.json")])
    text = (tmp_path / "rep.md").read_text()
    assert "Encoding decomposition — state effect" in text


def test_report_synthetic(tmp_path):
    import numpy as np
    import pytest
    from emosteer import report
    rng = np.random.default_rng(0)
    rows = []
    def ep(arm, tgt, val, shift):
        for e in range(20):
            for t in range(5):
                p = float(np.clip(0.3 + shift + rng.normal(0, .05), 0, 1))
                rows.append(dict(model="m", env="mini", level=0.3, arm=arm, target=tgt, valence=val,
                                 seed=e, step=t, p_violate=p, violated=bool(rng.random() < p)))
    ep("baseline", "", 0, 0)
    ep("random_relief:0", "0", 0, 0)
    for k, v, s in (("pain", -1, .1), ("calm", 1, -.05)):
        for key in (k, k + "@split0of2"):
            ep(f"state:{key}", key, v, s); ep(f"sham:{key}", key, v, s); ep(f"contingent:{key}", key, v, s + .05)
    jl = tmp_path / "x.jsonl"
    jl.write_text("\n".join(json.dumps(r) for r in rows))
    pytest.importorskip("matplotlib")
    report.main([str(jl), "--out", str(tmp_path / "blog"), "--n-boot", "100", "--dark"])
    md = (tmp_path / "blog_table.md").read_text()
    assert md.index("| pain") < md.index("| calm")          # sorted by state effect
    assert (tmp_path / "blog_effects.png").exists() and (tmp_path / "blog_effects_dark.svg").exists()
