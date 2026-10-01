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
