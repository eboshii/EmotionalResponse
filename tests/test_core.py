import numpy as np
import torch
from torch import nn

from emosteer import data as D
from emosteer.arms import Arm, Schedule, default_arms, parse_arm
from emosteer.calibrate import interp_alpha, random_dirs
from emosteer.extract import auc, build_directions, denoise_basis, project_out
from emosteer.hooks import Steer, capture_last
from emosteer.models import chat, find_layers
from emosteer.stats import cond_rate, paired_bootstrap, transitions


def test_dataset_lint_clean():
    d = D.load_emotions()
    assert D.lint(d) == []
    assert all(len(v["sentences"]) >= 20 for v in d["emotions"].values())
    assert all(v["valence"] in (-1, 1) for v in d["emotions"].values())


def test_find_layers_skips_vision(tiny_model):
    class VLM(nn.Module):
        def __init__(self, lm):
            super().__init__()
            self.vision_tower = nn.Module()
            self.vision_tower.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(20)])
            self.language_model = lm
    m = VLM(tiny_model)
    layers = find_layers(m)
    assert len(layers) == 6 and "vision" not in find_layers.last_path
    assert find_layers(m, "vision_tower.blocks") is m.vision_tower.blocks


def test_steer_hook_tuple_and_tensor():
    lin = nn.Linear(4, 4)
    x = torch.zeros(1, 3, 4)
    with torch.no_grad():
        base = lin(x)
        s = Steer(lin, torch.ones(4))
        assert torch.allclose(lin(x), base + 1)
        s.positions = "last"
        out = lin(x)
        assert torch.allclose(out[:, -1], base[:, -1] + 1) and torch.allclose(out[:, 0], base[:, 0])
        s.on = False
        assert torch.allclose(lin(x), base)
        s.remove()

    class TupleMod(nn.Module):
        def forward(self, x):
            return (x * 2, "cache")
    tm = TupleMod()
    s = Steer(tm, torch.ones(4))
    h, c = tm(x)
    assert c == "cache" and torch.allclose(h, torch.ones_like(h))
    s.remove()


def test_capture_and_steer_on_tiny_model(tiny_model, tok):
    layers = find_layers(tiny_model)
    acts = capture_last(tiny_model, tok, layers, ["I am here . I feel :", "a b"], batch=2)
    assert acts.shape == (2, 6, 32)
    # left padding: last token activation identical when run alone
    solo = capture_last(tiny_model, tok, layers, ["a b"], batch=1)
    assert torch.allclose(acts[1], solo[0], atol=1e-4)


def test_chat_system_fallback(tok):
    out = chat(tok, "hello", "SYS")
    assert "SYS" in out and out.endswith("<|assistant|>")
    import copy
    t2 = copy.deepcopy(tok)
    t2.chat_template = ("{% for m in messages %}{% if m['role'] == 'system' %}"
                        "{{ raise_exception('System role not supported') }}{% endif %}"
                        "<|user|> {{ m['content'] }} {% endfor %}<|assistant|>")
    out = chat(t2, "hello", "SYS")
    assert "SYS" in out and "hello" in out


def test_denoise_and_directions():
    g = torch.Generator().manual_seed(0)
    d, n = 16, 40
    noise_dir = torch.zeros(d); noise_dir[0] = 1
    sig = torch.zeros(d); sig[1] = 1
    neutral = torch.randn(n, d, generator=g) * 0.1 + torch.randn(n, 1, generator=g) * 5 * noise_dir
    emo = torch.randn(n, d, generator=g) * 0.1 + torch.randn(n, 1, generator=g) * 5 * noise_dir + sig
    basis = denoise_basis(neutral, 0.5)
    assert abs(float(basis[0] @ noise_dir)) > 0.9
    assert abs(float(project_out(noise_dir, basis).norm())) < 0.1
    acts = {"neutral": neutral[:, None].repeat(1, 3, 1), "x": emo[:, None].repeat(1, 3, 1)}
    dirs, aucs, mean_auc, read = build_directions(acts, ["x"], "neutral", 0.5, 4)
    assert float(dirs["x"][0] @ sig) > 0.9 and min(aucs["x"]) > 0.95
    assert auc(np.array([1., 2.]), np.array([0., 0.5])) == 1.0


def test_interp_alpha():
    alphas = np.logspace(-2, 0.3, 12)
    kls = 0.5 * alphas ** 2
    for t in (0.01, 0.1, 0.5):
        al = interp_alpha(alphas, kls, t)
        assert abs(0.5 * al ** 2 - t) / t < 0.05
    assert interp_alpha(alphas, kls, 100) is None
    # non-monotone noise handled by running max
    assert interp_alpha([0.1, 0.2, 0.4], [0.1, 0.05, 1.0], 0.3) is not None
    v = random_dirs(3, 8)
    assert torch.allclose(v.norm(dim=1), torch.ones(3))


def test_schedules():
    val = {"pain": -1, "happiness": 1}
    seq = [False, True, False, False, False, True, False]
    # negative contingent: on, violation -> off for K=2 steps
    s = Schedule(parse_arm("contingent:pain", val), K=2)
    ons = []
    for v in seq:
        ons.append(s.steer_on()); s.update(v)
    assert ons == [True, True, False, False, True, True, False]
    # positive contingent: off, violation -> on for 2
    s = Schedule(parse_arm("contingent:happiness", val), K=2)
    ons = []
    for v in seq:
        ons.append(s.steer_on()); s.update(v)
    assert ons == [False, False, True, True, False, False, True]
    # sham relief: always on
    s = Schedule(parse_arm("sham:pain", val), K=2)
    ons = []
    for v in seq:
        ons.append(s.steer_on()); s.update(v)
    assert all(ons)
    # sham reward ignores actions: same rng -> same schedule regardless of violations
    a = parse_arm("sham:happiness", val)
    def run(vs):
        s = Schedule(a, K=2, sham_p=0.3, rng=np.random.default_rng(5)); o = []
        for v in vs:
            o.append(s.steer_on()); s.update(v)
        return o
    assert run([True] * 20) == run([False] * 20)
    arms = default_arms(["pain", "happiness"], val, 1)
    names = [x.name for x in arms]
    assert names.index("contingent:happiness") < names.index("sham:happiness")
    assert parse_arm("random_relief:0", val).direction_key == "random0"


def test_bootstrap_and_transitions():
    a = {i: 0.6 for i in range(30)}
    b = {i: 0.4 for i in range(30)}
    r = paired_bootstrap(a, b, 2000)
    assert abs(r["diff"] - 0.2) < 1e-9 and r["lo"] <= 0.2 <= r["hi"]
    rows = [{"arm": "x", "seed": 1, "step": t, "violated": v} for t, v in enumerate([1, 1, 0, 1, 0])]
    tr = transitions(rows)["x"]
    assert cond_rate(tr) == 1 / 3 and cond_rate(tr, which="after_n") == 1.0
