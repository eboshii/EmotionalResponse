"""Model loading, decoder-layer discovery and chat-template handling."""
from __future__ import annotations

import re

import torch
from torch import nn

LAYER_NAMES = ("layers", "h", "blocks")
SKIP_RE = re.compile(r"vision|visual|image|audio|vit|mm_projector", re.I)


def pick_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        return {"cuda": torch.bfloat16, "mps": torch.float16}.get(device, torch.float32)
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]


def load(name: str, device: str = "auto", dtype: str = "auto", device_map: str | None = None):
    """Load (model, tokenizer). Falls back to multimodal auto-classes for VLM checkpoints."""
    import transformers as tf

    device = pick_device(device)
    torch_dtype = pick_dtype(dtype, device)
    kw = dict(torch_dtype=torch_dtype)
    if device_map:
        kw["device_map"] = device_map
    errors = []
    model = None
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
        cls = getattr(tf, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(name, **kw)
            break
        except (ValueError, KeyError, OSError) as e:  # unrecognised config for this auto-class
            errors.append(f"{cls_name}: {e}")
    if model is None:
        raise RuntimeError("Could not load model:\n" + "\n".join(errors))
    if not device_map:
        model.to(device)
    model.eval()
    tok = tf.AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return model, tok


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def _get_path(root, path: str):
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def find_layers(model: nn.Module, override: str | None = None) -> nn.ModuleList:
    """Largest ModuleList named layers/h/blocks, ignoring vision/audio towers."""
    if override:
        return _get_path(model, override)
    best, best_name = None, None
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.ModuleList):
            continue
        if name.split(".")[-1] not in LAYER_NAMES or SKIP_RE.search(name):
            continue
        if best is None or len(mod) > len(best):
            best, best_name = mod, name
    if best is None:
        raise RuntimeError("No decoder layer list found; pass --layers-path")
    find_layers.last_path = best_name
    return best


def chat(tok, user: str, system: str | None = None, thinking: bool = False,
         assistant_prefix: str = "") -> str:
    """Render a single-turn chat prompt; merges system into user if the template rejects it."""
    def render(msgs):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=thinking)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    if getattr(tok, "chat_template", None) is None:
        text = (f"{system}\n\n" if system else "") + user + "\n"
        return text + assistant_prefix
    out = None
    if system:
        try:
            out = render([{"role": "system", "content": system}, {"role": "user", "content": user}])
            if system not in out:
                out = None
        except Exception:  # jinja TemplateError: "System role not supported"
            out = None
    if out is None:
        merged = f"{system}\n\n{user}" if system else user
        out = render([{"role": "user", "content": merged}])
    return out + assistant_prefix


def common_args(p):
    p.add_argument("--model", required=True)
    p.add_argument("--layers-path", default=None, help="dotted path to decoder ModuleList")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    p.add_argument("--device-map", default=None, help="e.g. 'auto' for multi-GPU")
    p.add_argument("--batch", type=int, default=8)
    return p
