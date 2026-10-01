import json
import string

import pytest
import torch
from torch import nn


def make_tokenizer(chat_template=True):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from emosteer.data import load_emotions
    d = load_emotions()
    words = set(string.ascii_uppercase) | set(string.ascii_lowercase) | set(string.digits)
    import re, pathlib
    blob = json.dumps(d) + pathlib.Path("data/mini_scenarios.json").read_text()
    words |= set(re.findall(r"\w+|[^\w\s]", blob))
    words |= {"<pad>", "<eos>", "<unk>", "<|user|>", "<|system|>", "<|assistant|>"}
    vocab = {w: i for i, w in enumerate(sorted(words))}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(),
                                                pre_tokenizers.Punctuation()])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>",
                                  unk_token="<unk>")
    tok.padding_side = "left"
    if chat_template:
        tok.chat_template = ("{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} {% endfor %}"
                             "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    return tok


@pytest.fixture(scope="session")
def tok():
    return make_tokenizer()


@pytest.fixture(scope="session")
def tiny_model(tok):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=1024, pad_token_id=tok.pad_token_id)
    return LlamaForCausalLM(cfg).eval()
