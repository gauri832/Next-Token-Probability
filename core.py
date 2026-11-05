"""Core logic for the Next-Token Probability Lab (P5).

Everything here is plain PyTorch + Hugging Face transformers so it runs on a free CPU host.
Sampling (temperature / top-k / top-p) is implemented by hand in `adjust()` (Unit IV requirement).
"""
import math
import os
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

torch.set_grad_enabled(False)

# ----------------------------------------------------------------------------- models
_cache = {}


def get_gpt():
    if "gpt" not in _cache:
        tok = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").eval()
        _cache["gpt"] = (tok, model)
    return _cache["gpt"]


def get_bert():
    if "bert" not in _cache:
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").eval()
        _cache["bert"] = (tok, model)
    return _cache["bert"]


# ----------------------------------------------------------------------------- sampling
def adjust(logits, temperature=1.0, top_k=0, top_p=1.0):
    """Turn raw logits into the probability distribution actually sampled from.

    Order: temperature -> top-k -> top-p -> softmax (same order as Hugging Face).
    """
    logits = logits.clone().float() / max(float(temperature), 1e-3)
    if top_k and top_k > 0:
        k = min(int(top_k), logits.size(-1))
        kth = torch.topk(logits, k).values[-1]
        logits[logits < kth] = float("-inf")
    if top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        remove = (probs.cumsum(-1) - probs) > top_p  # keep the token that crosses top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(0, idx, sorted_logits)
    return F.softmax(logits, dim=-1)


def entropy_bits(p):
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum())


# ----------------------------------------------------------------------------- guardrail
# Deliberately simple demo blocklist (single-token variants only). See MODEL_CARD.md for limits.
BAD_WORDS = ["fuck", "shit", "bitch", "bastard", "asshole", "damn", "idiot", "stupid", "moron", "crap"]
BAD_RE = re.compile(r"\b(" + "|".join(BAD_WORDS) + r")\w*", re.I)


def banned_ids(tok):
    ids = set()
    for w in BAD_WORDS:
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize(), " " + w.upper()):
            enc = tok.encode(variant)
            if len(enc) == 1:
                ids.add(enc[0])
    return sorted(ids)


# ----------------------------------------------------------------------------- generation
FEW_SHOT = ("The capital of Russia is Moscow.\n"
            "The largest ocean on Earth is the Pacific.\n"
            "The author of Pride and Prejudice is Austen.\n")


def generate(prompt, max_new=40, temperature=1.0, top_k=0, top_p=1.0, seed=0, filter_toxic=False):
    """Sample `max_new` tokens from GPT-2 one at a time. Returns only the new text."""
    tok, model = get_gpt()
    g = torch.Generator().manual_seed(int(seed))
    ids = tok(prompt, return_tensors="pt").input_ids
    ban = banned_ids(tok) if filter_toxic else []
    new = []
    for _ in range(int(max_new)):
        logits = model(ids[:, -1024:]).logits[0, -1]
        if ban:
            logits[ban] = float("-inf")
        if temperature <= 0.05:
            nxt = torch.argmax(logits).view(1)
        else:
            nxt = torch.multinomial(adjust(logits, temperature, top_k, top_p), 1, generator=g)
        new.append(int(nxt))
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return tok.decode(new)


def _label(tok, i):
    return repr(tok.decode([int(i)]))


# ----------------------------------------------------------------------------- plots
def next_token_fig(prompt, temperature, top_k, top_p, n=15):
    tok, model = get_gpt()
    ids = tok(prompt, return_tensors="pt").input_ids[:, -1024:]
    logits = model(ids).logits[0, -1]
    base = F.softmax(logits, dim=-1)
    adj = adjust(logits, temperature, top_k, top_p)
    top = torch.topk(base, n).indices
    labels = [_label(tok, i) for i in top]
    x = range(n)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([i - 0.2 for i in x], base[top].tolist(), width=0.4, label="raw model (T=1)")
    ax.bar([i + 0.2 for i in x], adj[top].tolist(), width=0.4, label="after temperature/top-k/top-p")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("probability")
    ax.set_title(f"Next-token distribution (T={temperature}, top-k={top_k}, top-p={top_p})")
    ax.legend()
    fig.tight_layout()
    kept = int((adj > 0).sum())
    info = (f"**Entropy:** raw {entropy_bits(base):.2f} bits -> adjusted {entropy_bits(adj):.2f} bits  \n"
            f"**Top-1 token:** {_label(tok, base.argmax())} p={base.max():.3f} -> p={adj.max():.3f}  \n"
            f"**Tokens still possible after filtering:** {kept:,} of {base.numel():,}")
    return fig, info


def compare_temps_fig(prompt, t_low=0.2, t_high=1.2, n=10):
    tok, model = get_gpt()
    ids = tok(prompt, return_tensors="pt").input_ids[:, -1024:]
    logits = model(ids).logits[0, -1]
    base = F.softmax(logits, dim=-1)
    top = torch.topk(base, n).indices
    labels = [_label(tok, i) for i in top]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, t in zip(axes, (t_low, t_high)):
        p = adjust(logits, t)
        ax.bar(range(n), p[top].tolist())
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=60, ha="right")
        ax.set_title(f"T = {t}  (entropy {entropy_bits(p):.2f} bits)")
    axes[0].set_ylabel("probability")
    fig.tight_layout()
    samples = []
    for t in (t_low, t_high):
        samples.append(f"**T = {t}**")
        for s in range(3):
            samples.append(f"{s + 1}. {prompt}{generate(prompt, 30, t, seed=s)!r}")
    return fig, "\n\n".join(samples)


def attention_fig(prompt):
    tok, model = get_gpt()
    ids = tok(prompt, return_tensors="pt").input_ids[:, :40]
    att = model(ids, output_attentions=True).attentions[-1][0].mean(0)  # last layer, head-average
    labels = [_label(tok, i) for i in ids[0]]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(att.numpy(), cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("attended-to token (key)")
    ax.set_ylabel("current token (query)")
    ax.set_title("GPT-2 self-attention, last layer, mean of 12 heads\n(upper triangle is 0: causal mask)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def bert_vs_gpt(text, k=5):
    """Same blank, two models. BERT sees both sides; GPT-2 only the words before the blank."""
    if "[MASK]" not in text:
        return [["Put [MASK] in your sentence, e.g. The capital of France is [MASK].", "", "", ""]]
    btok, bert = get_bert()
    enc = btok(text, return_tensors="pt")
    pos = int((enc.input_ids[0] == btok.mask_token_id).nonzero()[0])
    bp = F.softmax(bert(**enc).logits[0, pos], dim=-1)
    btop = torch.topk(bp, k)
    gtok, gpt = get_gpt()
    left = text.split("[MASK]")[0].rstrip()
    gp = F.softmax(gpt(gtok(left, return_tensors="pt").input_ids).logits[0, -1], dim=-1)
    gtop = torch.topk(gp, k)
    return [[btok.decode([int(i)]), f"{float(p):.3f}", gtok.decode([int(j)]).strip(), f"{float(q):.3f}"]
            for i, p, j, q in zip(btop.indices, btop.values, gtop.indices, gtop.values)]


# ----------------------------------------------------------------------------- image
def text_card(text, size=(512, 512)):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, (30, 41, 59))
    d = ImageDraw.Draw(img)
    y = 30
    for line in textwrap.wrap(text, 44)[:20]:
        d.text((24, y), line, fill=(226, 232, 240))
        y += 22
    return img


def text_to_image(text):
    """Hosted sd-turbo if HF_TOKEN is set (Space secret); otherwise a plain text card."""
    token = os.environ.get("HF_TOKEN")
    if token and text.strip():
        try:
            from huggingface_hub import InferenceClient
            return InferenceClient(token=token).text_to_image(text.strip()[:300], model="stabilityai/sd-turbo")
        except Exception:
            pass
    return text_card(text)
