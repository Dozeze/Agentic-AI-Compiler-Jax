"""A small GPT-style decoder forward pass with its training loss: workload, inputs and float64 oracle."""

from __future__ import annotations

import numpy as np

from halo.tasks.base import Case

NAME = "tiny_gpt_loss"

VOCAB, MODEL, Q_HEADS, KV_HEADS, SEQ, BATCH, LAYERS, FFN = 8192, 128, 8, 2, 128, 4, 2, 512
HEAD_DIM = MODEL // Q_HEADS
GROUP = Q_HEADS // KV_HEADS
EPS = 1e-5

DESCRIPTION = f"""\
The forward pass of a small GPT-style decoder and its per-token training loss:
token + position embeddings, {LAYERS} pre-norm transformer blocks with causal
grouped-query attention and a GELU MLP, a final layer norm, a language-model head
tied to the token embedding, and softmax cross-entropy against the next-token labels.

Signature: candidate(params, ids, labels) -> losses
Shapes:    ids and labels are int32 ({BATCH}, {SEQ}) with values in [0, {VOCAB});
           losses is float32 ({BATCH}, {SEQ}). params is a dict:
             wte ({VOCAB}, {MODEL})  wpe ({SEQ}, {MODEL})  lnf_g, lnf_b ({MODEL},)
             layers: a list of {LAYERS} dicts, each with
               ln1_g, ln1_b ({MODEL},)   wq ({MODEL}, {MODEL})   wk, wv ({MODEL}, {KV_HEADS * HEAD_DIM})
               wo ({MODEL}, {MODEL})     ln2_g, ln2_b ({MODEL},)
               w1 ({MODEL}, {FFN})  b1 ({FFN},)  w2 ({FFN}, {MODEL})  b2 ({MODEL},)
Semantics: h = wte[ids] + wpe[:T]
           for each layer:
             a = layernorm(h, ln1_g, ln1_b)
             q = a @ wq -> ({Q_HEADS} heads of {HEAD_DIM}); k = a @ wk, v = a @ wv -> ({KV_HEADS} heads of {HEAD_DIM})
             query head i attends with key/value head i // {GROUP}, causally (position t
             sees positions <= t), softmax(q k^T / sqrt({HEAD_DIM})) with max-subtraction
             h = h + concat_heads(attention) @ wo
             h = h + gelu_tanh(layernorm(h, ln2_g, ln2_b) @ w1 + b1) @ w2 + b2
           logits = layernorm(h, lnf_g, lnf_b) @ wte^T
           losses[b, t] = logsumexp(logits[b, t]) - logits[b, t, labels[b, t]]
           layernorm uses the population variance and epsilon {EPS} inside the sqrt;
           gelu_tanh(u) = 0.5 u (1 + tanh(0.7978845608 (u + 0.044715 u^3))).
"""

#: Raised from the default 1e-6: two residual blocks, an 8192-way head and a
#: logsumexp over it, in float32. Worst measured absolute error 3.1e-06 (seed)
#: and 3.6e-06 (best known implementation), on losses of magnitude ~9.
ATOL = 1e-4


def _layer(rng: np.random.Generator) -> dict:
    n = lambda *shape, scale: (rng.standard_normal(shape, dtype=np.float32) * scale)
    return {
        "ln1_g": np.ones((MODEL,), np.float32), "ln1_b": np.zeros((MODEL,), np.float32),
        "wq": n(MODEL, MODEL, scale=MODEL ** -0.5),
        "wk": n(MODEL, KV_HEADS * HEAD_DIM, scale=MODEL ** -0.5),
        "wv": n(MODEL, KV_HEADS * HEAD_DIM, scale=MODEL ** -0.5),
        "wo": n(MODEL, MODEL, scale=MODEL ** -0.5),
        "ln2_g": np.ones((MODEL,), np.float32), "ln2_b": np.zeros((MODEL,), np.float32),
        "w1": n(MODEL, FFN, scale=MODEL ** -0.5), "b1": np.zeros((FFN,), np.float32),
        "w2": n(FFN, MODEL, scale=FFN ** -0.5), "b2": np.zeros((MODEL,), np.float32),
    }


def make_params(rng: np.random.Generator) -> dict:
    return {
        "wte": rng.standard_normal((VOCAB, MODEL), dtype=np.float32) * 0.1,
        "wpe": rng.standard_normal((SEQ, MODEL), dtype=np.float32) * 0.1,
        "layers": [_layer(rng) for _ in range(LAYERS)],
        "lnf_g": np.ones((MODEL,), np.float32), "lnf_b": np.zeros((MODEL,), np.float32),
    }


def make_inputs(rng: np.random.Generator) -> tuple:
    return (
        make_params(rng),
        rng.integers(0, VOCAB, size=(BATCH, SEQ), dtype=np.int32),
        rng.integers(0, VOCAB, size=(BATCH, SEQ), dtype=np.int32),
    )


def _f64(tree):
    if isinstance(tree, dict):
        return {k: _f64(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_f64(v) for v in tree]
    return tree.astype(np.float64)


def _layernorm(x, g, b):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS) * g + b


def _gelu(u):
    return 0.5 * u * (1.0 + np.tanh(0.7978845608028654 * (u + 0.044715 * u ** 3)))


def _attention(a, p):
    b, t, _ = a.shape
    q = (a @ p["wq"]).reshape(b, t, Q_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k = (a @ p["wk"]).reshape(b, t, KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v = (a @ p["wv"]).reshape(b, t, KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k = np.repeat(k, GROUP, axis=1)
    v = np.repeat(v, GROUP, axis=1)
    scores = np.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(HEAD_DIM)
    scores = np.where(np.tril(np.ones((t, t), bool)), scores, -np.inf)
    scores -= scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(axis=-1, keepdims=True)
    out = np.einsum("bhqk,bhkd->bhqd", w, v).transpose(0, 2, 1, 3).reshape(b, t, MODEL)
    return out @ p["wo"]


def reference(params: dict, ids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    p = _f64(params)
    h = p["wte"][ids] + p["wpe"][: ids.shape[1]]
    for layer in p["layers"]:
        h = h + _attention(_layernorm(h, layer["ln1_g"], layer["ln1_b"]), layer)
        a = _layernorm(h, layer["ln2_g"], layer["ln2_b"])
        h = h + _gelu(a @ layer["w1"] + layer["b1"]) @ layer["w2"] + layer["b2"]
    logits = _layernorm(h, p["lnf_g"], p["lnf_b"]) @ p["wte"].T
    m = logits.max(axis=-1, keepdims=True)
    lse = (m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True)))[..., 0]
    return lse - np.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]


def correctness_cases(rng: np.random.Generator) -> list[Case]:
    params = make_params(rng)
    # One token everywhere: only the position embedding varies along the sequence,
    # so a candidate that mixes up token and position lookups is caught here.
    same = np.full((BATCH, SEQ), 7, np.int32)
    # No MLP: the residual stream is embeddings plus attention only, which isolates
    # the attention path - grouping, causal mask, output projection.
    no_mlp = make_params(rng)
    for layer in no_mlp["layers"]:
        layer["w2"] = np.zeros_like(layer["w2"])
    ids = rng.integers(0, VOCAB, size=(BATCH, SEQ), dtype=np.int32)
    labels = rng.integers(0, VOCAB, size=(BATCH, SEQ), dtype=np.int32)
    # Extreme ids and labels: an off-by-one in the vocabulary shows up at the edges.
    edges = np.where(rng.random((BATCH, SEQ)) < 0.5, 0, VOCAB - 1).astype(np.int32)
    # Labels equal to the input ids: with a tied head the true token's logit is
    # the largest by construction, so the loss is small and any label/index
    # mix-up shows as a large one.
    return [
        Case("single_token", (params, same, labels)),
        Case("attention_only", (no_mlp, ids, labels)),
        Case("edge_ids", (params, edges, edges[:, ::-1].copy())),
        Case("labels_are_ids", (params, ids, ids.copy())),
    ]
