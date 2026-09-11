import jax
import jax.numpy as jnp

EPS = 1e-5
Q_HEADS, KV_HEADS = 8, 2


def layernorm(x, g, b):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + EPS) * g + b


def gelu(u):
    return 0.5 * u * (1.0 + jnp.tanh(0.7978845608028654 * (u + 0.044715 * u ** 3)))


def attention(a, p):
    b, t, d = a.shape
    head_dim = d // Q_HEADS
    group = Q_HEADS // KV_HEADS
    q = (a @ p["wq"]).reshape(b, t, Q_HEADS, head_dim)
    k = (a @ p["wk"]).reshape(b, t, KV_HEADS, head_dim)
    v = (a @ p["wv"]).reshape(b, t, KV_HEADS, head_dim)
    k = jnp.repeat(k, group, axis=2)
    v = jnp.repeat(v, group, axis=2)
    mask = jnp.tril(jnp.ones((t, t), dtype=bool))
    heads = []
    for h in range(Q_HEADS):
        qh, kh, vh = q[:, :, h, :], k[:, :, h, :], v[:, :, h, :]
        scores = jnp.einsum("bqd,bkd->bqk", qh, kh) / jnp.sqrt(jnp.float32(head_dim))
        scores = jnp.where(mask, scores, -jnp.inf)
        scores = scores - jnp.max(scores, axis=-1, keepdims=True)
        w = jnp.exp(scores)
        w = w / jnp.sum(w, axis=-1, keepdims=True)
        heads.append(jnp.einsum("bqk,bkd->bqd", w, vh))
    out = jnp.concatenate(heads, axis=-1)
    return out @ p["wo"]


def candidate(params, ids, labels):
    vocab = params["wte"].shape[0]
    h = jax.nn.one_hot(ids, vocab, dtype=jnp.float32) @ params["wte"]
    h = h + params["wpe"][: ids.shape[1]]
    for layer in params["layers"]:
        h = h + attention(layernorm(h, layer["ln1_g"], layer["ln1_b"]), layer)
        a = layernorm(h, layer["ln2_g"], layer["ln2_b"])
        h = h + gelu(a @ layer["w1"] + layer["b1"]) @ layer["w2"] + layer["b2"]
    logits = layernorm(h, params["lnf_g"], params["lnf_b"]) @ params["wte"].T
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    log_probs = shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))
    targets = jax.nn.one_hot(labels, vocab, dtype=jnp.float32)
    return -jnp.sum(targets * log_probs, axis=-1)
