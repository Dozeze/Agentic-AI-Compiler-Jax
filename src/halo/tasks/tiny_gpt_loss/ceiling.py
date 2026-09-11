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
    # Query heads grouped under their kv head: (b, kv, group, t, hd); k and v are
    # never repeated, every head is one batched matmul.
    q = (a @ p["wq"]).reshape(b, t, KV_HEADS, group, head_dim).transpose(0, 2, 3, 1, 4)
    k = (a @ p["wk"]).reshape(b, t, KV_HEADS, head_dim).transpose(0, 2, 1, 3)
    v = (a @ p["wv"]).reshape(b, t, KV_HEADS, head_dim).transpose(0, 2, 1, 3)
    scores = jnp.einsum("bgrqd,bgkd->bgrqk", q, k) / jnp.sqrt(jnp.float32(head_dim))
    mask = jnp.tril(jnp.ones((t, t), dtype=bool))
    scores = jnp.where(mask, scores, -jnp.inf)
    scores = scores - jnp.max(scores, axis=-1, keepdims=True)
    w = jnp.exp(scores)
    w = w / jnp.sum(w, axis=-1, keepdims=True)
    out = jnp.einsum("bgrqk,bgkd->bgrqd", w, v)  # (b, kv, group, t, hd)
    out = out.transpose(0, 3, 1, 2, 4).reshape(b, t, d)
    return out @ p["wo"]


def candidate(params, ids, labels):
    h = params["wte"][ids] + params["wpe"][: ids.shape[1]]
    for layer in params["layers"]:
        h = h + attention(layernorm(h, layer["ln1_g"], layer["ln1_b"]), layer)
        a = layernorm(h, layer["ln2_g"], layer["ln2_b"])
        h = h + gelu(a @ layer["w1"] + layer["b1"]) @ layer["w2"] + layer["b2"]
    logits = layernorm(h, params["lnf_g"], params["lnf_b"]) @ params["wte"].T
    m = jnp.max(logits, axis=-1, keepdims=True)
    lse = (m + jnp.log(jnp.sum(jnp.exp(logits - m), axis=-1, keepdims=True)))[..., 0]
    picked = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    return lse - picked
