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
    # All heads in one batched matmul over (b, heads). Found by the agent
    # (runs/flash/ablation/tiny_gpt_loss-algorithmic-r0, attempt 2): at this
    # size the repeated k/v are cheap and plain `@` on (b, h, t, hd) beats the
    # hand-written grouped 5-d einsum, which costs layout transposes - 1.058x
    # [1.051, 1.064] over it, on Apple M5.
    b, t, d = a.shape
    head_dim = d // Q_HEADS
    group = Q_HEADS // KV_HEADS
    q = (a @ p["wq"]).reshape(b, t, Q_HEADS, head_dim).transpose(0, 2, 1, 3)
    k = jnp.repeat((a @ p["wk"]).reshape(b, t, KV_HEADS, head_dim), group, axis=2)
    v = jnp.repeat((a @ p["wv"]).reshape(b, t, KV_HEADS, head_dim), group, axis=2)
    scores = q @ k.transpose(0, 2, 3, 1) / jnp.sqrt(jnp.float32(head_dim))
    mask = jnp.tril(jnp.ones((t, t), dtype=bool))
    scores = jnp.where(mask[None, None], scores, -jnp.inf)
    scores = scores - jnp.max(scores, axis=-1, keepdims=True)
    w = jnp.exp(scores)
    w = w / jnp.sum(w, axis=-1, keepdims=True)
    out = (w @ v.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3).reshape(b, t, d)
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
