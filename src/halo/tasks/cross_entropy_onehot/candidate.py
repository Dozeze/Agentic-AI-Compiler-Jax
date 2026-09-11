import jax
import jax.numpy as jnp


def candidate(logits, labels):
    vocab = logits.shape[-1]
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    log_probs = shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))
    targets = jax.nn.one_hot(labels, vocab, dtype=logits.dtype)
    return -jnp.sum(targets * log_probs, axis=-1)
