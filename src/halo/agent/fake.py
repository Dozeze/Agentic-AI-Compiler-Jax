"""Scripted agents. No network, no cost, fully deterministic.

These exist so the controller, the harness and the accept/reject policy can be
exercised end to end in CI. They are also the reason the :class:`Agent` protocol
is a real abstraction rather than a speculative one: there are two independent
implementations from the first commit.
"""

from __future__ import annotations

from halo.agent.protocol import Context
from halo.types import Proposal

#: Batches the per-head Python loop into two einsums. Genuinely faster.
FAST_ATTENTION = '''\
import jax.numpy as jnp


def candidate(q, k, v):
    dim = q.shape[-1]
    scale = 1.0 / jnp.sqrt(jnp.float32(dim))
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    scores = scores - jnp.max(scores, axis=-1, keepdims=True)
    weights = jnp.exp(scores)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, v)
'''

#: Drops the max-subtraction: fast, and wrong the moment scores get large.
UNSTABLE_ATTENTION = '''\
import jax.numpy as jnp


def candidate(q, k, v):
    dim = q.shape[-1]
    scale = 1.0 / jnp.sqrt(jnp.float32(dim))
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    weights = jnp.exp(scores)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, v)
'''


class ScriptedAgent:
    """Returns a fixed source, whatever the context says."""

    def __init__(self, source: str, name: str = "scripted") -> None:
        self.name = name
        self._source = source

    def propose(self, context: Context) -> Proposal:
        return Proposal(
            source=self._source,
            analysis=f"scripted agent '{self.name}', no analysis performed",
            hypothesis="fixed source supplied by the test harness",
        )


class EchoAgent:
    """Proposes the current source unchanged - the true no-op.

    Useful as a control: it measures how large a speedup the harness reports when
    nothing at all has changed, which is the noise floor the acceptance threshold
    has to clear.
    """

    name = "echo"

    def propose(self, context: Context) -> Proposal:
        return Proposal(
            source=context.current_source,
            analysis="no change proposed",
            hypothesis="control condition: measures the harness noise floor",
        )
