"""Acceptance decision (AIC-006): stability, the 1 + delta rule, one retry and fresh confirmation.

Sessions are synthetic (fixed nanosecond values), so these tests check the rule, not real speed.
"""
import itertools

from benchmark import Block, PairedSession, analyze_session
from contracts import (ACCEPTED, MEASUREMENT_UNSTABLE, VALID_NO_CHANGE, VALID_SLOWER, MeasurementProtocol,
                       protocol_hash)
from controller import decide, judge, verify_winner

P = MeasurementProtocol(warmup_calls=2, n_blocks=6, bootstrap_samples=500)
DELTA = 0.03


def analysis(b_over_a, noisy=False):
    """A session whose candidate side B takes `b_over_a` times as long as A, optionally noisy."""
    a_ns = [1_000_000] * 6
    b_ns = [round(1_000_000 * b_over_a * (1.5 if noisy and i % 2 else 1.0)) for i in range(6)]
    blocks = tuple(Block(i, "AB", 1, a, b, 2 * i, 2 * i + 1) for i, (a, b) in enumerate(zip(a_ns, b_ns)))
    return analyze_session(PairedSession(protocol_hash(P), 1, blocks), P)


def test_clear_win_is_accepted():
    assert decide(analysis(0.5), DELTA).state == ACCEPTED


def test_win_inside_delta_is_no_change():
    assert decide(analysis(1 / 1.02), DELTA).state == VALID_NO_CHANGE


def test_clear_loss_is_slower():
    assert decide(analysis(1.5), DELTA).state == VALID_SLOWER


def test_unstable_session_is_never_accepted():
    decision = decide(analysis(0.5, noisy=True), DELTA)
    assert decision.state == MEASUREMENT_UNSTABLE
    assert "IQR/median" in decision.reason


def _scripted(*analyses):
    it = iter(analyses)
    return lambda: next(it)


def test_judge_confirms_a_win_in_a_fresh_session():
    decision, used = judge(_scripted(analysis(0.5), analysis(0.5)), DELTA, P)
    assert decision.state == ACCEPTED and len(used) == 2
    assert "confirmed" in decision.reason


def test_judge_rejects_an_unconfirmed_win():
    decision, used = judge(_scripted(analysis(0.5), analysis(1.0)), DELTA, P)
    assert decision.state == VALID_NO_CHANGE and len(used) == 2
    assert "not confirmed" in decision.reason


def test_judge_retries_an_unstable_session_once_only():
    decision, used = judge(_scripted(analysis(0.5, True), analysis(0.5, True), analysis(0.5)), DELTA, P)
    assert decision.state == MEASUREMENT_UNSTABLE and len(used) == 2


def test_judge_retry_can_recover():
    decision, used = judge(_scripted(analysis(0.5, True), analysis(0.5), analysis(0.5)), DELTA, P)
    assert decision.state == ACCEPTED and len(used) == 3


def test_judge_no_confirmation_for_a_non_win():
    decision, used = judge(_scripted(analysis(1.0)), DELTA, P)
    assert decision.state == VALID_NO_CHANGE and len(used) == 1


def test_verify_winner_needs_every_session_on_every_shape():
    shapes = [(32, 128), (128, 1024)]
    ok, per_shape = verify_winner(lambda shape: analysis(0.5), shapes, lambda shape: DELTA, P)
    assert ok and all(len(per_shape[s]) == P.final_sessions for s in shapes)

    results = itertools.cycle([analysis(0.5), analysis(0.5), analysis(1.0)])
    ok, per_shape = verify_winner(lambda shape: next(results), shapes, lambda shape: DELTA, P)
    assert not ok
    assert VALID_NO_CHANGE in {d.state for decisions in per_shape.values() for d in decisions}
