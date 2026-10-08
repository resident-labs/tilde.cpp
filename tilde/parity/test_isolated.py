"""Tests for isolated_questions (pytest). Needs tilda-train (TILDA_CODE) and, for the data tests, a DEV file and the
checkpoint: TILDE_TEST_DEV, TILDE_TEST_BASE, TILDE_TEST_CHECKPOINT. The model test needs CUDA.

    TILDA_CODE=... TILDE_TEST_DEV=dev.jsonl TILDE_TEST_BASE=base TILDE_TEST_CHECKPOINT=ckpt pytest test_isolated.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
if os.environ.get("TILDA_CODE"):
    sys.path.insert(0, os.environ["TILDA_CODE"])

need = pytest.mark.skipif(not all(os.environ.get(k) for k in ("TILDA_CODE", "TILDE_TEST_DEV", "TILDE_TEST_BASE",
                                                              "TILDE_TEST_CHECKPOINT")),
                          reason="needs TILDA_CODE, TILDE_TEST_DEV, TILDE_TEST_BASE, TILDE_TEST_CHECKPOINT")


@pytest.fixture(scope="module")
def data():
    from tilda.data import load_examples
    from tilda.serialization import load_tokenizer
    tokenizer = load_tokenizer(Path(os.environ["TILDE_TEST_BASE"]), Path(os.environ["TILDE_TEST_CHECKPOINT"]))
    examples = load_examples(Path(os.environ["TILDE_TEST_DEV"]))["validation"]
    by_state = defaultdict(list)
    for ex in examples:
        if ex.kind not in ("point", "remark") and not getattr(ex, "images", ()):
            by_state[ex.state_id].append(ex)
    groups = [g for g in by_state.values() if 3 <= len(g) <= 6][:8]
    assert groups, "the DEV file needs states with 3-6 text questions"
    return tokenizer, groups


@need
def test_split_matches_one_question_encoding(data):
    from isolated_questions import split_record
    from tilda.clef_joint import encode_joint_record
    tokenizer, groups = data
    for group in groups:
        joint = encode_joint_record(group, tokenizer, 65536)
        for (single, _), ex in zip(split_record(joint, tokenizer), group, strict=True):
            ref = encode_joint_record([ex], tokenizer, 65536)
            assert single.input_ids == ref.input_ids
            assert single.questions[0].question_span == ref.questions[0].question_span
            assert single.questions[0].option_spans == ref.questions[0].option_spans


@need
@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs CUDA")
def test_answers_do_not_depend_on_other_questions(data):
    """Isolated logits of a question are identical whether it is asked alone, with others, or in another order; the
    joint logits do move (the test would be vacuous otherwise)."""
    import torch
    from isolated_questions import isolated_forward
    from tilda import joint_serving
    from tilda.clef_joint import encode_joint_record, forward_joint
    tokenizer, groups = data
    model, tok, _ = joint_serving.load_joint_checkpoint(os.environ["TILDE_TEST_BASE"],
                                                        os.environ["TILDE_TEST_CHECKPOINT"], device="cuda",
                                                        verify=False, fast_head=True)
    pad = tok.pad_token_id or 0
    moved = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for group in groups[:4]:
            first = group[0]
            variants = [group, list(reversed(group)), [first], group[:2], [group[-1], first]]
            answers, joint = [], []
            for v in variants:
                rec = encode_joint_record(v, tokenizer, 65536)
                at = v.index(first)
                answers.append(isolated_forward(model, [rec], pad, tokenizer)[0][at].float())
                joint.append(forward_joint(model, [rec], pad)[0][at].float())
            for a in answers[1:]:
                assert torch.equal(a, answers[0]), (answers[0], a)
            moved += int(any(not torch.allclose(j, joint[0], atol=1e-3) for j in joint[1:]))
    assert moved > 0, "joint answers never moved with the other questions; the invariance test proves nothing"
