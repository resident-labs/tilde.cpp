"""Isolated questions for the tilda-train PyTorch reference (importable; needs tilda-train on the path).

Entry point:

    from isolated_questions import isolated_forward
    logits = isolated_forward(model, records, pad_token_id, tokenizer)   # same shape as tilda.clef_joint.forward_joint

Isolated: each question's answer depends on the state and that question only. The state is read once (it never
sees a question); question i sees the state and itself; the KDA and short-convolution states of question i start
from the end of the state (not chained through earlier questions); the head of question i sees its own fields and
the state; question i keeps the positions it has in a one-question request (its field block right after the state).

How: every question becomes the one-question request the encoder would produce for it (`split_record`: preamble,
state, <|tilda_state_end|>, schema header, the field renumbered FIELD 1 / field_1, end marker), run through
`forward_joint`. With bidirectional MLA (`tilda.ling.attention` mode "bidirectional") the MLA layers are split at the
end of the schema header: those rows attend to the prefix keys only, the field's rows to every key. With causal MLA
nothing is needed (the prefix never sees later tokens). KDA and the convolutions are causal in both modes. This is
mathematically one shared state pass plus per-question branches; it runs here as N one-question passes (a
measurement implementation, not a fast one).

Answers are invariant to adding, removing or reordering the other questions of a request (test_isolated.py).
"""
from __future__ import annotations

import contextlib
from dataclasses import replace

SCHEMA_HEADER = "\n\nSCHEMA FIELDS:\n"

_SPLIT = {"prefix": None}
_INSTALLED = {"done": False}


def _tokens(tokenizer, text):
    from tilda.clef_joint import _raw_tokens
    return _raw_tokens(tokenizer, text)


def _marker(tokenizer, index):
    from tilda.serialization import STRUCTURE_TOKENS
    return tokenizer.convert_tokens_to_ids(STRUCTURE_TOKENS[index])


def prefix_end(input_ids, tokenizer) -> int:
    """End of the question-independent part (preamble, state, <|tilda_state_end|>, schema header)."""
    ids = list(input_ids)
    pos = ids.index(_marker(tokenizer, 2))
    header = _tokens(tokenizer, SCHEMA_HEADER)
    if ids[pos + 1:pos + 1 + len(header)] != header:
        raise ValueError("unexpected token layout after the state (state_first layout without question prefix "
                         "or state repeat is required)")
    return pos + 1 + len(header)


def _field_header(tokenizer, number, kind):
    return _tokens(tokenizer, f"\nFIELD {number}\nID: field_{number}\nTYPE: {kind}\nINSTRUCTION: ")


def split_record(record, tokenizer):
    """One-question EncodedRecords, token-identical to `encode_joint_record([example])` for each field."""
    from tilda.clef_joint import FIELD_TYPES, EncodedQuestion
    kinds = {v: k for k, v in FIELD_TYPES.items()}
    if getattr(record, "points", ()) or getattr(record, "image_spans", ()) or getattr(record, "remark", None):
        raise ValueError("isolated questions: text Clef fields only (no point fields, images or remark)")
    ids = list(record.input_ids)
    p = prefix_end(ids, tokenizer)
    end_marker = ids.index(_marker(tokenizer, 5), record.questions[-1].question_span[1])
    trailer = ids[end_marker:]
    starts = []
    for i, q in enumerate(record.questions):
        header = _field_header(tokenizer, i + 1, kinds[q.question_type])
        start = q.question_span[0] - 1 - len(header)  # header, <|tilda_instruction|> (marker 3), instruction
        if ids[start:start + len(header)] != header or ids[q.question_span[0] - 1] != _marker(tokenizer, 3):
            raise ValueError(f"field {i + 1}: unexpected field layout")
        starts.append(start)
    starts.append(end_marker)
    singles = []
    for i, q in enumerate(record.questions):
        header = _field_header(tokenizer, 1, kinds[q.question_type])
        body = ids[q.question_span[0] - 1:starts[i + 1]]  # marker 3, instruction, options, END FIELD
        new_ids = ids[:p] + header + body + trailer
        shift = (p + len(header) + 1) - q.question_span[0]
        question = EncodedQuestion("field_1", q.question_type,
                                   (q.question_span[0] + shift, q.question_span[1] + shift),
                                   tuple((a + shift, b + shift) for a, b in q.option_spans), q.option_ids)
        extra = {}
        if getattr(record, "examples", None):
            extra["examples"] = (record.examples[i],)
        if getattr(record, "row_indices", None):
            extra["row_indices"] = (record.row_indices[i],)
        singles.append((replace(record, input_ids=tuple(new_ids), questions=(question,), **extra), p))
    return singles


def _install():
    if _INSTALLED["done"]:
        return
    import torch
    from tilda.ling import attention
    original = attention.bidirectional_attention

    def split_attention(query, key, value, padding, scaling, dropout, active):
        p = _SPLIT["prefix"]
        if p is None:
            return original(query, key, value, padding, scaling, dropout, active)
        if query.shape[0] != 1 or p >= query.shape[2]:
            raise ValueError("isolated split: one row at a time, with tokens after the prefix")
        head = attention._noncausal(query[:, :, :p], key[:, :, :p], value[:, :, :p], scaling, dropout, active)
        tail = attention._noncausal(query[:, :, p:], key, value, scaling, dropout, active)
        return torch.cat([head, tail], dim=2)

    attention.bidirectional_attention = split_attention
    _INSTALLED["done"] = True


@contextlib.contextmanager
def mla_split(prefix):
    """Bidirectional MLA split at `prefix` for the forwards inside (one request at a time)."""
    _install()
    _SPLIT["prefix"] = prefix
    try:
        yield
    finally:
        _SPLIT["prefix"] = None


def isolated_forward(model, records, pad_token_id, tokenizer):
    """Per record, per question: the isolated logits (a list of tensors per record, like forward_joint)."""
    from tilda.clef_joint import forward_joint
    from tilda.ling import attention
    bidirectional = attention.attention_mode(model.base_backbone()) == "bidirectional"
    out = []
    for record in records:
        logits = []
        for single, p in split_record(record, tokenizer):
            with mla_split(p) if bidirectional else contextlib.nullcontext():
                logits.append(forward_joint(model, [single], pad_token_id)[0][0])
        out.append(logits)
    return out
