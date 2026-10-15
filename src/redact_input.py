"""Input redaction: oracle redaction (C2/C4/C5/C6) and detector redaction (C7/C8).

redact_text / REDACT_TOKENS: moved unchanged from pii_lib. redact_turns: moved unchanged from the C7 runner.
detector_spans: moved unchanged from the detector evaluation (Presidio + spaCy PERSON); it needs
presidio_analyzer and spacy (see configs/mitigation_conditions.yaml, detector_C7) and is only called by make_detector().
"""
from __future__ import annotations

from functools import lru_cache

REDACT_TOKENS = {
    "ssn": "[REDACTED_SSN]",
    "phone": "[REDACTED_PHONE]",
    "card": "[REDACTED_CARD]",
    "email": "[REDACTED_EMAIL]",
    "address": "[REDACTED_ADDRESS]",
    "name": "[REDACTED_NAME]",
}


def redact_text(text: str, spans: list[dict]) -> str:
    """Replace each PII span's literal text in `text` with its redaction token.

    `spans` entries: {"type","text","token"}. Longest-first to avoid
    partial overlaps.
    """
    for s in sorted(spans, key=lambda x: -len(x["text"])):
        text = text.replace(s["text"], s["token"])
    return text


def redact_turns(turns: list[dict], det_spans: list[dict]) -> list[str]:
    """Replace detector-reported spans (joined-prompt coordinates) with the
    type token — the realistic production behavior: only what the detector
    saw is redacted. Spans are applied longest-first, non-overlapping,
    right-to-left within each turn so offsets stay valid.

    Returns the list of (possibly redacted) turn contents.
    """
    contents = [t["content"] for t in turns]
    offsets = []
    pos = 0
    for c in contents:
        offsets.append(pos)
        pos += len(c) + 1  # "\n" joiner

    # attribute each span to the turn containing its start
    by_turn: dict[int, list[dict]] = {}
    for s in det_spans:
        for i, (o, cl) in enumerate(zip(offsets, contents)):
            if o <= s["start"] < o + len(cl):
                by_turn.setdefault(i, []).append(s)
                break
    red = list(contents)
    for ti, ss in by_turn.items():
        # right-to-left so earlier replacements don't shift later offsets
        for s in sorted(ss, key=lambda x: -x["start"]):
            o = offsets[ti]
            a, b = s["start"] - o, s["end"] - o
            if not (0 <= a < b <= len(red[ti])):
                continue  # span crosses the turn boundary; skip (rare)
            red[ti] = red[ti][:a] + REDACT_TOKENS[s["type"]] + red[ti][b:]
    return red


def detector_spans(text: str, analyzer, nlp) -> list[dict]:
    """Return [{type, start, end, text}] from Presidio + spaCy PERSON."""
    out = []
    res = analyzer.analyze(text, language="en")
    for r in res:
        t = r.entity_type
        if t == "EMAIL_ADDRESS":
            out.append({"type": "email", "start": r.start, "end": r.end,
                        "text": text[r.start:r.end]})
        elif t == "PHONE_NUMBER":
            out.append({"type": "phone", "start": r.start, "end": r.end,
                        "text": text[r.start:r.end]})
        elif t == "US_SSN":
            out.append({"type": "ssn", "start": r.start, "end": r.end,
                        "text": text[r.start:r.end]})
        elif t == "CREDIT_CARD":
            out.append({"type": "card", "start": r.start, "end": r.end,
                        "text": text[r.start:r.end]})
        elif t == "LOCATION":
            out.append({"type": "address", "start": r.start, "end": r.end,
                        "text": text[r.start:r.end]})
    for ent in nlp(text).ents:
        if ent.label_ == "PERSON":
            out.append({"type": "name", "start": ent.start_char,
                        "end": ent.end_char, "text": ent.text})
    return out


def make_detector():
    """Return detect(text) -> spans using the real Presidio/spaCy stack (lazy import; CPU)."""
    from presidio_analyzer import AnalyzerEngine
    import spacy
    analyzer, nlp = AnalyzerEngine(), spacy.load("en_core_web_sm")

    @lru_cache(maxsize=8192)
    def detect(text):
        if not isinstance(text, str):
            raise ValueError("Detector accepts text only")
        return detector_spans(text, analyzer, nlp)
    return detect
