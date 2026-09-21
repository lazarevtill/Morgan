"""The fingerprint, as arithmetic: five fixed strings and what it means for two embeddings of
them to be "the same space".

Morgan checks only a vector's width today: two models of the same width are indistinguishable
to ``store/vectors.py``, so swapping one in leaves every stored vector searched by a model that
never wrote it -- wrong answers, no error. Embedding these five strings once, storing the
result, and re-embedding them on a process's first embedding call is how a same-width model
swap is caught instead.

This module is pure: no I/O, no settings, no logging. It knows nothing of the database or the
embedding endpoint; ``store/spaces.py`` persists what this module computes, and
``checked_embedder.py`` calls ``compare`` to decide whether a database's vectors still match
the model in front of it.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

#: Frozen forever once committed: changing a single character below invalidates every
#: fingerprint already recorded in an owner's database, because the whole point is that the
#: same input text always produces the same vectors from the same model. Five strings, in a
#: fixed order that never changes, chosen to span what Morgan actually stores: an English
#: sentence, a Russian sentence with a capitalised name (entity extraction is cased-script
#: only), a line of code, a line of dates and numbers, and a longer mixed-language paragraph.
STRINGS: tuple[str, ...] = (
    "The quarterly report is due on Friday, and the numbers still need a second review.",
    "Мария закончила проект раньше срока и отправила отчёт всей команде.",
    "def total(items: list[float]) -> float:\n    return sum(items)",
    "Meeting rescheduled: 2026-09-21 14:30 UTC, room 12B, budget delta -3.5%, quorum 7/10.",
    (
        "The migration ran overnight without incident, and by morning every index had "
        "rebuilt cleanly. Миграция прошла ночью без сбоев, и утром все индексы "
        "пересобрались корректно. The team agreed to watch the dashboards for a full day "
        "before declaring it done, since a quiet log is not proof of a correct one. Команда "
        "решила понаблюдать за графиками ещё сутки, прежде чем закрывать задачу, потому что "
        "тихий лог -- это не доказательство корректности."
    ),
)


def digest_of(strings: tuple[str, ...]) -> str:
    """sha256 of the strings joined with newlines, so ``doctor`` can print which set of
    strings a running Morgan was built against without printing the strings themselves."""
    return hashlib.sha256("\n".join(strings).encode("utf-8")).hexdigest()


#: Computed once at import time from ``STRINGS`` above, never from a literal, so the two can
#: never drift apart.
DIGEST: str = digest_of(STRINGS)


@dataclass(frozen=True)
class Comparison:
    """The result of comparing a fresh embedding of ``STRINGS`` against a stored one.

    ``min_cosine`` is the tightest bound: a model swap that only spoils one of the five
    strings must still be caught, so the comparison is only as good as its worst string.
    """

    min_cosine: float
    per_string: list[float]
    worst_index: int


def _require_same_width(a: list[float], b: list[float]) -> None:
    if len(a) != len(b):
        raise ValueError(f"vectors of different widths cannot be compared: {len(a)} vs {len(b)}")


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of *a* and *b*.

    Raises ``ValueError`` naming both widths on a mismatch, and ``ValueError`` mentioning
    "zero" on either vector being the zero vector -- a zero vector has no direction, so its
    cosine similarity is mathematically undefined (0/0), not a legitimate 0.0 or ``nan``. A
    fake or broken embedding server that returns all zeros must fail a comparison loudly
    rather than pass it silently. The same holds for a NaN or infinite component, reported
    as "non-finite": the cosine would come out NaN, and NaN compares false against every
    threshold, so a test for failure written as ``cosine < tolerance`` would pass it.
    """
    _require_same_width(a, b)
    if not all(math.isfinite(x) for x in a) or not all(math.isfinite(y) for y in b):
        raise ValueError("cannot compute cosine similarity of a vector with a non-finite component")
    dot: float = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a: float = sum(x * x for x in a) ** 0.5
    norm_b: float = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("cannot compute cosine similarity against a zero vector")
    return dot / (norm_a * norm_b)


def compare(fresh: list[list[float]], stored: list[list[float]]) -> Comparison:
    """Compare a fresh embedding of ``STRINGS`` against a stored fingerprint, string by
    string, in the order of ``STRINGS``. Raises ``ValueError`` naming both widths when a pair
    of vectors does not match, before any cosine is computed.
    """
    if len(fresh) != len(stored):
        raise ValueError(f"expected {len(stored)} vectors to compare against, got {len(fresh)}")
    per_string = [cosine(f, s) for f, s in zip(fresh, stored, strict=True)]
    worst_index = min(range(len(per_string)), key=lambda i: per_string[i])
    return Comparison(
        min_cosine=per_string[worst_index],
        per_string=per_string,
        worst_index=worst_index,
    )


def pack(vectors: list[list[float]]) -> bytes:
    """Little-endian float32, every vector's components concatenated in the order given.

    Byte-compatible with ``store/spaces.py``'s ``_pack``/``unpack``, which this module's
    ``unpack`` below replaces: same format character (``f``), same byte order (``<``).
    """
    flat: list[float] = [component for vector in vectors for component in vector]
    return struct.pack(f"<{len(flat)}f", *flat)


def unpack(blob: bytes, *, dims: int) -> list[list[float]]:
    """The inverse of ``pack``: *dims*-wide vectors are not recoverable from the blob alone,
    since a flat run of float32s carries no boundary between vectors."""
    count = len(blob) // 4
    flat = struct.unpack(f"<{count}f", blob)
    return [list(flat[i : i + dims]) for i in range(0, count, dims)]
