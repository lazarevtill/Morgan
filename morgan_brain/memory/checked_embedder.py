"""The embedder that proves, on a process's first call, that the model answering is the one that
wrote the stored vectors.

Two embedding models of the same width are indistinguishable by width: swapping one in leaves
every stored vector searched by a model that never wrote it -- wrong answers, no error. The
active embedding space (``store/spaces.py``) records the vectors of five fixed strings
(``fingerprint.py``); this wrapper appends those strings to the first request the process
sends, splits them off the answer, and refuses by name when they fall outside the measured
tolerance. No extra round trip: the check rides on a request that was being sent anyway, so a
cold embedding host loads its model once, not twice.

A space whose fingerprint is not recorded yet is not trusted for going first. The same request
carries a sample of stored memories' texts, and the fingerprint is recorded only when their
fresh vectors match the stored ones; a space with no stored rows records it directly, since
there is nothing yet to be wrong about.

Once per process, keyed by (endpoint, model, dims, space id) at module level: ``morgan-mcp``
builds a context per tool call and checks once per server lifetime; the CLI once per command.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime

import structlog

from morgan_brain.config import Settings
from morgan_brain.memory import fingerprint
from morgan_brain.memory.embedder import Embedder
from morgan_brain.memory.store import spaces, vectors
from morgan_brain.providers.wire import EmbeddingSpaceMismatch

log = structlog.get_logger("embedding-space")

#: ``sample(n)``: up to *n* stored memories, each as (its text, the vector stored for it).
Sample = Callable[[int], list[tuple[str, list[float]]]]

#: The spaces this process has proven, as (endpoint, model, dims, space id). A key is added
#: only once its check passed: a refused check stays unproven, so every later call is checked
#: and refused again rather than handed a vector from the wrong model.
_checked: set[tuple[str, str, int, int]] = set()

#: The (endpoint, model) pairs this process has already reported as having no active space.
_unregistered: set[tuple[str, str]] = set()


def _utcnow() -> datetime:
    return datetime.now(UTC)


class CheckedEmbedder:
    """An ``Embedder`` that checks the active embedding space on the process's first call.

    *endpoint* is where *inner* sends its requests and *setting* the variable that addresses
    it; ``providers/factory.py::build_embedder``, which decided both, passes them in, and every
    refusal names that setting. *sample* is where the stored rows come from when the
    fingerprint is not recorded yet; ``None`` reads them from the database
    (``vectors.stored_sample`` over the active space's table).
    """

    def __init__(
        self,
        inner: Embedder,
        *,
        conn: sqlite3.Connection,
        settings: Settings,
        endpoint: str,
        setting: str,
        sample: Sample | None = None,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._inner = inner
        self._conn = conn
        self._url = endpoint
        self._setting = setting
        self._model = settings.embedding_model
        self._tolerance = settings.embedding_fingerprint_tolerance
        self._sample_rows = settings.embedding_fingerprint_sample_rows
        self._sample = sample
        self._clock = clock

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        space = spaces.active(self._conn)
        if space is None:
            if (self._url, self._model) in _unregistered:
                return await self._inner.embed_batch(texts)
        elif (self._url, self._model, space.dims, space.id) in _checked:
            return await self._inner.embed_batch(texts)
        return await self._first_call(space, texts)

    async def _first_call(
        self, space: spaces.EmbeddingSpace | None, texts: list[str]
    ) -> list[list[float]]:
        """Send *texts* with the five strings (and, for an unrecorded fingerprint, the stored
        sample) appended; check what came back; return the caller's own vectors."""
        rows = self._sample_of(space) if space is not None and space.fingerprint is None else []
        inputs = [*texts, *(text for text, _ in rows), *fingerprint.STRINGS]
        answered = await self._inner.embed_batch(inputs)
        if len(answered) != len(inputs):
            raise ValueError(
                f"the embedding model at {self._setting} answered {len(answered)} "
                f"vectors for {len(inputs)} inputs"
            )
        own = answered[: len(texts)]
        fresh_rows = answered[len(texts) : len(texts) + len(rows)]
        fresh_strings = answered[len(texts) + len(rows) :]

        if space is None:
            # Opening a writable database registers its space; one waiting for `morgan
            # migrate` has none until it runs. It is read-only until then, and recall must
            # keep working on it, unchecked as before.
            log.warning(
                "embedding-space.none-registered",
                endpoint=self._url,
                model=self._model,
                hint="run `morgan migrate`; embeddings are unchecked until a space is registered",
            )
            _unregistered.add((self._url, self._model))
            return own

        self._require_answers(space, answered)
        if space.fingerprint is not None:
            self._require_fingerprint(space, space.fingerprint, fresh_strings)
        else:
            self._record(space, fresh_strings, rows, fresh_rows)
        _checked.add((self._url, self._model, space.dims, space.id))
        return own

    def _sample_of(self, space: spaces.EmbeddingSpace) -> list[tuple[str, list[float]]]:
        if self._sample is None:
            return vectors.stored_sample(
                self._conn, table_name=space.table_name, n=self._sample_rows
            )
        return self._sample(self._sample_rows)[: self._sample_rows]

    def _require_answers(self, space: spaces.EmbeddingSpace, answered: list[list[float]]) -> None:
        """Every vector the model returned is the space's width and wholly finite.

        Checked before any cosine: a NaN component makes every cosine NaN, which no threshold
        can fail, and it would otherwise be handed to the caller or recorded as the space's
        fingerprint, after which every model would match it.
        """
        for vector in answered:
            if len(vector) != space.dims:
                raise EmbeddingSpaceMismatch.width(
                    space_id=space.id,
                    model=space.model,
                    dims=space.dims,
                    setting=self._setting,
                    got=len(vector),
                )
            for component in vector:
                if not math.isfinite(component):
                    raise EmbeddingSpaceMismatch.non_finite(
                        space_id=space.id,
                        model=space.model,
                        dims=space.dims,
                        setting=self._setting,
                        value=component,
                    )

    def _require_fingerprint(
        self, space: spaces.EmbeddingSpace, recorded: bytes, fresh_strings: list[list[float]]
    ) -> None:
        stored = spaces.unpack(recorded, dims=space.dims)
        comparison = fingerprint.compare(fresh_strings, stored)
        failed = len(comparison.per_string) - self._passing(comparison.per_string)
        if failed:
            raise EmbeddingSpaceMismatch.cosines(
                space_id=space.id,
                model=space.model,
                dims=space.dims,
                setting=self._setting,
                against="fingerprint",
                min_cosine=comparison.min_cosine,
                tolerance=self._tolerance,
                failed=failed,
                compared=len(comparison.per_string),
            )

    def _passing(self, cosines: list[float]) -> int:
        """How many reach the tolerance. Counted as passes, never as `c < tolerance` failures,
        so a cosine that is not a number counts as a failure."""
        return sum(1 for c in cosines if c >= self._tolerance)

    def _record(
        self,
        space: spaces.EmbeddingSpace,
        fresh_strings: list[list[float]],
        rows: list[tuple[str, list[float]]],
        fresh_rows: list[list[float]],
    ) -> None:
        """Record the fingerprint, but only if the stored sample says this model wrote them.

        With no sample, the fingerprint is recorded only when the space stores no vector at
        all: there is then nothing to be wrong about. A space that stores vectors none of which
        could be sampled is refused by name, and nothing is recorded.

        Runs after the embedding request returned: ``record_fingerprint`` takes its own short
        write transaction, and no lock is held across the await.
        """
        if not rows and vectors.holds_vectors(self._conn, table_name=space.table_name):
            raise EmbeddingSpaceMismatch.unverified(
                space_id=space.id,
                model=space.model,
                dims=space.dims,
                setting=self._setting,
            )
        cosines = [
            fingerprint.cosine(fresh, stored)
            for fresh, (_, stored) in zip(fresh_rows, rows, strict=True)
        ]
        min_cosine = min(cosines) if cosines else None
        failed = len(cosines) - self._passing(cosines)
        if failed:
            raise EmbeddingSpaceMismatch.cosines(
                space_id=space.id,
                model=space.model,
                dims=space.dims,
                setting=self._setting,
                against="stored-row",
                min_cosine=min(cosines),
                tolerance=self._tolerance,
                failed=failed,
                compared=len(cosines),
            )
        spaces.record_fingerprint(self._conn, space.id, fresh_strings, clock=self._clock)
        log.info(
            "embedding-space.fingerprint-recorded",
            space_id=space.id,
            model=space.model,
            min_cosine=min_cosine,
            sampled_rows=len(rows),
        )
