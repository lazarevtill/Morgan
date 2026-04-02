from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path


_MAX_MESSAGES = 100
_META_FILENAME = "_meta.json"


class SessionStore:
    """Persist conversation histories to disk as JSON files."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or Path.home() / ".morgan" / "sessions"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._key_map: dict[str, str] = {}  # hash -> conv_id
        self._load_meta()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, conv_id: str) -> list[dict]:
        """Return the message history for *conv_id*, or [] on any error."""
        path = self._key_to_path(conv_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        return []

    def save(self, conv_id: str, history: list[dict]) -> None:
        """Atomically persist *history* (capped to last 100 messages)."""
        trimmed = history[-_MAX_MESSAGES:]
        path = self._key_to_path(conv_id)
        tmp_path = path.with_suffix(".tmp")
        payload = json.dumps(trimmed, ensure_ascii=False)

        with self._lock:
            tmp_path.write_text(payload, encoding="utf-8")
            os.replace(tmp_path, path)
            h = self._hash(conv_id)
            if self._key_map.get(h) != conv_id:
                self._key_map[h] = conv_id
                self._save_meta()

    def load_all(self) -> dict[str, list[dict]]:
        """Scan the sessions directory and return every stored conversation."""
        self._load_meta()
        result: dict[str, list[dict]] = {}
        for hash_prefix, conv_id in self._key_map.items():
            path = self._base_dir / f"{hash_prefix}.json"
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    result[conv_id] = data
            except (OSError, json.JSONDecodeError, ValueError):
                continue
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(conv_id: str) -> str:
        return hashlib.sha256(conv_id.encode()).hexdigest()[:16]

    def _key_to_path(self, conv_id: str) -> Path:
        return self._base_dir / f"{self._hash(conv_id)}.json"

    def _meta_path(self) -> Path:
        return self._base_dir / _META_FILENAME

    def _load_meta(self) -> None:
        path = self._meta_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._key_map = data
        except (OSError, json.JSONDecodeError, ValueError):
            self._key_map = {}

    def _save_meta(self) -> None:
        path = self._meta_path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._key_map, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
