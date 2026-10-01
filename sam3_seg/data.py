"""Read access to the Phase 0 COCO dataset (``prepare_dataset.py`` output).

:class:`CocoData` resolves the annotation files of the standard splits and of
the K folds, returns the ISIC IDs they contain and verifies every file
against the SHA-256 recorded in ``meta.json`` — a training run never starts on
a modified or half-written dataset.

Layout::

    <data_dir>/images/                     shared image pool
    <data_dir>/annotations/{train,val,test}.json
    <data_dir>/folds/fold_<k>/{train,val}.json
    <data_dir>/meta.json                   fingerprints + fold summary
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from common import read_json, sha256_file


def ids_fingerprint(ids: list[str]) -> str:
    """SHA-256 of ``ids`` joined by newlines (same as the U-Net / Phase 0 manifests)."""
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


class CocoData:
    """The Phase 0 dataset rooted at ``data_dir`` (verified on construction)."""

    def __init__(self, data_dir: str | Path) -> None:
        self.root = Path(data_dir)
        self.meta: dict[str, Any] | None = read_json(self.root / "meta.json")
        if self.meta is None:
            raise FileNotFoundError(f"{self.root}/meta.json not found — run Phase 0 (prepare_dataset.py) first")
        self._verified: set[str] = set()

    @property
    def image_dir(self) -> Path:
        return self.root / "images"

    @property
    def k_folds(self) -> int:
        return int(self.meta["params"]["k_folds"])

    def split_json(self, split: str) -> Path:
        """Annotation file of ``train`` / ``val`` / ``test`` (verified)."""
        return self._verify(f"annotations/{split}.json")

    def fold_json(self, fold: int, part: str) -> Path:
        """Annotation file of fold ``fold`` (``part`` = ``train`` / ``val``, verified)."""
        return self._verify(f"folds/fold_{fold}/{part}.json")

    def ids(self, json_path: Path) -> list[str]:
        """ISIC IDs of the images in an annotation file, in file order."""
        return list(_ids(str(json_path)))

    def fingerprint(self) -> str:
        """Fingerprint of the whole dataset (the per-file hashes of ``meta.json``)."""
        return hashlib.sha256(json.dumps(self.meta["files_sha256"], sort_keys=True).encode()).hexdigest()[:16]

    def _verify(self, rel: str) -> Path:
        path = self.root / rel
        if rel not in self._verified:
            expected = self.meta["files_sha256"].get(rel)
            if expected is None:
                raise FileNotFoundError(f"{rel} is not part of the Phase 0 dataset {self.root}")
            if sha256_file(path) != expected:
                raise RuntimeError(f"{path} does not match meta.json — the dataset was modified; re-run Phase 0")
            self._verified.add(rel)
        return path


@lru_cache(maxsize=None)
def _ids(json_path: str) -> tuple[str, ...]:
    data = json.loads(Path(json_path).read_text())
    return tuple(im["isic_id"] for im in data["images"])
