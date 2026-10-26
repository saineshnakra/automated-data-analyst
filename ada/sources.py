"""Where a number comes from: the file, its checksum and the table read from it.

Every answer and claim check names its :class:`Source`, so a result can be
traced to the exact bytes it was computed from. Nothing here records a time or
a temporary path, so the same file gives the same record on every run.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    """The data an answer was computed from.

    ``path`` is as given (absolute only if it was given so), ``sha256`` is of
    the file's bytes, and ``rows`` and ``columns`` are of the table read from
    it before any row limit. A DataFrame passed in directly has no path or
    checksum: there are no bytes to point back to.
    """

    name: str
    path: str | None = None
    sha256: str | None = None
    rows: int | None = None
    columns: int | None = None
    sheet: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "sha256": self.sha256,
            "rows": self.rows,
            "columns": self.columns,
            "sheet": self.sheet,
        }


def sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def describe(name: str, *, path: str | Path | None = None, contents: bytes | None = None,
             rows: int | None = None, columns: int | None = None, sheet: str | None = None) -> Source:
    """A :class:`Source` for a table read from ``contents`` (the file's bytes) at ``path``."""
    return Source(
        name=name,
        path=None if path is None else str(path),
        sha256=None if contents is None else sha256_bytes(contents),
        rows=rows,
        columns=columns,
        sheet=sheet,
    )
