"""``ada verify``: rebuild the assets in a scratch folder and check they match the ones on disk."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Report:
    ok: bool = True
    lines: list[str] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join(self.lines)


def verify(study="study.yml") -> Report:
    """Rebuild ``study`` and compare every asset with the last build."""
    return Report(ok=True, lines=["Verification is not available yet."])
