"""The choices an answer rests on, ADA's default for each, and how a caller picks another.

Some steps of an answer can go more than one way: whether a World row counts
with the countries, whether a total is read from the file's own total row.
ADA takes each the same way every time (the default) and says why, so a run
with no one to ask gives the same answer twice. A caller who knows the goal
(a person, or a coding agent reasoning about the person's goal) passes another
option by the decision's key, and the answer records that the caller chose it.

Every decision an answer took is in its ``decisions``, with the options, the
default, the option used and the reason for the default. A decision that did
not arise (a table with no World row) is not listed, and passing a choice for
it changes nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

#: Each decision ADA can take, with its options; the first option is the default.
OPTIONS: dict[str, tuple[str, ...]] = {
    "aggregate_rows": ("leave_out", "keep"),
    "total_from": ("total_row", "sum_of_parts"),
    "join_links": ("sure", "all", "none"),
}


class UnknownDecision(ValueError):
    """A choice names a decision ADA does not take, or an option the decision does not have."""


@dataclass(frozen=True)
class Decision:
    """One choice an answer rests on: the question, its options, the default and the option used."""

    key: str
    question: str
    options: tuple[str, ...]
    default: str
    chosen: str
    reason: str

    @property
    def by_caller(self) -> bool:
        """Whether the caller chose the option, rather than ADA taking its default."""
        return self.chosen != self.default

    def to_dict(self) -> dict:
        return {
            "key": self.key, "question": self.question, "options": list(self.options),
            "default": self.default, "chosen": self.chosen,
            "decided_by": "caller" if self.by_caller else "default", "reason": self.reason,
        }

    def describe(self) -> str:
        """The decision in one line: "Leave out ...? leave_out (the default; also: keep)"."""
        others = [option for option in self.options if option != self.chosen]
        by = "chosen by the caller" if self.by_caller else "the default"
        return f"{self.question} {self.chosen} ({by}; also: {', '.join(others)})"


def checked_choices(choices: Mapping[str, str] | None) -> dict[str, str]:
    """``choices`` with every key and option checked against ``OPTIONS``; raises ``UnknownDecision``."""
    checked: dict[str, str] = {}
    for key, option in (choices or {}).items():
        if key not in OPTIONS:
            raise UnknownDecision(f"ADA takes no decision called {key!r}; it takes {', '.join(OPTIONS)}.")
        if option not in OPTIONS[key]:
            raise UnknownDecision(f"{key} is one of {', '.join(OPTIONS[key])}, not {option!r}.")
        checked[key] = option
    return checked


def parse_choices(items: Iterable[str]) -> dict[str, str]:
    """Choices written as ``key=option``, as the command line takes them, checked against ``OPTIONS``."""
    choices: dict[str, str] = {}
    for item in items:
        key, equals, option = item.partition("=")
        if not equals or not key.strip() or not option.strip():
            raise UnknownDecision(f"Write a choice as key=option, such as aggregate_rows=keep, not {item!r}.")
        choices[key.strip()] = option.strip()
    return checked_choices(choices)


def decide(key: str, choices: Mapping[str, str], question: str, reason: str) -> Decision:
    """The decision ``key``, taken as ``choices`` say, or by its default."""
    options = OPTIONS[key]
    return Decision(key, question, options, options[0], choices.get(key, options[0]), reason)


__all__ = ["OPTIONS", "Decision", "UnknownDecision", "checked_choices", "decide", "parse_choices"]
