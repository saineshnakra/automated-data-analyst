"""The researcher's own tables and figures, made by their code and recorded like ADA's.

List a Python file under ``custom:`` in ``study.yml``. In it, mark functions
that take the processed data and return a pandas table or a Matplotlib figure::

    from ada import research

    @research.table("adjusted-model", title="Viability adjusted for passage number")
    def adjusted(data, study):
        ...
        return frame

    @research.figure("dose-curve", title="Dose response", caption="Points are wells.")
    def dose_curve(data, study):
        fig, ax = plt.subplots()
        ...
        return fig, source_frame    # or just fig

``ada build`` runs each function on the processed data (after every exclusion),
saves the result in the study's formats, and records the script's checksum, so
a change to the script is reported like any other change.
"""

from __future__ import annotations

import runpy
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from .study import StudyError, sha256_of
from .tables import Table, decimals_for, number


@dataclass(frozen=True)
class Registered:
    kind: str
    ident: str
    title: str
    function: Callable
    caption: str = ""
    notes: tuple[str, ...] = ()
    section: str = "main"
    width: str | float = "single"
    aspect: float = 0.75


_REGISTERED: list[Registered] = []


def table(ident: str, *, title: str, notes: tuple[str, ...] | list[str] = (), section: str = "main"):
    """Mark a function returning a DataFrame as one of the study's tables."""

    def register(function: Callable) -> Callable:
        _REGISTERED.append(Registered("table", ident, title, function, notes=tuple(notes), section=section))
        return function

    return register


def figure(
    ident: str,
    *,
    title: str,
    caption: str = "",
    section: str = "main",
    width: str | float = "single",
    aspect: float = 0.75,
):
    """Mark a function returning a Matplotlib figure, or (figure, source data), as one of the study's figures.

    The function can call ``research.figure_size(study, width, aspect)`` for the
    journal's size in inches; Matplotlib's style is already set when it runs.
    """

    def register(function: Callable) -> Callable:
        _REGISTERED.append(
            Registered(
                "figure", ident, title, function, caption=caption, section=section, width=width, aspect=aspect
            )
        )
        return function

    return register


def _formatted(frame: pd.DataFrame) -> pd.DataFrame:
    """Numbers formatted for print, each column to the decimals its spread needs."""
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            decimals = decimals_for(display[column])
            display[column] = display[column].map(lambda value, d=decimals: number(value, d))
    return display.astype(str)


def run(result) -> None:
    """Run every function registered by the scripts under ``custom:``."""
    from .figures import resolve_style, save, styled  # noqa: PLC0415

    study = result.study
    for entry in study.custom:
        path = (study.root / entry.script).resolve()
        if not path.is_file():
            raise StudyError([f"`custom` lists {entry.script}, which is not there."], study.path)
        result.custom[entry.script] = sha256_of(path)
        _REGISTERED.clear()
        runpy.run_path(str(path), run_name="ada_custom")
        found = list(_REGISTERED)
        _REGISTERED.clear()
        if not found:
            result.warnings.append(
                f"{entry.script} marks no functions with @research.table or @research.figure."
            )
        depends = ("data", "variables", "processing", "custom")
        code = f"{entry.script}:{{}}"
        for item in found:
            if item.kind == "table":
                frame = item.function(result.data.copy(), study)
                if not isinstance(frame, pd.DataFrame):
                    raise StudyError(
                        [
                            f"{entry.script}: {item.function.__name__} should return a pandas DataFrame, "
                            f"not {type(frame).__name__}."
                        ],
                        study.path,
                    )
                table_ = Table(
                    id=item.ident,
                    title=item.title,
                    display=_formatted(frame),
                    data=frame,
                    notes=list(item.notes),
                )
                result.add_table(
                    table_,
                    section=item.section,
                    depends=depends,
                    code=code.format(item.function.__name__),
                    calculation=f"Computed by {item.function.__name__} in {entry.script}.",
                )
            else:
                style = result.style or resolve_style(study)
                result.style = style
                with styled(style):
                    made = item.function(result.data.copy(), study)
                    figure_, source = made if isinstance(made, tuple) else (made, None)
                    if not hasattr(figure_, "savefig"):
                        raise StudyError(
                            [
                                f"{entry.script}: {item.function.__name__} should return a Matplotlib "
                                f"figure, not {type(figure_).__name__}."
                            ],
                            study.path,
                        )
                    save(
                        result,
                        figure_,
                        ident=item.ident,
                        title=item.title,
                        caption=item.caption,
                        source=source,
                        depends=depends,
                        code=code.format(item.function.__name__),
                        calculation=f"Drawn by {item.function.__name__} in {entry.script}.",
                        section=item.section,
                    )


def figure_size(study, width: str | float = "single", aspect: float = 0.75) -> tuple[float, float]:
    """The journal's figure size in inches, for custom figures."""
    from .figures import figure_size as size  # noqa: PLC0415
    from .figures import resolve_style

    return size(resolve_style(study), width, aspect)
