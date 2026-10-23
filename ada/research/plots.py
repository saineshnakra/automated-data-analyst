"""The figure types: groups, paired, longitudinal, scatter, missingness, diagnostics and panels.

Every figure shows each observation, not only a summary, with the estimates and
their confidence intervals drawn over it in ink. The numbers behind each mark go
to the figure's source CSV, and the caption is written from what was drawn: what
a point, a line and an error bar are, n per group, the rows left out, the
confidence level, the linked analysis and whether it was planned.

``draw_all`` draws the figures listed under ``figures:`` in ``study.yml``, or one
per analysis when that list is empty. Each type has a ``draw_<type>(target,
result, spec)`` function that draws on a Matplotlib figure, subfigure or axes and
returns a :class:`Drawn`, so a panel figure can place several types side by side.
"""

from __future__ import annotations

import math
import re
import textwrap
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .figures import (
    INK,
    MM_PER_INCH,
    MUTED,
    PALETTE,
    figure_size,
    panel_label,
    resolve_style,
    save,
    small_font_size,
    tint,
)
from .processing import SOURCE_ROW
from .study import AnalysisSpec, FigureSpec

#: Parts of study.yml that every figure is drawn from.
DATA_SECTIONS = ("data", "variables", "processing", "design")
#: The missingness figure adds its row-by-variable map up to this many rows.
MAP_ROWS = 2000
#: Marker shapes, used in turn once the six colours run out so no two levels look alike.
MARKERS = ("o", "s", "^", "D", "v", "P")
#: Marker area in pt² (a 4 pt dot), the fill's opacity, and the width of thin lines in pt.
POINT_AREA = 16.0
POINT_ALPHA = 0.6
THIN = 0.5
SUMMARY_COLUMNS = ["n", "mean", "sd", "se", "ci_low", "ci_high"]
NO_TEST = "No analysis is linked to this figure, so it shows the data without a test."
#: Source-data columns that hold counts, written as whole numbers.
COUNT_COLUMNS = (SOURCE_ROW, "n", "missing", "rows")
VAN_DER_CORPUT = (
    "van der Corput, J. G. (1935). Verteilungsfunktionen. Proceedings of the Koninklijke Akademie van "
    "Wetenschappen te Amsterdam 38, 813–821."
)
QQ_PLOT = (
    "Wilk, M. B. and Gnanadesikan, R. (1968). Probability plotting methods for the analysis of data. "
    "Biometrika 55, 1–17."
)


class FigureError(ValueError):
    """A figure that cannot be drawn from this study's data, with the reason in plain words."""


@dataclass(frozen=True)
class Plan:
    """One figure to draw: what it is, where it goes in the paper, and whether ``study.yml`` lists it."""

    spec: FigureSpec
    section: str
    listed: bool


@dataclass
class Drawn:
    """What a drawing function drew: the data behind each mark and the caption sentences describing them."""

    title: str
    sentences: list[str]
    source: pd.DataFrame
    calculation: str
    used: int
    depends: tuple[str, ...] = ()
    numbers: dict = field(default_factory=dict)
    references: tuple[str, ...] = ()


def draw_all(result) -> None:
    """Draw each figure listed in ``study.yml``, or a default set chosen from the analyses."""
    for plan in planned(result):
        try:
            draw_figure(result, plan)
        except FigureError as error:
            result.warnings.append(f"Figure {plan.spec.id} was not drawn: {error}")


def planned(result) -> list[Plan]:
    """The figures to draw, in order. A figure listed as a panel of another is drawn only inside it."""
    study = result.study
    if not study.figures:
        return default_figures(result)
    inside = {ident for figure in study.figures if figure.type == "panel" for ident in figure.panels}
    return [
        Plan(figure, "supplement" if figure.type in ("missingness", "diagnostics") else "main", True)
        for figure in study.figures
        if figure.id not in inside
    ]


def default_figures(result) -> list[Plan]:
    """One figure per analysis, chosen by its kind and the design, then missing values and diagnostics.

    Planned analyses go in the main section and exploratory ones in the supplement, so
    the paper's figures are the ones decided before the data were seen.
    """
    study, data = result.study, result.data
    plans = []
    for analysis in study.analyses:
        kind = default_type(result, analysis)
        if kind is None:
            result.warnings.append(
                f"No default figure for analysis {analysis.id}: its outcome {analysis.outcome!r} is not "
                "numeric."
            )
            continue
        spec = FigureSpec(id=f"{analysis.id}-{kind}", type=kind, analysis=analysis.id)
        plans.append(Plan(spec, "main" if analysis.planned else "supplement", False))
    if any(data[name].isna().any() for name in study.variables if name in data):
        plans.append(Plan(FigureSpec(id="missing-values", type="missingness"), "supplement", False))
    for analysis in study.analyses:
        if _residuals(result.results.get(analysis.id)) is not None:
            spec = FigureSpec(id=f"{analysis.id}-diagnostics", type="diagnostics", analysis=analysis.id)
            plans.append(Plan(spec, "supplement", False))
    return plans


def default_type(result, analysis: AnalysisSpec) -> str | None:
    """The figure that shows an analysis's data: by its kind, and for comparisons by the design.

    Conditions measured within each subject get the paired figure, so the reader sees
    each subject's own change; groups of different subjects get the groups figure.
    """
    study, data = result.study, result.data
    if analysis.outcome not in data or not _is_numeric(data[analysis.outcome]):
        return None
    if analysis.change:
        return "longitudinal"
    if analysis.relate:
        return "scatter" if analysis.relate in data and _is_numeric(data[analysis.relate]) else "groups"
    if study.design in ("paired", "repeated") and _within_subjects(data, study.subject, analysis.compare):
        return "paired"
    return "groups"


def draw_figure(result, plan: Plan):
    """Draw one figure at the journal's size and save it with its source data, caption and provenance."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    spec = plan.spec
    figure = plt.figure(figsize=_size(result, spec))
    try:
        drawn = DRAWERS[spec.type](figure, result, spec)
    except Exception:
        plt.close(figure)
        raise
    caption = " ".join(part for part in (spec.caption.strip(), *drawn.sentences) if part)
    depends = (
        f"figures.{spec.id}" if plan.listed else "figures",
        *drawn.depends,
        *DATA_SECTIONS,
        "decisions",
        "output",
        "sharing",
    )
    return save(
        result,
        figure,
        ident=spec.id,
        title=spec.title or drawn.title,
        caption=caption,
        source=drawn.source,
        depends=tuple(dict.fromkeys(depends)),
        code=f"ada.research.plots.draw_{spec.type}",
        calculation=drawn.calculation,
        section=plan.section,
        references=drawn.references,
        rows=result.rows(drawn.used),
        numbers=drawn.numbers,
    )


# The figure types.


def draw_groups(target, result, spec: FigureSpec) -> Drawn:
    """Every observation per group, spread sideways without overlap, with the mean and its CI over them."""
    study, style = result.study, _style(result)
    analysis, found = _linked(result, spec)
    y = spec.y or (analysis.outcome if analysis else _first(result, "outcome"))
    x = spec.x or (analysis.predictor if analysis and not analysis.change else _first(result, "group"))
    by = spec.by
    _require(result, y, "y", numeric=True)
    _require(result, x, "x")
    used, left_out = _complete(result, [x, y, by, *_model_columns(study, analysis)])
    order = _levels(result.data, x)
    levels = _present(used, x, order)
    if len(levels) > 20:
        raise FigureError(f"{x} has {len(levels)} values; a scatter figure suits a numeric x.")
    by_order = _levels(result.data, by) if by else []
    by_levels = _present(used, by, by_order) if by else [None]
    model = None
    if analysis and not by and x == analysis.predictor:
        model = _model_means(result, spec, found, [(level, None) for level in levels])
    confidence = study.decisions.confidence
    ids = _id_columns(study, used, exclude=(x, y, by))
    ax = _axes(target)[0]
    step = 0.8 / len(by_levels)
    half, shift, beside = 0.3 * step, -0.08 * step, 0.35 * step
    x_labels = _labels(used, x)
    by_labels = _labels(used, by) if by else None
    points, summaries, counts, numbers = [], [], [], {}
    for i, level in enumerate(levels):
        for j, sub in enumerate(by_levels):
            mask = (x_labels == level) & (by_labels == sub) if by else x_labels == level
            cell = used.loc[mask]
            if cell.empty:
                continue
            position = i + (j - (len(by_levels) - 1) / 2) * step
            values = cell[y].to_numpy(dtype=float)
            colour, marker = _look(by_order, sub) if by else _look(order, level)
            _points(ax, position + shift + spread(values, half), values, colour, marker, style)
            summary = model[(level, None)] if model else t_summary(values, confidence)
            _estimate(ax, position + beside, summary)
            key = {x: level, **({by: sub} if by else {})}
            points += [
                {"element": "point", **key, **_ids(row, ids), y: row[y], SOURCE_ROW: int(row[SOURCE_ROW])}
                for _, row in cell.iterrows()
            ]
            summaries.append({"element": "mean", **key, **summary})
            counts.append((f"{level}, {sub}" if by else level, len(cell)))
            numbers[f"mean {level}" + (f", {sub}" if by else "")] = summary["mean"]
    if by and len(by_levels) > 1:
        _legend(ax, [(_look(by_order, sub), sub) for sub in by_levels], style)
    _category_axis(ax, levels)
    ax.set_xlabel(study.label(x))
    ax.set_ylabel(study.label(y))
    sentences = [
        f"{_each_point(study)}; {_n_list(counts)}.",
        f"Black dots and bars beside the points show the mean and {_ci(study)} confidence interval of each "
        + ("group" if not by else "cell")
        + (f", as estimated in analysis {analysis.id}." if model else " (t distribution)."),
        _dependence(study, used),
        f"Colours mark {study.label(by)}." if by else "",
        left_out,
        _analysis_sentence(analysis, found) or NO_TEST,
    ]
    if model:
        calculation = (
            f"Means and {_ci(study)} confidence intervals from analysis {analysis.id} (its group means)."
        )
    else:
        calculation = (
            f"Means with {_ci(study)} t intervals per group (scipy.stats.t.interval, standard error from "
            "scipy.stats.sem) over the rows drawn."
        )
    calculation += (
        " Points are offset sideways by a van der Corput sequence scaled by a Gaussian kernel density "
        "(scipy.stats.gaussian_kde), so the same data always give the same positions."
    )
    columns = ["element", x, *([by] if by else []), *ids, SOURCE_ROW, y, *SUMMARY_COLUMNS]
    return Drawn(
        title=f"{_name(study, y)} by {_name(study, x)}" + (f" and {_name(study, by)}" if by else ""),
        sentences=_sentences(sentences),
        source=_source(study, points + summaries, columns),
        calculation=calculation,
        used=len(used),
        depends=_analysis_depends(analysis),
        numbers={"n": len(used), **numbers},
        references=(VAN_DER_CORPUT,),
    )


def draw_paired(target, result, spec: FigureSpec) -> Drawn:
    """One line per subject joining its conditions, the condition means, and each subject's difference."""
    study, style = result.study, _style(result)
    analysis, found = _linked(result, spec)
    subject = study.subject
    if not subject or subject not in result.data:
        raise FigureError("a paired figure needs `design.subject`, the column naming who was measured.")
    y = spec.y or (analysis.outcome if analysis else _first(result, "outcome"))
    x = spec.x or (analysis.predictor if analysis and not analysis.change else _first(result, "group"))
    _require(result, y, "y", numeric=True)
    _require(result, x, "x")
    used, left_out = _complete(result, [x, y, subject, *_model_columns(study, analysis)])
    order = _levels(result.data, x)
    levels = _present(used, x, order)
    if len(levels) < 2:
        raise FigureError(f"{x} has fewer than two conditions in the rows with values.")
    frame = pd.DataFrame(
        {"subject": _labels(used, subject), "condition": _labels(used, x), "value": used[y].astype(float)}
    )
    if frame.duplicated(["subject", "condition"]).any():
        raise FigureError(
            f"some subjects have more than one row for the same {x}; a paired figure needs one row per "
            "subject and condition."
        )
    wide = frame.pivot(index="subject", columns="condition", values="value").reindex(columns=levels)
    wide = wide.reindex(sorted(wide.index, key=_natural))
    complete = wide.dropna()
    partial = [name for name in wide.index if name not in complete.index]
    if complete.empty:
        raise FigureError(f"no subject has a value in every condition of {x}.")
    drawn_rows = used.loc[frame["subject"].isin(complete.index).to_numpy()]
    k = len(levels)
    with_differences = k == 2 and not isinstance(target, Axes)
    axes = _axes(target, 2 if with_differences else 1, width_ratios=[2.3, 1])
    ax = axes[0]
    from matplotlib.collections import LineCollection  # noqa: PLC0415

    segments = [np.column_stack([np.arange(k), complete.loc[name].to_numpy()]) for name in complete.index]
    ax.add_collection(LineCollection(segments, colors=[tint(MUTED, 0.55, style)], linewidths=THIN, zorder=1))
    confidence = study.decisions.confidence
    model = None
    if analysis and x == analysis.predictor:
        model = _model_means(result, spec, found, [(level, None) for level in levels])
    summaries, numbers = [], {"subjects": len(complete)}
    for i, level in enumerate(levels):
        values = complete[level].to_numpy()
        colour, marker = _look(order, level)
        _points(ax, np.full(len(values), float(i)), values, colour, marker, style)
        summary = model[(level, None)] if model else t_summary(values, confidence)
        _estimate(ax, i + ((-0.22 if i == 0 else 0.22) if k == 2 else 0.0), summary)
        summaries.append({"element": "mean", x: level, **summary})
        numbers[f"mean {level}"] = summary["mean"]
    _category_axis(ax, levels)
    ax.autoscale_view()
    ax.set_xlabel(study.label(x))
    ax.set_ylabel(study.label(y))
    ordered = drawn_rows.assign(_rank=_labels(drawn_rows, x).map(levels.index).to_numpy())
    points = [
        {
            "element": "point",
            subject: row[subject],
            x: _label(row[x]),
            y: row[y],
            SOURCE_ROW: int(row[SOURCE_ROW]),
        }
        for _, row in ordered.sort_values(["_rank", SOURCE_ROW], kind="stable").iterrows()
    ]
    first, last = levels[0], levels[-1]
    sentences = [
        f"Each point is one measurement and thin grey lines join the measurements of the same subject "
        f"({subject}); n = {_count(len(complete), 'subject')} with "
        + (f"both {first} and {last}." if k == 2 else f"all {k} conditions."),
        f"Black dots and bars show the mean and {_ci(study)} confidence interval of each condition"
        + (f", as estimated in analysis {analysis.id}." if model else " (t distribution)."),
    ]
    differences = []
    if with_differences:
        change = complete[last] - complete[first]
        values = change.to_numpy()
        difference = t_summary(values, confidence)
        side = axes[1]
        side.axhline(0, color=MUTED, linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
        _points(side, spread(values, 0.28), values, MUTED, "o", style)
        _estimate(side, 0.0, difference)
        side.set_xticks([0], [_wrap(f"{last} − {first}", 12)])
        side.set_xlim(-0.6, 0.6)
        unit = study.variable(y).unit
        side.set_ylabel("Difference" + (f" ({unit})" if unit else ""))
        differences = [
            {"element": "difference", subject: name, "difference": float(change[name])}
            for name in change.index
        ]
        differences.append({"element": "mean difference", **difference})
        numbers.update(
            {
                "mean difference": difference["mean"],
                "difference ci_low": difference["ci_low"],
                "difference ci_high": difference["ci_high"],
            }
        )
        sentences.append(
            f"Right: each subject's difference ({last} − {first}) with the mean difference and its "
            f"{_ci(study)} confidence interval (paired t interval); the dotted line marks no difference."
        )
    if partial:
        named = f" ({', '.join(partial)})" if len(partial) <= 5 else ""
        one = len(partial) == 1
        sentences.append(
            f"{len(partial)} {'subject' if one else 'subjects'} without a value in every condition "
            f"{'is' if one else 'are'} left out{named}."
        )
    sentences += [left_out, _analysis_sentence(analysis, found) or NO_TEST]
    if model:
        calculation = f"Condition means and {_ci(study)} confidence intervals from analysis {analysis.id}."
    else:
        calculation = (
            f"Condition means with {_ci(study)} t intervals (scipy.stats.t.interval, scipy.stats.sem) over "
            "the subjects with a value in every condition."
        )
    if with_differences:
        calculation += (
            " Differences per subject, with the paired t interval (the t interval of the differences)."
        )
    columns = ["element", subject, x, SOURCE_ROW, y, "difference", *SUMMARY_COLUMNS]
    return Drawn(
        title=f"{_name(study, y)} by {_name(study, x)}, paired within {subject}",
        sentences=_sentences(sentences),
        source=_source(study, points + summaries + differences, columns),
        calculation=calculation,
        used=len(drawn_rows),
        depends=_analysis_depends(analysis),
        numbers=numbers,
    )


def draw_longitudinal(target, result, spec: FigureSpec) -> Drawn:
    """A light line per subject over time, with the mean and its CI at each time point per group.

    Lines join only a subject's own measurements, so the repeated measures stay visible.
    """
    study, style = result.study, _style(result)
    analysis, found = _linked(result, spec)
    time = spec.x or (analysis.change if analysis and analysis.change else None) or study.time
    if not time:
        raise FigureError(
            "a longitudinal figure needs a time: `x`, the analysis's `change`, or `design.time`."
        )
    y = spec.y or (analysis.outcome if analysis else _first(result, "outcome"))
    by = spec.by or (analysis.by if analysis else None)
    _require(result, y, "y", numeric=True)
    _require(result, time, "x")
    subject = study.subject if study.subject and study.subject in result.data else None
    used, left_out = _complete(result, [time, y, by, subject, *_model_columns(study, analysis)])
    numeric_time = _is_numeric(result.data[time])
    if numeric_time:
        moments = sorted(set(used[time].astype(float)))
        position = {_label(moment): float(moment) for moment in moments}
        span = (moments[-1] - moments[0]) or 1.0
    else:
        moments = _present(used, time, _levels(result.data, time))
        position = {moment: float(index) for index, moment in enumerate(moments)}
        span = max(len(moments) - 1, 1)
    time_labels = _labels(used, time)
    xs = time_labels.map(position).to_numpy(dtype=float)
    group_order = _levels(result.data, by) if by else []
    groups = _present(used, by, group_order) if by else [None]
    group_labels = _labels(used, by) if by else pd.Series([None] * len(used), index=used.index)
    ax = _axes(target)[0]
    confidence = study.decisions.confidence

    def look(group):
        return _look(group_order, group) if by else (INK, "o")

    if subject:
        from matplotlib.collections import LineCollection  # noqa: PLC0415

        subjects = _labels(used, subject)
        segments, colours, singles = [], [], []
        for name in sorted(subjects.unique(), key=_natural):
            mask = (subjects == name).to_numpy()
            line_x, line_y = xs[mask], used[y].to_numpy(dtype=float)[mask]
            ordering = np.argsort(line_x, kind="stable")
            colour = look(group_labels[mask].iloc[0])[0] if by else MUTED
            if len(line_x) == 1:
                singles.append((line_x[0], line_y[0], colour))
                continue
            segments.append(np.column_stack([line_x[ordering], line_y[ordering]]))
            colours.append(tint(colour, 0.4, style))
        ax.add_collection(LineCollection(segments, colors=colours, linewidths=THIN, zorder=1))
        for single_x, single_y, colour in singles:
            ax.scatter([single_x], [single_y], s=9, color=[tint(colour, 0.5, style)], linewidths=0, zorder=1)
    else:
        for group in groups:
            mask = (group_labels == group).to_numpy() if by else np.ones(len(used), bool)
            for moment in moments:
                at = mask & (xs == position[_label(moment)])
                values = used[y].to_numpy(dtype=float)[at]
                colour, marker = look(group)
                _points(
                    ax,
                    xs[at] + spread(values, 0.18 * span / max(len(moments), 1)),
                    values,
                    colour,
                    marker,
                    style,
                )
    model = None
    if analysis and analysis.change == time and (by or None) == (analysis.by or None):
        model = _model_means(
            result, spec, found, [(group, _label(moment)) for group in groups for moment in moments]
        )
    delta = 0.022 * span if numeric_time else 0.09
    points, summaries, numbers, sizes = [], [], {}, []
    for j, group in enumerate(groups):
        offset = (j - (len(groups) - 1) / 2) * delta
        colour, marker = look(group)
        line_x, line_mean, line_low, line_high = [], [], [], []
        for moment in moments:
            key = _label(moment)
            mask = (time_labels == key) & ((group_labels == group) if by else True)
            values = used.loc[mask, y].to_numpy(dtype=float)
            if not len(values):
                continue
            summary = model[(group, key)] if model else t_summary(values, confidence)
            line_x.append(position[key] + offset)
            line_mean.append(summary["mean"])
            line_low.append(summary["ci_low"])
            line_high.append(summary["ci_high"])
            summaries.append({"element": "mean", **({by: group} if by else {}), time: key, **summary})
            sizes.append(len(values))
            numbers[f"mean {group + ', ' if by else ''}{time} {key}"] = summary["mean"]
        mean = np.array(line_mean)
        errors = np.array([mean - np.array(line_low), np.array(line_high) - mean])
        ax.plot(line_x, mean, color=colour, linewidth=1.0, zorder=3)
        ax.errorbar(
            line_x,
            mean,
            yerr=np.where(np.isfinite(errors), errors, 0),
            fmt=marker,
            color=INK,
            ecolor=INK,
            elinewidth=0.9,
            capsize=0,
            markersize=4.5,
            markerfacecolor=colour,
            markeredgecolor=INK if by else "white",
            markeredgewidth=0.5,
            zorder=4,
        )
    for _, row in used.iterrows():
        points.append(
            {
                "element": "point",
                **({subject: row[subject]} if subject else {}),
                **({by: _label(row[by])} if by else {}),
                time: _label(row[time]),
                y: row[y],
                SOURCE_ROW: int(row[SOURCE_ROW]),
            }
        )
    if numeric_time:
        if len(moments) <= 12:
            ax.set_xticks(moments, [_label(moment) for moment in moments])
        ax.set_xlim(moments[0] - 0.06 * span, moments[-1] + 0.06 * span)
    else:
        _category_axis(ax, moments)
    if by and len(groups) > 1:
        _legend(ax, [(look(group), group) for group in groups], style, lines=True)
    ax.set_xlabel(study.label(time))
    ax.set_ylabel(study.label(y))
    low, high = min(sizes), max(sizes)
    per_mean = f"n = {low:,} per mean" if low == high else f"n = {low:,} to {high:,} per mean"
    if subject:
        if by:
            per_group = used.assign(_g=group_labels.to_numpy()).groupby("_g", sort=False)[subject].nunique()
            split = " (" + ", ".join(f"{per_group[group]:,} {group}" for group in groups) + ")"
        else:
            split = ""
        first = (
            f"Thin lines join the measurements of each subject ({subject}); "
            f"{_count(used[subject].nunique(), 'subject')}{split}."
        )
    else:
        first = f"{_each_point(study)}."
    second = (
        ("Markers" if by else "Black dots")
        + f" and bars show the mean and {_ci(study)} confidence interval at each {_name(study, time).lower()}"
        + (
            f" for each {_name(study, by).lower()}, offset slightly so the groups do not overlap"
            if by and len(groups) > 1
            else ""
        )
        + f" ({per_mean}"
        + (f"; as estimated in analysis {analysis.id})." if model else "; t distribution).")
    )
    sentences = [first, second, left_out, _analysis_sentence(analysis, found) or NO_TEST]
    if model:
        calculation = f"Means and {_ci(study)} confidence intervals at each time from analysis {analysis.id}."
    else:
        calculation = (
            f"Means with {_ci(study)} t intervals (scipy.stats.t.interval, scipy.stats.sem) at each time"
            + (" and group" if by else "")
            + " over the rows drawn."
        )
    columns = ["element", *([subject] if subject else []), *([by] if by else []), time, SOURCE_ROW, y]
    return Drawn(
        title=f"{_name(study, y)} over {_name(study, time).lower()}"
        + (f" by {_name(study, by)}" if by else ""),
        sentences=_sentences(sentences),
        source=_source(study, points + summaries, columns + SUMMARY_COLUMNS),
        calculation=calculation,
        used=len(used),
        depends=_analysis_depends(analysis),
        numbers={"n": len(used), **numbers},
    )


def draw_scatter(target, result, spec: FigureSpec) -> Drawn:
    """Points with a fitted line and its confidence band for the mean.

    With a linked relate analysis the line is that model's prediction, covariates held at reference
    values; otherwise it is least squares over the points, with no band when they are not independent.
    """
    import statsmodels.api as sm  # noqa: PLC0415

    study, style = result.study, _style(result)
    analysis, found = _linked(result, spec)
    x = spec.x or (analysis.relate if analysis and analysis.relate else None)
    if not x:
        raise FigureError("a scatter figure needs `x`, a numeric variable, or a linked `relate` analysis.")
    y = spec.y or (analysis.outcome if analysis else _first(result, "outcome"))
    by = spec.by
    _require(result, x, "x", numeric=True)
    _require(result, y, "y", numeric=True)
    used, left_out = _complete(result, [x, y, by, *_model_columns(study, analysis)])
    if len(used) < 3:
        raise FigureError(f"only {len(used)} rows have values for both {x} and {y}.")
    xs, ys = used[x].to_numpy(dtype=float), used[y].to_numpy(dtype=float)
    if np.ptp(xs) == 0:
        raise FigureError(f"{x} has a single value in the rows drawn.")
    ax = _axes(target)[0]
    by_order = _levels(result.data, by) if by else []
    groups = _present(used, by, by_order) if by else [None]
    labels = _labels(used, by) if by else None
    counts = []
    for group in groups:
        mask = (labels == group).to_numpy() if by else np.ones(len(used), bool)
        colour, marker = _look(by_order, group) if by else (PALETTE[0], "o")
        _points(ax, xs[mask], ys[mask], colour, marker, style)
        counts.append((group, int(mask.sum())))
    if by and len(groups) > 1:
        _legend(ax, [(_look(by_order, group), group) for group in groups], style)
    confidence = study.decisions.confidence
    grid = np.linspace(xs.min(), xs.max(), 101)
    linked = _model_line(analysis, found, x, y)
    clustering = _clustering(study, used)
    mixed = study.design != "independent"
    held = {}
    if linked:
        # The linked model's own prediction, with its covariates held at the values the analysis gives.
        line, reported = linked
        held = line["held"]
        weights = np.column_stack([np.ones(len(grid)), grid])
        mean = line["intercept"] + line["slope"] * grid
        half = line["critical"] * np.sqrt(np.einsum("ij,jk,ik->i", weights, line["covariance"], weights))
        low, high = mean - half, mean + half
        offset = line["critical"] * math.sqrt(line["covariance"][0, 0])
        intercept = (line["intercept"], line["intercept"] - offset, line["intercept"] + offset)
        slope = (reported["estimate"], reported["ci_low"], reported["ci_high"])
        n, banded = found.n, True
    else:
        fit = sm.OLS(ys, sm.add_constant(xs)).fit()
        prediction = fit.get_prediction(sm.add_constant(grid, has_constant="add"))
        band = prediction.summary_frame(alpha=1 - confidence)
        mean, low, high = (band[name].to_numpy() for name in ("mean", "mean_ci_lower", "mean_ci_upper"))
        # conf_int() has one row per coefficient, (low, high): the intercept first, then the slope.
        (intercept_low, intercept_high), (slope_low, slope_high) = fit.conf_int(alpha=1 - confidence)
        intercept = (fit.params[0], intercept_low, intercept_high)
        slope = (fit.params[1], slope_low, slope_high)
        n, banded = len(used), not clustering
        if not banded:
            # Least squares treats every point as independent, so its intervals would be too narrow here.
            low = high = np.full(len(grid), math.nan)
            intercept, slope = (intercept[0], math.nan, math.nan), (slope[0], math.nan, math.nan)
    if banded:
        ax.fill_between(grid, low, high, color=tint(INK, 0.13, style), linewidth=0, zorder=1)
    ax.plot(grid, mean, color=INK, linewidth=1.0, zorder=3)
    ax.set_xlabel(study.label(x))
    ax.set_ylabel(study.label(y))
    ids = _id_columns(study, used, exclude=(x, y, by))
    points = [
        {
            "element": "point",
            **_ids(row, ids),
            **({by: _label(row[by])} if by else {}),
            x: row[x],
            y: row[y],
            SOURCE_ROW: int(row[SOURCE_ROW]),
        }
        for _, row in used.iterrows()
    ]
    fitted = [
        {"element": "fit", x: value, **held, "fit": fit_value, "ci_low": fit_low, "ci_high": fit_high}
        for value, fit_value, fit_low, fit_high in zip(grid, mean, low, high, strict=True)
    ]
    interval = ("estimate", "ci_low", "ci_high")
    estimates = [
        {"element": "intercept", **held, "n": n, **dict(zip(interval, intercept, strict=True))},
        {"element": "slope", "n": n, **dict(zip(interval, slope, strict=True))},
    ]
    sentences = [
        f"{_each_point(study)}; {_n_list(counts) if by else f'n = {len(used):,}'}.",
        f"Colours mark {study.label(by)}." if by else "",
    ]
    if linked:
        kind = "fixed-effect" if mixed else "least-squares"
        sentences.append(
            f"The line and shaded band show the {kind} fit of analysis {analysis.id} ({found.test}) and its "
            f"{_ci(study)} confidence interval for the mean"
            + (f", holding {_held(study, held)}." if held else ".")
        )
        if held:
            sentences.append(
                f"The points are the observed values, not adjusted for {', '.join(held)}, so the line need "
                "not follow them."
            )
        if mixed and clustering:
            sentences.append(
                f"Because {clustering}, the line and band come from the mixed model rather than from least "
                "squares over the points."
            )
    elif banded:
        sentences.append(
            f"The line and shaded band show the least-squares fit and its {_ci(study)} confidence interval "
            "for the mean (ordinary least squares, no covariates)."
        )
    else:
        accounted = analysis is not None and found is not None and getattr(found, "supported", False)
        sentences.append(
            "The line is the least-squares fit through the points (ordinary least squares, no covariates), "
            f"drawn only to describe them: {clustering}, so the points are not independent and least "
            "squares gives no valid confidence band."
            + (f" Analysis {analysis.id} accounts for this." if accounted else "")
        )
    if analysis and analysis.covariates and not linked:
        sentences.append(
            f"Analysis {analysis.id} also adjusts for {', '.join(analysis.covariates)}; the line does not."
        )
    sentences += [left_out, _analysis_sentence(analysis, found) or NO_TEST]
    if linked:
        calculation = (
            f"Line and band from analysis {analysis.id} ({found.method}): its prediction across {x}"
            + (f", holding {_held(study, held)}" if held else "")
            + f", with the {_ci(study)} confidence interval for that prediction from the covariance of the "
            + ("model's fixed-effect coefficients." if mixed else "model's coefficients.")
        )
    elif banded:
        calculation = (
            f"Ordinary least squares of {y} on {x} (statsmodels OLS); the band is the {_ci(study)} "
            "confidence interval for the mean from get_prediction().summary_frame()."
        )
    else:
        calculation = (
            f"Ordinary least squares of {y} on {x} (statsmodels OLS), drawn without a band because the "
            "points are not independent."
        )
    columns = [
        "element",
        *ids,
        *([by] if by else []),
        SOURCE_ROW,
        x,
        y,
        *held,
        "fit",
        "estimate",
        "n",
        "ci_low",
        "ci_high",
    ]
    return Drawn(
        title=f"{_name(study, y)} against {_name(study, x).lower()}",
        sentences=_sentences(sentences),
        source=_source(study, points + fitted + estimates, columns),
        calculation=calculation,
        used=len(used),
        depends=_analysis_depends(analysis),
        numbers={
            "n": len(used),
            "slope": slope[0],
            **({"slope ci_low": slope[1], "slope ci_high": slope[2]} if banded else {}),
        },
    )


def draw_missingness(target, result, spec: FigureSpec) -> Drawn:
    """The share of missing values per variable, and a map of where they are when the data are small."""
    study, style, data = result.study, _style(result), result.data
    names = [name for name in study.variables if name in data]
    if not names:
        raise FigureError("none of the described variables is in the data.")
    total = len(data)
    counts = data[names].isna().sum()
    names = sorted(names, key=lambda name: (-int(counts[name]), names.index(name)))
    with_map = 0 < total <= MAP_ROWS and counts.sum() > 0 and not isinstance(target, Axes)
    axes = _axes(target, 2 if with_map else 1, sharey=True, width_ratios=[1, 1.5])
    ax = axes[0]
    positions = np.arange(len(names))[::-1]
    shares = np.array([100 * counts[name] / total if total else 0.0 for name in names])
    ax.barh(positions, shares, height=0.62, color=PALETTE[0], linewidth=0, zorder=2)
    reach = max(shares.max() * 1.45, 5.0)
    small = small_font_size(style)
    for place, share, name in zip(positions, shares, names, strict=True):
        ax.text(share + reach * 0.03, place, f"{int(counts[name]):,}", va="center", ha="left", fontsize=small)
    ax.set_xlim(0, reach)
    ax.set_ylim(-0.5, len(names) - 0.5)
    ax.set_yticks(positions, [_wrap(study.label(name), 22) for name in names])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Missing (% of rows)")
    source = [
        {
            "element": "variable",
            "variable": name,
            "missing": int(counts[name]),
            "rows": total,
            "share": counts[name] / total if total else 0.0,
        }
        for name in names
    ]
    sentences = [
        f"Bars show the share of the {total:,} rows with no value for each variable; labels give the count."
    ]
    codes = list(study.missing_codes) + [
        code for name in names for code in study.variable(name).missing_codes
    ]
    if codes:
        sentences.append(
            f"Missing-value codes ({', '.join(map(str, dict.fromkeys(codes)))}) count as missing."
        )
    if study.steps:
        sentences.append("Counted after the processing steps in the processing record.")
    if with_map:
        from matplotlib.colors import ListedColormap  # noqa: PLC0415

        ordered = data.sort_values(SOURCE_ROW, kind="stable")
        matrix = ordered[names].isna().to_numpy().T.astype(float)
        side = axes[1]
        side.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=ListedColormap([_blend(MUTED, 0.16), INK]),
            vmin=0,
            vmax=1,
            extent=(0.5, total + 0.5, -0.5, len(names) - 0.5),
            origin="upper",
        )
        side.tick_params(axis="y", left=False)
        for spine in ("left", "top", "right"):
            side.spines[spine].set_visible(False)
        side.set_xlabel("Row, in file order")
        sentences.append("Right: each row (in file order) against each variable; dark cells are missing.")
        for column, name in enumerate(names):
            for place in np.flatnonzero(matrix[column]):
                source.append(
                    {
                        "element": "missing cell",
                        "variable": name,
                        SOURCE_ROW: int(ordered[SOURCE_ROW].iloc[place]),
                    }
                )
    return Drawn(
        title="Missing values",
        sentences=_sentences(sentences),
        source=_source(study, source, ["element", "variable", SOURCE_ROW, "missing", "rows", "share"]),
        calculation="Missing values per variable in the processed data (pandas isna), after missing-value "
        "codes and processing steps.",
        used=total,
        numbers={f"missing {name}": int(counts[name]) for name in names},
    )


def draw_diagnostics(target, result, spec: FigureSpec) -> Drawn:
    """Residuals against fitted values, and a normal Q–Q plot of the residuals, side by side."""
    from scipy import stats  # noqa: PLC0415

    study, style = result.study, _style(result)
    analysis, found = _linked(result, spec)
    residuals = _residuals(found)
    if analysis is None or residuals is None:
        raise FigureError("a diagnostics figure needs a linked `analysis` with residuals.")
    residuals = residuals.dropna()
    fitted = found.fitted.reindex(residuals.index) if found.fitted is not None else None
    with_fitted = fitted is not None and fitted.notna().all()
    count = 2 if with_fitted and not isinstance(target, Axes) else 1
    axes = _axes(target, count)
    values = residuals.to_numpy(dtype=float)
    (theoretical, ordered), (slope, intercept, _) = stats.probplot(values, dist="norm")
    quantiles = np.empty(len(values))
    quantiles[np.argsort(values, kind="stable")] = theoretical
    unit = study.variable(analysis.outcome).unit
    if count == 2 or with_fitted:
        ax = axes[0]
        ax.axhline(0, color=MUTED, linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
        _points(ax, fitted.to_numpy(dtype=float), values, PALETTE[0], "o", style)
        ax.set_xlabel("Fitted value" + (f" ({unit})" if unit else ""))
        ax.set_ylabel("Residual" + (f" ({unit})" if unit else ""))
    if count == 2 or not with_fitted:
        ax = axes[-1]
        ends = np.array([theoretical[0], theoretical[-1]])
        ax.plot(ends, intercept + slope * ends, color=INK, linewidth=0.8, zorder=3)
        _points(ax, theoretical, ordered, PALETTE[0], "o", style)
        ax.set_xlabel("Normal quantile")
        ax.set_ylabel("Ordered residual" + (f" ({unit})" if unit else ""))
    rows = _source_rows(result, residuals.index)
    source = [
        {
            "element": "residual",
            SOURCE_ROW: rows[i],
            "fitted": float(fitted.iloc[i]) if with_fitted else None,
            "residual": values[i],
            "normal_quantile": quantiles[i],
        }
        for i in range(len(values))
    ]
    test = f" ({found.test})" if found.test else ""
    parts = []
    if count == 2:
        parts.append(
            f"Left: residuals of analysis {analysis.id}{test} against its fitted values; the dotted line "
            "marks zero. Right: normal Q–Q plot of the residuals (scipy.stats.probplot) with its "
            "least-squares line"
        )
    elif with_fitted:
        parts.append(f"Residuals of analysis {analysis.id}{test} against its fitted values")
    else:
        parts.append(
            f"Normal Q–Q plot of the residuals of analysis {analysis.id}{test} (scipy.stats.probplot)"
        )
    sentences = [
        "; ".join(parts) + f"; n = {len(values):,}.",
        "If the model's assumptions hold, the residuals spread evenly around zero at every fitted value and "
        "the points lie close to the line.",
        _analysis_sentence(analysis, found),
    ]
    return Drawn(
        title=f"Model checks for analysis {analysis.id}",
        sentences=_sentences(sentences),
        source=_source(study, source, ["element", SOURCE_ROW, "fitted", "residual", "normal_quantile"]),
        calculation="Residuals and fitted values reported by the analysis; normal quantiles from "
        "scipy.stats.probplot.",
        used=len(values),
        depends=_analysis_depends(analysis),
        numbers={"n": len(values), "qq_r": float(np.corrcoef(theoretical, ordered)[0, 1])},
        references=(QQ_PLOT,),
    )


def draw_panel(target, result, spec: FigureSpec) -> Drawn:
    """Several figures side by side, each labelled with a bold letter at its top left."""
    study, style = result.study, _style(result)
    if isinstance(target, Axes):
        raise FigureError("a panel figure needs a whole figure to draw on, not one axes.")
    known = {figure.id: figure for figure in study.figures}
    parts = [known[ident] for ident in spec.panels if ident in known]
    if not parts:
        raise FigureError("it lists no `panels` to combine.")
    if any(part.type == "panel" for part in parts):
        raise FigureError("a panel figure cannot contain another panel figure.")
    columns, rows = _grid(len(parts))
    cells = target.subfigures(rows, columns, squeeze=False, wspace=0.04, hspace=0.04)
    sources, sentences, depends, numbers, references, used = [], [], [], {}, [], 0
    titles, calculations = [], []
    for index, part in enumerate(parts):
        cell = cells.flat[index]
        drawn = DRAWERS[part.type](cell, result, part)
        letter, size = panel_label(style, index)
        cell.suptitle(letter, x=0.0, y=1.0, ha="left", va="top", fontweight="bold", fontsize=size)
        frame = drawn.source.copy()
        frame.insert(0, "figure", part.id)
        frame.insert(0, "panel", letter)
        sources.append(frame)
        title = part.title or drawn.title
        titles.append(title)
        caption = " ".join(piece for piece in (part.caption.strip(), *drawn.sentences) if piece)
        sentences.append(f"({letter}) {title}. {caption}")
        depends += [f"figures.{part.id}", *drawn.depends]
        numbers.update({f"{letter}: {key}": value for key, value in drawn.numbers.items()})
        references += drawn.references
        used = max(used, drawn.used)
        calculations.append(f"({letter}) {drawn.calculation}")
    combined = pd.concat(sources, ignore_index=True)
    return Drawn(
        title="; ".join(titles),
        sentences=sentences,
        source=combined,
        calculation=" ".join(calculations),
        used=used,
        depends=tuple(depends),
        numbers=numbers,
        references=tuple(dict.fromkeys(references)),
    )


DRAWERS = {
    "groups": draw_groups,
    "paired": draw_paired,
    "longitudinal": draw_longitudinal,
    "scatter": draw_scatter,
    "missingness": draw_missingness,
    "diagnostics": draw_diagnostics,
    "panel": draw_panel,
}


# Statistics drawn when the analysis gives none.


def t_summary(values, confidence: float) -> dict:
    """Mean, SD, standard error and the t-based confidence interval of ``values``."""
    from scipy import stats  # noqa: PLC0415

    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values)) if n else math.nan
    sd = float(np.std(values, ddof=1)) if n > 1 else math.nan
    se = float(stats.sem(values)) if n > 1 else math.nan
    if n > 1 and se > 0:
        low, high = stats.t.interval(confidence, n - 1, loc=mean, scale=se)
    elif n > 1:
        low = high = mean
    else:
        low = high = math.nan
    return {"n": n, "mean": mean, "sd": sd, "se": se, "ci_low": float(low), "ci_high": float(high)}


def spread(values, half_width: float) -> np.ndarray:
    """Sideways offsets for a strip of points, wider where the values are dense, the same on every build.

    The points are ranked by value and given the terms of a van der Corput sequence
    in turn, so neighbouring values land on alternate sides; the offsets are then
    scaled by a Gaussian kernel density, so the strip takes the shape of the data.
    """
    values = np.asarray(values, dtype=float)
    count = len(values)
    if count < 2:
        return np.zeros(count)
    offsets = np.empty(count)
    offsets[np.argsort(values, kind="stable")] = _van_der_corput(count) - 0.5
    density = np.ones(count)
    if count >= 3 and np.ptp(values) > 0:
        from scipy.stats import gaussian_kde  # noqa: PLC0415

        try:
            density = gaussian_kde(values)(values)
            density = density / density.max()
        except np.linalg.LinAlgError:
            density = np.ones(count)
    width = half_width * min(1.0, 0.35 + count / 25)
    return 2 * offsets * (0.25 + 0.75 * density) * width


def _van_der_corput(count: int) -> np.ndarray:
    """The first ``count`` terms of the base-2 van der Corput sequence: 1/2, 1/4, 3/4, 1/8, …"""
    terms = np.empty(count)
    for index in range(count):
        value, denominator, rest = 0.0, 1.0, index + 1
        while rest:
            rest, digit = divmod(rest, 2)
            denominator *= 2
            value += digit / denominator
        terms[index] = value
    return terms


# Reading the study, the data and the analysis results.


def _style(result):
    if getattr(result, "style", None) is None:
        result.style = resolve_style(result.study)
    return result.style


def _linked(result, spec: FigureSpec):
    """The figure's analysis and, when the statistics step has run it, its result."""
    if not spec.analysis:
        return None, None
    analysis = next((item for item in result.study.analyses if item.id == spec.analysis), None)
    found = (getattr(result, "results", None) or {}).get(spec.analysis) if analysis else None
    return analysis, found


def _model_means(result, spec: FigureSpec, found, cells: list[tuple]) -> dict | None:
    """The analysis's means and intervals for each (group, time) cell drawn, or None to use the rows."""
    means = getattr(found, "group_means", None)
    if found is None or not getattr(found, "supported", True) or means is None or len(means) == 0:
        return None
    table = {}
    for row in means.to_dict("records"):
        table[(_cell(row.get("group")), _cell(row.get("time")))] = row
    groups = {group for group, _ in table}
    single = next(iter(groups)) if len(groups) == 1 else None
    chosen = {}
    for group, time in cells:
        row = table.get((group, time))
        if row is None and group is None and len(groups) == 1:
            row = table.get((single, time))
        if row is None:
            result.warnings.append(
                f"Figure {spec.id}: analysis {spec.analysis} gives no mean for "
                f"{', '.join(str(part) for part in (group, time) if part is not None) or 'every cell'}, "
                "so the figure shows means computed from the rows drawn."
            )
            return None
        chosen[(group, time)] = {column: _number(row.get(column)) for column in SUMMARY_COLUMNS}
        chosen[(group, time)]["n"] = int(row["n"]) if _number(row.get("n")) == _number(row.get("n")) else 0
    return chosen


def _model_line(analysis, found, x: str, y: str) -> tuple[dict, pd.Series] | None:
    """The linked analysis's fitted line of y on x and its reported slope, when the statistics step gave both.

    The line is the model's prediction with its covariates held at reference values (see
    ``AnalysisResult.line``), so the figure shows the relationship the analysis estimated.
    """
    if analysis is None or found is None or not getattr(found, "supported", False):
        return None
    line, estimates = getattr(found, "line", None), getattr(found, "estimates", None)
    if not line or analysis.relate != x or analysis.outcome != y or estimates is None:
        return None
    if "measure" not in estimates or estimates.empty:
        return None
    slope = estimates[(estimates["term"].astype(str) == x) & (estimates["measure"] == "slope")]
    return (line, slope.iloc[0]) if len(slope) else None


def _residuals(found) -> pd.Series | None:
    residuals = getattr(found, "residuals", None)
    if found is None or residuals is None or len(residuals) < 3:
        return None
    return pd.Series(residuals)


def _source_rows(result, index) -> list:
    """Each residual's line in the original file, when its index points into the processed data."""
    data = result.data
    return [int(data.at[label, SOURCE_ROW]) if label in data.index else None for label in index]


def _complete(result, columns) -> tuple[pd.DataFrame, str]:
    """The rows with a value in every column the figure needs, and a sentence on the rows left out."""
    data = result.data
    columns = [column for column in dict.fromkeys(columns) if column]
    mask = data[columns].notna().all(axis=1)
    used = data.loc[mask]
    if used.empty:
        raise FigureError(f"no row has a value for every one of {', '.join(columns)}.")
    dropped = data.loc[~mask, columns]
    if dropped.empty:
        return used, ""
    gaps = {column: int(dropped[column].isna().sum()) for column in columns if dropped[column].isna().any()}
    count = len(dropped)
    rows = "1 row" if count == 1 else f"{count:,} rows"
    verb = "is" if count == 1 else "are"
    if len(gaps) == 1:
        return used, f"{rows} with no value for {next(iter(gaps))} {verb} left out."
    detail = "; ".join(f"{column}: {number}" for column, number in gaps.items())
    return used, f"{rows} with a missing value {verb} left out ({detail})."


def _model_columns(study, analysis) -> list[str]:
    """The columns the linked analysis needs, so the points are the rows it used."""
    if analysis is None:
        return []
    columns = [analysis.outcome, analysis.predictor, analysis.by, *analysis.covariates]
    if study.design in ("paired", "repeated"):
        columns.append(study.subject)
    if study.design == "clustered":
        columns.append(study.cluster)
    return [column for column in columns if column]


def _require(result, name: str | None, key: str, numeric: bool = False) -> None:
    if not name:
        raise FigureError(f"it needs `{key}`, and no linked analysis or study variable supplies one.")
    if name not in result.data:
        raise FigureError(f"the data have no column {name!r}.")
    if numeric and not _is_numeric(result.data[name]):
        raise FigureError(f"{name} is not numeric.")


def _first(result, role: str) -> str | None:
    return next((name for name in result.study.with_role(role) if name in result.data), None)


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not isinstance(series.dtype, pd.CategoricalDtype)


def _within_subjects(data: pd.DataFrame, subject: str | None, column: str | None) -> bool:
    """Whether ``column`` changes within subjects, with one row per subject and level."""
    if not subject or not column or subject not in data or column not in data:
        return False
    pairs = data[[subject, column]].dropna().astype(str)
    if pairs.empty or pairs.duplicated().any():
        return False
    return bool(pairs.groupby(subject)[column].nunique().max() > 1)


def _levels(data: pd.DataFrame, column: str) -> list[str]:
    """Every level of ``column`` in study order. Colours follow this order, so a level keeps its colour
    whichever other levels a figure shows."""
    series = data[column]
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [_label(level) for level in series.cat.categories]
    values = series.dropna().unique()
    if _is_numeric(series):
        values = sorted(values)
    return list(dict.fromkeys(_label(value) for value in values))


def _present(frame: pd.DataFrame, column: str, order: list[str]) -> list[str]:
    present = set(_labels(frame, column))
    return [level for level in order if level in present]


def _labels(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype(object).map(_label)


def _label(value) -> str:
    """A value as it reads on an axis: 7 rather than 7.0, dates without a time of day."""
    if isinstance(value, bool | np.bool_):
        return str(bool(value))
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        return str(int(value)) if float(value).is_integer() else f"{float(value):g}"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d") if value == value.normalize() else value.isoformat()
    return str(value)


def _cell(value) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA or value == "":
        return None
    return _label(value)


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _natural(text: str):
    """Sort P2 before P10."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(text))]


def _id_columns(study, frame: pd.DataFrame, exclude=()) -> list[str]:
    """The columns that identify each point: the subject or cluster, else the first identifier."""
    names = [name for name in (study.subject, study.cluster) if name]
    if not names:
        names = study.with_role("identifier")[:1]
    return [
        name
        for name in dict.fromkeys(names)
        if name in frame and name not in exclude and name not in study.sharing.withhold
    ]


def _ids(row, ids: list[str]) -> dict:
    return {name: row[name] for name in ids}


def _source(study, records: list[dict], columns: list[str]) -> pd.DataFrame:
    """The figure's source data: one row per drawn mark, without the columns ``sharing.withhold`` names."""
    frame = pd.DataFrame.from_records(records)
    ordered = [column for column in dict.fromkeys(columns) if column in frame]
    frame = frame[ordered + [column for column in frame.columns if column not in ordered]]
    frame = frame.drop(columns=[column for column in study.sharing.withhold if column in frame])
    frame = frame.dropna(axis=1, how="all")
    for column in frame.columns:
        if column in COUNT_COLUMNS:
            frame[column] = frame[column].astype("Int64")
        elif pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(
                lambda value: float(f"{value:.12g}") if pd.notna(value) else value
            )
    return frame


def _blend(colour: str, alpha: float) -> tuple[float, float, float]:
    """``colour`` at ``alpha`` over white, as an opaque colour."""
    from matplotlib.colors import to_rgb  # noqa: PLC0415

    return tuple(alpha * value + 1 - alpha for value in to_rgb(colour))


def _analysis_depends(analysis) -> tuple[str, ...]:
    return (f"analyses.{analysis.id}",) if analysis else ()


# Words.


def _ci(study) -> str:
    return f"{study.decisions.confidence * 100:g}%"


def _name(study, column: str) -> str:
    """A variable's label without its unit, for titles: "Viability" for "Viability (%)"."""
    variable = study.variable(column)
    label = variable.label or column
    return re.sub(r"\s*\([^()]*\)$", "", label) or label


def _each_point(study) -> str:
    observation = study.observation.strip().rstrip(".")
    if not observation:
        return "Each point is one row of the data"
    if "row" in observation.lower():
        return f"Each point is one row of the data ({observation})"
    return f"Each point is {observation}"


def _n_list(counts: list[tuple]) -> str:
    if len(counts) == 1:
        return f"n = {counts[0][1]:,}"
    parts = [f"{count:,} ({label})" for label, count in counts]
    return "n = " + ", ".join(parts[:-1]) + " and " + parts[-1]


def _dependence(study, used: pd.DataFrame) -> str:
    """A note that points are not independent when subjects or clusters contribute several."""
    clustering = _clustering(study, used)
    return f"Points are not independent: {clustering}." if clustering else ""


def _clustering(study, used: pd.DataFrame) -> str:
    if study.design in ("paired", "repeated") and study.subject in used:
        subjects = used[study.subject].nunique()
        if subjects < len(used):
            return (
                f"the {len(used):,} points come from {_count(subjects, 'subject')} ({study.subject}) "
                "measured more than once"
            )
    if study.design == "clustered" and study.cluster in used:
        clusters = used[study.cluster].nunique()
        if clusters < len(used):
            return f"the {len(used):,} points come from {_count(clusters, 'cluster')} ({study.cluster})"
    return ""


def _count(number: int, noun: str) -> str:
    """A count with its noun: "1 subject", "42 subjects"."""
    return f"{number:,} {noun}" + ("" if number == 1 else "s")


def _held(study, held: dict) -> str:
    """Where a model's line holds each covariate: "age at its mean in the rows analysed (64.8 years)"."""
    parts = []
    for name, value in held.items():
        if isinstance(value, str):
            parts.append(f"{name} at its reference level ({value})")
        else:
            unit = study.variable(name).unit
            number = (f"{value:.4g}" if abs(value) < 10_000 else f"{value:,.0f}").replace("-", "−")
            parts.append(f"{name} at its mean in the rows analysed ({number}{f' {unit}' if unit else ''})")
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _analysis_sentence(analysis, found) -> str:
    """The linked analysis, its test, and whether it was planned before the data were seen."""
    if analysis is None:
        return ""
    test = getattr(found, "test", "") if found is not None and getattr(found, "supported", True) else ""
    name = f"Analysis {analysis.id}" + (f" ({test})" if test else "")
    if analysis.planned:
        sentence = f"{name} was planned before the data were seen."
    else:
        sentence = f"{name} is exploratory: it was not planned before the data were seen."
    if found is not None and not getattr(found, "supported", True):
        sentence += f" ADA did not fit it: {found.reason}"
    return sentence


def _sentences(parts: list[str]) -> list[str]:
    return [part.strip() for part in parts if part and part.strip()]


def _wrap(text: str, width: int = 12) -> str:
    return textwrap.fill(text, width, break_long_words=False) if len(text) > width else text


# Drawing.


def _size(result, spec: FigureSpec) -> tuple[float, float]:
    """The figure's size in inches from the journal preset, the type, and any size set for the figure."""
    style = _style(result)
    width, aspect = {
        "groups": ("single", 0.78),
        "paired": ("single", 0.62),
        "longitudinal": ("single", 0.75),
        "scatter": ("single", 0.8),
        "missingness": ("single", 0.6),
        "diagnostics": ("single", 0.48),
        "panel": ("double", 0.4),
    }[spec.type]
    if spec.type == "missingness":
        count = sum(1 for name in result.study.variables if name in result.data)
        width_mm = figure_size(style, width)[0] * MM_PER_INCH
        aspect = min(max((24 + 6 * count) / width_mm, 0.42), 1.2)
    if spec.type == "panel":
        columns, rows = _grid(len(spec.panels))
        aspect = 0.78 * rows / columns
    wide, high = figure_size(style, width, aspect)
    if spec.width_mm:
        wide = spec.width_mm / MM_PER_INCH
        high = min(wide * aspect, style.max_height_mm / MM_PER_INCH)
    if spec.height_mm:
        high = min(spec.height_mm, style.max_height_mm) / MM_PER_INCH
    return wide, high


def _grid(count: int) -> tuple[int, int]:
    """Columns and rows for ``count`` panels: one row up to three, then rows of two or three."""
    columns = count if count <= 3 else 2 if count == 4 else 3
    return columns, math.ceil(count / columns)


def _axes(target, count: int = 1, **options) -> list:
    """Axes to draw on: the one given, or ``count`` side by side on a figure or subfigure."""
    if isinstance(target, Axes):
        return [target]
    if count == 1:
        return [target.add_subplot()]
    sharey = options.pop("sharey", False)
    return list(target.subplots(1, count, sharey=sharey, gridspec_kw=options))


def _look(order: list[str], level: str) -> tuple[str, str]:
    """A level's colour and marker, fixed by its place in the study's order of levels."""
    index = order.index(level) if level in order else 0
    return PALETTE[index % len(PALETTE)], MARKERS[(index // len(PALETTE)) % len(MARKERS)]


def _points(ax, xs, ys, colour: str, marker: str, style) -> None:
    ax.scatter(
        xs,
        ys,
        s=POINT_AREA,
        marker=marker,
        facecolors=[tint(colour, POINT_ALPHA, style)],
        edgecolors=[colour],
        linewidths=0.5,
        zorder=2,
    )


def _estimate(ax, x: float, summary: dict) -> None:
    """A mean and its confidence interval in ink, with a white rim so it reads over the points."""
    mean, low, high = summary["mean"], summary["ci_low"], summary["ci_high"]
    if not np.isfinite(mean):
        return
    errors = [[mean - low], [high - mean]] if np.isfinite(low) and np.isfinite(high) else None
    ax.errorbar(
        [x],
        [mean],
        yerr=errors,
        fmt="o",
        color=INK,
        ecolor=INK,
        elinewidth=1.0,
        capsize=0,
        markersize=5,
        markerfacecolor=INK,
        markeredgecolor="white",
        markeredgewidth=0.6,
        zorder=4,
    )


def _category_axis(ax, levels: list[str]) -> None:
    ax.set_xticks(range(len(levels)), [_wrap(level) for level in levels])
    ax.set_xlim(-0.6, len(levels) - 0.4)
    ax.tick_params(axis="x", length=0)


def _legend(ax, entries: list[tuple], style, lines: bool = False) -> None:
    """A legend above the axes, one entry per level, in study order."""
    from matplotlib.lines import Line2D  # noqa: PLC0415

    handles = [
        Line2D(
            [],
            [],
            color=colour if lines else "none",
            linewidth=1.0 if lines else 0,
            marker=marker,
            markersize=4.5,
            markerfacecolor=colour if lines else tint(colour, POINT_ALPHA, style),
            markeredgecolor=INK if lines else colour,
            markeredgewidth=0.5,
        )
        for (colour, marker), _ in entries
    ]
    ax.legend(
        handles,
        [label for _, label in entries],
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncols=min(len(entries), 4),
        borderaxespad=0.2,
        handlelength=1.4 if lines else 0.8,
        handletextpad=0.4,
        columnspacing=1.2,
    )
