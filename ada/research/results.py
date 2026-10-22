"""What one analysis found: the shape shared by the analyses, the figures and the methods draft."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .study import AnalysisSpec

#: Columns of :attr:`AnalysisResult.estimates`, one row per reported term or contrast.
ESTIMATE_COLUMNS = ["term", "label", "estimate", "ci_low", "ci_high", "statistic", "df", "p", "p_adjusted"]
#: Columns of :attr:`AnalysisResult.group_means`, one row per group, time point or group × time cell.
MEAN_COLUMNS = ["group", "time", "n", "mean", "sd", "se", "ci_low", "ci_high"]


@dataclass
class AnalysisResult:
    spec: AnalysisSpec
    design: str
    #: False when ADA does not fit this analysis; ``reason`` says why and what to do instead.
    supported: bool = True
    reason: str = ""
    #: The test in words, such as "Welch's t-test" or "Linear mixed model".
    test: str = ""
    #: The library call behind it, such as "scipy.stats.ttest_ind(equal_var=False)".
    method: str = ""
    #: The model in formula notation, or "" for a test without one.
    formula: str = ""
    #: Rows used, and rows left out because a value the analysis needs was missing.
    n: int = 0
    n_missing: int = 0
    subjects: int | None = None
    clusters: int | None = None
    #: Rows used per group level.
    groups: dict[str, int] = field(default_factory=dict)
    #: One row per term or contrast; see ESTIMATE_COLUMNS. "statistic" is named by ``statistic_name``.
    estimates: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=ESTIMATE_COLUMNS))
    statistic_name: str = ""
    #: The standardised effect size: {"name", "value", "ci_low", "ci_high"}.
    effect: dict = field(default_factory=dict)
    #: Means with confidence intervals per group (and time); see MEAN_COLUMNS. Figures draw these.
    group_means: pd.DataFrame | None = None
    #: The fitted values and residuals, for the diagnostic plots.
    fitted: pd.Series | None = None
    residuals: pd.Series | None = None
    #: Checks of the model's assumptions, in plain sentences, and anything the reader should know.
    diagnostics: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    references: tuple[str, ...] = ()
    #: The model's full printed summary, for the supplement.
    summary: str = ""

    @property
    def planned(self) -> bool:
        return self.spec.planned

    @property
    def family(self) -> str:
        return self.spec.family
