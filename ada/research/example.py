"""``ada demo``: a small example study to try ``ada build`` on.

Sixteen mice, eight on chow and eight on a high-fat diet, weighed once a week
for eight weeks. The data are simulated by seeded code here, so the example
ships as a few lines of Python rather than a data file. The study file uses
every part of study.yml: authors, units, a missing-value code, two exclusions
with their reasons, a planned and an exploratory analysis, figures, a custom
table and sharing settings with a withheld column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .study import sha256_of

SEED = 11
DATA = "data/mouse-weights.csv"
SCRIPT = "analysis/weight-gain.py"
WEEKS = range(9)
#: Initials of the people who weighed the mice: the column the study withholds.
HANDLERS = ("AR", "BK", "CL")


def mouse_weights(seed: int = SEED) -> pd.DataFrame:
    """Weekly weights and food intake; the high-fat mice gain about 0.6 g a week more than chow mice."""
    rng = np.random.default_rng(seed)
    rows = []
    for number in range(1, 17):
        diet = "Chow" if number % 2 else "High fat"
        sex = "Female" if number <= 8 else "Male"
        start = rng.normal(19.5 if sex == "Female" else 22.5, 1.0)
        slope = rng.normal(0.35 if diet == "Chow" else 0.95, 0.08)
        appetite = rng.normal(3.4 if diet == "Chow" else 2.8, 0.15)
        for week in WEEKS:
            rows.append(
                {
                    "Mouse": f"M{number:02d}",
                    "Sex": sex,
                    "Diet": diet,
                    "Week": week,
                    "Body weight (g)": round(start + slope * week + rng.normal(0, 0.3), 2),
                    "Food intake (g/day)": round(appetite + rng.normal(0, 0.2), 2) if week else np.nan,
                    "Handler": HANDLERS[int(rng.integers(len(HANDLERS)))],
                }
            )
    frame = pd.DataFrame(rows)
    missed = (frame["Mouse"] == "M04") & (frame["Week"] == 5)
    frame.loc[missed, "Body weight (g)"] = -99
    slipped = (frame["Mouse"] == "M11") & (frame["Week"] == 2)
    frame.loc[slipped, "Body weight (g)"] = 2.31
    return frame


STUDY = """\
# study.yml for the example study written by `ada demo`. The data are simulated.
# `ada build` reads this file and never changes it. Edit anything, then build again;
# each setting is explained in docs/research.md.

study:
  title: Body weight of mice on a high-fat diet
  description: >-
    Sixteen mice, eight on standard chow and eight on a high-fat diet, weighed once a week
    for eight weeks. An example study written by `ada demo`; the data are simulated.
  authors:
    - name: Ada Researcher
      orcid: 0000-0002-1825-0097
      affiliation: Example Institute of Physiology
    - name: Ben Analyst
      affiliation: Example Institute of Physiology
  keywords: [mice, diet, body weight, repeated measures]
  license: ""          # for the shared data, such as CC-BY-4.0; you choose it
  code_license: ""     # for the shared code, such as MIT

data:
  path: {data}
  sha256: {sha256}
  observation: one mouse at one weekly weighing
  missing_codes: [-99]   # a weighing that was missed

design:
  type: repeated
  subject: Mouse
  time: Week

variables:
  Mouse:
    role: identifier
    type: text
    description: Ear-tag code
  Sex:
    role: covariate
    type: categorical
    levels: [Female, Male]
  Diet:
    role: group
    type: categorical
    levels: [Chow, High fat]   # chow is the reference
    description: Diet from week 0
  Week:
    role: time
    type: numeric
    unit: weeks
    description: Weeks since the diet started
  Body weight (g):
    role: outcome
    type: numeric
    unit: g
    range: [10, 60]
    description: Weighed on the same scale each Monday morning
  Food intake (g/day):
    role: outcome
    type: numeric
    unit: g/day
    range: [0, 10]
    description: Mean daily intake over the week before the weighing
  Handler:
    role: other
    type: categorical
    description: Initials of the person who weighed the mouse

processing:
  - exclude:
      ids: [M07]
    reason: M07 was given the wrong diet in weeks 2 and 3 (feeding log, entry 12).
  - exclude:
      where: {{column: Body weight (g), below: 10}}
    reason: A reading below 10 g is a scale error; the scale was re-zeroed that morning (lab notebook p. 41).

decisions:
  missing_data: complete-case
  outliers: flag
  multiple_comparisons: holm
  alpha: 0.05
  confidence: 0.95

# Decided before the data were seen: weight gain by diet. Food intake was looked at afterwards.
analyses:
  - id: weight-gain
    question: Does body weight rise faster on the high-fat diet?
    outcome: Body weight (g)
    change: Week
    by: Diet
    covariates: []
    planned: true
    family: primary
  - id: food-intake
    question: Do mice on the high-fat diet eat a different amount?
    outcome: Food intake (g/day)
    compare: Diet
    covariates: []
    planned: false
    family: exploratory

figures:
  - id: weight-curves
    type: longitudinal
    y: Body weight (g)
    x: Week
    by: Diet
    analysis: weight-gain
  - id: food-intake
    type: groups
    y: Food intake (g/day)
    x: Diet
    analysis: food-intake

output:
  dir: paper-assets
  journal: default
  tables: [csv, docx, tex]
  figures: [pdf, svg, png]

sharing:
  data: processed
  withhold: [Handler]    # never leaves this folder; `ada preview` and `ada share` check it
  row_level: true

custom: [{script}]
"""

CUSTOM = '''\
"""A table of each diet's weight gain, made by the researcher's own code."""

import pandas as pd

from ada import research


@research.table(
    "weight-gain-by-diet",
    title="Weight gained from week 0 to week 8, by diet",
    notes=["Gain is each mouse's week 8 weight minus its week 0 weight; mice missing either are left out."],
    section="supplement",
)
def weight_gain(data, study):
    weights = data.dropna(subset=["Body weight (g)"])
    first = weights[weights["Week"] == 0].set_index("Mouse")["Body weight (g)"]
    last = weights[weights["Week"] == 8].set_index("Mouse")["Body weight (g)"]
    gain = (last - first).dropna()
    diet = weights.groupby("Mouse", observed=True)["Diet"].first().astype(str)
    frame = pd.DataFrame({"Diet": diet.loc[gain.index], "Gain (g)": gain})
    summary = frame.groupby("Diet", sort=False)["Gain (g)"].agg(["count", "mean", "std"])
    return summary.rename(columns={"count": "Mice", "mean": "Mean gain (g)", "std": "SD (g)"}).reset_index()
'''


def write_example(folder: str | Path = "ada-demo", overwrite: bool = False) -> list[Path]:
    """Write the example study into ``folder`` and return the files written."""
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{folder} already exists and is not empty. Choose another folder, or pass --force."
        )
    data = folder / DATA
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_bytes(mouse_weights().to_csv(index=False, lineterminator="\n").encode("utf-8"))
    script = folder / SCRIPT
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_bytes(CUSTOM.encode("utf-8"))
    study = folder / "study.yml"
    study.write_bytes(STUDY.format(data=DATA, sha256=sha256_of(data), script=SCRIPT).encode("utf-8"))
    return [study, data, script]
