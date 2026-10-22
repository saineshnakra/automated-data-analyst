"""Small studies for the research tests: a data file and a study.yml in a scratch folder."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def assay(rows_per_group: int = 12, seed: int = 3) -> pd.DataFrame:
    """Wells under three treatments, viability falling with dose, one -99 code and one negative reading."""
    rng = np.random.default_rng(seed)
    frames = []
    for treatment, shift in (("Control", 0.0), ("Low", -5.0), ("High", -12.0)):
        frames.append(
            pd.DataFrame(
                {
                    "Well": [f"{treatment[0]}{i:02d}" for i in range(rows_per_group)],
                    "Treatment": treatment,
                    "Passage": rng.integers(3, 12, rows_per_group),
                    "Viability (%)": np.round(90 + shift + rng.normal(0, 3, rows_per_group), 2),
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    frame.loc[2, "Viability (%)"] = -99
    frame.loc[15, "Viability (%)"] = -4.0
    return frame


def weights(mice: int = 12, seed: int = 5) -> pd.DataFrame:
    """Mice weighed on days 0 to 28 on two diets; the high-fat mice gain faster."""
    rng = np.random.default_rng(seed)
    rows = []
    for mouse in range(mice):
        diet = "Chow" if mouse < mice // 2 else "High fat"
        start = rng.normal(20, 1)
        for day in (0, 7, 14, 21, 28):
            gain = (0.3 if diet == "Chow" else 0.8) * day / 7
            rows.append(
                {
                    "Mouse": f"M{mouse:02d}",
                    "Diet": diet,
                    "Day": day,
                    "Weight (g)": round(start + gain + rng.normal(0, 0.3), 2),
                }
            )
    return pd.DataFrame(rows)


def assay_study(**changes) -> dict:
    study = {
        "study": {"title": "Viability assay", "authors": [{"name": "A. Researcher"}], "license": "CC-BY-4.0"},
        "data": {"path": "assay.csv", "observation": "one well", "missing_codes": [-99]},
        "design": {"type": "independent"},
        "variables": {
            "Well": {"role": "identifier", "type": "text"},
            "Treatment": {"role": "group", "type": "categorical", "levels": ["Control", "Low", "High"]},
            "Passage": {"role": "covariate", "type": "numeric"},
            "Viability (%)": {"role": "outcome", "type": "numeric", "unit": "%", "range": [0, 120]},
        },
        "processing": [],
        "analyses": [
            {
                "id": "primary",
                "outcome": "Viability (%)",
                "compare": "Treatment",
                "planned": True,
                "family": "primary",
            }
        ],
    }
    for key, value in changes.items():
        study[key] = value
    return study


@contextmanager
def study_folder(frame: pd.DataFrame, study: dict, data_name: str = "assay.csv"):
    """A scratch folder holding the data as CSV and the study as study.yml; yields the study.yml path."""
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        frame.to_csv(root / data_name, index=False)
        path = root / "study.yml"
        path.write_text(yaml.safe_dump(study, sort_keys=False, allow_unicode=True), encoding="utf-8")
        yield path
