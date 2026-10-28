"""Published tables for the question and claim tests."""

from __future__ import annotations

import pandas as pd


def migration() -> pd.DataFrame:
    """A CSO-style table of migration in thousands, every count under each level and again under its total.

    By hand, 2024: immigrants by age band 20, 15, 50, 10 and 5 (100 in all), emigrants 5, 10, 20, 4 and 1
    (40); 2023: immigrants 10, 10, 30, 8 and 2 (60), emigrants 4, 8, 16, 2 and 2 (32). Men are 60% of each.
    """
    bands = ["0 - 14 years", "15 - 24 years", "25 - 44 years", "45 - 64 years", "65 years and over"]
    counts = {
        (2024, "Immigrants: All origins"): [20.0, 15.0, 50.0, 10.0, 5.0],
        (2024, "Emigrants: All destinations"): [5.0, 10.0, 20.0, 4.0, 1.0],
        (2023, "Immigrants: All origins"): [10.0, 10.0, 30.0, 8.0, 2.0],
        (2023, "Emigrants: All destinations"): [4.0, 8.0, 16.0, 2.0, 2.0],
    }
    rows = []
    for (year, flow), values in counts.items():
        for age, value in [("All ages", sum(values)), *zip(bands, values, strict=True)]:
            for sex, part in (("Both sexes", 1.0), ("Male", 0.6), ("Female", 0.4)):
                rows.append({"Year": year, "Age Group": age, "Sex": sex, "Inward or Outward Flow": flow,
                             "UNIT": "Thousand", "VALUE": round(value * part, 1)})
    return pd.DataFrame(rows)



def births() -> pd.DataFrame:
    """A StatCan-style table of births: counts beside percentages, a nation over its provinces, and labels
    that all start "Age of mother, ".

    By hand, 2025: Canada 100 births, 12 to mothers aged 15 to 19, 50 aged 20 to 34, 30 aged 35 to 39 and 8
    aged 40 to 44; Alberta 40 and Nunavut 60, Nunavut's 12 all to mothers aged 15 to 19. 2024 halves each.
    """
    ages = ["Age of mother, all ages", "Age of mother, 15 to 19 years", "Age of mother, 20 to 34 years",
            "Age of mother, 35 to 39 years", "Age of mother, 40 to 44 years"]
    places = {"Alberta, place of residence of mother": [40, 0, 26, 10, 4],
              "Nunavut, place of residence of mother": [60, 12, 24, 20, 4]}
    rows = []
    for year, part in ((2025, 1.0), (2024, 0.5)):
        canada = [sum(column) for column in zip(*places.values(), strict=True)]
        for geo, counts in {"Canada, place of residence of mother": canada, **places}.items():
            for age, count in zip(ages, counts, strict=True):
                number = count * part
                share = number / (counts[0] * part) * 100
                rows.append({"REF_DATE": year, "GEO": geo, "Age of mother": age,
                             "Characteristics": "Number of live births", "UOM": "Number",
                             "COORDINATE": f"{len(rows) % 3 + 1}.{len(rows) % 5 + 1}.1", "VALUE": number})
                rows.append({"REF_DATE": year, "GEO": geo, "Age of mother": age,
                             "Characteristics": "Percentage of live births", "UOM": "Percentage",
                             "COORDINATE": f"{len(rows) % 3 + 1}.{len(rows) % 5 + 1}.2",
                             "VALUE": round(share, 1)})
    return pd.DataFrame(rows)
