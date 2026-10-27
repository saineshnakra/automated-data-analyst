"""Independent reference values for the fourth fresh holdout: chocolate, HairEyeColor, Olympics. pandas only.

Written before ADA read these files. Run with the folder that holds chocolate/, haireyecolor/ and olympics/.
"""
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])

bars = pd.read_csv(root / "chocolate" / "chocolate.csv")
cocoa = bars["cocoa_percent"].str.rstrip("%").astype(float)
print("chocolate", bars.shape, "rating mean", bars["rating"].mean(), "cocoa mean", cocoa.mean())
usa = bars["company_location"] == "U.S.A."
print("rating mean by company_location (U.S.A.)", bars.loc[usa, "rating"].mean(), "n", int(usa.sum()))
print("reviews in 2019", int((bars["review_date"] == 2019).sum()))

eyes = pd.read_csv(root / "haireyecolor" / "HairEyeColor.csv")
print("haireyecolor", eyes.shape, "people", eyes["Freq"].sum(),
      "combinations unique", not eyes.duplicated(["Hair", "Eye", "Sex"]).any())
print("share with blue eyes", eyes.loc[eyes["Eye"] == "Blue", "Freq"].sum() / eyes["Freq"].sum(),
      "row share", (eyes["Eye"] == "Blue").mean())
print("share female", eyes.loc[eyes["Sex"] == "Female", "Freq"].sum() / eyes["Freq"].sum())

games = pd.read_csv(root / "olympics" / "olympics.csv")
regions = pd.read_csv(root / "olympics" / "regions.csv")
print("olympics", games.shape, "athletes", games["id"].nunique(), "regions", regions.shape,
      "NOC unique", regions["NOC"].is_unique)
unmatched = ~games["noc"].isin(regions["NOC"])
print("noc rows unmatched", int(unmatched.sum()), "values", sorted(games.loc[unmatched, "noc"].unique()))
print("age mean (rows)", games["age"].mean(), "n", games["age"].notna().sum(),
      "age mean (one row per athlete)", games.drop_duplicates("id")["age"].mean())
print("gold medal rows", int((games["medal"] == "Gold").sum()),
      "share of rows with a medal", games["medal"].notna().mean())
print("rows in 2016", int((games["year"] == 2016).sum()))
