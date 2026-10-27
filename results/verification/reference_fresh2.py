"""Independent reference values for the third fresh holdout: deforestation and mtcars. pandas only.

Written before ADA read these files. Run with the folder that holds deforestation/ and mtcars/.
"""
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])

D = root / "deforestation"
names = ("forest", "forest_area", "brazil_loss", "soybean_use", "vegetable_oil")
tables = {name: pd.read_csv(D / f"{name}.csv") for name in names}
for name, table in tables.items():
    print(name, table.shape, "entity-year unique", not table.duplicated(["entity", "year"]).any(),
          "code missing", int(table["code"].isna().sum()), "years", table["year"].min(), table["year"].max())
area = tables["forest_area"]
for name in ("forest", "brazil_loss", "soybean_use", "vegetable_oil"):
    pairs = tables[name][["entity", "year"]].drop_duplicates()
    joined = pairs.merge(area[["entity", "year"]], how="left", indicator=True)
    unmatched = joined["_merge"].eq("left_only").sum()
    merged = tables[name].merge(area[["entity", "year"]], how="left", indicator=True)
    rows = merged["_merge"].eq("left_only").sum()
    print(f"  {name}.(entity, year) -> forest_area: pairs unmatched {unmatched}, rows unmatched {rows}")
forest = tables["forest"]
print("forest entities", forest["entity"].nunique(), "has World", "World" in set(forest["entity"]))
print("net_forest_conversion mean", forest["net_forest_conversion"].mean(), "n", len(forest))
loss = tables["brazil_loss"]
print("brazil pasture 2004", loss.loc[loss["year"] == 2004, "pasture"].tolist(),
      "pasture mean", loss["pasture"].mean())
world = (area["entity"] == "World") & (area["year"] == 2020)
print("world forest_area 2020", area.loc[world, "forest_area"].tolist())

cars = pd.read_csv(root / "mtcars" / "mtcars.csv")
print("mtcars", cars.shape)
print("mpg mean", cars["mpg"].mean(), "hp mean", cars["hp"].mean(), "wt mean", cars["wt"].mean())
print("mpg by cyl", cars.groupby("cyl")["mpg"].mean().round(3).to_dict(),
      "4-cyl n", int((cars["cyl"] == 4).sum()))
print("share automatic (am == 0)", (cars["am"] == 0).mean())
