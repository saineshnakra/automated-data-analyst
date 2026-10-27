"""Independent reference values for the Formula 1 holdout (Ergast via TidyTuesday 2021-09-07). pandas only."""
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
def read(name):
    return pd.read_csv(root / name, na_values=["\\N"], keep_default_na=True)


results, races, drivers = read("results.csv"), read("races.csv"), read("drivers.csv")
constructors, status, pits = read("constructors.csv"), read("status.csv"), read("pit_stops.csv")
print("results shape", results.shape)
print("missing per column (\\N):", results.isna().sum()[lambda s: s > 0].to_dict())
print(
    "points total",
    results["points"].sum(),
    "mean",
    results["points"].mean(),
    "n",
    results["points"].notna().sum(),
)
speed = pd.to_numeric(results["fastestLapSpeed"])
print("fastestLapSpeed mean", speed.mean(), "n", speed.notna().sum(), "missing", speed.isna().sum())
print("races per year 2020", int((races["year"] == 2020).sum()), "2019", int((races["year"] == 2019).sum()))
print("distinct drivers in results", results["driverId"].nunique())
names = drivers[["driverId", "forename", "surname"]]
joined = results.merge(names, on="driverId", how="left", validate="many_to_one")
ham = joined[(joined.forename == "Lewis") & (joined.surname == "Hamilton")]
print(
    "Hamilton results",
    len(ham),
    "points",
    ham["points"].sum(),
    "wins",
    int((ham["positionOrder"] == 1).sum()),
)
# key coverage both ways
pairs = (
    (results, races, "raceId"), (results, drivers, "driverId"), (results, constructors, "constructorId"),
    (results, status, "statusId"), (races, read("circuits.csv"), "circuitId"),
)
for left, right, key in pairs:
    unmatched = int((~left[key].isin(right[key])).sum())
    unused = int((~right[key].isin(left[key])).sum())
    print(key, "left unmatched", unmatched, "right unused", unused, "right unique", right[key].is_unique)
drives = results[["raceId", "driverId"]].drop_duplicates()
pk = pits.merge(drives, on=["raceId", "driverId"], how="left", indicator=True)
print("pit stops without a result row", int((pk["_merge"] == "left_only").sum()), "pit rows", len(pits))
print("results rows per (raceId, driverId) max", results.groupby(["raceId", "driverId"]).size().max())
print("grid mean", results["grid"].mean(), "median laps", results["laps"].median())
