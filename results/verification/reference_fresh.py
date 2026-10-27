"""Independent reference values for the second fresh holdout: Himalayan expeditions and Lego. pandas only.

Written before ADA read these files. Run with the folder that holds himalaya/ and lego/.
"""
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])


def coverage(left, right, left_key, right_key):
    values = left[left_key].dropna()
    unmatched = int((~values.isin(right[right_key])).sum())
    print(f"  {left_key} -> {right_key}: right unique {right[right_key].is_unique}, "
          f"left rows unmatched {unmatched}, left repeats {not values.is_unique}")


H = root / "himalaya"
peaks, expeditions, members = (pd.read_csv(H / f"{name}.csv") for name in ("peaks", "expeditions", "members"))
print("himalaya shapes", peaks.shape, expeditions.shape, members.shape)
coverage(members, expeditions, "expedition_id", "expedition_id")
coverage(expeditions, peaks, "peak_id", "peak_id")
coverage(members, peaks, "peak_id", "peak_id")
print("  member_id unique", members["member_id"].is_unique)
print("peak height mean", peaks["height_metres"].mean(), "n", peaks["height_metres"].notna().sum())
print("member age mean", members["age"].mean(), "n", members["age"].notna().sum(),
      "missing", members["age"].isna().sum())
print("member success share", members["success"].mean(), "n", len(members))
print("members per expedition max", members.groupby("expedition_id").size().max(),
      "expeditions with members", members["expedition_id"].nunique())
print("expeditions in 2019", int((expeditions["year"] == 2019).sum()))
print("member deaths total (expeditions)", expeditions["member_deaths"].sum(),
      "died rows (members)", int(members["died"].sum()))

L = root / "lego"
sets, themes, inventories = (pd.read_csv(L / f"{name}.csv") for name in ("sets", "themes", "inventories"))
parts, colors, categories = (
    pd.read_csv(L / f"{name}.csv") for name in ("parts", "colors", "part_categories")
)
lines = pd.read_csv(L / "inventory_parts.csv", usecols=["inventory_id", "part_num", "color_id", "quantity"])
print("lego shapes", sets.shape, themes.shape, inventories.shape, lines.shape, parts.shape, colors.shape,
      categories.shape)
coverage(sets, themes, "theme_id", "id")
coverage(inventories, sets, "set_num", "set_num")
coverage(lines, inventories, "inventory_id", "id")
coverage(lines, parts, "part_num", "part_num")
coverage(lines, colors, "color_id", "id")
coverage(parts, categories, "part_cat_id", "id")
print("sets num_parts mean", sets["num_parts"].mean(), "n", sets["num_parts"].notna().sum())
print("sets in 2020", int((sets["year"] == 2020).sum()))
print("2019 num_parts mean", sets.loc[sets["year"] == 2019, "num_parts"].mean())
print("set_num unique", sets["set_num"].is_unique, "inventories per set_num max",
      inventories.groupby("set_num").size().max())
