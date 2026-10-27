import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.simplefilter("ignore")
H = Path(sys.argv[1])

g = pd.read_csv(H / "gapminder/gapminder.csv")
print("gapminder", g.shape, g.country.nunique(), sorted(g.year.unique()))
y7 = g[g.year == 2007]
print("2007 mean lifeExp by continent", y7.groupby("continent").lifeExp.mean().round(3).to_dict())
print(
    "2007 unweighted mean",
    y7.lifeExp.mean(),
    "pop-weighted",
    np.average(y7.lifeExp, weights=y7["pop"]),
    "n",
    len(y7),
)
print(
    "pooled r log gdp vs lifeExp",
    stats.pearsonr(np.log(g.gdpPercap), g.lifeExp),
    "2007 only",
    stats.pearsonr(np.log(y7.gdpPercap), y7.lifeExp),
)

c = pd.read_csv(H / "chickweight/ChickWeight.csv")
print(
    "chick",
    c.shape,
    c.Chick.nunique(),
    c.groupby("Diet").Chick.nunique().to_dict(),
    sorted(c.Time.unique()),
)
d21 = c[c.Time == 21]
print("day21 mean weight by diet", d21.groupby("Diet").weight.agg(["mean", "count"]).round(3).to_dict())
mm = smf.mixedlm("weight ~ Time * C(Diet)", c, groups=c["Chick"], re_formula="~Time").fit(reml=True)
print("mixed time:diet", mm.params.filter(like=":").round(3).to_dict(), mm.pvalues.filter(like=":").to_dict())
print("naive anova all rows", stats.f_oneway(*[x.weight for _, x in c.groupby("Diet")]))
print("day21 anova", stats.f_oneway(*[x.weight for _, x in d21.groupby("Diet")]))

t = pd.read_csv(H / "titanic/Titanic.csv")
print("titanic", t.shape, t.columns.tolist(), "sum Freq", t.Freq.sum())
surv = t.groupby("Survived").Freq.sum()
print("survival rate weighted", surv["Yes"] / surv.sum(), "row-count share", (t.Survived == "Yes").mean())
print(
    "by class weighted",
    (t[t.Survived == "Yes"].groupby("Class").Freq.sum() / t.groupby("Class").Freq.sum()).round(4).to_dict(),
)
print(
    "by sex weighted",
    (t[t.Survived == "Yes"].groupby("Sex").Freq.sum() / t.groupby("Sex").Freq.sum()).round(4).to_dict(),
)

W = H / "worldbank-lifeexp"
data = next(W.glob("API_*.csv"))
head = data.read_text(encoding="utf-8-sig").splitlines()[:5]
print("wb head lines", [h[:60] for h in head])
wb = pd.read_csv(data, skiprows=4)
meta = pd.read_csv(next(W.glob("Metadata_Country_*.csv")))
print(
    "wb",
    wb.shape,
    wb.columns[:6].tolist(),
    wb.columns[-3:].tolist(),
    "meta",
    meta.shape,
    meta.columns.tolist(),
)
agg = meta[meta.Region.isna()]
print("aggregates (no region)", len(agg), agg.TableName.head(8).tolist())
print(
    "Japan 2021",
    wb.loc[wb["Country Code"] == "JPN", "2021"].tolist(),
    "World 2021",
    wb.loc[wb["Country Code"] == "WLD", "2021"].tolist(),
)
cols = [c for c in wb.columns if c.isdigit()]
last = max(c for c in cols if wb[c].notna().any())
print("latest year with data", last, "non-null", wb[last].notna().sum())

B = H / "bikesharing"
day = pd.read_csv(B / "day.csv")
hour = pd.read_csv(B / "hour.csv")
print(
    "day",
    day.shape,
    "hour",
    hour.shape,
    "cnt==casual+registered",
    (day.cnt == day.casual + day.registered).all(),
    (hour.cnt == hour.casual + hour.registered).all(),
)
by_year = day.groupby("yr").cnt.sum()
print("totals by yr", by_year.to_dict(), "growth", by_year[1] / by_year[0] - 1)
hs = hour.groupby("dteday").cnt.sum()
print("hour sums match day", (day.set_index("dteday").cnt == hs).all())
print(
    "r cnt~registered",
    stats.pearsonr(day.cnt, day.registered)[0],
    "r cnt~temp",
    stats.pearsonr(day.cnt, day.temp)[0],
)
