import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

D = Path(sys.argv[1])

p = pd.read_csv(D / "penguins/penguins.csv")
print("penguins", p.shape, p.isna().sum().to_dict())
print(p.species.value_counts().to_dict())
print(p.groupby("species").body_mass_g.agg(["mean", "count"]).round(3).to_dict())
print("overall mass mean", round(p.body_mass_g.mean(), 3), p.body_mass_g.count())
q = p.dropna(subset=["bill_length_mm", "bill_depth_mm"])
print("pooled r", stats.pearsonr(q.bill_length_mm, q.bill_depth_mm))
for s, g in q.groupby("species"):
    print("  within", s, stats.pearsonr(g.bill_length_mm, g.bill_depth_mm))
raw = pd.read_csv(D / "penguins/penguins_raw.csv")
print("raw", raw.shape, raw["Individual ID"].nunique(), raw["Sample Number"].nunique())

f = pd.read_csv(D / "nycflights13/flights.csv")
print("flights", f.shape, "dep_time NA", f.dep_time.isna().sum(), "dep_delay NA", f.dep_delay.isna().sum())
print(f.origin.value_counts().to_dict())
print("mean dep_delay", f.dep_delay.mean(), "mean arr_delay", f.arr_delay.mean())
planes = pd.read_csv(D / "nycflights13/planes.csv")
air = pd.read_csv(D / "nycflights13/airports.csv")
al = pd.read_csv(D / "nycflights13/airlines.csv")
w = pd.read_csv(D / "nycflights13/weather.csv")
tn = f.tailnum.dropna()
print(
    "tailnum not in planes: flights",
    (~tn.isin(planes.tailnum)).sum(),
    "distinct",
    tn[~tn.isin(planes.tailnum)].nunique(),
)
print("dest not in airports", sorted(set(f.dest) - set(air.faa)), (~f.dest.isin(air.faa)).sum())
print("origin not in airports", sorted(set(f.origin) - set(air.faa)))
print("carrier not in airlines", sorted(set(f.carrier) - set(al.carrier)))
print(
    "planes tailnum unique",
    planes.tailnum.is_unique,
    "airports faa unique",
    air.faa.is_unique,
    "airlines unique",
    al.carrier.is_unique,
)
print(
    "weather (origin,time_hour) unique",
    not w.duplicated(["origin", "time_hour"]).any(),
    w.duplicated(["origin", "time_hour"]).sum(),
)
hours = w[["origin", "time_hour"]].drop_duplicates()
fk = f[["origin", "time_hour"]].merge(hours, how="left", indicator=True)
print("flights without weather match", (fk._merge == "left_only").sum())

b = pd.read_csv(D / "lahman/Batting.csv")
pe = pd.read_csv(D / "lahman/People.csv")
t = pd.read_csv(D / "lahman/Teams.csv")
s = pd.read_csv(D / "lahman/Salaries.csv")
print(
    "batting",
    b.shape,
    "max year",
    b.yearID.max(),
    "people",
    pe.shape,
    "teams",
    t.shape,
    "salaries",
    s.shape,
    s.yearID.min(),
    s.yearID.max(),
)
bonds = pe[(pe.nameFirst == "Barry") & (pe.nameLast == "Bonds")]
print(bonds[["playerID", "nameGiven", "birthYear"]].to_dict("records"))
print("bonds 2001 HR", b[(b.playerID == "bondsba01") & (b.yearID == 2001)].HR.sum())
season = b.groupby(["playerID", "yearID"]).HR.sum().sort_values(ascending=False)
print("top seasons", season.head(3).to_dict())
print(
    "batting rows with stint>1",
    (b.stint > 1).sum(),
    "player-years",
    b.groupby(["playerID", "yearID"]).ngroups,
)
print(
    "people playerID unique",
    pe.playerID.is_unique,
    "batting players not in people",
    (~b.playerID.isin(pe.playerID)).sum(),
)
print(
    "teams (teamID,yearID) unique",
    not t.duplicated(["teamID", "yearID"]).any(),
    "teamID alone unique",
    t.teamID.is_unique,
)

ss = pd.read_csv(D / "sleepstudy/sleepstudy.csv")
print("sleep", ss.shape, ss.Subject.nunique(), ss.Days.unique())

m = smf.mixedlm("Reaction ~ Days", ss, groups=ss["Subject"], re_formula="~Days").fit(reml=True)
print("mixed slope", m.params["Days"], m.bse["Days"], m.pvalues["Days"])
ols = smf.ols("Reaction ~ Days", ss).fit()
print("ols slope", ols.params["Days"], ols.bse["Days"], ols.pvalues["Days"])
slopes = ss.groupby("Subject").apply(lambda g: np.polyfit(g.Days, g.Reaction, 1)[0])
print("per-subject slope mean", slopes.mean(), stats.ttest_1samp(slopes, 0))

o = pd.read_csv(D / "owid-co2/owid-co2-data.csv")
print(
    "owid",
    o.shape,
    "no iso rows",
    o.iso_code.isna().sum(),
    "countries w/o iso",
    o[o.iso_code.isna()].country.nunique(),
)
print(o[o.iso_code.isna()].country.drop_duplicates().head(40).tolist())
y = o[o.year == 2022]
print(
    "2022 World co2",
    y[y.country == "World"].co2.tolist(),
    "sum all rows",
    y.co2.sum(),
    "sum iso rows",
    y[y.iso_code.notna()].co2.sum(),
    "sum iso no OWID_",
    y[y.iso_code.notna() & ~y.iso_code.fillna('').str.startswith('OWID')].co2.sum(),
)
print("max year", o.year.max(), o[o.country=="World"].dropna(subset=["co2"]).year.max())

pk = pd.read_csv(D / "parkinsons/parkinsons_updrs.data")
print("parkinsons", pk.shape, pk["subject#"].nunique(), pk.groupby("subject#").sex.nunique().max())
mm = smf.mixedlm("Q('Jitter(%)') ~ C(sex)", pk, groups=pk["subject#"]).fit(reml=True)
print("mixed jitter~sex", mm.params.filter(like="sex").tolist(), mm.pvalues.filter(like="sex").tolist())
per = pk.groupby("subject#").agg(sex=("sex", "first"), j=("Jitter(%)", "mean"))
print("subject-mean welch", stats.ttest_ind(per[per.sex == 0].j, per[per.sex == 1].j, equal_var=False))
print(
    "naive welch rows",
    stats.ttest_ind(pk[pk.sex == 0]["Jitter(%)"], pk[pk.sex == 1]["Jitter(%)"], equal_var=False),
)
