import sys
import warnings

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.simplefilter("ignore")
pk = pd.read_csv(sys.argv[1])
def plain(column):
    return column.replace("(", "_").replace(")", "").replace(":", "_").replace("%", "pct").replace("#", "")


pk = pk.rename(columns=plain)
for col in ["Jitter_pct", "Jitter_Abs", "Jitter_RAP", "Shimmer", "HNR", "total_UPDRS", "motor_UPDRS"]:
    a, b = pk[pk.sex == 0][col], pk[pk.sex == 1][col]
    welch = stats.ttest_ind(a, b, equal_var=False).pvalue
    mw = stats.mannwhitneyu(a, b).pvalue
    ols = smf.ols(f"{col} ~ C(sex)", pk).fit().pvalues.iloc[1]
    mixed = smf.mixedlm(f"{col} ~ C(sex)", pk, groups=pk["subject"]).fit(reml=True).pvalues.iloc[1]
    print(f"{col:12s} welch={welch:.3g} mannwhitney={mw:.3g} ols={ols:.3g} mixed={mixed:.3g}")
