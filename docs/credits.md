# Credits

ADA is a thin layer of rules and wording over open-source scientific Python.
The libraries below do the computing; ADA decides what to compute, checks it
and says it in plain words. Each calculation line in the brief names the
method and the library function behind the figure, so a reader can reproduce
it. If you publish results from ADA, please cite the libraries that produced
them.

ADA is MIT-licensed and depends only on permissively licensed libraries.

## The analysis package

| Library | Licence | What ADA uses it for |
|---|---|---|
| [pandas](https://pandas.pydata.org) | BSD-3-Clause | Reading CSV and Excel files, cleaning, grouping and every aggregation |
| [NumPy](https://numpy.org) | BSD-3-Clause | The array arithmetic behind the vectorised group tests, correlations and trendlines |
| [SciPy](https://scipy.org) | BSD-3-Clause | The t, F, chi-squared and normal distributions behind every p-value; `scipy.stats.chi2_contingency` and `fisher_exact` for rate comparisons; `contingency.association` for Cramér's V between two groupings; `studentized_range` for Tukey's HSD on the pairs of groups that differ; `ttest_rel`, `wilcoxon` and `friedmanchisquare` for groups that hold the same subjects; `theilslopes` and `kendalltau` for trend intervals and the Mann-Kendall test; ranks; `kstwo`, `kurtosis` and `binom` for the data checks and the outlier counts |
| [statsmodels](https://www.statsmodels.org) | BSD-3-Clause | The Benjamini-Hochberg false-discovery-rate adjustment (`multipletests`); Newcombe's interval for a difference in rates (`confint_proportions_2indep`); ordinary least squares for the joint model, for what readings add within a grouping's levels and for a comparison of groups with other readings held fixed (with its F test, `f_test`), with `variance_inflation_factor` for collinearity; logistic regression (`Logit`) for a yes/no measure; the repeated-measures ANOVA follows `AnovaRM`, whose figures it reproduces for every reading at once; Kaplan-Meier survival (`SurvfuncRight`), the log-rank test (`survdiff`) and the Cox model (`PHReg`) for follow-up times |
| [ruptures](https://centre-borelli.github.io/ruptures-docs/) | BSD-2-Clause | Binary segmentation (`Binseg`) to find a lasting change in the level of the measure's period series |
| [python-calamine](https://github.com/dimastbk/python-calamine) | MIT | Reading `.xlsx` and `.xlsm` workbooks through pandas, built on the Rust [calamine](https://github.com/tafia/calamine) crate (MIT) |
| [Matplotlib](https://matplotlib.org) | Matplotlib licence (PSF-based) | The paper figures from `ada build` |
| [python-docx](https://python-docx.readthedocs.io) | MIT | The Word tables, methods draft and supplement from `ada build` |
| [PyYAML](https://pyyaml.org) | MIT | Reading and writing `study.yml` |

## The app

| Library | Licence | What ADA uses it for |
|---|---|---|
| [Streamlit](https://streamlit.io) | Apache-2.0 | The web app in `app.py` and `ada/ui.py` |
| [Plotly](https://plotly.com/python/) | MIT | The charts in the app |

The optional AI layer uses the [OpenAI Python library](https://github.com/openai/openai-python)
(Apache-2.0) and [Pydantic](https://docs.pydantic.dev) (MIT).

## How to cite them

- **NumPy:** Harris, C. R., Millman, K. J., van der Walt, S. J. et al. (2020).
  Array programming with NumPy. *Nature* 585, 357–362.
  https://doi.org/10.1038/s41586-020-2649-2
- **SciPy:** Virtanen, P., Gommers, R., Oliphant, T. E. et al. (2020). SciPy
  1.0: fundamental algorithms for scientific computing in Python. *Nature
  Methods* 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2
- **pandas:** McKinney, W. (2010). Data structures for statistical computing
  in Python. *Proceedings of the 9th Python in Science Conference*, 56–61.
  https://doi.org/10.25080/Majora-92bf1922-00a; and The pandas development
  team, pandas-dev/pandas: Pandas, Zenodo.
  https://doi.org/10.5281/zenodo.3509134
- **statsmodels:** Seabold, S. and Perktold, J. (2010). statsmodels:
  Econometric and statistical modeling with Python. *Proceedings of the 9th
  Python in Science Conference*, 92–96.
  https://doi.org/10.25080/Majora-92bf1922-011
- **ruptures:** Truong, C., Oudre, L. and Vayatis, N. (2020). Selective review
  of offline change point detection methods. *Signal Processing* 167, 107299.
  https://doi.org/10.1016/j.sigpro.2019.107299

## Methods

The methods ADA applies, with the papers that introduced them:

| Method | Where | Reference |
|---|---|---|
| Welch's t-test and ANOVA | Group comparisons | Welch, B. L. (1947), *Biometrika* 34, 28–35; Welch, B. L. (1951), *Biometrika* 38, 330–336 |
| Mann-Whitney U test | Two groups compared on ranks | Mann, H. B. and Whitney, D. R. (1947), *Annals of Mathematical Statistics* 18, 50–60 |
| Kruskal-Wallis test | Three or more groups compared on ranks | Kruskal, W. H. and Wallis, W. A. (1952), *JASA* 47, 583–621 |
| Tukey's honestly significant difference | Which pairs of groups differ | Tukey, J. W. (1949), *Biometrics* 5, 99–114 |
| Paired t-test | Two groups holding the same subjects | Student (1908), *Biometrika* 6, 1–25 |
| Wilcoxon signed-rank test | Two groups holding the same subjects, on ranks | Wilcoxon, F. (1945), *Biometrics Bulletin* 1, 80–83 |
| Repeated-measures ANOVA | Three or more groups holding the same subjects | Fisher, R. A. (1925), *Statistical Methods for Research Workers*, Oliver and Boyd |
| Friedman test | Three or more groups holding the same subjects, on ranks | Friedman, M. (1937), *JASA* 32, 675–701 |
| Kendall's W | The effect size of the Friedman test | Kendall, M. G. and Babington Smith, B. (1939), *Annals of Mathematical Statistics* 10, 275–287 |
| Hedges' g | Group comparisons | Hedges, L. V. (1981), *Journal of Educational Statistics* 6, 107–128 |
| Spearman's rank correlation | Relationships, trend tests across ordered levels | Spearman, C. (1904), *American Journal of Psychology* 15, 72–101 |
| Fisher's z-transform | Confidence intervals for r | Fisher, R. A. (1915), *Biometrika* 10, 507–521 |
| Pearson's chi-squared test | Rate comparisons | Pearson, K. (1900), *Philosophical Magazine* 50, 157–175 |
| Cramér's V | Rate comparisons | Cramér, H. (1946), *Mathematical Methods of Statistics*, Princeton University Press |
| Holm's adjustment | Scans of many comparisons | Holm, S. (1979), *Scandinavian Journal of Statistics* 6, 65–70 |
| Benjamini-Hochberg false discovery rate | Screens of hundreds of readings | Benjamini, Y. and Hochberg, Y. (1995), *Journal of the Royal Statistical Society B* 57, 289–300 |
| Newcombe's interval for a difference in proportions | Rate comparisons | Newcombe, R. G. (1998), *Statistics in Medicine* 17, 873–890 |
| Inverse-transform sampling | Synthetic demo data: `numpy.quantile` (linear interpolation) of uniform draws for numbers, `numpy.random.Generator.permutation` for categories | Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, chapter 2 |
| ISO 7064 MOD 11-2 check digit | Checking authors' ORCID iDs for the repository files | ISO 7064:2003; ORCID, *Structure of the ORCID Identifier* |
| Citation File Format 1.2.0 | `repository/CITATION.cff` | Druskat, S., Spaaks, J. H., Chue Hong, N. et al. (2021), https://doi.org/10.5281/zenodo.5171937 |
| Binary segmentation | Changes in level | Scott, A. J. and Knott, M. (1974), *Biometrics* 30, 507–512 |
| Bias of the lag-1 autocorrelation | Changes in level | Kendall, M. G. (1954), *Biometrika* 41, 403–404 |
| Variance inflation factor | The joint model | Marquardt, D. W. (1970), *Technometrics* 12, 591–612 |
| Logistic regression and McFadden's pseudo R² | The model of a yes/no measure | Cox, D. R. (1958), *Journal of the Royal Statistical Society B* 20, 215–242; McFadden, D. (1974), in *Frontiers in Econometrics*, Academic Press, 105–142 |
| Fences at 1.5 × IQR | Exceptional records | Tukey, J. W. (1977), *Exploratory Data Analysis*, Addison-Wesley |
| Kolmogorov-Smirnov distance | Data checks: columns that look simulated | Kolmogorov, A. N. (1933), *Giornale dell'Istituto Italiano degli Attuari* 4, 83–91; Smirnov, N. (1948), *Annals of Mathematical Statistics* 19, 279–281 |
| Otsu's threshold | Splitting a reading with two modes | Otsu, N. (1979), *IEEE Transactions on Systems, Man, and Cybernetics* 9, 62–66 |
| Ashman's D | How far apart two modes are | Ashman, K. M., Bird, C. M. and Zepf, S. E. (1994), *Astronomical Journal* 108, 2348–2361 |
| Kaplan-Meier estimator | Survival from censored follow-up times | Kaplan, E. L. and Meier, P. (1958), *JASA* 53, 457–481 |
| Log-rank test | Survival compared across groups | Mantel, N. (1966), *Cancer Chemotherapy Reports* 50, 163–170; Peto, R. and Peto, J. (1972), *JRSS A* 135, 185–207 |
| Cox proportional-hazards model | Hazard ratios for survival | Cox, D. R. (1972), *JRSS B* 34, 187–220 |
| Theil-Sen slope | Trendlines, anomalies, forecasts, trend tests (`scipy.stats.theilslopes` for the interval) | Theil, H. (1950), *Indagationes Mathematicae* 12; Sen, P. K. (1968), *JASA* 63, 1379–1389 |
| Mann-Kendall trend test | Long-run trends and trend answers (`scipy.stats.kendalltau`) | Mann, H. B. (1945), *Econometrica* 13, 245–259; Kendall, M. G. (1975), *Rank Correlation Methods*, 4th ed., Griffin |
| Effective sample size for autocorrelated trends | Trend tests | Yue, S. and Wang, C. Y. (2004), *Water Resources Management* 18, 201–218 |
| MASE against a seasonal no-change forecast | Forecast backtests | Hyndman, R. J. and Koehler, A. B. (2006), *International Journal of Forecasting* 22, 679–688 |

## Paper figures

| Library or method | Licence or reference | What ADA uses it for |
|---|---|---|
| [Matplotlib](https://matplotlib.org) | Matplotlib licence (PSF-based) | Drawing every figure and writing PDF, SVG, PNG and EPS; Hunter, J. D. (2007), *Computing in Science & Engineering* 9, 90–95 |
| [Pillow](https://python-pillow.org) | MIT-CMU | Writing TIFF files as RGB with LZW compression, through Matplotlib's own dependency |
| Okabe–Ito colours | Okabe, M. and Ito, K. (2008), *Color Universal Design*, https://jfly.uni-koeln.de/color/ | The group colours, in an order that keeps neighbours apart for colour-blind readers |
| van der Corput sequence | van der Corput, J. G. (1935), *Proceedings of the Koninklijke Akademie van Wetenschappen te Amsterdam* 38, 813–821 | Spreading the points of each group sideways the same way at every build, scaled by `scipy.stats.gaussian_kde` |
| Normal Q–Q plot | Wilk, M. B. and Gnanadesikan, R. (1968), *Biometrika* 55, 1–17 | The model checks, through `scipy.stats.probplot` |
| t confidence interval | `scipy.stats.t.interval` with `scipy.stats.sem` | Means and their intervals when an analysis gives none |
| Confidence band for the mean | The linked model's coefficient covariance (the fixed-effect covariance for a mixed model); `statsmodels` OLS `get_prediction().summary_frame()` without a linked analysis | The band in the scatter figure |

## Checking a rebuild

`ada verify` compares each number in a rebuilt CSV file with the one on disk
with NumPy's `isclose`: they match when |rebuilt − disk| ≤ atol + rtol × |disk|
(by default rtol is 1e-9 and atol 1e-12). It finds the lines of text that
differ with Python's `difflib`, and reads the text of Word files with
[python-docx](https://python-docx.readthedocs.io) (MIT).

## The methods draft

The methods draft (`ada build`, in `methods/`) cites the libraries above that
the build used, and also:

- **Python:** Van Rossum, G. and Drake, F. L. (2009). *Python 3 Reference
  Manual*. CreateSpace.
- **Matplotlib** (figures): Hunter, J. D. (2007). Matplotlib: a 2D graphics
  environment. *Computing in Science & Engineering* 9, 90–95.
  https://doi.org/10.1109/MCSE.2007.55
- **The Bonferroni correction**, with `multiple_comparisons: bonferroni`:
  Dunn, O. J. (1961). Multiple comparisons among means. *Journal of the
  American Statistical Association* 56, 52–64.
  https://doi.org/10.1080/01621459.1961.10482090
- **Holm's method** and **the Benjamini-Hochberg procedure**, with `holm` or
  `fdr_bh`: the references in the table above.

[python-docx](https://python-docx.readthedocs.io) (MIT) writes the Word files.

## The analyses behind the paper assets

`ada build` fits each analysis in `study.yml` with these methods. The model
output in `supplement/models/` names the library call behind every number.

| Method | Library function | Reference |
|---|---|---|
| Welch's t-test, with its interval | `scipy.stats.ttest_ind(equal_var=False)` | Welch, B. L. (1947), *Biometrika* 34, 28–35 |
| Welch's ANOVA | `statsmodels.stats.oneway.anova_oneway(use_var="unequal")` | Welch, B. L. (1951), *Biometrika* 38, 330–336 |
| Games–Howell pairwise contrasts | `scipy.stats.studentized_range` | Games, P. A. and Howell, J. F. (1976), *Journal of Educational Statistics* 1, 113–125 |
| Tukey–Kramer pairwise contrasts (`test: anova`) | `scipy.stats.tukey_hsd` | Tukey, J. W. (1949), *Biometrics* 5, 99–114; Kramer, C. Y. (1956), *Biometrics* 12, 307–310 |
| Hedges' g with an exact small-sample correction | `scipy.special.gammaln` | Hedges, L. V. (1981), *Journal of Educational Statistics* 6, 107–128; Hedges, L. V. and Olkin, I. (1985), *Statistical Methods for Meta-Analysis* |
| Noncentral-t intervals for g and Cohen's dz | `scipy.stats.nct`, `scipy.optimize.brentq` | Steiger, J. H. and Fouladi, R. T. (1997), in *What If There Were No Significance Tests?*, 221–257; Cohen, J. (1988), *Statistical Power Analysis for the Behavioral Sciences* |
| ω² and partial ω² with a noncentral-F interval | `statsmodels.stats.oneway.confint_noncentrality` | Olejnik, S. and Algina, J. (2003), *Psychological Methods* 8, 434–447; Steiger, J. H. (2004), *Psychological Methods* 9, 164–182 |
| Paired t-test | `scipy.stats.ttest_rel` | Student (1908), *Biometrika* 6, 1–25 |
| Mann–Whitney, Kruskal–Wallis and Wilcoxon signed-rank tests | `scipy.stats.mannwhitneyu`, `kruskal`, `wilcoxon` | Mann and Whitney (1947); Kruskal and Wallis (1952); Wilcoxon, F. (1945), *Biometrics Bulletin* 1, 80–83 |
| Hodges–Lehmann estimates with intervals from inverting the rank test | the SciPy tests above | Hodges, J. L. and Lehmann, E. L. (1963), *Annals of Mathematical Statistics* 34, 598–611; Bauer, D. F. (1972), *JASA* 67, 687–690 |
| Linear regression, F-tests of model terms | `statsmodels.formula.api.ols` | |
| Linear mixed models with a random intercept, fitted by REML | `statsmodels.formula.api.mixedlm` | Laird, N. M. and Ware, J. H. (1982), *Biometrics* 38, 963–974; Patterson, H. D. and Thompson, R. (1971), *Biometrika* 58, 545–554 |
| Intraclass correlation from the variance components | | Snijders, T. A. B. and Bosker, R. J. (2012), *Multilevel Analysis* (2nd ed.) |
| Pearson's r with a Fisher z interval; Spearman's ρ | `scipy.stats.pearsonr`, `spearmanr` | Pearson, K. (1896), *Phil. Trans. R. Soc. A* 187, 253–318; Fisher (1915); Spearman (1904) |
| Risk difference with Newcombe's interval | `statsmodels.stats.proportion.confint_proportions_2indep(method="newcomb")` | Newcombe, R. G. (1998), *Statistics in Medicine* 17, 873–890 |
| Odds ratio with Woolf's interval | `statsmodels.stats.contingency_tables.Table2x2` | Woolf, B. (1955), *Annals of Human Genetics* 19, 251–253 |
| χ² and Fisher's exact tests (Freeman–Halton by seeded Monte Carlo for larger tables) | `scipy.stats.chi2_contingency`, `fisher_exact` | Pearson (1900); Fisher, R. A. (1922), *JRSS* 85, 87–94; Freeman, G. H. and Halton, J. H. (1951), *Biometrika* 38, 141–149 |
| Wilson intervals for proportions | `statsmodels.stats.proportion.proportion_confint(method="wilson")` | Wilson, E. B. (1927), *JASA* 22, 209–212 |
| Logistic regression | `statsmodels.formula.api.logit` | Cox, D. R. (1958), *JRSS B* 20, 215–242 |
| Holm, Bonferroni and Benjamini–Hochberg adjustment | `statsmodels.stats.multitest.multipletests` | Holm, S. (1979), *Scandinavian Journal of Statistics* 6, 65–70; Dunn, O. J. (1961), *JASA* 56, 52–64; Benjamini and Hochberg (1995) |
| Shapiro–Wilk, Levene (median-centred) and Breusch–Pagan tests; variance inflation factors | `scipy.stats.shapiro`, `levene`; `statsmodels.stats.diagnostic.het_breuschpagan`, `variance_inflation_factor` | Shapiro and Wilk (1965); Brown and Forsythe (1974); Breusch and Pagan (1979); Marquardt (1970) |
| Durbin–Watson statistic of the residuals in time order | NumPy | Durbin and Watson (1950) |
