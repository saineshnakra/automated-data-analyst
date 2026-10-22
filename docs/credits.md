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
| [SciPy](https://scipy.org) | BSD-3-Clause | The t, F, chi-squared and normal distributions behind every p-value; `scipy.stats.chi2_contingency` and `fisher_exact` for rate comparisons; `contingency.association` for Cramér's V between two groupings; `studentized_range` for Tukey's HSD on the pairs of groups that differ; `ttest_rel`, `wilcoxon` and `friedmanchisquare` for groups that hold the same subjects; ranks; `kstwo`, `kurtosis` and `binom` for the data checks and the outlier counts |
| [statsmodels](https://www.statsmodels.org) | BSD-3-Clause | The Benjamini-Hochberg false-discovery-rate adjustment (`multipletests`); Newcombe's interval for a difference in rates (`confint_proportions_2indep`); ordinary least squares for the joint model, for what readings add within a grouping's levels and for a comparison of groups with other readings held fixed (with its F test, `f_test`), with `variance_inflation_factor` for collinearity; logistic regression (`Logit`) for a yes/no measure; the repeated-measures ANOVA follows `AnovaRM`, whose figures it reproduces for every reading at once |
| [ruptures](https://centre-borelli.github.io/ruptures-docs/) | BSD-2-Clause | Binary segmentation (`Binseg`) to find a lasting change in the level of the measure's period series |
| [python-calamine](https://github.com/dimastbk/python-calamine) | MIT | Reading `.xlsx` and `.xlsm` workbooks through pandas, built on the Rust [calamine](https://github.com/tafia/calamine) crate (MIT) |

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
| Binary segmentation | Changes in level | Scott, A. J. and Knott, M. (1974), *Biometrics* 30, 507–512 |
| Bias of the lag-1 autocorrelation | Changes in level | Kendall, M. G. (1954), *Biometrika* 41, 403–404 |
| Variance inflation factor | The joint model | Marquardt, D. W. (1970), *Technometrics* 12, 591–612 |
| Logistic regression and McFadden's pseudo R² | The model of a yes/no measure | Cox, D. R. (1958), *Journal of the Royal Statistical Society B* 20, 215–242; McFadden, D. (1974), in *Frontiers in Econometrics*, Academic Press, 105–142 |
| Fences at 1.5 × IQR | Exceptional records | Tukey, J. W. (1977), *Exploratory Data Analysis*, Addison-Wesley |
| Kolmogorov-Smirnov distance | Data checks: columns that look simulated | Kolmogorov, A. N. (1933), *Giornale dell'Istituto Italiano degli Attuari* 4, 83–91; Smirnov, N. (1948), *Annals of Mathematical Statistics* 19, 279–281 |
| Theil-Sen slope | Trendlines, anomalies, forecasts | Theil, H. (1950), *Indagationes Mathematicae* 12; Sen, P. K. (1968), *JASA* 63, 1379–1389 |
