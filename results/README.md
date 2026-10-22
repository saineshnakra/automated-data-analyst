# Results on 60 research datasets

We tested ADA on 60 lab and research datasets drawn at random from Kaggle,
in three rounds of 20. After each round we fixed what the reviews found and
checked every dataset again. The table below gives the verdicts on the first
run of each dataset and on the version of ADA in this repository now.

This page is about the quick first look: what `ada analyze` and the web app
say about a table nobody has described. It does not test `ada build`, which
works from a `study.yml` in which the researcher states the design and the
planned analyses.

| Round | Datasets | First run (good / partly useful / poor) | Now |
|---|---|---|---|
| 1 | 20 | 0 / 5 / 15 | 11 / 7 / 2 |
| 2 | 20 | 1 / 14 / 5 | 9 / 10 / 1 |
| 3 | 20 | 1 / 8 / 11 | 9 / 10 / 1 |
| All | 60 | 2 / 27 / 31 | 29 / 27 / 4 |

**Good** means a researcher could use the brief as it stands: right measure,
sound findings, and nothing misleading. **Partly useful** means it is right
in parts, but it leads with the wrong thing, misses something obvious, needs
an override to get there, or gets an answer wrong that a careful reader would
catch. **Poor** means it picks the
wrong column, leads with something circular, or would mislead someone who
trusted it.

The verdicts grade what a new user would see today, not how much a dataset
improved.

## How the datasets were picked

We searched Kaggle with 56 lab and research queries, such as "blood test",
"soil analysis", "gene expression", "concrete strength", "enzyme",
"water quality", "calibration" and "psychology experiment". That returned
3,120 datasets, or 1,786 once duplicates were removed. We shuffled that list
with a fixed seed (20261020) and went down it in order, keeping each dataset
that passed four checks until we had 60:

- it holds lab, field or experimental measurements, not business or web data;
- it has a CSV, TSV or Excel table with at least 30 rows, 3 columns and 2
  numeric measurement columns;
- the uploader doesn't describe it as synthetic or simulated;
- it isn't a copy of a dataset already kept.

We examined 140 datasets to keep 60.
[selection-log.csv](selection-log.csv) lists every one and why it was kept
or skipped. Nothing was picked because ADA did well or badly on it. The 60
went through in three rounds of 20, in the order they were drawn.

The reviews were done by AI agents, each told to act as a sceptical
scientist who knows that kind of data. For each dataset the reviewer:

- ran ADA on the main table;
- checked every figure in the headline and findings against pandas, SciPy or
  statsmodels;
- judged whether ADA had picked the right measure and grouping;
- asked it three to five questions a researcher would ask.

After each round we fixed what the reviews found, as general rules rather
than per-dataset patches, and re-ran ADA on every dataset so far. A final
review graded all 60 on the code in this repository. It went issue by issue
through the first review of each dataset, and every quoted number can be
reproduced by running ADA again.

## What changed because of them

Every change was a general rule, tested on small synthetic tables and
checked so that it didn't make any other dataset or the bundled samples
worse. The main ones:

- **Reading.**
  - Missing-value codes such as `-99`, `-999`, `ND` and `n/a` are set to
    missing. Below-detection and trace values (`<0.5`, `T`) are read as
    censored, with a note.
  - Title rows, two-row headers and duplicate headers are handled.
  - Decimal commas are decided per column, and numbers written with a spread
    and a unit (`5.66 ± 1.69 items/g`) are read.
  - Columns spelled two ways (`Hipanic`, `Hispanic`) are merged. Large Excel
    files read several times faster.
- **Choosing what to analyse.**
  - The measure is the outcome a researcher would care about: a reading
    rather than a set dose, the label column in a classification table, the
    column the file is named for.
  - Codes, IDs, coordinates, counters and experiment numbers are kept out of
    the comparisons.
  - Count matrices, long tables (one row per parameter) and files that stack
    several tables are recognised.
- **Statistics.**
  - Groups are compared with Welch's t-test or ANOVA, or rank tests for
    skewed readings, with effect sizes, and the pairs that differ are found
    with Tukey's HSD.
  - Rates are compared with χ² or Fisher's exact test.
  - p-values are adjusted for the number of tests run.
  - A joint least-squares model (or a logistic model for a yes/no outcome)
    is reported when it adds to single correlations.
  - Subjects measured more than once are compared within subjects.
  - Follow-up times get Kaplan-Meier survival, log-rank tests and Cox hazard
    ratios.
  - Long records get a Theil-Sen trend with a Mann-Kendall test that allows
    for autocorrelation.
- **Data checks.** These flag:
  - impossible values, from a column's name or a physical limit;
  - caps, floors and piles;
  - swapped and copied columns;
  - failed records and placeholder blocks;
  - signs that a table was simulated;
  - gaps that follow a group or another column.
- **Saying less when less is known.**
  - Findings that hold by construction are named instead of reported: a
    total against its parts, a band cut from a reading, a ratio against its
    numerator.
  - A gap within the noise is called no clear difference.
  - A forecast is refused when it would do no better than assuming no
    change.
- **Questions.**
  - Ask reads column names before question words.
  - It answers year ranges, seasons, rates of a yes/no outcome and
    comparisons that hold another reading fixed.
  - It uses the same rows and data checks as the report.

## What ADA still gets wrong

The calculations themselves matched pandas, SciPy and statsmodels wherever
the reviewers re-ran them. What is still wrong is what ADA chooses to lead
with and the caveats it leaves out. These are the problems the final reviews
found most often:

- **It can miss the comparison the study was designed for.** Without being
  told the design, ADA can lead with a true but minor finding:
  - cancer type rather than drug in a drug screen;
  - fish counts by region rather than by protection status per transect;
  - a grand mean of raw counts rather than autism against control in a
    microbiome table.

  Three of the four datasets still rated poor are of this kind.
- **Some findings still hold by construction.** Systolic against diastolic
  pressure, haemoglobin against haematocrit, a daily maximum against the
  daily mean: these are flagged in some tables and not in others.
- **Answers to questions don't always use the design the report uses.**
  - Survival questions can ignore censoring.
  - A repeated visit can be counted as a new reading.
  - An answer can run on a sample of a very large table while the report
    uses every row.
- **Pooling can mislead.**
  - Readings pooled across groups that cover different ranges (a dye's
    concentration, a set of stations that grows over time) can show a
    pattern no group has.
  - The joint model sometimes leaves out a real input as another possible
    outcome, or fits shares that sum to 100.
- **Some questions are refused.** Examples:
  - an interaction between two factors;
  - when a sensor first responded;
  - which month is highest when the months are columns.
- **One table at a time.** Datasets made of several related files are
  analysed one file at a time.
- **Speed.** Very large or wide tables take 12 to 46 seconds, against under
  a few seconds for most.

These are the limits of a first look at a table nobody has described. For
results you will report, describe the study with `ada init`. `ada build`
then fits the design and the plan you state, and runs nothing else.

## Every dataset

| Dataset | Field | Rows | First run | Now |
|---|---|---|---|---|
| [Human gut microbiome of the children with ASD](datasets/antaresnyc__human-gut-microbiome-with-asd.md) | biology | 5,619 | poor | poor |
| [Food Dyes RGB data for Digital Image Colorimetry](datasets/yozeftjandra__food-dyes-rgb-data-for-digital-image-colorimetry.md) | chemistry | 1,228 | poor | partly useful |
| [FI_Optimun_Frecuency](datasets/davidhguerrero__fi-optimun-frecuency.md) | sensor | 3,108,545 | poor | partly useful |
| [Adult NHANES 2011-2012](datasets/rehoyt__adult-nhanes-2011-2012.md) | clinical | 4,571 | poor | good |
| [SuperCam calibration dataset 20221218](datasets/erikkepes__supercam-calibration-dataset-20221218.md) | sensor | 1,139 | poor | poor |
| [Weather Conditions in World War Two](datasets/smid80__weatherww2.md) | environmental | 119,040 | poor | good |
| [Social Learning in House Sparrows](datasets/thedevastator__social-learning-in-house-sparrows.md) | behavioural | 351 | poor | good |
| [Soil Analysis Data at Colombia](datasets/pablorodrguezn__soil-analysis-data-at-colombia.md) | chemistry | 46,745 | partly useful | partly useful |
| [Compressive strength foamed/normal concrete](datasets/imtkaggleteam__compressive-strength-foamednormal-concrete.md) | materials | 1,133 | poor | good |
| [Air Quality Data - India](datasets/amandeepvasistha__air-quality-data.md) | environmental | 737,406 | partly useful | partly useful |
| [Heart Attack Dataset](datasets/fatemehmohammadinia__heart-attack-dataset-tarik-a-rashid.md) | clinical | 1,319 | partly useful | good |
| [Indoor Laboratory Fire Dataset](datasets/saurabhshahane__indoor-laboratory-fire-dataset.md) | sensor | 2,974 | poor | partly useful |
| [Infant Numeracy Experiment](datasets/junianlim__infant-numeracy-experiment.md) | behavioural | 30 | poor | good |
| [Crop Health and Environmental Stress Dataset](datasets/datasetengineer__crop-health-and-environmental-stress-dataset.md) | agriculture | 212,019 | poor | good |
| [Soil Compositions](datasets/ramakrishnanthiyagu__soil-compositions.md) | chemistry | 48 | partly useful | partly useful |
| [Workplace Risk-Taking: A 2×2 Behavioral Experiment](datasets/nikitasfilosofof__workplace-risk-taking-a-22-behavioral-experiment.md) | behavioural | 231 | poor | good |
| [Anemia Types Classification](datasets/ehababoelnaga__anemia-types-classification.md) | clinical | 1,281 | poor | good |
| [England Water Quality Sampling Dataset (2025)](datasets/krishnanshverma__england-water-quality.md) | environmental | 81,026 | poor | partly useful |
| [IAQ Baquba Hospital dataset – University of Diyala](datasets/haiderqassim__iaq-baquba-hospital-dataset-university-of-diyala.md) | environmental | 523,524 | partly useful | good |
| [Beach Weather Stations 🏖️🌊👓](datasets/sanjanchaudhari__beach-weather-stations.md) | environmental | 156,051 | poor | good |
| [🌤️ Weather Data Munich 1954-2022](datasets/mexwell__weather-data-munich-1954-2022.md) | environmental | 25,051 | poor | good |
| [Data for weld of 5052 Aluminium Alloy](datasets/nutanjain__data-for-weld-of-5052-aluminium-alloy.md) | materials | 31 | partly useful | partly useful |
| [Gene Expression in Brain Cancer](datasets/khushim10__gene-expression-in-brain-cancer.md) | biology | 54,675 | poor | good |
| [Smog City Diaries: Unveiling Air Quality Trends📈](datasets/tanisha1416__aqi-bangalore-2023.md) | environmental | 147 | partly useful | good |
| [Gene Expression Dataset](datasets/ziya07__gene-expression-dataset.md) | biology | 1,000 | partly useful | good |
| [Metagenomic Pipeline Diversity Analysis Phyloseq](datasets/mannekuntanagendra__metagenomic-pipeline-diversity-analysis-phyloseq.md) | biology | 2,000 | poor | poor |
| [Genomics of Drug Sensitivity in Cancer (GDSC)](datasets/samiraalipour__genomics-of-drug-sensitivity-in-cancer-gdsc.md) | biology | 242,035 | partly useful | partly useful |
| [Rocky Intertidal Data](datasets/ashsummers__rocky-intertidal-data.md) | environmental | 120 | partly useful | partly useful |
| [Reef Ecological Assessment: Del Norte](datasets/jocelyndumlao__reef-ecological-assessment-del-norte.md) | environmental | 2,431 | partly useful | partly useful |
| [vital signs of breast cancer](datasets/asmasalema__vital-signs-of-breast-cancer.md) | clinical | 2,435 | partly useful | partly useful |
| [Age Prediction Using Genomic Information](datasets/thedevastator__age-prediction-for-individuals-using-multi-omic.md) | biology | 104 | partly useful | partly useful |
| [Air Quality Index in Jakarta](datasets/senadu34__air-quality-index-in-jakarta-2010-2021.md) | environmental | 5,538 | partly useful | good |
| [Anemia Dataset](datasets/biswaranjanrao__anemia-dataset.md) | clinical | 1,421 | partly useful | good |
| [Microplastics: Food to Human Bloodstream](datasets/ahsanneural__microplastics-food-to-human-bloodstream.md) | environmental | 354 | poor | partly useful |
| [Soil Nutrient Dataset of Southern Indian States](datasets/manojk2002__soil-nutrient-dataset-of-southern-indian-states.md) | agriculture | 875 | partly useful | good |
| [Steel Strength Data 2018](datasets/fuarresvij__steel-test-data.md) | materials | 312 | partly useful | good |
| [Air Quality from Multiple Districts of Bangladesh.](datasets/mdshobuj__air-quality-dataset-of-bangladesh.md) | sensor | 4,566 | partly useful | partly useful |
| [Ruminant Microbiome Dataset — Pakistan](datasets/krishnanivja__ruminant-microbiome-dataset-pakistan.md) | biology | 10,000 | partly useful | partly useful |
| [Harvest Time_Traits_Yield](datasets/willianoliveiragibin__harvest-time-traits-yield.md) | agriculture | 107 | poor | partly useful |
| [Corn Crop Growth](datasets/miguelh65__corn-crop-growth.md) | agriculture | 1,000 | good | good |
| [abitlity to fast ramadan](datasets/raidathmane__abitlity-to-fast-ramadan.md) | clinical | 104,125 | partly useful | good |
| [Chronic KIdney Disease dataset](datasets/mansoordaku__ckdisease.md) | clinical | 400 | good | good |
| [NASA Milling Dataset](datasets/vinayak123tyagi__milling-data-set-prognostic-data.md) | sensor | 167 | poor | partly useful |
| [Self compacting concrete dataset](datasets/gaming919__self-compacting-concrete-dataset.md) | materials | 555 | partly useful | partly useful |
| [Landslide dataset](datasets/rajumavinmar__landslide-dataset.md) | environmental | 2,000 | poor | partly useful |
| [Heart Attack](datasets/pritsheta__heart-attack.md) | clinical | 303 | partly useful | good |
| [Untargeted LC-MS/MS](datasets/willianoliveiragibin__untargeted-lc-msms.md) | chemistry | 1,230 | poor | poor |
| [Ivermectin's Influence on Gut Microbiota](datasets/kirbysasuke__ivermectin-impacts-on-gut-microbial-ecosystem.md) | biology | 198 | partly useful | partly useful |
| [Air Handler Sound Levels](datasets/mikeboucher__air-handler-sound-levels.md) | sensor | 178,280 | poor | good |
| [High Entropy Alloys - Properties](datasets/sethpointaverage__high-entropy-alloys-properties.md) | materials | 1,545 | partly useful | partly useful |
| [AQI India](datasets/yashdogra__aqi-india.md) | environmental | 3,327 | poor | partly useful |
| [Temperature Time-Series for some Brazilian cities](datasets/volpatto__temperature-timeseries-for-some-brazilian-cities.md) | environmental | 110 | partly useful | good |
| [Secchi Depth](datasets/jacobsharples__secchi-depth.md) | environmental | 404 | poor | good |
| [🧬 Gene Expression - Bioinformatics 🗃️ Dataset](datasets/samira1992__gene-expression-bioinformatics-dataset.md) | biology | 4,381 | poor | good |
| [Cancer Statistics in US States](datasets/nancyalaswad90__cancer-statistics-in-us-states.md) | biology | 272 | poor | partly useful |
| [Disease Recovery PLOS paper  😷🙄](datasets/qusaybtoush1990__disease-recovery-plos-paper.md) | biology | 168 | poor | partly useful |
| [Endangered fishes dataset](datasets/harshithgupta__endangered-fish-data.md) | environmental | 42,336 | poor | partly useful |
| [Enhanced Gait Biomechanics Dataset](datasets/anitarostami__enhanced-gait-biomechanics-dataset.md) | sensor | 178,200 | partly useful | good |
| [Prediction of Mohs Hardness with Machine Learning](datasets/jocelyndumlao__prediction-of-mohs-hardness-with-machine-learning.md) | materials | 622 | poor | partly useful |
| [Water Quality Data 🌊](datasets/supriyoain__water-quality-data.md) | environmental | 2,371 | partly useful | good |

## Reproducing

The data is not stored in this repository. Each dataset's page links to it
on Kaggle and gives its licence. [datasets.csv](datasets.csv) lists the
file and worksheet ADA read for each one, so after downloading:

```bash
pip install automated-data-analyst
ada analyze "path/to/the/main/file.csv"
ada analyze "path/to/workbook.xlsx" --sheet "Data"
```

Running ADA again reproduces every figure the reviews quote. The verdicts
are judgement calls about those figures.
