# Documentation Sources

Single inventory of the official documentation this project relies on. Any best-practice
claim written into `PROJECT_ISSUES.md`, `LEARNINGS.md`, notebook markdown cells, or code
comments must trace back to a link in **Official documentation** below — never to a
tutorial alone. Tutorials (CampusX, Baker, StatQuest) are fine for building intuition
first, per the "one lesson first, then docs" rule already in `PROJECT_ISSUES.md`, but they
never substitute for a doc citation when a methodology decision is being justified.

## How to use this file

- Before writing a claim about "correct" usage of a library, find the relevant link below
  and cite that, not a tutorial or course note.
- If the page you need isn't listed, add it here first, then cite it elsewhere — this file
  should stay the complete inventory, not a partial one.
- If a tutorial and a doc page below disagree, the doc wins (existing rule, see
  `PROJECT_ISSUES.md` "Source rule").
- `scikit-learn` is unpinned in `requirements.txt`, so these links point at *current stable*
  docs. Once the version gets pinned (Issue 11), re-check links still match that version.

## Official documentation (authoritative — cite these)

### pandas
- [`read_csv`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) — loading the interim CSV
- [`to_datetime`](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html) — parsing `FlightDate`
- [Time series / date functionality user guide](https://pandas.pydata.org/docs/user_guide/timeseries.html) — date-based filtering for the temporal split

### scikit-learn
- [Cross-validation user guide](https://scikit-learn.org/stable/modules/cross_validation.html) (includes the `TimeSeriesSplit` section) — source of "a test set should still be held out for final evaluation, but the validation set is no longer needed when doing CV"
- [`TimeSeriesSplit` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) — note the "samples must be equally spaced" requirement, which is why Issue 2 splits the date array rather than the rows
- [`cross_val_score`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) / [`cross_validate`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html) — scoring a `Pipeline` across the folds
- [`train_test_split` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) — note its `shuffle=True` default, the exact pitfall Issue 2 exists to avoid
- [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html) (data leakage, "never fit on test")
- [`OneHotEncoder` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [`ColumnTransformer` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [Preprocessing data user guide](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Pipelines and composite estimators user guide](https://scikit-learn.org/stable/modules/compose.html)
- [Linear Models user guide](https://scikit-learn.org/stable/modules/linear_model.html) (§1.1.1 Ordinary Least Squares)
- [Time-related feature engineering example](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html) — why raw HHMM-style ordinal time features hurt linear models but not tree models, and the periodic/cyclical encoding fix
- [Ensembles user guide](https://scikit-learn.org/stable/modules/ensemble.html)
- [`RandomForestRegressor` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Model evaluation: metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html) (MAE, RMSE)
- [Model persistence](https://scikit-learn.org/stable/model_persistence.html) (security/version-compatibility warnings)

### numpy
Declared in `requirements.txt`, not yet imported anywhere in the repo.
- [NumPy documentation home](https://numpy.org/doc/stable/) — add specific pages here once numeric code beyond pandas is written

### matplotlib
Declared in `requirements.txt`, not yet imported anywhere in the repo.
- [Matplotlib documentation home](https://matplotlib.org/stable/index.html)

### seaborn
Declared in `requirements.txt`, not yet imported anywhere in the repo.
- [Seaborn documentation home](https://seaborn.pydata.org/)

### Streamlit
Planned for Gate 3 (Issues 14-16); **not yet in `requirements.txt`** — add it when this work starts.
- [Get started](https://docs.streamlit.io/get-started)
- [Deploy overview](https://docs.streamlit.io/deploy)
- [Prep and deploy on Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
- [App dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)

### Data source (not software docs, but the authoritative reference for every raw column)
- [BTS/DOT TranStats — Reporting Carrier On-Time Performance](https://transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr) — field definitions for every column in the raw dataset; check here before asserting what a column means

## Teaching sources (background only — never the sole citation for a best-practice claim)

- CampusX — [100 Days of Machine Learning](https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH) playlist + [companion notebook repo](https://github.com/campusx-official/100-days-of-machine-learning)
- Baker, CMPT 353 course notes ([ggbaker.ca/data-science](https://ggbaker.ca/data-science/content/ml.html))
- StatQuest (YouTube)

Useful for meeting a concept for the first time. Not useful as the reason a methodology
decision was made — that reason must trace to a link in "Official documentation" above.
