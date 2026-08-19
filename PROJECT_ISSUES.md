# Project 1 — Issue Backlog (Gate 1 → Gate 3)

Covers the Seattle Two-Flight Reliability Comparison Tool from the outstanding linear
baseline through deployment on **August 30**. Nothing past September 1 is in scope.

Each issue below is written to be pasted into GitHub as a separate issue. Suggested
labels are given per issue. **Learn this first** lists external sources only — docs,
videos, and the CMPT 353 decks already on disk.

**Gate dates:** Gate 1 (was Aug 7, now overdue) · Gate 2 — Aug 24 · Gate 3 — Aug 30

**Primary resource:** [CampusX — 100 Days of Machine Learning](https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH)
(134 videos), with its companion notebook repo at
[campusx-official/100-days-of-machine-learning](https://github.com/campusx-official/100-days-of-machine-learning).
CampusX is Python/sklearn-first, which suits the mechanics issues better than concept-first
sources. Two warnings: **playlist position ≠ day number** (playlist item 50 is Day 48), so
search the playlist by day number rather than scrolling to a position; and the series has
**no episode on temporal or time-aware splitting**, which is the single most important
evaluation decision in this project — that one stays on the sklearn docs.

**Trust rule:** where CampusX code disagrees with the current sklearn docs, the docs win.
The videos are good teaching material but the API has moved since they were recorded.

---

## How to read the source lists

Each issue's **Learn this first** block is split into **MUST** (read before starting that
issue) and **Optional** (only if you get stuck). Where MUST lists items in order, follow
the order: learn the concept from the lesson first, *then* open the API page to translate
it to your problem. API references are for keeping open while you code, not for learning a
technique cold.

Don't front-load. Read what the next issue actually needs, then go implement it.

---

## Milestone: Gate 1 — Data & honest baseline

### Issue 1 — Create the baseline notebook and load the clean dataset

**Labels:** `gate-1`, `setup`

Create `notebooks/linear_baseline.ipynb` and load the 324,490-row clean interim
dataset. The interim file deliberately retains all 110 columns; column selection was
deferred to this notebook.

**Done when**
- Notebook exists in `notebooks/` and runs top to bottom without manual cell reordering
- Clean CSV loads with the expected row count
- Date column parsed as a datetime dtype, not a string
- Modelling columns narrowed to the target plus the pre-booking features from the
  feature audit; the exclusion of post-flight columns is restated in a markdown cell

**Learn this first**

**MUST:** nothing new — `reports/feature_audit.md` already contains the
pre-booking/post-flight classification.

**Optional:** CampusX **Day 34 — Handling Date and Time Variables**
([notebook](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day34-handling-date-and-time)),
if datetime parsing needs a refresher.

---

### Issue 2 — Build the temporal holdout split

**Labels:** `gate-1`, `evaluation`

Split into train / validation / test by calendar date. Not by row position — a single
date must not be divided across two partitions.

**Done when**
- Split is defined by date boundaries, and those boundaries are stated in the notebook
- No calendar date appears in more than one partition (asserted in code, not assumed)
- Partitions are strictly ordered in time: train earliest, test latest
- Row counts and date ranges printed for each partition

**Learn this first**

**MUST**
- [sklearn — Cross-validation user guide](https://scikit-learn.org/stable/modules/cross_validation.html)
  — the TimeSeriesSplit section states the core argument: ordinary CV on time-ordered
  data trains on the future and evaluates on the past. Read it for the principle; you are
  building a single date-cut holdout, not rolling folds.

**Optional**
- [sklearn — TimeSeriesSplit reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- Baker, [Training and Validation](https://ggbaker.ca/data-science/content/ml.html#trainvalid)
  — see the warning below before using it

> **CampusX gap.** The 100 Days series covers train/test splitting generically but has no
> episode on time-aware splitting. This issue has no CampusX equivalent — use the docs.
>
> **Baker gap too.** Baker's [Training and Validation](https://ggbaker.ca/data-science/content/ml.html#trainvalid)
> section is worth reading for the overfitting argument, but he demonstrates
> `train_test_split` with its default random shuffle and never covers temporal splits.
> Read it for the reasoning, not the split call — copying that pattern here is the exact
> mistake this issue exists to prevent.

---

### Issue 3 — Build the naive per-profile baseline

**Labels:** `gate-1`, `evaluation`

Build the naive comparator the model has to beat: a per-profile historical median
computed from training data only.

**Done when**
- Baseline is fit on training rows only
- Profiles present in validation but absent from training are handled explicitly, and
  the fallback is documented
- Baseline produces a prediction for every validation row — no silent nulls
- Validation MAE recorded

**Learn this first**

**MUST:** nothing new. This is the groupby → median → merge → fillna pattern;
`pandas/pandas_numpy_practice.ipynb` (the BikeIndia set) covers the mechanics.

---

### Issue 4 — Encode categorical features

**Labels:** `gate-1`, `preprocessing`

One-hot encode the categorical pre-booking features. Encoding is fit on training data
only. One-hot was chosen over label encoding (imposes false ordinal structure on
airport codes) and target encoding (adds leakage risk) — that decision is closed.

**Done when**
- Encoder fit on train, applied to validation and test without refitting
- Categories appearing in validation or test but not in training do not crash the
  transform; the handling is a deliberate, documented choice
- Resulting feature count recorded — high-cardinality columns expand fast and this
  number matters for the write-up
- Fitted preprocessing object retained so it can later be incorporated into the serialized
  Pipeline (Issue 11)

**Learn this first**

**MUST — in this order**
1. CampusX **Day 27 — One Hot Encoding**
   ([notebook](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day27-one-hot-encoding)).
   Day 26 covers ordinal/label encoding if you want to see concretely why it was rejected here.
2. CampusX **Day 28 — ColumnTransformer**
   ([notebook](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day28-column-transformer))
   — the practical answer to applying different transforms to different column groups
3. [sklearn — `OneHotEncoder` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
   — the authority on current behaviour for categories not seen during fit. CampusX teaches
   the concept; this page governs the implementation.
4. [sklearn — Common pitfalls: data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
   — states the rule plainly: split first, `fit_transform` on train, plain `transform`
   on test, never `fit` on test

**Optional**
- CampusX **Day 29 — ML Pipelines A-Z**
  ([notebook](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day29-sklearn-pipelines))
  — can wait until Issue 11, where it becomes required
- [sklearn — Preprocessing data user guide](https://scikit-learn.org/stable/modules/preprocessing.html),
  the categorical features section

---

### Issue 5 — Fit the linear regression and score it

**Labels:** `gate-1`, `modelling`

Fit the linear model on the training partition and score it on validation. This is the
first genuinely new modelling step in the project.

**Done when**
- Model fit on training partition only
- Validation MAE recorded alongside the Issue 3 baseline MAE
- Coefficients inspected; you can say in one sentence what the model is doing and what
  a coefficient means here
- Nothing has touched the test partition yet

**Learn this first**

**MUST**
- CampusX **Day 50 — Multiple Linear Regression** (geometric intuition + code)
  ([notebook](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day50-multiple-linear-regression))
  — the main lesson for this project, because the model uses many features
- CampusX **Day 49 — Regression Metrics** (MSE, MAE, RMSE, R², adjusted R²)
  ([notebook](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day49-regression-metrics))
  — watch before writing Issue 6
- Baker, [Machine Learning — Linear Regression](https://ggbaker.ca/data-science/content/ml.html#linregress)
  through [ML Pipelines](https://ggbaker.ca/data-science/content/ml.html#pipeline). Short,
  sklearn-first, and it names the vocabulary precisely (model / hyperparameters /
  parameters / fit / predict). Fifteen minutes, and it covers what a coefficient is.

**Optional**
- CampusX **Day 48 — Simple Linear Regression**
  ([notebook](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day48-simple-linear-regression))
  if you want the one-feature intuition before Day 50
- StatQuest [Linear Regression, Clearly Explained](https://www.youtube.com/watch?v=nk2CQITm_eo)
  (27 min) if the mechanism still feels thin after CampusX. Concept-first and demonstrated
  in R, so it complements rather than repeats.
- [sklearn — Linear Models user guide](https://scikit-learn.org/stable/modules/linear_model.html), §1.1.1

---

### Issue 6 — Write the evaluation note

**Labels:** `gate-1`, `docs`

Write `reports/baseline_evaluation.md`: what was compared, on what split, with what
result, and whether the linear model actually beat the naive baseline.

**Done when**
- Split design and date boundaries stated
- Both MAE figures reported, with lift over the naive baseline
- Honest target restated: delay conditional on the flight operating and arriving,
  carrying no cancellation or diversion signal
- If the model did **not** beat the naive baseline, the note diagnoses which cause
  applies rather than declaring failure — the execution plan §8 lists the five
  candidate causes and this is not an automatic trigger to switch projects

**Learn this first**

**MUST:** nothing new.

**Optional:** Baker, [Communicating](https://ggbaker.ca/data-science/content/communicating.html)
— "Asking the Right Question" and "Communicating Results" are the relevant sections.

---

### Issue 7 — Gate 1 review

**Labels:** `gate-1`, `review`

Bring completed evidence for review: quality/provenance memo, feature audit table,
honest-target definition, baseline metric vs the naive per-profile baseline.

**Pass criteria:** honest target defensible; enough pre-booking features carry signal;
baseline beats the naive per-profile baseline; data scoping within budget.

---

## Milestone: Gate 2 — Model & evaluation (Aug 24)

### Issue 8 — RandomForest candidate

**Labels:** `gate-2`, `modelling`

Fit a RandomForest on the same split and preprocessing. This is "attempt, cut if
needed" — if it overruns, ship the linear baseline and note RF as deferred.

**Done when**
- Trained on the identical training partition and preprocessing as the linear model
- Validation MAE recorded on the same basis, so the comparison is real
- Training time and any memory constraints noted — one-hot expansion makes this heavier
  than it looks

**Learn this first**

**MUST — in this order**
1. CampusX **Day 65 — Random Forest**
   ([notebook](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day65-random-forest))
   — for what a forest is and why it beats a single tree
2. [sklearn — `RandomForestRegressor` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
   — to translate that understanding to your problem. Your target is continuous, and the
   regressor exposes different split criteria from the classifier.

**Optional**
- Baker, [Ensembles](https://ggbaker.ca/data-science/content/ml-classif.html#ensemble) /
  [Random Forests](https://ggbaker.ca/data-science/content/ml-classif.html#rand-forest)
- [sklearn — Ensembles user guide](https://scikit-learn.org/stable/modules/ensemble.html),
  the forest section

> Both CampusX and Baker teach forests through **classification** examples. The forest
> concept transfers directly; the estimator, target, and evaluation metrics do not.

---

### Issue 9 — Choose the final model on validation

**Labels:** `gate-2`, `evaluation`

Compare candidates on **validation**. The test set is untouched until the choice is
already made.

**Done when**
- All candidate comparisons documented on validation
- One model selected, with the reason written down before the test run
- Test set has still not been used

**Learn this first**

**MUST:** nothing new — [sklearn Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
was already read at Issue 4. Reread the one line if useful: test data should never be used
to make choices about the model.

---

### Issue 10 — Honest final test-set evaluation

**Labels:** `gate-2`, `evaluation`

Evaluate the chosen model on the test partition and record the result. The prohibition is
on changing the modelling process in response to what you see — not on the mechanical act
of calling predict.

**Done when**
- Test MAE and RMSE recorded for the chosen model and the naive baseline
- Train-vs-test gap reported and interpreted
- Lift over the naive baseline reported, including on thin-support flight profiles
- Test performance is not used to alter model selection, preprocessing, features, or
  hyperparameters. Any rerun after the first evaluation is for reproducibility or to
  correct a documented implementation bug — never to improve the number.

**Learn this first**

**MUST**
- Baker, [Training and Validation](https://ggbaker.ca/data-science/content/ml.html#trainvalid)
  — states the criterion this issue turns on: a big drop from training score to validation
  score is usually a sign of overfitting

**Optional:** CampusX **Day 49 — Regression Metrics**
([notebook](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day49-regression-metrics))
— already seen at Issue 5; rewatch only if MAE-versus-RMSE interpretation is unclear.

---

### Issue 11 — Serialize model and preprocessing

**Labels:** `gate-2`, `deployment-prep`

Persist the fitted preprocessing and estimator so the app reproduces training-time
transformations exactly.

**Done when**
- Fitted preprocessing and estimator persisted **together as a single sklearn Pipeline**
  where practical. If they must stay separate, the notebook states why and shows how the
  app is guaranteed to apply the identical transforms used in training.
- A fresh process can load the artifacts and produce a prediction without refitting
- Library versions pinned in `requirements.txt`
- Artifact paths and the gitignore decision documented

**Learn this first**

**MUST**
- CampusX **Day 29 — ML Pipelines A-Z**
  ([notebook](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day29-sklearn-pipelines))
  — demonstrates sklearn pipelines and persisted preprocessing/model artifacts. Use it for
  the mechanics; for this project, prefer persisting the fitted preprocessing and estimator
  together as one Pipeline rather than as separate objects.
- [sklearn — Model persistence](https://scikit-learn.org/stable/model_persistence.html),
  including its security and version-compatibility warnings

---

### Issue 12 — Model-choice justification

**Labels:** `gate-2`, `docs`

Write the justification: what was compared, what won, why, and what the honest
limitations are.

**Done when**
- Written before Gate 2, not retrofitted after
- Names the leakage controls actually in place
- States what the model cannot do

---

### Issue 13 — Gate 2 review

**Labels:** `gate-2`, `review`

**Pass criteria:** no leakage; honest temporal holdout; ML beats the naive baseline
(especially on thin-support flights); choice justified.
**Fail path:** ship the linear baseline, drop RF.

---

## Milestone: Gate 3 — Deployed (Aug 30)

### Issue 14 — Two-flight comparison app

**Labels:** `gate-3`, `deployment`

Build the Streamlit app: user picks two flights, app returns a reliability comparison.

**Done when**
- Runs locally end-to-end
- Accepts two flight profiles and returns both predictions plus a comparison
- Inputs restricted to pre-booking fields only — the interface must not ask for
  anything unavailable at booking time
- Unseen categories produce a graceful message, not a stack trace

**Learn this first**

**MUST**
- [Streamlit — Get started](https://docs.streamlit.io/get-started)
- [Streamlit — Deploy overview](https://docs.streamlit.io/deploy)

---

### Issue 15 — Guardrail and disclaimer text

**Labels:** `gate-3`, `deployment`, `docs`

Add honest framing to the app surface itself.

**Done when**
- BTS attribution present
- Disclaimer states the model predicts delay conditional on the flight operating and
  arriving, and says nothing about cancellation or diversion risk
- Competitive-tools acknowledgment present
- No "safe", "guaranteed", or equivalent claims anywhere in the interface

---

### Issue 16 — Deploy to Streamlit Community Cloud

**Labels:** `gate-3`, `deployment`

**Done when**
- App is live at a public URL
- A two-flight comparison runs successfully on the deployed instance
- Dependencies resolve from `requirements.txt` in the cloud environment
- Data-file strategy works remotely — the interim CSV is gitignored, so the app must
  not depend on a file that was never pushed

**Learn this first**

**MUST**
- [Streamlit — Prep and deploy on Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
- [Streamlit — App dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)

---

### Issue 17 — Complete the README

**Labels:** `gate-3`, `docs`

**Done when** it covers: honest target definition; feature audit summary and the
pre-booking/post-flight split; cancelled and diverted row handling; temporal holdout
design; baseline comparison and final metrics; competitive-tools acknowledgment;
limitations; live app link; reproduction steps.

**Learn this first**

**MUST:** nothing new.

**Optional:** Baker, [Communicating](https://ggbaker.ca/data-science/content/communicating.html),
especially "Communicating Results" and "Visualizing Data".

---

### Issue 18 — Demo media and résumé bullets

**Labels:** `gate-3`, `docs`

**Done when**
- Screenshot or short screen recording of a real two-flight comparison
- Media embedded in the README
- 2–3 résumé bullets drafted, each stating a concrete decision or number rather than a
  tool list

---

### Issue 19 — Gate 3 review

**Labels:** `gate-3`, `review`

**Pass criteria:** app link works; two-flight comparison runs; attribution, honest
framing, and competitive acknowledgment all present; no guarantee-style claims.
**Fail path:** simplify the interface until it works — protect the deployment.

---

## Notes

- Issues 1–7 are the overdue Gate 1 work and block everything below them.
- Issue 8 is the designated first cut if Week 4 runs short.
- Cut order when late: stretch features → model complexity → duplicate docs.
  Never cut evaluation integrity, a working deployment, or explanation quality.
