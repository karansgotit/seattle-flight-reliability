# Stack & Issue Alignment Review

Technical review of the Seattle Two-Flight Reliability project: what the stack should be,
what the official docs actually say, and where `PROJECT_ISSUES.md` diverges from them.

Every documentation claim below was fetched from the live page and quoted verbatim, per the
`CLAUDE.md` rule. Where a belief is *not* backed by documentation, it is labelled as a
reasoned judgement or a deliberate tradeoff rather than dressed up as a doc citation.

Reviewed against the working tree as of 2026-08-21. Gate 2 is Aug 24, Gate 3 is Aug 30.

---

## Phase 1 — Project understanding

**Problem.** A traveler choosing between two SEA-departing domestic flights wants an honest
pre-booking comparison of historical arrival-delay reliability.

**ML task.** Supervised **regression** on tabular data. Target `ArrDelay` in minutes.

**A distinction that drives the whole stack:** this is *not* a forecasting problem. There is
no autoregressive structure being modelled, no per-series future path being projected. It is
tabular regression whose rows happen to carry timestamps, with a deployment requirement to
generalise to *future* dates. That distinction justifies temporal cross-validation while
ruling out the classical time-series toolchain (`statsmodels`, `prophet`) entirely — see
Phase 2.

**Data.** BTS/DOT TranStats Reporting Carrier On-Time Performance, 2024–2025, SEA-origin.
14,080,680 raw rows → 328,559 after SEA filter → 324,490 after dropping null `ArrDelay`.
731 distinct dates, 2024-01-01 → 2025-12-31. 13 pre-booking features plus the target.

Measured properties relevant to the review:

| property | measured value |
|---|---|
| flights/day | 242 – 555 (2.29×), summer ≈ 50% above winter |
| `ArrDelay` | mean 4.91, std 43.58, p50 −4, p75 11, max 3359 |
| `Flight_Number_Reporting_Airline` | 2,591 distinct |
| `Dest` | 96 distinct |
| `IATA_CODE_Reporting_Airline` | 11 distinct |
| `CRSDepTime` / `CRSArrTime` | HHMM-encoded ints, range 1–2359, zero values with minute-part ≥ 60 |
| naive one-hot width (3 categoricals) | ≈ 2,698 columns |

**Expected product.** A deployed Streamlit app taking two pre-booking flight profiles and
returning a reliability comparison, plus honest documentation of scope and limits.

**Current stage.** Data provenance, quality audit, feature audit and cleaning are complete.
`linear_baseline.ipynb` loads the data, parses `FlightDate`, and drops post-flight and
redundant columns. **No model has been trained, no split has been implemented.** Everything
from Issue 2 onward is plan, not code.

**Intended workflow.** clean CSV → temporal split → per-fold preprocessing → naive baseline
→ linear model → RF candidate → select on CV → single test evaluation → persist Pipeline →
Streamlit app → deploy.

---

## Phase 2 — Required technical stack

### Necessary

| Technology | Role | Workflow stage | Justification |
|---|---|---|---|
| **Python 3** | Implementation language | all | Only language in the repo; sklearn/Streamlit are Python-native. |
| **pandas** | Load, filter, date parsing, groupby for the naive baseline, date-array split construction | data → preprocessing | Already the sole library in use. 121 MB CSV, 324k rows — comfortably in-memory. |
| **numpy** | Unique-date array, fold index arrays, `isin` masks, metric arithmetic | preprocessing → evaluation | Needed directly by the Issue 2 date-axis split (`np.sort` on unique dates, integer fold indices). Currently declared but unimported — it *will* be imported. |
| **scikit-learn** | `TimeSeriesSplit`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`, `LinearRegression`, `RandomForestRegressor`, metrics, persistence | preprocessing → modeling → evaluation → persistence | The core of the project. |
| **joblib** | Pipeline serialisation | persistence | Ships as a sklearn dependency; named explicitly in the persistence docs. Should be pinned even though it arrives transitively. |
| **Streamlit** | Two-flight comparison UI, deployment target | deployment | The specified product surface. **Currently missing from `requirements.txt`.** |
| **matplotlib** | Residual and predicted-vs-actual diagnostics for Issues 6 and 12; demo media for Issue 18 | evaluation → communication | Justified narrowly — a heavy-tailed target (max 3359 min) cannot be diagnosed from a single MAE number. |
| **Jupyter** | Development environment | all | Existing workflow. Dev-only dependency. |

### Unnecessary complexity — recommend removing

**`seaborn`** — declared in `requirements.txt`, never imported. It is a styling and
statistical-plotting layer over matplotlib. This project needs perhaps three diagnostic
plots, all of which are a few lines of plain matplotlib, and Streamlit renders its own
charts natively in the app. Carrying it adds an install and a version to pin on a
memory-constrained deployment target for no capability gain. **Recommend dropping it unless
a specific plot needs it.**

### Deliberately excluded

- **`statsmodels` / `prophet` / `sktime`** — these serve autoregressive forecasting of a
  series over time. This project predicts a per-flight target from pre-booking attributes;
  there is no series to forecast and no lag structure. Including them would be complexity
  with no role.
- **`xgboost` / `lightgbm`** — plausibly better than RandomForest on this data, but the
  backlog already flags RF as "attempt, cut if needed" with Gate 2 three days out. Adding a
  third model class is scope risk, not value. Legitimate to revisit post-Gate 3.
- **A database** — one 121 MB CSV, read once. A database is unnecessary.

---

## Phase 3 — Documentation sources

All fetched live and quoted verbatim in Phase 4. Full inventory is maintained in
`DOCUMENTATION_SOURCES.md`.

| # | Source | Page |
|---|---|---|
| D1 | sklearn — Cross-validation user guide | `/stable/modules/cross_validation.html` |
| D2 | sklearn — `TimeSeriesSplit` API reference | `/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html` |
| D3 | sklearn — Common pitfalls and recommended practices | `/stable/common_pitfalls.html` |
| D4 | sklearn — `OneHotEncoder` API reference | `/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html` |
| D5 | sklearn — Linear Models user guide §1.1.1 | `/stable/modules/linear_model.html` |
| D6 | sklearn — Time-related feature engineering (example) | `/stable/auto_examples/applications/plot_cyclical_feature_engineering.html` |
| D7 | sklearn — `RandomForestRegressor` API reference | `/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html` |
| D8 | sklearn — Model persistence | `/stable/model_persistence.html` |
| D9 | sklearn — Metrics and scoring | `/stable/modules/model_evaluation.html` |
| D10 | Streamlit — Caching | `/develop/concepts/architecture/caching` |
| D11 | Streamlit — Manage your app (resource limits) | `/deploy/streamlit-community-cloud/manage-your-app` |

**A negative result worth recording.** D9 does **not** contain guidance on MAE-vs-RMSE
robustness or which to prefer for heavy-tailed targets. That belief is widespread and
correct as mathematics, but it is not in the sklearn metrics documentation. It is treated
below as a reasoned tradeoff, not a doc-backed rule.

---

## Phase 4 — Issue-by-issue alignment

### Issue 1 — Create baseline notebook, load clean dataset
**Asks:** load the 324,490-row CSV, parse dates, narrow to target + pre-booking features.
**Affects:** data loading. **Applies:** pandas.
**Verdict: aligned, with one carried-forward risk.** Implemented and correct as far as it
goes. The risk is in *which* columns survive — see Finding 5 on `Year` and `FlightDate`,
which the current drop list retains as model features.

### Issue 2 — Temporal split and CV strategy
**Asks:** date-boundary train pool / test split; `TimeSeriesSplit` over the train pool with
folds acting as validation; test touched once.
**Affects:** the single most consequential evaluation decision. **Applies:** D1, D2.
**Verdict: aligned** (as rewritten earlier in this session).

D1 states the no-validation-partition rule directly:
> "A test set should still be held out for final evaluation, but the validation set is no
> longer needed when doing CV."

and why ordinary k-fold is wrong here:
> "classical cross-validation techniques such as `KFold` and `ShuffleSplit` assume the
> samples are independent and identically distributed, and would result in unreasonable
> correlation between training and testing instances (yielding poor estimates of
> generalization error) on time series data."

The split-on-dates decision is forced by D2:
> "To ensure comparable metrics across folds, samples must be equally spaced. Once this
> condition is met, each test set covers the same time duration, while the train set size
> accumulates data from previous splits."

Verified against the data: splitting raw rows puts all 5 fold boundaries mid-date and
produces folds spanning 106–136 days at identical row counts. Splitting the 731 unique
dates gives exactly 106 validation days per fold with zero date overlap.

### Issue 3 — Naive per-profile baseline
**Asks:** per-profile historical median, computed per fold, scored on the same folds.
**Affects:** the comparator everything else is judged against. **Applies:** pandas, D7.
**Verdict: aligned, and better-motivated than the issue itself states.**

The median is the L1-minimising point estimate, so a median baseline is precisely the right
naive comparator for an MAE-evaluated project. D7 confirms sklearn treats the pairing the
same way, describing `criterion="absolute_error"` as
> "mean absolute error, which minimizes the L1 loss using the median of each terminal node"

Worth writing this justification into the issue — right now the choice of median over mean
reads as arbitrary when it is actually the correct pairing.

**Gap:** the unseen-profile fallback here and the unseen-category handling in Issue 14 are
the same problem at two layers, specified independently. See Finding 10.

### Issue 4 — Encode categorical features
**Asks:** one-hot encode inside a Pipeline, fit per fold, handle unseen categories, record
feature count.
**Affects:** preprocessing; the leakage surface. **Applies:** D3, D4.
**Verdict: partially aligned — correct on leakage, silent on three decisions that matter.**

The leakage discipline is exactly right and D3 supports it:
> "Always split the data into train and test subsets first, particularly before any
> preprocessing steps."

and
> "The scikit-learn pipeline is a great way to prevent data leakage as it ensures that the
> appropriate method is performed on the correct data subset. The pipeline is ideal for use
> in cross-validation and hyper-parameter tuning functions."

What is missing: **Findings 1, 2 and 3** below — time encoding, the `drop` parameter, and
2,591-category flight numbers. All three are preprocessing decisions this issue owns and
none is currently mentioned.

### Issue 5 — Fit linear regression and score it
**Asks:** pipeline + `cross_val_score` on the Issue 2 folds; inspect coefficients.
**Affects:** modeling. **Applies:** D5, D6.
**Verdict: not aligned — the coefficient-interpretation goal is undermined by Issue 4's
omissions.** See Findings 1, 2, 6.

### Issue 6 — Evaluation note
**Asks:** report split design, both MAEs, honest scope, and diagnose if the model loses.
**Affects:** communication. **Applies:** none (writing task).
**Verdict: aligned in structure, incomplete in the diagnosis list.** The issue lists five
candidate causes for a linear model failing to beat the naive baseline. Raw HHMM time
encoding (Finding 1) is a *likely* cause here and is not among them, so the diagnosis could
plausibly land on the wrong explanation.

### Issue 7 — Gate 1 review
**Verdict: aligned.** Process gate, no technical claim to check.

### Issue 8 — RandomForest candidate
**Asks:** RF on the same folds and pipeline; record CV MAE, training time, memory.
**Affects:** modeling, and — unexpectedly — deployment feasibility. **Applies:** D7, D11.
**Verdict: not aligned — the default configuration cannot be deployed.** See Finding 4.

### Issue 9 — Choose final model on CV
**Asks:** compare candidates on CV MAE; test untouched.
**Affects:** model selection. **Applies:** D3.
**Verdict: aligned in protocol, confounded in substance.** The protocol is right, and D3
backs it:
> "Test data should never be used to make choices about the model. The general rule is to
> never call `fit` on the test data."

But the comparison it performs is not a clean one. Per D6, tree models absorb raw ordinal
time features while linear models cannot — so RF would win this comparison partly because
of feature encoding rather than model class. See Finding 1.

### Issue 10 — Honest final test evaluation
**Asks:** refit on full train pool, evaluate once, report MAE/RMSE and train-vs-test gap.
**Affects:** final evaluation. **Applies:** D9.
**Verdict: aligned, with an interpretation caveat.** The refit-then-evaluate-once protocol
is correct. The caveat is RMSE on a target with max 3359 and p75 11 — a handful of extreme
delays will dominate it, so RMSE and MAE will tell different stories. Report both, but say
which one the product claim rests on. See Finding 6.

### Issue 11 — Serialize model and preprocessing
**Asks:** persist the Issue 10 refit as one Pipeline; pin versions.
**Affects:** persistence → deployment. **Applies:** D8.
**Verdict: aligned in approach, under-specified on metadata and now urgent on pinning.**

D8 is blunter about versions than the issue is:
> "There are no supported ways to load a model trained with a different version of
> scikit-learn. While using skops.io, joblib, pickle, or cloudpickle, models saved using one
> version of scikit-learn might load in other versions, however, this is entirely
> unsupported and inadvisable."

`requirements.txt` is currently **fully unpinned**. Community Cloud installs from it, so the
cloud can silently resolve a different sklearn than the training laptop. See Finding 8.

D8 also lists metadata the issue does not require:
> "In order to rebuild a similar model with future versions of scikit-learn, additional
> metadata should be saved along the pickled model: The training data, e.g. a reference to
> an immutable snapshot — The Python source code used to generate the model — The versions
> of scikit-learn and its dependencies — The cross validation score obtained on the training
> data"

### Issue 12 — Model-choice justification
**Verdict: aligned.** Should additionally record the Phase 5 tradeoffs.

### Issue 13 — Gate 2 review
**Verdict: aligned.** Recommend adding a deployability criterion (artifact size) given
Finding 4 — discovering at Gate 3 that the Gate 2 winner cannot deploy would be costly with
six days to spare.

### Issue 14 — Two-flight comparison app
**Asks:** Streamlit app, pre-booking inputs only, graceful unseen-category handling.
**Affects:** deployment. **Applies:** D10, D11.
**Verdict: not aligned — no caching strategy.** See Finding 9.

### Issue 15 — Guardrail and disclaimer text
**Verdict: aligned, and genuinely good practice.** Recommend one addition: the seasonal
caveat. A temporal holdout makes the test partition Q4-only, so the headline accuracy number
describes winter operations, not a year-round average.

### Issue 16 — Deploy to Community Cloud
**Asks:** live URL, working comparison, dependencies resolve, data-file strategy works.
**Affects:** deployment. **Applies:** D11.
**Verdict: partially aligned — checks dependencies and data, not artifact size or memory.**
See Findings 4 and 9. D11:
> "CPU: 0.078 cores minimum, 2 cores maximum" · "Memory: 690MB minimum, 2.7GBs maximum"
> "If your app meets or exceeds its limits, it may slow down from throttling or become
> nonfunctional."

### Issues 17, 18 — README and demo media
**Verdict: aligned.** README should carry the Phase 5 tradeoff list.

### Missing issue — hyperparameter tuning
There is no tuning step anywhere in the backlog: linear → RF → choose → test. RF defaults
(`max_depth=None`) are exactly what produces the Finding 4 blow-up, so *some* bounding is
required for deployability regardless of whether tuning-for-accuracy is in scope. If tuning
is deliberately cut for time, that is defensible — but it should be a stated tradeoff rather
than an omission.

---

## Findings

### Finding 1 — `CRSDepTime` / `CRSArrTime` are HHMM integers, not quantities (HIGH)
**Issues 4, 5, 6, 8, 9.**

Verified: both range 1–2359 with zero values whose minute-part is ≥ 60 — confirming HHMM
encoding. Fed to a linear model as raw numbers, 10:59 → 11:00 is a **+41 unit** jump for one
minute of real time, and 23:59 → 00:01 is a −2358 unit jump for two minutes. The scale is
non-linear in time *and* discontinuous at midnight.

D6 addresses exactly this, and notes the asymmetry between model families:
> "Note that the time related features are passed as is, i.e. without processing them. But
> this is not much of a problem for tree-based models as they can learn a non-monotonic
> relationship between ordinal input features and the target."

In D6's own benchmark the naive linear pipeline scores MAE 0.142 ± 0.014 against gradient
boosting's 0.044 ± 0.003 on the same features.

**Why it matters here.** Departure time is likely one of the strongest real predictors of
delay — late-day flights inherit the day's accumulated disruption. Handing the linear model
a broken encoding of the most predictive feature risks it losing to the naive baseline in
Issue 6 for a reason the issue's diagnosis list does not contain, and then losing to RF in
Issue 9 for a reason that is about encoding rather than model class.

**Recommended.** Convert to minutes-since-midnight, then for the linear model add a periodic
encoding — sine/cosine, or `SplineTransformer` with `extrapolation="periodic"`, which D6
introduces for this purpose. Tree models can take the ordinal form as-is.

### Finding 2 — One-hot + unregularised OLS creates perfect collinearity (MEDIUM)
**Issues 4, 5.**

Issue 4 never mentions the `drop` parameter. D4 is explicit about when it is needed:
> "Specifies a methodology to use to drop one of the categories per feature. This is useful
> in situations where perfectly collinear features cause problems, such as when feeding the
> resulting data into an unregularized linear regression model."

`LinearRegression` is unregularised — D5: it minimises `min_w ||Xw - y||₂²`. D5 on the
consequence:
> "The coefficient estimates for Ordinary Least Squares rely on the independence of the
> features. When features are correlated and some columns of the design matrix X have an
> approximately linear dependence, the design matrix becomes close to singular and as a
> result, the least-squares estimate becomes highly sensitive to random errors in the
> observed target, producing a large variance."

**Why it matters here.** Issue 5 requires "coefficients inspected; you can say in one
sentence what a coefficient means." Under full dummy encoding with an intercept the
coefficients are not uniquely determined, so that deliverable is not achievable as written.

**Recommended.** `drop="first"` for the OLS baseline. Note the tradeoff D4 also states:
> "However, dropping one category breaks the symmetry of the original representation and can
> therefore induce a bias in downstream models, for instance for penalized linear
> classification or regression models."

So `drop="first"` is right for `LinearRegression` and should be reconsidered if the project
ever switches to Ridge/Lasso.

### Finding 3 — 2,591 flight numbers, no mitigation specified (HIGH)
**Issue 4.**

Measured: 2,591 distinct flight numbers, 96 destinations, 11 carriers → ≈ 2,698 one-hot
columns, the overwhelming majority carrying a handful of rows each. Issue 4 says the feature
count should be "recorded" but specifies no mitigation.

D4 provides the mechanisms directly:
> **min_frequency**: "Specifies the minimum frequency below which a category will be
> considered infrequent. If int, categories with a smaller cardinality will be considered
> infrequent. If float, categories with a smaller cardinality than `min_frequency *
> n_samples` will be considered infrequent."

> **max_categories**: "Specifies an upper limit to the number of output features for each
> input feature when considering infrequent categories."

and the `handle_unknown` value that composes with them:
> **'infrequent_if_exist'**: "When an unknown category is encountered during transform, the
> resulting one-hot encoded columns for this feature will map to the infrequent category if
> it exists."

**Why it matters here.** Two compounding reasons. First, under `TimeSeriesSplit` the early
folds train on small date windows, so a large share of flight numbers appearing in a fold's
validation rows were never seen in that fold's training rows — the encoding degrades exactly
where the CV estimate is formed. Second, carrier + flight number + destination is close to
the *identity* of the Issue 3 profile, so one-hot flight numbers push the linear model toward
re-implementing the naive baseline as a lookup rather than learning generalisable structure.

**Recommended.** Decide explicitly: either drop `Flight_Number_Reporting_Airline` from the
model features (keeping it only as the profile key for Issue 3), or retain it with
`min_frequency` / `max_categories` set and the threshold justified. Either is defensible;
silence is not.

### Finding 4 — Default RandomForest will not fit the deployment target (HIGH)
**Issues 8, 11, 13, 16.**

Measured on this dataset: a 10-tree forest on 40,000 rows pickles to 34.9 MB (484,772
nodes). Scaling to the 285,649-row train pool at sklearn's default `n_estimators=100`
extrapolates to a **≈ 2.5 GB artifact**.

D11 gives the ceiling:
> "Memory: 690MB minimum, 2.7GBs maximum"
> "If your app meets or exceeds its limits, it may slow down from throttling or become
> nonfunctional."

A 2.5 GB model exceeds the minimum allocation nearly fourfold and brushes the maximum before
Python, pandas or the app itself is loaded. D7 anticipates the cause:
> "The default values for the parameters controlling the size of the trees (e.g. `max_depth`,
> `min_samples_leaf`, etc.) lead to fully grown and unpruned trees which can potentially be
> very large on some data sets. To reduce memory consumption, the complexity and size of the
> trees should be controlled by setting those parameter values."

**Why it matters here.** Gate 2 selects a model and Gate 3 deploys it, six days apart. As
written, nothing between them checks artifact size — so the failure surfaces at deployment,
after the choice is locked.

**Recommended.** Set an explicit artifact budget (≈ 200 MB is comfortable) as a Gate 2 pass
criterion; constrain `max_depth` and `min_samples_leaf` and treat deployability as a
selection constraint, not an afterthought.

### Finding 5 — `Year` and `FlightDate` do not extrapolate to prediction time (HIGH)
**Issues 1, 4, 14.** *Reasoned judgement — see the caveat below.*

`Year` takes exactly two values (2024, 2025). The app predicts for flights the user is about
to book — 2026 and later. That value never appears in training. Numeric, the model
extrapolates a two-point trend off the end of its support; one-hot with
`handle_unknown="ignore"`, D4 says the row becomes all-zeros:
> "When an unknown category is encountered during transform, the resulting one-hot encoded
> columns for this feature will be all zeros."

Either way the feature contributes nothing valid at prediction time, having absorbed
variance during training. `FlightDate` has the same defect if used as a predictor — it is
required for constructing the split, which is not the same as being a model input.

The feature audit justifies keeping `Year`/`Quarter`/`Month`/`DayofMonth` "for convenience"
so they need not be regenerated from `FlightDate`. That is a fine reason to keep a *column*
and not a reason to feed it to the *model*. `Month` and `DayOfWeek` are genuinely cyclical
and recur in future years — they should stay. `Year` and raw `FlightDate` do not recur.

**Caveat on grounding.** D3 defines leakage as using "information that would not be available
at prediction time," which is the mirror image of this problem — here the information *is*
available but carries no valid meaning outside the training range. I could not find a
sklearn page stating this rule directly, so treat it as reasoning about the deployment
contract, not a doc citation.

**Recommended.** Drop `Year` and raw `FlightDate` from the model feature set; keep
`FlightDate` in the dataframe as the split key and the Issue 3 grouping key. Also consider
dropping `Quarter` as redundant with `Month` (Finding 2's collinearity concern).

### Finding 6 — Training loss and evaluation metric disagree (MEDIUM — tradeoff)
**Issues 5, 6, 10.**

`ArrDelay` is severely right-skewed: mean 4.91, std 43.58, median −4, p75 11, max 3359.

`LinearRegression` minimises squared error (D5: `min_w ||Xw - y||₂²`) while the project
evaluates on MAE. The model optimises one thing and is scored on another, and with a max of
3359 the squared-error objective is heavily influenced by a small number of extreme delays.
D7 shows sklearn offers the aligned option for the forest:
> "**absolute_error**: mean absolute error, which minimizes the L1 loss using the median of
> each terminal node"

though it is substantially slower to fit.

**Honesty note.** D9 does *not* contain guidance on MAE-versus-RMSE robustness or heavy
tails. The reasoning above rests on the definitions of the loss functions, not on a
documented sklearn recommendation. Recording it as a deliberate tradeoff is the correct
treatment — not as a doc-backed best practice.

**Recommended.** Keep MAE as the headline metric (it is the honest one for a traveler-facing
claim — "typically within N minutes"), keep `LinearRegression` as-is for the baseline, and
state the mismatch explicitly in Issue 12. Consider `criterion="absolute_error"` for RF only
if fit time allows, which given Gate 2's date it likely does not.

### Finding 7 — Unpinned `requirements.txt` is now a deployment risk (MEDIUM)
**Issue 11, Issue 16.**

`requirements.txt` lists six bare package names with no versions, and omits `streamlit`
entirely. Per D8, models are only supported when loaded by the same sklearn version that
wrote them. Community Cloud resolves this file independently of the training environment, so
an unpinned file invites a silent version mismatch between the notebook that trained the
model and the app that loads it.

**Recommended.** Pin all versions before Issue 11 persists anything, add `streamlit`, drop
`seaborn` (Phase 2). D8's `InconsistentVersionWarning` pattern is worth adopting at load
time in the app.

### Finding 8 — Persistence metadata under-specified (LOW)
**Issue 11.** Covered under Issue 11 above — adopt D8's metadata list, in particular
recording the CV score and dependency versions alongside the artifact.

### Finding 9 — No Streamlit caching strategy (MEDIUM)
**Issues 14, 16.**

Streamlit's execution model makes this load-bearing. D10:
> Streamlit executes scripts top-to-bottom at every user interaction — "Long-running
> functions run again and again, which slows down your app" and "Objects get recreated again
> and again, which makes it hard to persist them across reruns or sessions."

and the specific remedy:
> "`st.cache_resource` is the recommended way to cache global resources like ML models or
> database connections."

**Why it matters here.** Without `@st.cache_resource` the model artifact is deserialised on
every widget interaction. Combined with Finding 4's artifact size and D11's 690 MB floor,
this is not a performance nicety — it is the difference between an app that runs and one
that shows "🤯 This app has gone over its resource limits."

**Recommended.** Add to Issue 14: model loading wrapped in `@st.cache_resource`; any
reference data (e.g. the Issue 3 profile-median lookup) wrapped in `@st.cache_data`.

### Finding 10 — Unseen-category handling specified twice, independently (LOW)
**Issues 3, 4, 14.** Issue 3's unseen-profile fallback, Issue 4's `handle_unknown` choice,
and Issue 14's "graceful message, not a stack trace" are one decision surfacing at three
layers. Specify the policy once and cross-reference, so the app's user-facing behaviour and
the encoder's configuration cannot drift apart.

---

## Phase 5 — Alignment report

### 5.1 Recommended stack
Python 3 · pandas · numpy · scikit-learn · joblib · Streamlit · matplotlib · Jupyter (dev).
**Remove** `seaborn`. **Add** `streamlit`. **Pin every version.** Excluded by design:
statsmodels/prophet (no forecasting task), xgboost/lightgbm (scope risk before Gate 2), any
database.

### 5.2 Documentation sources
D1–D11 in Phase 3, all fetched live and quoted verbatim. Inventory maintained in
`DOCUMENTATION_SOURCES.md`. One documented negative result: sklearn's metrics page carries no
MAE-vs-RMSE outlier guidance, so that reasoning is labelled a tradeoff rather than cited.

### 5.3 Issue-by-issue summary

| Issue | Verdict | Findings |
|---|---|---|
| 1 Load data | Aligned (carried risk) | F5 |
| 2 Temporal split + CV | **Aligned** | — |
| 3 Naive baseline | Aligned (strengthen rationale) | F10 |
| 4 Categorical encoding | **Partially aligned** | F1, F2, F3 |
| 5 Linear regression | **Not aligned** | F1, F2, F6 |
| 6 Evaluation note | Aligned (incomplete diagnosis list) | F1 |
| 7 Gate 1 review | Aligned | — |
| 8 RandomForest | **Not aligned** | F4 |
| 9 Model selection | Aligned protocol, confounded comparison | F1 |
| 10 Final test eval | Aligned (interpretation caveat) | F6 |
| 11 Persistence | Partially aligned | F7, F8 |
| 12 Justification | Aligned | — |
| 13 Gate 2 review | Aligned (add deployability criterion) | F4 |
| 14 Streamlit app | **Not aligned** | F9, F10 |
| 15 Disclaimers | Aligned (add seasonal caveat) | — |
| 16 Deploy | Partially aligned | F4, F7, F9 |
| 17 README | Aligned | — |
| 18 Demo media | Aligned | — |
| — Tuning | **Missing** | F4 |

### 5.4 Issues that should be modified

**Before Gate 2 (Aug 24) — these change the numbers:**
1. **Issue 4** — add time-feature encoding (F1), `drop="first"` (F2), and a
   `Flight_Number` cardinality decision (F3).
2. **Issue 1 / feature list** — remove `Year` and raw `FlightDate` from model features (F5).
3. **Issue 8** — add an artifact-size budget and tree-size constraints (F4).
4. **Issue 13** — add deployability to the Gate 2 pass criteria (F4).
5. **Issue 6** — add broken time encoding to the diagnosis list (F1).

**Before Gate 3 (Aug 30):**
6. **Issue 11** — pin versions, adopt D8's metadata list (F7, F8).
7. **Issue 14** — specify `@st.cache_resource` / `@st.cache_data` (F9).
8. **Issue 16** — verify artifact size and memory headroom, not just dependency resolution.
9. **`requirements.txt`** — pin, add `streamlit`, drop `seaborn`.

**Lower priority:** unify unseen-category policy (F10); strengthen Issue 3's median
rationale; add the seasonal caveat to Issue 15.

### 5.5 Deliberate tradeoffs to record in Issue 12 and the README

1. **MAE headline over RMSE**, despite `LinearRegression` optimising squared error. Chosen
   because a traveler-facing claim is about the typical case, not the tail. Not a
   documented sklearn recommendation — a project judgement (F6).
2. **Q4-only test partition.** Inherent to an honest temporal holdout on two years of data.
   The final number describes winter operations; the alternative (random split) would be
   dishonest.
3. **No hyperparameter tuning**, if cut for time. Acceptable for a baseline-versus-RF
   comparison; must be stated, and does not exempt the tree-size bounding F4 requires.
4. **RandomForest possibly dropped entirely** if the artifact budget or Gate 2 timing
   forces it. The backlog already anticipates this; F4 makes it more likely.
5. **`Flight_Number_Reporting_Airline` dropped or frequency-capped**, accepting some loss of
   route-specific signal in exchange for a model that generalises rather than memorises (F3).
6. **Scope excludes cancellation and diversion risk** — already well documented, and the
   single most important honesty guardrail in the project.

---

## Open questions

1. **Is `Flight_Number_Reporting_Airline` intended as a model feature, or only as the Issue 3
   profile key?** This changes the encoding design materially (F3).
2. **Is the app's profile identity carrier+flight-number+destination, or
   carrier+destination+time-of-day?** It determines both the naive baseline's grouping and
   what the UI must ask for.
3. **Is hyperparameter tuning in scope before Gate 2**, or deliberately cut?
4. **Is RandomForest still wanted** given F4's deployment constraint and three days to
   Gate 2 — or should the linear baseline be hardened and shipped instead?
