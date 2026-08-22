# Project 1 — Issue Backlog (Gate 1 → Gate 3)

Covers the Seattle Two-Flight Reliability Comparison Tool from the outstanding linear
baseline through deployment on **August 30**. Nothing past September 1 is in scope.

Each issue below is written to be pasted into GitHub as a separate issue. Suggested
labels are given per issue. **Learn this first** lists external sources only — docs,
videos, and the CMPT 353 decks already on disk.

**Gate dates:** Gate 1 (was Aug 7, now overdue) · Gate 2 — Aug 24 · Gate 3 — Aug 30

**Status:** this file is the single source of truth. It supersedes
`reports/stack_and_alignment_review.md`, which was a point-in-time technical review; the
findings that survived validation are folded into the issues below and the review is kept
only as the audit trail. See `PROJECT_ISSUES_CHANGELOG.md` for what changed and why.

**Primary resource:** [CampusX — 100 Days of Machine Learning](https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH)
(134 videos), with its companion notebook repo at
[campusx-official/100-days-of-machine-learning](https://github.com/campusx-official/100-days-of-machine-learning).
CampusX is Python/sklearn-first, which suits the mechanics issues better than concept-first
sources. Two warnings: **playlist position ≠ day number** (playlist item 50 is Day 48), so
search the playlist by day number rather than scrolling to a position; and the series has
**no episode on temporal or time-aware splitting**, which is the single most important
evaluation decision in this project — that one stays on the sklearn docs.

**Source rule:** official documentation is the authority. Where CampusX or Baker disagrees
with the current sklearn docs, the docs win — the tutorials are good teaching material but
the API has moved since they were recorded.

Docs are not always the best place to meet a concept for the *first* time, though: reference
pages are written for someone who already knows what the thing is. So the rule is one lesson
first, then docs, and only for genuinely new concepts. In this backlog that is four videos
total — Days 27, 28, 50, and 65. Everything else goes straight to documentation.

For those four, the companion notebook in the CampusX repo is usually faster than the video.
Skim the notebook first; watch only if the code alone doesn't land.

**Version caution.** `scikit-learn` is pinned to **1.7.2**, but `scikit-learn.org/stable/`
now serves **1.9.0** docs. Every sklearn behaviour asserted in this file was re-checked
against the installed 1.7.2 before being written down. If you read a doc page and it
disagrees with an issue here, check the version before assuming the issue is wrong.

---

## How to read the source lists

Each issue's **Learn this first** block is split into **MUST** (read before starting that
issue) and **Optional** (only if you get stuck). Where MUST lists items in order, follow
the order: learn the concept from the lesson first, *then* open the API page to translate
it to your problem. API references are for keeping open while you code, not for learning a
technique cold.

Don't front-load. Read what the next issue actually needs, then go implement it.

---

## Closed decisions

These were open questions. They are now settled, so no issue below re-litigates them. Each
is reversible; the cost of reversing is noted.

**D-1 — `Flight_Number_Reporting_Airline` is not a model feature.**
It stays in the dataframe as the Issue 3 profile key and an Issue 14 app input, but it is
never handed to the estimator. 2,591 distinct values against 96 destinations and 11
carriers; one-hot encoding all three produces ≈2,698 columns, and carrier + flight number +
destination is so close to the *identity* of a profile that a linear model fed it would
re-implement the Issue 3 naive baseline as a lookup table rather than learn transferable
structure. It also degrades worst exactly where the CV estimate is formed: under
`TimeSeriesSplit` the early folds train on short date windows, so a large share of flight
numbers in a fold's validation rows were never seen in its training rows.
*Reversing:* re-add it with `min_frequency` set, and rerun Issues 4, 5, 8, 9. Cheap before
Gate 2, expensive after.

**D-2 — A "profile" is carrier + flight number + destination.**
That is what a traveler actually compares ("Alaska 1234 to SFO" versus "Delta 2201 to
SFO"), so the product requires this grain, and Issue 3 groups on it. Measured on the clean
data: **4,483 profiles, median 23 rows each, 36.4% with fewer than 10 rows, 10th percentile
= 1 row.** That thin tail is the reason Issue 3 needs an explicit fallback ladder rather
than a single groupby. The coarser grain (carrier + destination) is **216 groups, median
974 rows, 8.3% under 10 rows** — that is the first rung of the fallback, not the primary
grain.
*Reversing:* changing the grain changes both the baseline number and the app's input form.

**D-3 — No hyperparameter tuning before Gate 2.**
Gate 2 is Aug 24 and the comparison is baseline-vs-linear-vs-RF, not a search for the best
possible RF. This is a stated tradeoff, recorded in Issue 12, not an omission.
**It does not exempt the tree-size bounding in Issue 8** — that is a deployability
constraint with a measured artifact-size justification, not an accuracy optimisation.
Issue 8b holds the tuning work if time appears; it is the first thing cut.

**D-4 — RandomForest stays in scope.**
The deployment objection against it dissolves once the trees are bounded: measured on the
full train pool, a default forest is ≈2.5 GB while `max_depth=12, min_samples_leaf=50` at
100 trees is **8.7 MB and fits in 11 seconds**. See Issue 8. It remains the designated
first cut if Week 4 runs short.

**D-5 — Unseen categories have one policy, not three.**
Issue 3's unseen-profile fallback, Issue 4's encoder configuration, and Issue 14's
user-facing message are one decision surfacing at three layers. The policy is defined once
in **Issue 4** and the other two cross-reference it. They must not drift apart.

---

## The model feature set

Fixed here so Issues 4, 5, 8 and 14 all mean the same thing by "the features". Columns kept
in the dataframe but *not* handed to the estimator are listed separately — that distinction
is the substance of Finding 5 in the review and it is easy to lose.

**Handed to the estimator (9):**

| Column | Treated as | Note |
|---|---|---|
| `IATA_CODE_Reporting_Airline` | categorical, one-hot | 11 values |
| `Dest` | categorical, one-hot | 96 values |
| `Month` | categorical, one-hot | 12 values — one-hot, not numeric, so Dec→Jan is not a cliff |
| `DayOfWeek` | categorical, one-hot | 7 values — same reasoning |
| `DayofMonth` | numeric | kept for the holiday mechanism (with `Month`, locates Thanksgiving/Christmas) |
| `CRSDepTime` → `dep_minutes` | cyclical numeric | converted from HHMM, then periodically encoded — see Issue 4 |
| `CRSArrTime` → `arr_minutes` | cyclical numeric | same |
| `CRSElapsedTime` | numeric | scheduled block time |
| `Distance` | numeric | |

One-hot width before `drop="first"` is 11 + 96 + 12 + 7 = **126 columns**, against ≈2,698
under the naive scheme that included flight numbers.

**Kept in the dataframe, never a model input:**

| Column | Why it's kept | Why it isn't a feature |
|---|---|---|
| `FlightDate` | Issue 2 split key; Issue 3 grouping | A date does not recur. The app predicts 2026+ dates, which lie outside the training range entirely. |
| `Flight_Number_Reporting_Airline` | Issue 3 profile key; Issue 14 input | D-1 |
| `Year` | provenance | Takes exactly two values, 2024 and 2025. Every prediction the app makes is for a year the model has never seen. Numeric, it extrapolates a two-point trend off the end of its support; one-hot with `handle_unknown="ignore"`, the row silently becomes all-zeros. Either way it absorbed variance during training and contributes nothing valid at prediction time. |
| `Quarter` | provenance | A deterministic function of `Month`. Redundant, and redundancy is what Issue 4's `drop="first"` exists to control. |

> **Grounding note.** The `Year` / `FlightDate` argument is *reasoning about the deployment
> contract*, not a doc citation. sklearn's
> [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) page defines
> leakage as using information unavailable at prediction time; this is the mirror image —
> the information *is* available, it just carries no valid meaning outside the training
> range. No sklearn page states this rule directly, so it is recorded as a judgement.

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
- **The 9-column model feature set above is what reaches the estimator.** `Year`,
  `Quarter`, `FlightDate` and `Flight_Number_Reporting_Airline` survive in the dataframe
  but are excluded from the feature matrix, and a markdown cell says why — "kept for
  convenience" is a fine reason to keep a *column* and not a reason to feed it to a *model*
- `reports/feature_audit.md` updated so its Pre-Booking table distinguishes
  "pre-booking" from "model input" — they are not the same set

**Learn this first**

**MUST:** nothing new — `reports/feature_audit.md` already contains the
pre-booking/post-flight classification.

**Optional:** CampusX **Day 34 — Handling Date and Time Variables**
([notebook](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day34-handling-date-and-time)),
if datetime parsing needs a refresher.

---

### Issue 2 — Build the temporal train/test split and CV strategy

**Labels:** `gate-1`, `evaluation`

Split into **train pool / test** by calendar date. Not by row position — a single date
must not be divided across the two partitions. There is no separate fixed validation
partition: every candidate comparison in Gate 1–2 (Issues 3, 4, 5, 8, 9) is scored with
`TimeSeriesSplit` cross-validation over the train pool, so each fold's held-out rows act
as validation. The test partition is touched exactly once, at Issue 10.

**Run `TimeSeriesSplit` over the array of unique dates, not over the rows.** This is the
part that is easy to get wrong. `TimeSeriesSplit` splits on row position and knows nothing
about `FlightDate`; the docs require that "samples must be equally spaced" so that "each
test set covers the same time duration." Flight rows are *not* equally spaced — this
dataset runs 242–555 flights/day (2.29×), with summer months carrying ~34% more traffic
than winter (Jun–Aug daily mean 513 vs Dec–Feb 383). Splitting raw rows therefore produces both defects at once, verified against
the real data:

- every one of the 5 fold boundaries lands mid-date, splitting one calendar date across
  train and validation — the exact thing this issue forbids
- folds hold identical row counts (47,608) but cover 94–122 days, so per-fold MAEs are not
  comparable

Both figures are measured on the train pool — 639 dates, 285,649 rows — since that is the
only partition the splitter ever sees. The defect is not an artifact of that choice:
splitting all 324,490 rows produces the same mid-date boundaries and spans of 106–136 days.

Splitting the unique dates instead fixes both: dates *are* equally spaced (one per
day), so every fold covers an identical span and no date can straddle a boundary. Take the
fold indices from the date array, then map them back to row masks with
`df['FlightDate'].isin(dates[idx])`.

**Verified design** (recompute rather than trusting these numbers):

| partition | range | dates | rows |
|---|---|---|---|
| train pool | 2024-01-01 → 2025-09-30 | 639 | 285,649 |
| test | 2025-10-01 → 2025-12-31 | 92 | 38,841 (12.0%) |

`TimeSeriesSplit(n_splits=5)` over the 639 train-pool dates gives five folds of exactly
106 validation days each, with train windows expanding 109 → 533 days (successive train
sets are supersets, as the docs describe).

**Done when**
- Split is defined by a single date boundary (train pool / test), stated in the notebook
- The splitter is run over unique dates and mapped back to rows — asserted in code by
  checking that no date appears in both sides of any fold, and none in both partitions
- Train pool is strictly earlier in time than test
- A splitter is instantiated once over the train pool (fixed `n_splits`) and reused for
  every candidate comparison in Issues 3, 4, 5, 8, 9 — same folds, so comparisons are
  apples-to-apples
- Row counts, date counts, and date ranges printed per fold and per partition
- `gap=` considered and the choice recorded. Default `gap=0` is defensible here because
  no feature is lagged or rolling — every predictor is known at booking time — so adjacent
  dates share no constructed value. Revisit if a lagged feature is ever added.
- The seasonal limitation is written down: a temporal holdout forces the test partition to
  be Q4-only, so the final number reflects winter operations. Keep month and day-of-week
  in the feature set, and restate this in the Issue 6 note and the Issue 15 disclaimer.
- `FlightDate` is used here as the **split key only**. It does not enter the feature
  matrix — see the feature-set table above

**Learn this first**

**MUST**
- [sklearn — Cross-validation user guide](https://scikit-learn.org/stable/modules/cross_validation.html)
  — two lines govern this issue. On why there is no validation partition:
  > "A test set should still be held out for final evaluation, but the validation set is
  > no longer needed when doing CV."

  And on why ordinary k-fold is wrong here:
  > "classical cross-validation techniques such as `KFold` and `ShuffleSplit` assume the
  > samples are independent and identically distributed, and would result in unreasonable
  > correlation between training and testing instances (yielding poor estimates of
  > generalization error) on time series data."
- [sklearn — `TimeSeriesSplit` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
  — the constraint that drives the split-on-dates decision above:
  > "To ensure comparable metrics across folds, samples must be equally spaced. Once this
  > condition is met, each test set covers the same time duration, while the train set
  > size accumulates data from previous splits."

  Also read `n_splits`, `gap`, and `max_train_size`.

**Optional**
- Baker, [Training and Validation](https://ggbaker.ca/data-science/content/ml.html#trainvalid)
  — see the warning below before using it

> **CampusX gap.** The 100 Days series covers train/test splitting generically but has no
> episode on time-aware splitting or `TimeSeriesSplit`. This issue has no CampusX
> equivalent — use the docs.
>
> **Baker gap too.** Baker's [Training and Validation](https://ggbaker.ca/data-science/content/ml.html#trainvalid)
> section is worth reading for the overfitting argument, but he demonstrates
> `train_test_split` with its default random shuffle and never covers temporal splits or
> cross-validation. Read it for the reasoning, not the split call — copying that pattern
> here is the exact mistake this issue exists to prevent.

---

### Issue 3 — Build the naive per-profile baseline

**Labels:** `gate-1`, `evaluation`

Build the naive comparator the model has to beat: a per-profile historical median,
scored with the same `TimeSeriesSplit` folds from Issue 2 as every other candidate.

**Why the median and not the mean.** This is not an arbitrary choice — it is the *correct*
partner for an MAE-scored project. The median is the point estimate that minimises absolute
error, so a median baseline is the strongest possible naive comparator under MAE, which is
what makes beating it meaningful. `ArrDelay` also makes the difference concrete: mean 4.91
against median −4, because 57.2% of SEA departures arrive early and a thin right tail
(p99 = 154, max = 3359) drags the average up. A mean baseline would predict a flight that
essentially never happens. sklearn pairs the two the same way — per the
[`RandomForestRegressor` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html),
`criterion="absolute_error"` is
> "mean absolute error, which minimizes the L1 loss using the median of each terminal node"

**The thin-support problem, measured.** Grouping on the D-2 profile grain gives 4,483
profiles with a **median of 23 rows each, 36.4% holding fewer than 10 rows, and a 10th
percentile of 1 row**. Inside an early `TimeSeriesSplit` fold — whose training window is
only 109 days — that tail is much worse. So the fallback is not an edge case to handle
defensively; it is a routine path that will fire on a large fraction of held-out rows, and
its design materially affects the baseline number.

**Fallback ladder** (this is the D-5 policy at the baseline layer):
1. median for (carrier, flight number, destination), if the profile has at least *n* rows
   in this fold's training data — pick *n*, justify it, keep it fixed across folds
2. else median for (carrier, destination) — 216 groups, median 974 rows
3. else the global median of this fold's training rows

**Done when**
- Within each fold, the baseline is fit on that fold's training rows only
- Profiles present in a fold's held-out rows but absent from that fold's training rows
  are handled explicitly by the ladder above, and the chosen *n* is documented
- **How often each rung fires is counted and reported per fold** — a baseline that lands on
  rung 3 for most rows is a global-median baseline wearing a per-profile label, and Issue 6
  must not describe it as something it isn't
- Baseline produces a prediction for every held-out row in every fold — no silent nulls
- Mean and std MAE across folds recorded
- Lift is also reported **restricted to thin-support profiles**, since Issue 10 and Issue
  13 both turn on that subgroup

**Learn this first**

**MUST:** nothing new. This is the groupby → median → merge → fillna pattern;
`pandas/pandas_numpy_practice.ipynb` (the BikeIndia set) covers the mechanics. Apply it
inside each CV fold instead of once.

---

### Issue 4 — Encode categorical features

**Labels:** `gate-1`, `preprocessing`

Build the preprocessing `ColumnTransformer` that every model shares. One-hot encode the
categorical pre-booking features, and fix the time encoding. Encoding is fit on training
data only. One-hot was chosen over label encoding (imposes false ordinal structure on
airport codes) and target encoding (adds leakage risk) — that decision is closed.

This issue grew: it now owns three preprocessing decisions the original version was silent
on. Each is here because getting it wrong changes the Issue 5 and Issue 9 numbers.

#### 4a — `CRSDepTime` / `CRSArrTime` are clock readings, not quantities

Verified in the data: both range 1–2359, with **zero** occurrences of `0`, `2400`, or a
minute-part ≥ 60 — so the values are clean HHMM, and the problem is purely the encoding's
shape. Handed to a linear model as raw integers, 10:59 → 11:00 is a **+41 unit** jump for
one minute of real time, and 23:59 → 00:01 is **−2358** for two minutes. The scale is
non-linear in time and discontinuous at midnight.

Departure time is plausibly the single strongest pre-booking predictor here — late-day
flights inherit the day's accumulated disruption. Handing the linear model a broken
encoding of its best feature is how it loses to the naive baseline for a reason that has
nothing to do with linear models.

The [Time-related feature engineering example](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html)
addresses exactly this, and notes that it does *not* hit all model families equally:
> "Note that the time related features are passed as is, i.e. without processing them. But
> this is not much of a problem for tree-based models as they can learn a non-monotonic
> relationship between ordinal input features and the target."

**Do:** convert both to minutes-since-midnight (`(t // 100) * 60 + (t % 100)`, range
0–1439). For the linear model, add a periodic encoding on top — sine/cosine, or
[`SplineTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html)
with `extrapolation="periodic"`, which exists for this purpose:
> "If 'periodic', periodic splines with a periodicity equal to the distance between the
> first and last knot are used. Periodic splines enforce equal function values and
> derivatives at the first and last knot. For example, this makes it possible to avoid
> introducing an arbitrary jump between Dec 31st and Jan 1st in spline features derived
> from a naturally periodic "day-of-year" input feature. In this case it is recommended to
> manually set the knot values to control the period."

Confirmed working on the pinned 1.7.2. The tree model in Issue 8 can take the plain
minutes-since-midnight form without the periodic step — but see Issue 9 on keeping the
comparison honest.

`Month` and `DayOfWeek` have the same wrap-around property in principle, but at 12 and 7
levels one-hot encoding handles it for free and yields interpretable coefficients. Periodic
encoding is reserved for the 1,440-value time features, where one-hot would be absurd.

#### 4b — `drop="first"`, because `LinearRegression` is unregularised

Full dummy encoding plus an intercept makes the columns perfectly collinear, and
[`OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
names the situation exactly:
> "Specifies a methodology to use to drop one of the categories per feature. This is useful
> in situations where perfectly collinear features cause problems, such as when feeding the
> resulting data into an unregularized linear regression model."

This is not cosmetic. Issue 5 requires that you can say in one sentence what a coefficient
means — and under full dummy encoding with an intercept the coefficients are **not uniquely
determined**, so that deliverable is unachievable as written. The
[Linear Models user guide](https://scikit-learn.org/stable/modules/linear_model.html) spells
out the consequence:
> "The coefficient estimates for Ordinary Least Squares rely on the independence of the
> features. When features are correlated and some columns of the design matrix X have an
> approximately linear dependence, the design matrix becomes close to singular and as a
> result, the least-squares estimate becomes highly sensitive to random errors in the
> observed target, producing a large variance."

Record the tradeoff the same page states, because it decides what happens if this project
ever moves to Ridge or Lasso:
> "However, dropping one category breaks the symmetry of the original representation and
> can therefore induce a bias in downstream models, for instance for penalized linear
> classification or regression models."

So: `drop="first"` for the OLS baseline, revisit if the estimator ever gains a penalty.

#### 4c — The unseen-category policy (the D-5 single source)

**Use `handle_unknown="infrequent_if_exist"` with `min_frequency` set — not
`handle_unknown="ignore"`.**

There is a trap here, verified on the pinned 1.7.2. `drop="first"` combined with
`handle_unknown="ignore"` is *accepted* by sklearn, but it is silently ambiguous: an unseen
category encodes to all-zeros, which is **byte-identical to the encoding of the dropped
reference category**. The docs say so plainly for `'ignore'` —
> "When an unknown category is encountered during transform, the resulting one-hot encoded
> columns for this feature will be all zeros."

— and with `drop="first"` the reference category *is* all zeros. An unfamiliar carrier
would therefore be predicted as if it were Alaska, with no warning. `'infrequent_if_exist'`
avoids this by giving unknowns their own column:
> "When an unknown category is encountered during transform, the resulting one-hot encoded
> columns for this feature will map to the infrequent category if it exists."

Tested: with `drop="first"`, `min_frequency=2`, categories `{AS×3, DL×2, F9×1}`, an unseen
`ZZ` transforms to `[0, 1]` — the infrequent column — while the dropped reference `AS`
transforms to `[0, 0]`. Distinct, as required. The interaction with dropping is documented:
> "When `max_categories` or `min_frequency` is configured to group infrequent categories,
> the dropping behavior is handled after the grouping."

Because `Flight_Number_Reporting_Airline` is out (D-1), the remaining categoricals are
low-cardinality and `min_frequency` mostly serves to *manufacture* the unknown bucket
rather than to compress width. Set it deliberately and say so — a rare destination served a
handful of times is genuinely better pooled than given its own coefficient. The mechanisms,
if width ever becomes a problem again:
> **min_frequency**: "Specifies the minimum frequency below which a category will be
> considered infrequent. If int, categories with a smaller cardinality will be considered
> infrequent. If float, categories with a smaller cardinality than `min_frequency *
> n_samples` will be considered infrequent."

> **max_categories**: "Specifies an upper limit to the number of output features for each
> input feature when considering infrequent categories."

This configuration is the policy Issue 3's ladder and Issue 14's user-facing message both
refer back to. Change it here, and update both.

**Done when**
- Encoder lives inside a `Pipeline` (preprocessing + estimator), so `cross_val_score` /
  `cross_validate` refits it fresh on each fold's training rows — never fit once on the
  whole train pool. This is the same `Pipeline` object Issue 11 later persists.
- Time columns converted to minutes-since-midnight, with a periodic encoding for the linear
  model — and a markdown cell showing the 23:59 → 00:01 jump the conversion fixes
- `drop="first"` set, with the reason recorded
- `handle_unknown="infrequent_if_exist"` set, with `min_frequency` chosen and justified;
  a cell demonstrates that an unseen category and the dropped reference category produce
  **different** encodings
- `Year`, `Quarter`, `FlightDate` and `Flight_Number_Reporting_Airline` are absent from the
  transformer's column lists
- Resulting feature count recorded and compared against the ≈2,698 the naive scheme would
  have produced — this number matters for the write-up

**Learn this first**

**MUST — in this order**
1. CampusX **Day 27 — One Hot Encoding**
   ([notebook](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day27-one-hot-encoding)).
   Day 26 covers ordinal/label encoding if you want to see concretely why it was rejected here.
2. CampusX **Day 28 — ColumnTransformer**
   ([notebook](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day28-column-transformer))
   — the practical answer to applying different transforms to different column groups
3. CampusX **Day 29 — ML Pipelines A-Z**
   ([notebook](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day29-sklearn-pipelines))
   — required starting here, not at Issue 11: the encoder only avoids fold-to-fold leakage
   if it's wrapped in a `Pipeline` before it's handed to cross-validation
4. [sklearn — `OneHotEncoder` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
   — the authority on current behaviour for `drop`, `min_frequency`, and categories not seen
   during fit. CampusX teaches the concept; this page governs the implementation.
5. [sklearn — Common pitfalls: data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
   — states the rule plainly:
   > "Always split the data into train and test subsets first, particularly before any
   > preprocessing steps."

   and why the `Pipeline` is the mechanism rather than discipline:
   > "The scikit-learn pipeline is a great way to prevent data leakage as it ensures that
   > the appropriate method is performed on the correct data subset. The pipeline is ideal
   > for use in cross-validation and hyper-parameter tuning functions."
6. [sklearn — Time-related feature engineering example](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html)
   — read the sections on raw ordinal time features and on periodic spline features

**Optional**
- [sklearn — Preprocessing data user guide](https://scikit-learn.org/stable/modules/preprocessing.html),
  the categorical features section
- [sklearn — `SplineTransformer` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html)
  if you take the spline route over sine/cosine

---

### Issue 5 — Fit the linear regression and score it

**Labels:** `gate-1`, `modelling`

Put the linear model inside the Issue 4 `Pipeline` and score it with `cross_val_score`
(or `cross_validate`) over the Issue 2 `TimeSeriesSplit` folds. This is the first
genuinely new modelling step in the project.

The coefficient-interpretation goal below depends on Issue 4 being done properly: without
`drop="first"` the coefficients are not uniquely determined, and without the time fix the
`dep_minutes` coefficient is a slope through a broken scale. Do not attempt the
interpretation on a pipeline that skipped 4a and 4b.

**A mismatch to state, not to fix.** `LinearRegression` minimises *squared* error while
this project reports MAE. With `ArrDelay` reaching 3359, a single 300-minute miss
contributes as much to the objective (90,000) as one hundred 30-minute misses combined
(100 × 900) — so the fit chases catastrophic delays that MAE barely notices. Keep
`LinearRegression` as-is; the baseline's job is to be a plain, well-understood reference.
Record the mismatch in Issue 12 as a deliberate tradeoff. It is *not* a doc-backed sklearn
recommendation — the metrics page carries no guidance on metric choice under heavy tails —
it follows from the definitions of the two loss functions.

**Done when**
- Model only ever sees each fold's training rows during that fold's fit — never the
  whole train pool at once
- Mean and std CV MAE recorded alongside the Issue 3 baseline's mean/std MAE, same folds
- Coefficients inspected from a separate copy of the pipeline refit on the *full* train
  pool, for interpretation only — that refit is not what CV scored, and is not the model
  compared in Issue 9. You can say in one sentence what the model is doing and what a
  coefficient means here, **including which category each one-hot coefficient is measured
  relative to** (the `drop="first"` reference)
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
- [sklearn — `cross_val_score` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)

---

### Issue 6 — Write the evaluation note

**Labels:** `gate-1`, `docs`

Write `reports/baseline_evaluation.md`: what was compared, on what split, with what
result, and whether the linear model actually beat the naive baseline.

**Done when**
- Split design stated: train pool / test date boundary, plus the `TimeSeriesSplit`
  fold count used for CV
- Mean ± std CV MAE reported for both baseline and model, on the same folds, with lift
  over the naive baseline
- Honest target restated: delay conditional on the flight operating and arriving,
  carrying no cancellation or diversion signal
- Seasonal caveat stated (from Issue 2): the temporal holdout makes the test partition
  Q4-only, so the headline test number describes winter operations and is not a
  year-round average
- **What the naive baseline actually was** is stated honestly, using the Issue 3 rung
  counts — if most held-out rows fell through to the global median, say so
- If the model did **not** beat the naive baseline, the note diagnoses which cause
  applies rather than declaring failure — the execution plan §8 lists five candidate
  causes; check these two first, because both are specific to this dataset and neither is
  on that list:
  - **broken time encoding** — if Issue 4a was skipped or half-done, the model's best
    feature was fed to it on a scale that jumps −2358 across midnight. Confirm 4a landed
    before diagnosing anything else.
  - **the naive baseline is genuinely strong here.** A per-profile median on 4,483
    profiles is close to a lookup of the answer. Losing to it is informative, not
    embarrassing — but say which of the two it is.

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

Fit a RandomForest inside the same `Pipeline` shape and score it with the same
`TimeSeriesSplit` folds as the linear model. This is "attempt, cut if needed" — if it
overruns, ship the linear baseline and note RF as deferred.

**Bound the trees. This is mandatory, and it is a deployability constraint rather than
tuning.** Measured on this project's own train pool (285,649 rows, sklearn 1.7.2):

| configuration | trees | artifact | nodes | fit time |
|---|---|---|---|---|
| sklearn defaults | 10 | **251.6 MB** | 3,493,834 | 75.4 s |
| sklearn defaults | 100 (the default) | **≈2.5 GB** (extrapolated ×10) | ≈35 M | ~12 min |
| `max_depth=12, min_samples_leaf=50` | 100 | **8.7 MB** | 119,936 | 11.4 s |

A default forest is a **290× larger artifact and 6.6× slower to fit than the bounded one at
ten times the tree count.** The [`RandomForestRegressor` reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
names the cause:
> "The default values for the parameters controlling the size of the trees (e.g.
> `max_depth`, `min_samples_leaf`, etc.) lead to fully grown and unpruned trees which can
> potentially be very large on some data sets. To reduce memory consumption, the complexity
> and size of the trees should be controlled by setting those parameter values."

And [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
gives the ceiling that makes 2.5 GB fatal rather than merely wasteful:
> "Memory: 690MB minimum, 2.7GBs maximum"

> "If your app meets or exceeds its limits, it may slow down from throttling or become
> nonfunctional."

2.5 GB is nearly four times the minimum allocation and brushes the maximum before Python,
pandas, or the app itself has loaded. Without this bound, the failure surfaces at Gate 3
deployment — six days after Gate 2 locked the choice.

**Artifact budget: 200 MB.** Comfortable against the 690 MB floor with room for the
interpreter and dataframes. The bounded config above uses 4% of it.

**Done when**
- Scored on the identical folds and preprocessing `Pipeline` as the linear model
- `max_depth` and `min_samples_leaf` set explicitly, with the chosen values recorded — do
  not ship sklearn's defaults
- **Serialized artifact size measured and recorded**, and under the 200 MB budget
- Mean and std CV MAE recorded on the same basis, so the comparison is real
- Training time noted — it now trains once per fold instead of once

**Learn this first**

**MUST — in this order**
1. CampusX **Day 65 — Random Forest**
   ([notebook](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day65-random-forest))
   — for what a forest is and why it beats a single tree
2. [sklearn — `RandomForestRegressor` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
   — to translate that understanding to your problem. Your target is continuous, and the
   regressor exposes different split criteria from the classifier. Read `n_estimators`,
   `max_depth`, `min_samples_leaf`, and the memory-consumption note.

**Optional**
- Baker, [Ensembles](https://ggbaker.ca/data-science/content/ml-classif.html#ensemble) /
  [Random Forests](https://ggbaker.ca/data-science/content/ml-classif.html#rand-forest)
- [sklearn — Ensembles user guide](https://scikit-learn.org/stable/modules/ensemble.html),
  the forest section

> Both CampusX and Baker teach forests through **classification** examples. The forest
> concept transfers directly; the estimator, target, and evaluation metrics do not.

---

### Issue 8b — Hyperparameter tuning *(optional — first to cut)*

**Labels:** `gate-2`, `modelling`, `optional`

Per **D-3** there is no tuning before Gate 2. This issue exists so that the absence is a
recorded decision rather than a gap someone notices at review.

Open it only if Issues 1–9 are done with time in hand. If it stays closed, Issue 12 states
"no hyperparameter search was performed" as a tradeoff, and that is a complete answer for a
baseline-versus-RF comparison.

**If it is opened, done when**
- Search runs over the Issue 2 folds via the same `Pipeline` — the splitter object is
  passed as `cv=`, not re-created
- The search space respects the Issue 8 artifact budget: `max_depth` and `min_samples_leaf`
  bounded so no candidate can exceed 200 MB
- Best configuration reported with its CV MAE, next to the untuned RF's, so the value of
  tuning is visible rather than assumed
- The test partition is still untouched

**Learn this first**

**MUST:** [sklearn — Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html)
— now listed in `DOCUMENTATION_SOURCES.md`.

---

### Issue 9 — Choose the final model on cross-validated MAE

**Labels:** `gate-2`, `evaluation`

Compare candidates on **cross-validated MAE** (mean ± std across the Issue 2
`TimeSeriesSplit` folds). The test set is untouched until the choice is already made.

**Keep the comparison about model class, not encoding.** Issue 4a gives the linear model a
periodic time encoding that the tree does not need. If the RF is scored on raw ordinal
minutes while the linear model is scored on splines, a win for RF is partly a win for
*feature engineering*, and the write-up would attribute it to the wrong cause. Either score
both on the identical transformed features, or score the RF both ways and report both.
State which you did. The
[time-related feature engineering example](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html)
is the reason the asymmetry exists at all.

**Done when**
- All candidate comparisons documented on CV MAE, same folds for every candidate
- The encoding each candidate received is stated, and the asymmetry above is either
  removed or reported
- **Artifact size recorded per candidate**, so Issue 13's deployability criterion can be
  checked before the choice is locked
- One model selected, with the reason written down before the test run
- Test set has still not been used

**Learn this first**

**MUST:** nothing new — [sklearn Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
was already read at Issue 4. Reread the one line if useful:
> "Test data should never be used to make choices about the model. The general rule is to
> never call `fit` on the test data."

---

### Issue 10 — Honest final test-set evaluation

**Labels:** `gate-2`, `evaluation`

Refit the chosen model's `Pipeline` on the **entire train pool** (all folds combined —
Issue 9's CV scoring never saw the full pool at once), then evaluate that single refit
model on the test partition and record the result. The prohibition is on changing the
modelling process in response to what you see — not on the mechanical act of calling
predict.

**Done when**
- Chosen `Pipeline` refit on the full train pool before touching test — this is a
  different fit than any of the Issue 9 CV folds
- Test MAE and RMSE recorded for the chosen model and the naive baseline
- **Which of the two the product claim rests on is stated, and it is MAE.** They will
  disagree here and the disagreement is not a bug: with p75 = 11 and max = 3359, RMSE is
  dominated by a handful of extreme delays, while MAE describes the typical flight. A
  traveler comparing two flights is asking about the typical case. Report both, lead with
  MAE, and say why — this is a project judgement, not a documented sklearn rule.
- Train-vs-test gap reported and interpreted (train score = refit on full train pool;
  compare against the Issue 9 CV mean, not a single validation number)
- Lift over the naive baseline reported, including on thin-support flight profiles
- Test performance is not used to alter model selection, preprocessing, features, or
  hyperparameters. Any rerun after the first evaluation is for reproducibility or to
  correct a documented implementation bug — never to improve the number.

**Learn this first**

**MUST**
- Baker, [Training and Validation](https://ggbaker.ca/data-science/content/ml.html#trainvalid)
  — states the criterion this issue turns on: a big drop from training score to CV score
  is usually a sign of overfitting

**Optional:** CampusX **Day 49 — Regression Metrics**
([notebook](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day49-regression-metrics))
— already seen at Issue 5; rewatch only if MAE-versus-RMSE interpretation is unclear.

---

### Issue 11 — Serialize model and preprocessing

**Labels:** `gate-2`, `deployment-prep`

Persist the exact `Pipeline` object Issue 10 refit on the full train pool, so the app
reproduces training-time transformations exactly.

**Pinning is already done** — `requirements.txt` was pinned and split from
`requirements-dev.txt` in commit `d50f21e`, with `streamlit` added and `seaborn` removed.
This issue no longer has to do that work; it has to *not undo* it. If the training
environment changes, re-pin from `pip freeze` before persisting anything, because
[Model persistence](https://scikit-learn.org/stable/model_persistence.html) is blunt about
the consequence:
> "There are no supported ways to load a model trained with a different version of
> scikit-learn. While using skops.io, joblib, pickle, or cloudpickle, models saved using
> one version of scikit-learn might load in other versions, however, this is entirely
> unsupported and inadvisable."

The same page lists metadata to store alongside the artifact — adopt the list:
> "In order to rebuild a similar model with future versions of scikit-learn, additional
> metadata should be saved along the pickled model: The training data, e.g. a reference to
> an immutable snapshot — The Python source code used to generate the model — The versions
> of scikit-learn and its dependencies — The cross validation score obtained on the
> training data"

**Done when**
- The Issue 10 full-train-pool refit — preprocessing and estimator together as one
  `Pipeline`, the same shape built back in Issue 4 — is what gets persisted, not a
  separate refit
- A fresh process can load the artifact and produce a prediction without refitting
- A sidecar metadata file records: the interim CSV's path and row count, the notebook that
  produced the model, `sklearn` / `pandas` / `numpy` / `joblib` versions, the Issue 9 CV
  MAE, and the train/test date boundary
- Artifact size recorded and checked against the Issue 8 budget
- Library versions confirmed still pinned and matching the training environment
- Artifact paths and the gitignore decision documented

**Learn this first**

**MUST**
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
- **Records these deliberate tradeoffs**, each as a choice with a reason rather than an
  oversight:
  1. **MAE as the headline metric**, despite `LinearRegression` optimising squared error.
     A traveler-facing claim is about the typical case, not the tail. A project judgement,
     not a documented sklearn recommendation.
  2. **Q4-only test partition** — inherent to an honest temporal holdout on two years of
     data. The alternative, a random split, would be dishonest.
  3. **No hyperparameter tuning** (D-3), and why that is acceptable for a
     baseline-versus-RF comparison.
  4. **Tree size bounded for deployability** (Issue 8) — note that this may have cost
     accuracy, and that the cost was accepted knowingly.
  5. **`Flight_Number_Reporting_Airline` dropped as a feature** (D-1) — accepting the loss
     of route-specific signal in exchange for a model that generalises rather than
     memorises.
  6. **`Year` and `FlightDate` dropped as features** — they do not recur at prediction time.
  7. **Scope excludes cancellation and diversion risk** — the single most important
     honesty guardrail in the project.

---

### Issue 13 — Gate 2 review

**Labels:** `gate-2`, `review`

**Pass criteria:** no leakage; honest temporal CV (`TimeSeriesSplit`) plus untouched
final holdout; ML beats the naive baseline (especially on thin-support flights); choice
justified; **the chosen model's serialized artifact is under the 200 MB budget and the
number was measured, not assumed.**

That last criterion is here because Gate 2 selects and Gate 3 deploys, six days apart.
Discovering at Gate 3 that the Gate 2 winner cannot fit in Community Cloud's 690 MB floor
would mean re-running model selection with two days left.

**Fail path:** ship the linear baseline, drop RF.

---

## Milestone: Gate 3 — Deployed (Aug 30)

### Issue 14 — Two-flight comparison app

**Labels:** `gate-3`, `deployment`

Build the Streamlit app: user picks two flights, app returns a reliability comparison.

**Cache the model, or the app will not survive its own resource limits.** Streamlit reruns
the entire script top-to-bottom on every widget interaction, so without caching the model
artifact is deserialised from disk each time a user changes a dropdown. Per
[Streamlit's caching docs](https://docs.streamlit.io/develop/concepts/architecture/caching):
> "Long-running functions run again and again, which slows down your app. Objects get
> recreated again and again, which makes it hard to persist them across reruns or sessions."

and the specific remedy:
> "st.cache_resource is the recommended way to cache global resources like ML models or
> database connections."

with the other decorator for the data side:
> "st.cache_data is the recommended way to cache computations that return data: loading a
> DataFrame from CSV, transforming a NumPy array, querying an API, or any other function
> that returns a serializable data object."

Combined with the 690 MB memory floor, this is not a performance nicety — it is the
difference between a working app and one showing "🤯 This app has gone over its resource
limits."

**Done when**
- Runs locally end-to-end
- Model loading wrapped in `@st.cache_resource`; the Issue 3 profile-median lookup and any
  reference CSV wrapped in `@st.cache_data`
- Accepts two flight profiles and returns both predictions plus a comparison
- Inputs restricted to pre-booking fields only — the interface must not ask for
  anything unavailable at booking time, and must not ask for `Year` or `FlightDate` as
  model inputs since neither is one (a date picker is still fine for deriving `Month`,
  `DayOfWeek`, and `DayofMonth`)
- Unseen categories produce a graceful message, not a stack trace — and the message is
  consistent with the **Issue 4c policy**, which is the single source of truth for this
  behaviour (D-5). The encoder maps an unknown to the infrequent bucket rather than
  silently to the reference category; the UI should say the prediction is based on limited
  history for that carrier or destination.
- Flight-number input is used to look up the Issue 3 profile, not passed to the model (D-1)

**Learn this first**

**MUST**
- [Streamlit — Get started](https://docs.streamlit.io/get-started)
- [Streamlit — Caching](https://docs.streamlit.io/develop/concepts/architecture/caching)
- [Streamlit — Deploy overview](https://docs.streamlit.io/deploy)

---

### Issue 15 — Guardrail and disclaimer text

**Labels:** `gate-3`, `deployment`, `docs`

Add honest framing to the app surface itself.

**Done when**
- BTS attribution present
- Disclaimer states the model predicts delay conditional on the flight operating and
  arriving, and says nothing about cancellation or diversion risk
- **Seasonal caveat present**: the accuracy figure was measured on an October–December test
  partition, so it describes winter operations rather than a year-round average. This is
  the same caveat as Issue 2 and Issue 6, stated where a user will actually see it.
- Competitive-tools acknowledgment present
- No "safe", "guaranteed", or equivalent claims anywhere in the interface
- Any headline number is phrased against the median, not the mean — "typically lands about
  4 minutes early" is defensible; "average delay 4.9 minutes" is arithmetically true and
  quietly misleading, because 57.2% of these flights arrive early

---

### Issue 16 — Deploy to Streamlit Community Cloud

**Labels:** `gate-3`, `deployment`

**Done when**
- App is live at a public URL
- A two-flight comparison runs successfully on the deployed instance
- Dependencies resolve from `requirements.txt` in the cloud environment — note that
  `jupyter` is deliberately in `requirements-dev.txt` only, so it must not appear here
- Python version set in Community Cloud's advanced settings to match the training
  environment (3.13.x), since `requirements.txt` cannot pin it
- **Deployed memory footprint checked against the limits**, not just assumed:
  > "CPU: 0.078 cores minimum, 2 cores maximum" · "Memory: 690MB minimum, 2.7GBs maximum"

  Confirm the loaded artifact plus pandas plus the app sits well inside the floor. If it
  does not, the fallback is the linear pipeline, which is kilobytes.
- Data-file strategy works remotely — the interim CSV is gitignored, so the app must
  not depend on a file that was never pushed

**Learn this first**

**MUST**
- [Streamlit — Prep and deploy on Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
- [Streamlit — App dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Streamlit — Manage your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
  (resource limits)

---

### Issue 17 — Complete the README

**Labels:** `gate-3`, `docs`

**Done when** it covers: honest target definition; feature audit summary and the
pre-booking/post-flight split; **the further distinction between pre-booking columns and
model inputs** (`Year`, `Quarter`, `FlightDate`, flight number are the first but not the
second, and why); cancelled and diverted row handling; temporal holdout design; baseline
comparison and final metrics; the Issue 12 tradeoff list; competitive-tools
acknowledgment; limitations; live app link; reproduction steps.

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
- Issue 8b is the designated first cut, then Issue 8, if Week 4 runs short.
- Cut order when late: stretch features → model complexity → duplicate docs.
  Never cut evaluation integrity, a working deployment, or explanation quality.
- **Do not cut Issue 8's tree bounding as if it were tuning.** It is the difference between
  an 8.7 MB artifact and a 2.5 GB one, and Gate 3 depends on it.
