# Learnings & Gotchas

Decisions made while building this project, the evidence that drove each one, and what each
one costs. Every number here was computed against the actual dataset rather than recalled —
several entries exist specifically because a plausible-sounding assumption did not survive
that check.

**How to use this file before an interview:** read [The arc](#the-arc) and
[Three stories with a reversal](#three-stories-with-a-reversal) — that's the whole project in
about a page, and it's enough to talk through unprompted. Everything below the divider is
backup for follow-up questions; find the section by name when you're asked something specific.

---

## The arc

**The project in one sentence:** given two SEA-departing flights a traveler is choosing
between, predict how late each one will land, using only what's knowable at booking time.
324,490 flights, 2024–2025, from US DOT BTS on-time data.

The order things happened, and the decision at each step:

| # | Step | Decision | The number behind it |
|---|---|---|---|
| 1 | Framed the problem | Regression on tabular rows, **not** forecasting — but evaluation still has to be time-aware | 731 dates × 242–555 rows/day; a series has one row per step |
| 2 | Audited the columns | Dropped ~70 post-flight columns as leakage; dropped a second group as redundant | `DepDelay` would leak; `Origin` is `SEA` in 100% of rows |
| 3 | Handled the nulls | Dropped 4,069 null-target rows — which *scopes the model* to "delay given the flight operates" | 3,161 cancelled + 908 diverted = all 4,069, zero unexplained |
| 4 | Split the data | Split on the **unique-date array**, not rows; single cutoff `2025-10-01` | Row splits put all 5 fold boundaries mid-date; folds spanned 108–136 days at equal row counts |
| 5 | Built a baseline | Per-profile **median** (not mean), with a 3-rung fallback ladder at n ≥ 10 | MAE **19.95** min (std 1.21); n=10 excludes 5.0% of rows but 39.9% of profiles |
| 6 | Tested the baseline's parts | Searched 11 groupings for rung 2 instead of guessing; measured the ceiling | carrier+hour lift **0.415** vs carrier+dest 0.149; total headroom only **1.70** min |
| 7 | Encoded the features | Cyclical time encoding; `drop="first"`; two encoders split by open vs closed category sets | 23:59→00:01 is 1,438 apart raw, **0.0087** apart as sin/cos |
| 8 | Found what CV couldn't | An unseen carrier encodes identically to the dropped reference — fixed at the app layer | Rarest carrier 744 rows, so no bucket at any usable threshold; 0 unseen carriers in all 5 folds |

**If you only remember one line about each:** the split had a bug that looked fine, the
baseline was made deliberately hard to beat, and the encoder had a failure mode that
cross-validation is structurally incapable of catching.

---

## Three stories with a reversal

These are the ones worth telling, because in each case the obvious thing was wrong and there
is a number proving it.

### 1. "The cross-validation was splitting mid-day and I didn't notice"

`TimeSeriesSplit` sounds like it splits on time. It splits on **array position**. Handed raw
flight rows — which run 242–555 per day — its fold boundaries land in the middle of calendar
dates, putting the same day on both sides of a train/validation split.

I caught it because the docs carry a qualifier I'd have skipped if I'd cited from memory:
*"samples must be equally spaced."* Flight rows aren't; calendar dates are. The fix is to run
the splitter over the array of unique dates and map back to rows with `.isin()`.

**Proof it mattered:** on raw rows, all 5 fold boundaries fell mid-date and validation folds
spanned 108/125/136/106/124 distinct days despite identical row counts. On the date array,
exactly equal folds, zero date overlap.

*(Detail: [§3](#3-splitting-for-honest-evaluation))*

### 2. "My baseline's middle rung was doing almost nothing"

The fallback ladder's rung 2 originally grouped by carrier + destination. Comparing the full
ladder's MAE against a version without it showed almost no difference — but that comparison
is diluted, because most rows never reach rung 2 and contribute identical error to both sides.

Scoped correctly — only the rows that actually needed a fallback — rung 2 was earning a lift
of **0.149 minutes**. Basically nothing.

Rather than guess a replacement I scored 11 candidate groupings on the same folds. Carrier +
departure hour won at **0.415**, nearly triple. The surprise: **destination was actively
hurting.** Carrier alone (0.210) beat carrier + destination (0.149), and adding month
collapsed it to −0.001. Destination wasn't adding signal, it was fragmenting groups into
smaller, noisier ones. Departure hour carries a real ~10-minute effect across the day that
the route doesn't — early flights run ahead of schedule, later ones inherit accumulated
delay.

**The honest footnote I put in the write-up:** I selected that grouping on the same CV folds
used for every other comparison. That's mild selection bias. It's defensible because a
stronger baseline makes the later "does the model beat it?" test *harder* — but it's stated
rather than hidden.

*(Detail: [§4](#4-building-a-baseline-worth-beating))*

### 3. "The bug cross-validation is structurally unable to find"

With `drop="first"`, the dropped reference category encodes as all-zeros. With sklearn's
default `handle_unknown="ignore"`, an unseen category *also* encodes as all-zeros. They are
byte-identical — so an airline the model has never seen gets silently predicted as whichever
carrier happened to be dropped.

The documented fix, `handle_unknown="infrequent_if_exist"`, routes unknowns to a dedicated
column — **but only if that column exists**, which requires something to have fallen below
`min_frequency` during fit. I set `min_frequency=2` first. It built no bucket at all, because
the thinnest destination in the early folds has 3 rows. sklearn fell straight back to
all-zeros with no error.

Then the deeper problem: `min_frequency` is one threshold shared across every column, and the
columns need different ones. The rarest carrier has **744** rows in the train pool, so no sane
threshold ever pools one. Carrier can never get a bucket.

**And cross-validation cannot surface any of this** — zero unseen carriers appear in any of
the five validation folds. The bug is real and never fires on 2024–2025 data. But the app
predicts 2026+, where an airline starting SEA service is ordinary. Worse, this tool compares
*two* flights, so an unrecognised carrier doesn't return an obviously-wrong number, it returns
a confident comparison where one side is secretly another airline's delay profile.

**Resolution:** no encoder setting fixes it honestly — forcing a bucket would train it on the
rarest *existing* carriers, which is a different population from *new* ones. So it's closed as
input validation in the app layer, checked against the fitted encoder's `categories_` before
predicting, and documented so the two halves of the policy can't drift apart.

*(Detail: [§6](#6-encoding-the-features))*

---

## Reference

Everything below is detail for follow-up questions. Doc citations follow the repo rule in
`CLAUDE.md`: the verbatim sentence carrying the claim is quoted, and only pages listed in
`DOCUMENTATION_SOURCES.md` are cited.

1. [Framing the problem](#1-framing-the-problem)
2. [Loading and cleaning](#2-loading-and-cleaning)
3. [Splitting for honest evaluation](#3-splitting-for-honest-evaluation)
4. [Building a baseline worth beating](#4-building-a-baseline-worth-beating)
5. [Deciding what the model actually sees](#5-deciding-what-the-model-actually-sees)
6. [Encoding the features](#6-encoding-the-features)
7. [Method notes](#7-method-notes)

---

## 1. Framing the problem

### This is regression on tabular rows, not time-series forecasting

Every row carries a `FlightDate`, which makes the time-series toolchain (`statsmodels`,
`prophet`, `sktime`) look relevant. It isn't. Those libraries model **autoregression** —
predicting the next value of a series from earlier values of *that same series*. The test to
apply: *does making a prediction require knowing recent observed values of the thing being
predicted?* Here it doesn't. The app receives a description of one flight and predicts that
row's `ArrDelay` from its own attributes. There is no series to project forward, and no way
to obtain "last Tuesday's actual delay for this flight" at prediction time anyway, since the
user is booking months ahead against a static pickled model.

Structurally the data is a panel, not a series: 731 dates carrying 242–555 rows each, where
a time series has exactly one observation per time step. `FlightDate` is an attribute of a
row, not an index along which the target evolves.

**The consequence cuts both ways, and the second half is the easy one to miss.** Ruling out
forecasting rules those libraries out of the *stack*. It does not rule out time-aware
*evaluation*. Training data is 2024–2025 and users will book 2026, so the honest question is
whether a model fit on the past generalises to the future. Rows near each other in time are
also genuinely correlated — a snowstorm or an ATC ground stop hits every flight that day —
which is the correlation the docs warn about, even though no autoregressive model is
involved.

Per the [Cross-validation user guide](https://scikit-learn.org/stable/modules/cross_validation.html):
> "Time series data is characterized by the correlation between observations that are near
> in time (autocorrelation). However, classical cross-validation techniques such as `KFold`
> and `ShuffleSplit` assume the samples are independent and identically distributed, and
> would result in unreasonable correlation between training and testing instances (yielding
> poor estimates of generalization error) on time series data."

So time-awareness enters through the **evaluation protocol**, not the model form. The common
beginner error is the exact reverse: reaching for a forecasting model and then evaluating it
with a shuffled split.

*Reasoned judgement, not a doc citation — this is a framing decision about the problem, not
a claim about how a library behaves.*

### `ArrDelay` is right-skewed, so the mean describes a flight that rarely happens

Measured on the 324,490 clean rows: mean **4.91**, median **−4**, p75 **11**, std **43.58**,
max **3359** (56 hours). **57.2%** of SEA departures arrive early; **59.2%** arrive early or
exactly on time. When the mean sits ~9 minutes above the median, a long right tail is
dragging the average up while most of the mass sits below it.

Three separate decisions fall out of this one shape:

- **The naive baseline uses the median, not the mean.** The median minimises absolute error,
  so it is the correct partner for an MAE-scored project rather than an arbitrary choice.
  sklearn pairs them the same way — per the
  [`RandomForestRegressor` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html),
  `criterion` offers `"absolute_error"` for
  > "the mean absolute error, which minimizes the L1 loss using the median of each terminal node"
- **Training loss and reporting metric disagree.** `LinearRegression` minimises *squared*
  error while the project reports MAE. A single 300-minute miss contributes as much to the
  objective (90,000) as one hundred 30-minute misses combined (100 × 900). With a max of
  3,359 the fit will chase a handful of catastrophic delays that MAE barely notices.
- **What the app may honestly claim.** "Typically lands 4 minutes early" is defensible.
  "Average delay 4.9 minutes" is arithmetically true and quietly misleading, because the
  average describes a flight that essentially never occurs.

---

## 2. Loading and cleaning

### Unnamed columns from CSV round-tripping

`to_csv()` writes the row index as the first column by default. On reload that index becomes
`Unnamed: 0`, since it had no header. Fix: `to_csv(index=False)` when saving, or drop the
column on load. A trailing comma in each row can also create a phantom empty column at the
end (`Unnamed: 109` here).

### `FlightDate` loads as a string, not a date

`read_csv` loads date columns as `object` dtype. `pd.to_datetime()` or `parse_dates=` is
required — without it, chronological comparison and temporal splitting silently do the wrong
thing rather than erroring.

### A null `ArrDelay` means the flight never arrived, which scopes the whole model

Verified across all 24 raw monthly files: of 328,559 SEA-origin rows, 4,069 have a null
`ArrDelay`, and **every one is accounted for** — 3,161 cancelled, 908 diverted, zero
unexplained. These are not random missingness to be imputed; they are flights that had no
arrival at all.

Dropping them (leaving 324,490) is right, since there is no arrival time to regress on. But
it defines what the model *is*: **a predictor of arrival delay conditional on the flight
operating.** It says nothing about cancellation or diversion risk — which, for a traveler
comparing two flights, is often the more important risk. That boundary belongs in the app's
disclaimer, not buried in a dropped-rows line.

### Post-flight columns are leakage, and "duplicate" is a separate reason from "leaky"

The raw dataset has ~110 columns. Most (`DepDelay`, `DepTime`, `TaxiOut`,
`ActualElapsedTime`, delay-cause breakdowns, cancellation/diversion flags) only exist after
the flight lands. A model trained on them learns "flights that depart late arrive late" and
scores beautifully while being useless, because none of it is knowable at booking time.

A second, distinct group was dropped for redundancy rather than leakage:
`Reporting_Airline`, `OriginCityName`, `DistanceGroup`, `DepTimeBlk` all restate information
already present in a column that was kept. `Origin` was dropped because every row is `SEA` —
a column with one distinct value carries no signal for separating a delayed flight from an
on-time one.

Keeping these two reasons separate matters: the leakage group would corrupt the evaluation,
the redundancy group would only waste width.

---

## 3. Splitting for honest evaluation

### A row-count split doesn't give you the time split you think it does

Flights/day varies 242–555 across the year, with summer running ~34% above winter (Jun–Aug
daily mean 513 vs Dec–Feb 383; the peak-to-trough month gap is +43%, Jul 519 vs Jan 361). The
data is not evenly spread across the calendar, so a fixed *row-count* split ("first 60% of
rows") does not land on a clean date boundary and does not cover 60% of the two-year span —
it covers more or less time depending on which season those rows fall in.

**Fix:** split on the **array of unique dates**, not the rows. Decide the split as a
boundary in time, then let row counts follow from it.

Per the [`TimeSeriesSplit` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html):
> "To ensure comparable metrics across folds, samples must be equally spaced. Once this
> condition is met, each test set covers the same time duration, while the train set size
> accumulates data from previous splits."

That qualifier is the whole point, and it is why the first version of this issue was wrong.
`TimeSeriesSplit` splits by **array position**, not by date — it has no concept of a
`FlightDate` column. Handed raw flight rows, which run 242–555 per day, its boundaries land
mid-calendar-date.

Verified on this dataset (324,490 rows, 731 dates, `TimeSeriesSplit(n_splits=5)`, sklearn
1.7.2): splitting on raw rows put all 5 fold boundaries mid-date and produced validation
folds spanning 108/125/136/106/124 distinct days at identical row counts. Splitting the date
array instead gave exactly equal folds with zero date overlap, because dates *are* evenly
spaced — one per calendar day, no gaps across 2024-01-01 to 2025-12-31 — even though the
flight rows aren't.

**The general mechanism:** dates array → fold indices → actual dates → row boolean mask via
`.isin()` → filtered dataframe. This was extracted into a `make_cv_folds()` generator so
every candidate model in the project scores on identical folds. One pitfall worth naming:
storing the *result* of `.split()` gives you a one-shot generator that silently produces zero
iterations on a second use — recreate the splitter inside the function instead.

### The actual split, and two things it deliberately gives up

Single cutoff `2025-10-01`. Train pool: 2024-01-01 → 2025-09-30, 639 dates, 285,649 rows.
Test: 2025-10-01 → 2025-12-31, 92 dates, 38,841 rows (~12%), untouched until final
evaluation. Five expanding-window folds over the train pool's unique dates, 106 validation
days each.

- **`gap=0` is a choice, not an oversight.** No feature in this project is lagged or rolling
  and every predictor is known at booking time, so adjacent dates share no constructed value
  that a zero gap could leak across a boundary. This has to be revisited if a lagged feature
  is ever added.
- **The test partition is entirely Q4.** Because the split is temporal, the final holdout
  number describes winter operations only, not a full-year average. `Month` and `DayOfWeek`
  stay in the feature set partly for this reason, and the caveat has to be repeated wherever
  that number is reported — including in the app.

---

## 4. Building a baseline worth beating

The goal was a naive baseline strong enough that beating it means something: a per-flight-
profile (carrier + flight number + destination) historical median of `ArrDelay`.

### The first version hid its own weakness

Grouping by profile and merging medians into the validation fold left **32%** of fold 0's
validation rows with no match at all. But that number understated the problem, because a
median was being computed for every profile that appeared even *once* — profiles with one or
two rows were producing "predictions" instead of counting as no-match. The real fallback rate
is higher than 32% once a minimum-support threshold is applied.

### Choosing the threshold by measuring instead of guessing

The question is how many training rows a profile needs before its median is trustworthy.
Measured on fold 0 — the thinnest fold, where the choice matters most:

| n | % of profiles below | % of training rows below |
|---|---|---|
| 5 | 21.7 | 1.2 |
| 10 | 39.9 | 5.0 |
| 15 | 48.1 | 7.7 |
| 20 | 52.9 | 10.2 |

**`n = 10` was chosen.** At that threshold only 5.0% of the fold's training rows sit in
profiles too thin to use, even though 39.9% of *individual profiles* fall below it — most
thin profiles carry very little row weight, so raising the bar costs almost no data. `n = 5`
excludes even less (1.2%) but trusts a median computed from as few as 5 points, which is
risky given the target's fat right tail (train-pool p99 = 152, max = 3,359): one outlier can
swing a median that small. Going higher costs data without buying stability — n = 20 doubles
the n = 10 exclusion rate and n = 15 is about one and a half times it.

The profile/row split in that table is the actual insight. "40% of profiles are too thin"
sounds alarming and "5% of rows are affected" sounds fine, and both describe the same
threshold. Row weight is the number that matters for a prediction task.

### The fallback ladder, and tracking which rung fired

Three rungs, threshold applied identically at each:

1. per-flight-profile median, if the profile has ≥ 10 training rows in that fold
2. else carrier + scheduled-departure-hour median, also ≥ 10 rows
3. else the fold's flat global median

**Rung usage is counted per fold rather than assumed**, so the baseline cannot silently
become "a global-median model wearing a per-profile label." The counting uses a
boolean-mask snapshot: capture `unresolved = col.isna()` immediately *before* a fill step,
then `unresolved & col.notna()` after it isolates exactly the rows that step resolved.

Result: mean cross-validated MAE **19.95 minutes**, std **1.21** — small variation across
folds, so performance is stable as seasons pass.

### Measuring whether rung 2 earns its complexity

The obvious way to test rung 2 is to compare the ladder's overall MAE against a version that
falls straight from rung 1 to the global median. That dilutes the signal, because the
majority of rows never left rung 1 and contribute identical error to both sides.

**The correct scope is the subgroup where the question is live:** rows that needed a fallback
at all. Restricted that way, the original rung 2 (carrier + destination) produced a lift of
only **0.149 minutes** over falling straight to the flat median — essentially nothing, with
both sitting around 20.7–20.8 MAE.

Rather than guessing a replacement, 11 candidate groupings were scored on the same 5 folds:

| grouping | lift over flat median (min) |
|---|---|
| carrier + departure hour | **0.415** |
| carrier alone | 0.210 |
| carrier + destination | 0.149 |
| carrier + destination + month | −0.001 |

Two findings. **Destination actively hurts** — carrier alone beat carrier + destination, and
adding month collapsed the lift to nothing. Destination isn't adding signal, it is
fragmenting groups into smaller, noisier ones. **Departure hour carries a real effect** the
route doesn't: median `ArrDelay` by scheduled departure hour runs about −10 minutes at 6am
against roughly 0 from late morning onward — a ~10-minute swing across the day, larger than
any route-level difference, consistent with delay propagation compounding through an
aircraft's later legs.

One encoding note that looks like a contradiction but isn't: `DepHour` here is a `groupby`
key, not a model feature, so the HHMM midnight-wraparound problem (§6) does not apply. Group
labels carry no ordering — hour 23 and hour 0 being numerically far apart is irrelevant when
the number is only being used to form a bucket.

### How good could any median-based baseline ever be?

Worth bounding before treating a later model's score as good or bad. Predicting the flat
global median for every row gives **20.337** MAE. A *cheating* oracle — each profile's true
median computed directly on the validation rows it is predicting, not obtainable in reality
— gives **18.636**.

So the entire headroom available to any per-group median approach is about **1.70 minutes**,
because `ArrDelay`'s spread around its own median (mean absolute deviation ≈ 20.4 minutes,
p10 = −22, p90 = +36) dwarfs the difference between any two groups' centres. The ladder at
19.95 is already close to that ceiling.

**Implication carried forward:** a real model that fails to beat this baseline by much is not
automatically broken. That has to be said explicitly in the write-up rather than discovered
as a disappointment.

### The caveat that has to be stated, not buried

The winning rung-2 grouping was selected by scoring 11 candidates on the same 5 folds used
for every other comparison in the project. That is a mild form of selection on the CV folds.
It is defensible here — a stronger baseline makes the later "does the model beat it?"
comparison *harder*, not easier — but it is the kind of thing that has to appear in the
write-up in plain language instead of being left implicit.

---

## 5. Deciding what the model actually sees

Four columns survive in the dataframe but are never handed to the estimator, each for a
different reason. Keeping a column because it is useful for splitting, grouping, or the app
is a fine reason to keep a *column*; it is not a reason to feed it to a *model*.

- **`Year`** takes exactly two values here, 2024 and 2025, and the app predicts 2026 onward
  — so every prediction it will ever make is for a year the model has never seen. As a
  number it extrapolates a two-point trend off the end of its support; one-hot it lands in
  the unknown bucket or all-zeros. Either way it absorbs variance during training and
  contributes nothing valid at prediction time.
- **`Quarter`** is a deterministic function of `Month` (verified: each month maps to exactly
  one quarter). Its dummy columns are exactly reconstructible from `Month`'s — perfect
  collinearity, the same problem `drop="first"` exists to fix, added back deliberately.
- **`FlightDate`** never recurs. 2024-03-15 is not happening again, so there is nothing in it
  to generalise from. It stays because it is the split key and a grouping key.
- **`Flight_Number_Reporting_Airline`** is a cardinality problem: 2,591 distinct values on the
  full dataset, 2,383 on the train pool, against about 120 columns for all four categoricals
  that were kept. Beyond width, a key this granular risks the model memorising "this exact
  flight tends to be late" rather than learning a generalisable pattern. It stays as the
  Issue 3 profile key and the app's user input.

*The `Year`/`FlightDate` argument is reasoning about the deployment contract, not a doc
citation. sklearn's [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
page defines leakage as using information unavailable at prediction time; this is the mirror
image — the information is available, it just carries no valid meaning outside the training
range.*

---

## 6. Encoding the features

### HHMM time columns are clock readings, not quantities

`CRSDepTime`/`CRSArrTime` store times as integers that look numeric (`1435` = 2:35pm) but
aren't. Two distinct problems:

**The within-day scale is non-linear.** `10:59 → 11:00` is one real minute but a
`1100 − 1059 = 41`-unit jump in the raw integer.

**The scale is discontinuous at midnight.** `23:59 → 00:01` is two real minutes and a
−2,358-unit jump.

Checked for dirty values first and found none: in both the clean CSV and the raw monthly
files, both columns range 1–2359 with **zero** occurrences of `0`, `2400`, or a minute-part
≥ 60. (The `2400`-for-midnight quirk does appear, but only in `DepTime`/`ArrTime` —
post-flight columns already excluded as leakage.) So the problem is purely the encoding's
shape.

This does not hit all model families equally. Per the
[Time-related feature engineering example](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html):
> "Note that the time related features are passed as is, i.e. without processing them. But
> this is not much of a problem for tree-based models as they can learn a non-monotonic
> relationship between ordinal input features and the target."

**The fix is two steps, and step one alone is not enough.**

*Step 1 — minutes since midnight,* `(t // 100) * 60 + (t % 100)`, range 0–1439. The 10:59 →
11:00 step becomes a clean `+1`. (A useful sanity check: the resulting minimum is 1, not 0,
because `0001` is a real scheduled time — 12:01am.)

*Step 2 — cyclical encoding.* Minutes-since-midnight does **not** fix the wraparound: 23:59
→ 1439 and 00:01 → 1 are still 1,438 apart for two real minutes. Encoding the angle
`2π · minutes / 1440` as a `sin`/`cos` pair puts the time on a circle instead of a line.
Verified directly:

| time | minutes | sin | cos |
|---|---|---|---|
| 23:59 | 1439 | −0.0044 | 1.0000 |
| 00:01 | 1 | 0.0044 | 1.0000 |

0.0087 apart in `(sin, cos)` space instead of 1,438 apart in raw minutes. **Both** components
are required — sine alone repeats each value twice per cycle, so 6am and 6pm would encode
identically; the pair uniquely locates a point on the circle.

Only the `sin`/`cos` columns are fed to the linear model. Including raw minutes alongside
them would reintroduce the midnight discontinuity as its own feature.

[`SplineTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html)
with `extrapolation="periodic"` is the documented alternative to a `sin`/`cos` pair, and its
docs describe exactly this situation:
> "If 'periodic', periodic splines with a periodicity equal to the distance between the first
> and last knot are used. Periodic splines enforce equal function values and derivatives at
> the first and last knot. For example, this makes it possible to avoid introducing an
> arbitrary jump between Dec 31st and Jan 1st in spline features derived from a naturally
> periodic "day-of-year" input feature."

`Month` and `DayOfWeek` wrap around too, in principle, but at 12 and 7 levels one-hot handles
it for free and yields interpretable coefficients. Periodic encoding is reserved for the
1,440-value time features, where one-hot would be absurd.

### `drop="first"`, because unregularised OLS has no unique answer without it

One-hot encoding the 11 carriers gives 11 dummy columns that always sum to 1 per row. The
intercept is a coefficient on a hidden column that is always 1. So the intercept column and
the sum of the dummies are identical, row for row.

The consequence: any amount can be subtracted from the intercept and added to every dummy
coefficient without changing a single prediction. There are infinitely many equally valid
coefficient settings, so "carrier X adds 5 minutes" is meaningless — it could equally have
been 8, or −200. Dropping one category removes the thing that absorbs the compensating shift;
that category becomes the all-zeros reference, and the remaining coefficients read as
"relative to it."

Per the [`OneHotEncoder` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html):
> "Specifies a methodology to use to drop one of the categories per feature. This is useful
> in situations where perfectly collinear features cause problems, such as when feeding the
> resulting data into an unregularized linear regression model."

And the tradeoff to record, from the [Linear Models user guide](https://scikit-learn.org/stable/modules/linear_model.html),
in case this project ever gains a penalty term:
> "However, dropping one category breaks the symmetry of the original representation and can
> therefore induce a bias in downstream models, for instance for penalized linear
> classification or regression models."

### The unseen-category collision, and the fix that silently didn't work

`drop="first"` makes the reference category encode as all-zeros. sklearn's
`handle_unknown="ignore"` makes any *unseen* category encode as all-zeros too:
> "When an unknown category is encountered during transform, the resulting one-hot encoded
> columns for this feature will be all zeros."

The two are byte-identical, so an unfamiliar carrier gets predicted as the reference carrier
with no error and no warning. `handle_unknown="infrequent_if_exist"` is the documented fix:
> "When an unknown category is encountered during transform, the resulting one-hot encoded
> columns for this feature will map to the infrequent category if it exists."

**"If it exists" is doing real work in that sentence.** The infrequent bucket is built
**per column**, and only if some category actually fell below `min_frequency` during `fit`.
If none did, `infrequent_categories_` is `None` for that column and sklearn 1.7.2 falls
straight back to all-zeros — exactly the behaviour the setting was chosen to avoid, with no
error raised.

This was not hypothetical. `min_frequency=2` was set first and **built no bucket at all** in
folds 0 and 1, because the thinnest destination in those folds has 3 rows and nothing was
below 2. `min_frequency=4` clears it.

### The real failure was `Month`, not `Dest` — and it isn't a frequency problem

`min_frequency` is a *single* threshold applied to every column handed to the encoder, and
the columns need different treatment. Measured per fold, which categories can actually turn
up unseen in validation:

| column | distinct | rarest (fold 0) | pooled at `min_frequency=4` | unseen in validation, by fold |
|---|---|---|---|---|
| carrier | 11 | 148 | never | 0, 0, 0, 0, 0 |
| `Dest` | 85–93 | 3 | 1–5 per fold | 1, 3, 3, 1, 0 |
| `Month` | 4–12 | 7,606 | never | **4, 3, 1, 0, 0** |
| `DayOfWeek` | 7 | 5,055 | never | 0, 0, 0, 0, 0 |

`Month` is the column that actually breaks. The folds are temporal, so fold 0 trains on
January–April only and May–August arrive in its validation set as categories the encoder has
never seen — in 3 of 5 folds. No `min_frequency` can fix that, because those months aren't
*rare*, they're *absent*.

**So the right axis to split on is whether the category set is closed, not cardinality.**
Cardinality actively misleads here: `Month` has 12 values and fails, carrier has 11 and
doesn't.

- **Closed sets** — `Month` and `DayOfWeek` can be enumerated today (1–12, 1–7). Passing an
  explicit `categories=` list removes the problem at the source: no value is ever unknown,
  and `handle_unknown` becomes irrelevant for them.
- **Open sets** — `Dest` and carrier cannot be enumerated in advance, since an airline can
  add a route or begin service at SEA. They keep `min_frequency=4` +
  `handle_unknown="infrequent_if_exist"`.

Two encoders, routed by a `ColumnTransformer`.

### The carrier gap: a real bug that CV cannot surface

Carrier never gets an infrequent bucket at any usable threshold. Its rarest is 148 rows in
fold 0 and **744** across the whole train pool, so forcing a bucket would take
`min_frequency=745` to capture even one carrier and ~2,000 to capture four.

Verified by transforming a row unseen in both open columns: the unknown `Dest` correctly
routes to `Dest_infrequent_sklearn`, while the unknown carrier comes back all-zeros —
identical to `AA`, the dropped reference.

No encoder setting fixes this honestly:
- raising `min_frequency` to ~745 would train the bucket on the *rarest existing* carriers at
  SEA and burn a real carrier's coefficient to do it. Rare-at-SEA and new-to-SEA are
  different populations, so the resulting number would describe nothing.
- a sentinel category via `categories=` is all-zero across training, so least squares returns
  a zero coefficient for it and it predicts as the reference anyway.

**Cross-validation cannot catch this**: zero unseen carriers appear in any of the five
validation folds. The bug is real in principle and never fires on 2024–2025 data — but the
app predicts 2026+, where a carrier beginning SEA service is an ordinary event. And the
severity is specific to this product: the tool compares *two* flights, so an unrecognised
carrier doesn't just return a wrong number, it returns a comparison where one side carries an
unrelated carrier's delay profile while presenting as authoritative.

**Resolution: the gap is closed as input validation in the app layer, not in the encoder** —
check the carrier against the fitted encoder's `categories_` before predicting, and branch to
an explanatory message. Documented as such so the two halves of the policy cannot drift apart.

### `ColumnTransformer` mechanics worth knowing

Built from `(name, transformer, columns)` triples. The name is a label used to prefix output
feature names and to reach a fitted sub-transformer later — which is what the carrier check
above needs.

- **What decides where a column goes isn't dtype, it's label-versus-quantity.** `Month` and
  `DayOfWeek` are integers and still go to a one-hot encoder. `Distance` 2400 genuinely is
  twice 1200; `Month` 12 is not twice `Month` 6. Same distinction as the HHMM case, where
  `1435` looked numeric but was a clock reading. Handing an encoder the whole dataframe would
  treat each distinct `Distance` value as its own category — 100 distinct values, plus 364 for
  `CRSElapsedTime`, so roughly 460 columns of nonsense.
- **`remainder` defaults to `'drop'`.** Any column not listed is discarded, so the exclusions
  in §5 are enforced by construction rather than by remembering to drop them.
- **Output is a numpy array, so column names are gone.** `get_feature_names_out()` returns
  them in matching order, which is what lets names line up against `coef_` later.
- **Encoded width legitimately differs per fold.** Fold 0's 41,901 rows contain 85
  destinations; fold 4's 231,754 contain 93. That is correct, not a bug — if fold 0's encoder
  knew all 93, destinations from 2025 would have leaked backwards into a model that should
  only know early 2024, the same mistake as shuffling a temporal split.

### Why the encoder has to live inside a `Pipeline`

`fit` scans rows and stores the category table on the object (`categories_`,
`infrequent_categories_` — the trailing underscore is sklearn's convention for "only exists
after fitting"). `transform` uses that stored table as a lookup, so a given category always
lands in the same column position.

That two-step split is exactly what makes leakage preventable, and fitting the preprocessor
once on the whole train pool and then slicing folds out of the result would defeat it. Inside
a `Pipeline`, `fit` runs `fit_transform` on only that fold's training rows and `predict` runs
`transform` on the validation rows — so validation is always encoded using categories learned
from training alone, guaranteed rather than remembered.

Per [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html):
> "Always split the data into train and test subsets first, particularly before any
> preprocessing steps."

---

## 7. Method notes

Cross-cutting habits that caught real errors in this project.

### Verify numbers against the data; don't recall them

Almost every entry above exists because a plausible assumption failed a check. A partial list
of claims that looked right and weren't:

- `min_frequency=2` would protect against unseen categories — it built no bucket at all
- a rare carrier had "count 1" — the rarest carrier has 744 rows in the train pool
- the thin destinations were seasonal ski routes — the actually-pooled set is four one-off
  flights to Cedar Rapids, Buffalo, Harrisburg and Green Bay
- destination counts grow across folds because the window reaches later winters — the fold-4
  addition is a *summer* route
- "≈2,698 columns" was flight numbers alone — it's carrier + flight number + destination
  combined; flight numbers alone are 2,591

### A library's own warning can be wrong

Transforming a row unknown in both open columns raised:

> "Found unknown categories in columns [0, 1] during transform. These unknown categories
> will be encoded as all zeros"

Column 1 demonstrably received a `1` in its infrequent bucket. The warning flags any column
containing an unknown without checking whether that column had a bucket to absorb it. Read
the output, not the message.

### Check per block, not across the whole vector

Asking whether the full 94-wide output vector is all-zeros returns "no", because `Dest` got
its `1` — which would wrongly imply the carrier block was protected too. The check has to be
scoped to the block being tested. This is the same scoping error as measuring rung-2 lift
across all rows instead of only the rows that needed a fallback (§4): in both cases, mixing
in rows that were never in question hides the answer.

### Order of operations bites in pandas

A Series must be `.rename()`d *before* `.reset_index()` if the resulting dataframe's value
column needs a specific name — reversing them either errors or silently does the wrong thing,
since `DataFrame.rename()` doesn't accept a bare string the way `Series.rename()` does.

---

## Open items

- Under `categories=`, May's column is all-zeros across every one of fold 0's training rows,
  so its coefficient *should* come out 0 and it should still predict like the dropped
  reference month. **Not yet checked against a fitted `coef_`** — it should not be asserted
  until it has been. The deployed model doesn't have this problem at all, since it fits on
  the whole train pool where every month has at least 12,398 rows; `categories=` here is
  mostly CV hygiene.
