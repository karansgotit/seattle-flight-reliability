# PROJECT_ISSUES.md — Changelog

**2026-08-21 — reconciled with `reports/stack_and_alignment_review.md`.**

`PROJECT_ISSUES.md` is now the single source of truth. The review is retained as the audit
trail for how these conclusions were reached, but it is no longer a document to work from.

Every sklearn behaviour asserted below was re-checked against the **installed 1.7.2**, not
the 1.9.0 that `scikit-learn.org/stable/` currently serves. Every dataset number was
recomputed from `data/interim/seattle_ontime_clean.csv` rather than copied.

---

## Accepted from the review

| # | Change | Issues | Why | Review finding |
|---|---|---|---|---|
| 1 | Time features converted to minutes-since-midnight, then periodically encoded for the linear model | 4a, 5, 6, 9 | `CRSDepTime`/`CRSArrTime` are HHMM clock readings. Verified: both range 1–2359 with zero malformed values, so the problem is the encoding's *shape* — 23:59 → 00:01 is a −2358 jump for two minutes of real time. Departure time is plausibly the strongest pre-booking predictor; feeding it to OLS on a broken scale is how the model loses to the naive baseline for a non-reason. | F1 |
| 2 | `drop="first"` on the one-hot encoder | 4b, 5 | `LinearRegression` is unregularised, so full dummy encoding plus an intercept leaves coefficients **not uniquely determined** — which makes Issue 5's "say what a coefficient means" deliverable unachievable as written, not merely imprecise. | F2 |
| 3 | `Flight_Number_Reporting_Airline` dropped as a model feature (kept as profile key) | D-1, 4c, 14 | 2,591 distinct values → ≈2,698 one-hot columns, and carrier+flight+dest is near-identity, so OLS would re-implement the naive baseline as a lookup. Verified cardinalities. | F3 |
| 4 | Mandatory tree bounding + a 200 MB artifact budget | 8, 9, 13, 16 | See "Strengthened" below — the review extrapolated; this was measured. | F4 |
| 5 | `Year`, `Quarter`, `FlightDate` removed from the model feature set | 1, feature table, 12, 17 | `Year` holds exactly two values (2024, 2025) and every prediction the app makes is for a year outside that support. Kept as columns, excluded as inputs — the feature audit conflated "pre-booking" with "model input". | F5 |
| 6 | MAE named as the metric the product claim rests on; loss/metric mismatch recorded as a tradeoff | 5, 10, 12 | `LinearRegression` minimises squared error; with max 3359 and p75 11, RMSE and MAE will disagree and the disagreement is structural. Labelled a project judgement — sklearn's metrics page carries no heavy-tail guidance. | F6 |
| 7 | `@st.cache_resource` / `@st.cache_data` required | 14, 16 | Streamlit reruns the whole script per interaction, so an uncached model is re-deserialised on every dropdown change. Against a 690 MB floor this decides whether the app runs. | F9 |
| 8 | Unseen-category policy defined once in Issue 4c; Issues 3 and 14 cross-reference it | D-5, 3, 4c, 14 | It was specified independently in three places and could drift. | F10 |
| 9 | Persistence metadata list adopted | 11 | Directly from the model-persistence docs: data reference, source, dependency versions, CV score. | F8 |
| 10 | Deployability added to Gate 2 pass criteria | 13 | Gate 2 selects, Gate 3 deploys, six days apart. Finding the winner undeployable at Gate 3 means re-running selection with two days left. | F4 |
| 11 | Broken time encoding + "the baseline is genuinely strong" added to the Issue 6 diagnosis list | 6 | The original five candidate causes did not include the two most likely ones for this dataset. | F1 |
| 12 | Seasonal caveat added to the app's own disclaimer text | 15 | It was already in Issues 2 and 6 but nowhere a user would see it. | — |
| 13 | Median-over-mean rationale written into the baseline | 3 | It read as arbitrary; it is actually the L1-minimising estimator and therefore the correct partner for an MAE-scored project. | F-Issue3 |

## Strengthened beyond the review

| # | Change | Why |
|---|---|---|
| 14 | **RF artifact sizes measured, not extrapolated.** Full train pool, sklearn defaults, 10 trees → **251.6 MB / 3,493,834 nodes / 75.4 s**. At the default `n_estimators=100` that is **≈2.5 GB**. With `max_depth=12, min_samples_leaf=50` at 100 trees → **8.7 MB / 119,936 nodes / 11.4 s**. | The review inferred 2.5 GB from a 40,000-row sample. The inference was right, but a Gate 2 pass/fail criterion should not rest on an extrapolation. The measured bounded config also **resolves** the review's open question of whether RF is still viable: it is, comfortably. |
| 15 | **`drop="first"` must be paired with `handle_unknown="infrequent_if_exist"`, never `"ignore"`.** | Not in the review, and it is a live correctness trap. Tested on 1.7.2: the combination `drop="first"` + `handle_unknown="ignore"` is *accepted*, but an unseen category encodes to all-zeros — **byte-identical to the dropped reference category**. An unfamiliar carrier would silently be predicted as Alaska. With `infrequent_if_exist` + `min_frequency`, an unseen value transforms to `[0, 1]` while the reference is `[0, 0]`. Adopting the review's F2 and F10 recommendations independently would have produced exactly the broken pairing. |
| 16 | **Profile grain settled with measured support, and a fallback ladder required.** Profile = carrier + flight number + destination: **4,483 groups, median 23 rows, 36.4% under 10 rows, p10 = 1 row.** Coarser (carrier + destination): **216 groups, median 974 rows, 8.3% under 10.** | The review raised the profile-identity question but left it open. The measurement shows the fallback is a routine path, not an edge case — it will fire on a large share of held-out rows and materially move the baseline number. Issue 3 now also requires **counting how often each rung fires**, so the write-up cannot describe a global-median baseline as "per-profile". |
| 17 | **Month/DayOfWeek treated as one-hot categoricals, not cyclically encoded.** | The review's F1 logic implies all cyclical features need periodic encoding. At 12 and 7 levels, one-hot handles the wrap-around for free *and* keeps coefficients interpretable, which Issue 5 requires. Periodic encoding is reserved for the 1,440-value time features where one-hot would be absurd. |
| 18 | **Issue 9 must state which encoding each candidate got.** | The review noted the RF-vs-linear comparison is confounded by encoding but did not turn it into a requirement. Now it is: either score both on identical features, or score RF both ways and report both. |
| 19 | **New Issue 8b — hyperparameter tuning, explicitly optional and first to cut.** | The review flagged tuning as a missing issue. Rather than adding scope before Aug 24, it is added as a closed decision (D-3) with a placeholder issue, so the absence is a recorded tradeoff rather than a gap. Crucially it is separated from Issue 8's tree bounding, which is **not** tuning and must not be cut alongside it. |
| 20 | **A "Closed decisions" section (D-1 … D-5) at the top**, each with its reversal cost. | The review ended with four open questions. Leaving them open would leave the roadmap unable to act as a source of truth. Each is decided and each says what reversing costs. |

## Rejected / already done

| # | Review item | Disposition |
|---|---|---|
| 21 | **F7 — pin `requirements.txt`, add `streamlit`, drop `seaborn`** | **Already done** in commit `d50f21e`, before the review was written. The review is stale here. Issue 11 no longer asks for this work; it asks only that the pins are not undone and are re-checked if the training environment changes. |
| 22 | Drop `DayofMonth` (implied by the review's redundancy argument) | **Rejected.** Together with `Month` it locates fixed-date holidays (Thanksgiving, Christmas), which is a real delay mechanism. `Quarter` was dropped instead — it is a deterministic function of `Month` with no independent mechanism. |
| 23 | `criterion="absolute_error"` for RandomForest | **Deferred, as the review itself suggested.** It aligns the forest's loss with the reporting metric but is substantially slower to fit, and Gate 2 is Aug 24. Recorded as a tradeoff in Issue 12 rather than adopted. |
| 24 | Add `xgboost` / `lightgbm` | **Rejected**, consistent with the review. A third model class before Gate 2 is scope risk, not value. |

---

## What did not change

The original structure is intact: all 19 issues keep their numbers, titles, purposes, and
gate assignments; the milestone layout, the "Learn this first" MUST/Optional split, the
CampusX-then-docs sequencing, and the cut-order notes are unchanged. Issue 2 needed no
correction — the review confirmed it was already aligned.

Learning value was preserved deliberately. Nothing was replaced with a black-box shortcut:
the linear baseline stays a plain `LinearRegression` so its coefficients can be read, the
naive baseline stays a hand-built groupby rather than a `DummyRegressor`, and every added
requirement carries the reasoning for *why* rather than just the instruction. The additions
create new things to learn — periodic encoding, dummy-variable collinearity, the
memory/depth tradeoff in tree ensembles, Streamlit's rerun model — rather than removing
them.

## Side effects

- `DOCUMENTATION_SOURCES.md`: added `SplineTransformer`, Streamlit **Caching**, Streamlit
  **Manage your app** (resource limits), and sklearn **grid search** (for the optional 8b).
  Added a note that `stable` docs now serve 1.9.0 against the project's 1.7.2 pin.
- `reports/feature_audit.md`: Issue 1 now requires its Pre-Booking table to separate
  "pre-booking" from "model input". Not yet edited — that is Issue 1's work.
