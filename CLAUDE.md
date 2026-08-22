# Project instructions

**Documentation-first rule:** when writing or editing `PROJECT_ISSUES.md`, `LEARNINGS.md`,
notebook markdown cells, or any other content that states a "best practice" or "correct way"
to use a library, cite only links from `DOCUMENTATION_SOURCES.md`. Tutorials (CampusX, Baker,
StatQuest) may be linked for background reading but must never be the sole justification for
a methodology decision.

If the claim needs a doc page that isn't in `DOCUMENTATION_SOURCES.md` yet, add it there
first (with a one-line note on what it's for), then cite it. Keep that file the complete,
current inventory rather than letting citations drift back into scattered per-issue links.

**Fetch and quote, don't cite from memory.** Before asserting that a library's docs
recommend a specific pattern (e.g. a CV strategy, an encoding approach), actually fetch the
page and paste the **verbatim sentence** that carries the claim into the doc being written.
A link alone is not enough: a correct-looking citation can sit on top of an implementation
the page does not actually endorse, which is the failure this rule exists to catch.

Two reasons the verbatim quote specifically, rather than a paraphrase:
- API docs change between versions, and `scikit-learn` is unpinned in this project's
  `requirements.txt`, so recalled behaviour may be stale.
- Paraphrasing from memory reproduces the *gist* of a page while dropping the qualifying
  clause that decides the implementation. Copying the sentence out forces the qualifier
  into view.

That is not hypothetical here. Issue 2 was once rewritten to use `TimeSeriesSplit`, citing
the right page, and was still wrong: `TimeSeriesSplit` splits on row position, and the API
reference's qualifier — "samples must be equally spaced" — rules out feeding it raw flight
rows, which run 242–555 per day. The error only surfaced when the sentence was quoted
rather than recalled. Issue 2 now splits the unique-date array instead.

**Verify quantitative claims against the data.** When a doc's applicability depends on a
property of this dataset (spacing, cardinality, class balance, missingness), check that
property in the data before relying on it, and record the number in the issue.
