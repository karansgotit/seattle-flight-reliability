"""Regenerate the figures embedded in README.md.

Run from the repository root, after data_cleaning.ipynb has written
data/interim/seattle_ontime_clean.csv:

    python reports/figures/make_figures.py

Uses the same split, folds and 3-rung ladder as notebooks/linear_baseline.ipynb,
so the figures cannot drift from the numbers reported in the README.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

OUT = "reports/figures"
INK, MUTED, GRID = "#1b1b1b", "#6b6b6b", "#d9d9d9"
BARS = ["#3b6ea5", "#7fa8cd", "#c2d4e6"]

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})

PROFILE = ["IATA_CODE_Reporting_Airline", "Flight_Number_Reporting_Airline", "Dest"]
CUTOFF, N_SPLITS, MIN_ROWS = pd.Timestamp("2025-10-01"), 5, 10


def make_cv_folds(train_pool, dates, n_splits=N_SPLITS):
    for fold, (tr, va) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(dates)):
        train_dates, val_dates = dates[tr], dates[va]
        assert not set(train_dates) & set(val_dates)
        yield (fold,
               train_pool[train_pool["FlightDate"].isin(train_dates)],
               train_pool[train_pool["FlightDate"].isin(val_dates)],
               train_dates, val_dates)


def load():
    df = pd.read_csv("data/interim/seattle_ontime_clean.csv", low_memory=False)
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    df["DepHour"] = df["CRSDepTime"] // 100
    return df[df["FlightDate"] < CUTOFF], df[df["FlightDate"] >= CUTOFF]


def fig_departure_hour(train_pool):
    med = train_pool.groupby("DepHour")["ArrDelay"].median()
    med = med[(med.index >= 5) & (med.index <= 23)]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.axhline(0, color=GRID, lw=1)
    ax.plot(med.index, med.values, color=BARS[0], lw=2, marker="o", ms=3.5)
    lo, hi = med.idxmin(), med.idxmax()
    ax.set_ylim(med.min() - 3.5, med.max() + 1.5)
    # mark the two extremes on the line, and describe them in the empty
    # lower-right corner so no text ever crosses the series
    ax.scatter([lo, hi], [med[lo], med[hi]], s=42, zorder=5,
               facecolor="white", edgecolor=BARS[0], linewidth=1.8)
    rows = [(f"{lo:02d}:00", f"{med[lo]:.0f} min"),
            (f"{hi:02d}:00", f"{med[hi]:.0f} min"),
            ("spread", f"{med.max()-med.min():.0f} min")]
    w = max(len(v) for _, v in rows)
    ax.text(0.985, 0.06,
            "\n".join(f"{k:<7}{v:>{w}}" for k, v in rows),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=INK, linespacing=1.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=GRID, linewidth=0.8))
    ax.set_title("Median arrival delay by scheduled departure hour", pad=12)
    ax.set_xlabel("Scheduled departure hour", labelpad=8)
    ax.set_ylabel("Median ArrDelay (min)", labelpad=8)
    ax.set_xticks(range(5, 24, 2)); ax.grid(axis="y", color=GRID, lw=0.6)
    ax.margins(x=0.03)
    ax.text(0, -0.24, f"Train pool only, {len(train_pool):,} flights. "
            "Early departures run ahead of schedule; later ones absorb the day's delay.",
            transform=ax.transAxes, ha="left", fontsize=8, color=MUTED)
    fig.savefig(f"{OUT}/delay_by_departure_hour.png"); plt.close(fig)
    return med


def rung_counts(train_pool, dates):
    out = []
    for fold, tr, va, _, _ in make_cv_folds(train_pool, dates):
        st = tr.groupby(PROFILE)["ArrDelay"].agg(["median", "count"])
        va = va.merge(st[st["count"] >= MIN_ROWS]["median"].rename("pred").reset_index(),
                      on=PROFILE, how="left")
        va["rung"] = np.where(va["pred"].notna(), 1, np.nan)
        cs = tr.groupby(["IATA_CODE_Reporting_Airline", "DepHour"])["ArrDelay"].agg(["median", "count"])
        va = va.merge(cs[cs["count"] >= MIN_ROWS]["median"].rename("coarse").reset_index(),
                      on=["IATA_CODE_Reporting_Airline", "DepHour"], how="left")
        unresolved = va["pred"].isna()
        va["pred"] = va["pred"].fillna(va["coarse"])
        va.loc[unresolved & va["pred"].notna(), "rung"] = 2
        va.loc[va["pred"].isna(), "rung"] = 3
        c = va["rung"].value_counts()
        out.append((fold, int(c.get(1, 0)), int(c.get(2, 0)), int(c.get(3, 0))))
    return out


def fig_rung_usage(counts):
    folds = [f"Fold {c[0]}" for c in counts]
    r1 = np.array([c[1] for c in counts], float)
    r2 = np.array([c[2] for c in counts], float)
    r3 = np.array([c[3] for c in counts], float)
    tot = r1 + r2 + r3
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    labels = ["Rung 1 — flight profile", "Rung 2 — carrier + departure hour", "Rung 3 — global median"]
    bottom = np.zeros(len(folds))
    for vals, colour, label in zip((r1, r2, r3), BARS, labels):
        pct = vals / tot * 100
        ax.barh(folds, pct, left=bottom, color=colour, label=label, height=0.62)
        for i, (p, b) in enumerate(zip(pct, bottom)):
            if p > 4:
                ax.text(b + p / 2, i, f"{p:.0f}%", ha="center", va="center",
                        fontsize=8.5, color="white" if colour == BARS[0] else INK)
        bottom += pct
    ax.set_xlim(0, 100); ax.invert_yaxis()
    ax.set_title("Which fallback rung produced each prediction", pad=12)
    ax.set_xlabel("Share of validation rows (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=3,
              frameon=False, fontsize=8)
    fig.savefig(f"{OUT}/rung_usage_by_fold.png"); plt.close(fig)


def fig_fold_structure(train_pool, test, dates):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for fold, _, _, train_dates, val_dates in make_cv_folds(train_pool, dates):
        y = fold
        ax.barh(y, train_dates.max() - train_dates.min(), left=train_dates.min(),
                color=BARS[1], height=0.55)
        ax.barh(y, val_dates.max() - val_dates.min(), left=val_dates.min(),
                color=BARS[0], height=0.55)
    ax.barh(N_SPLITS, test["FlightDate"].max() - test["FlightDate"].min(),
            left=test["FlightDate"].min(), color="#b5523f", height=0.55)
    ax.set_yticks(list(range(N_SPLITS)) + [N_SPLITS])
    ax.set_yticklabels([f"Fold {i}" for i in range(N_SPLITS)] + ["Held-out test"])
    ax.set_ylim(N_SPLITS + 1.15, -0.75)          # inverted, with room under the bars
    ax.axvline(CUTOFF, color=INK, ls="--", lw=1, zorder=1)
    ax.annotate("cutoff 2025-10-01", xy=(CUTOFF, N_SPLITS + 0.85),
                xytext=(-6, 0), textcoords="offset points",
                fontsize=8, color=INK, va="center", ha="right")
    ax.set_title("Cross-validation folds are cut on dates, never inside a day", pad=12)
    ax.grid(axis="x", color=GRID, lw=0.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BARS[1], BARS[0], "#b5523f")]
    ax.legend(handles, ["Train", "Validation", "Test (untouched)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
              frameon=False, fontsize=8)
    fig.savefig(f"{OUT}/cv_fold_structure.png"); plt.close(fig)


if __name__ == "__main__":
    train_pool, test = load()
    dates = np.sort(train_pool["FlightDate"].unique())
    med = fig_departure_hour(train_pool)
    counts = rung_counts(train_pool, dates)
    fig_rung_usage(counts)
    fig_fold_structure(train_pool, test, dates)
    print(f"train pool {len(train_pool):,} rows / {len(dates)} dates | test {len(test):,} rows")
    print(f"departure-hour spread: {med.max()-med.min():.1f} min "
          f"(min {med.min():.0f} at {med.idxmin():02d}:00, max {med.max():.0f} at {med.idxmax():02d}:00)")
    for f, a, b, c in counts:
        print(f"  fold {f}: rung1={a} rung2={b} rung3={c}")
