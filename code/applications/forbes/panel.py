"""Forbes archived daily wealth and billionaire panels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mgp_core import (
    DAILY_RETURN_MAX_GAP,
    ROOT,
    clean_panel_cache_path,
    fill_short_internal_return_gaps,
    read_clean_panel_cache,
    standardize_panel,
    write_clean_panel_cache,
)

from applications.panels import (
    SourcePanel,
    WEIGHT_TIMING,
    load_source_panel_cache,
    save_source_panel_cache,
    source_panel_cache_path,
)


FORBES_ARCHIVE_DAILY = ROOT / "data" / "forbes_billionaires" / "archive_realtime_daily.csv"
BAD_FORBES_PERSON_NAMES = {"RTB", "Real-Time Billionaires"}
FORBES_USECOLS = [
    "archive_timestamp",
    "snapshot_date",
    "archive_capture_date",
    "natural_id",
    "person_name",
    "final_worth_billion_usd",
]


def forbes_archive_frequency(path: Path) -> str:
    name = path.name
    if "daily" in name:
        return "daily"
    if "weekly" in name:
        return "weekly"
    if "monthly" in name:
        return "monthly"
    return "archive-snapshot"


def forbes_summary_path(path: Path) -> Path:
    frequency = forbes_archive_frequency(path)
    return path.parent / f"archive_fetch_summary_{frequency}.csv"


def forbes_consistent_timestamps(path: Path) -> set[str]:
    """Archive timestamps whose snapshot is internally consistent.

    A capture is dropped when, after the standard row cleaning, its published
    rank ordering is inconsistent with the reported worths (worth increasing in
    rank) -- a symptom of a corrupted or partially replayed archive capture.
    The scan is chunked and the resulting set is cached against the archive's
    content fingerprint.
    """
    cache_path = clean_panel_cache_path("forbes_consistent_timestamps", {"version": 1}, [path])
    cached = read_clean_panel_cache(cache_path)
    if cached is not None:
        return set(cached["timestamps"])

    pieces = []
    usecols = ["archive_timestamp", "person_name", "rank", "final_worth_billion_usd"]
    for chunk in pd.read_csv(path, usecols=usecols, dtype=str, chunksize=500_000):
        worth = pd.to_numeric(chunk["final_worth_billion_usd"], errors="coerce")
        rank = pd.to_numeric(chunk["rank"], errors="coerce")
        keep = worth.gt(0) & rank.notna() & ~chunk["person_name"].isin(BAD_FORBES_PERSON_NAMES)
        pieces.append(
            pd.DataFrame(
                {
                    "archive_timestamp": chunk.loc[keep, "archive_timestamp"].astype(str),
                    "rank": rank[keep].to_numpy(dtype=np.float32),
                    "worth": worth[keep].to_numpy(dtype=np.float32),
                }
            )
        )
    d = pd.concat(pieces, ignore_index=True)
    d = d.sort_values(["archive_timestamp", "rank"], kind="stable")
    good: set[str] = set()
    for ts, g in d.groupby("archive_timestamp", sort=False):
        if not np.any(np.diff(g["worth"].to_numpy()) > 1e-9):
            good.add(str(ts))
    n_bad = d["archive_timestamp"].nunique() - len(good)
    print(f"[forbes_billionaires] consistency filter: dropped {n_bad} inconsistent snapshots", flush=True)
    write_clean_panel_cache(cache_path, {"timestamps": sorted(good)})
    return good


def forbes_archive_timestamps(path: Path, level_count: int | None = None) -> list[str]:
    summary_path = forbes_summary_path(path)
    if summary_path.exists():
        summary = pd.read_csv(summary_path, dtype={"archive_timestamp": str})
        if "status" in summary.columns:
            summary = summary[summary["status"].astype(str).str.startswith("ok")]
        if "rows_written" in summary.columns:
            summary = summary[pd.to_numeric(summary["rows_written"], errors="coerce").fillna(0).gt(0)]
        candidates = summary["archive_timestamp"].dropna().astype(str).drop_duplicates().sort_values().tolist()
    else:
        seen: set[str] = set()
        for chunk in pd.read_csv(path, usecols=["archive_timestamp"], dtype=str, chunksize=500_000):
            seen.update(chunk["archive_timestamp"].dropna().astype(str).unique())
        candidates = sorted(seen)

    consistent = forbes_consistent_timestamps(path)
    timestamps = [ts for ts in candidates if ts in consistent and ts[:8] <= "20251231"]
    if level_count is None:
        return timestamps
    if len(timestamps) < level_count:
        raise ValueError(f"{path} has only {len(timestamps)} consistent snapshots, fewer than {level_count}.")
    return timestamps[-level_count:]


def clean_forbes_archive_chunk(chunk: pd.DataFrame, timestamps: set[str] | None = None) -> pd.DataFrame:
    if timestamps is not None:
        chunk = chunk[chunk["archive_timestamp"].isin(timestamps)]
    if chunk.empty:
        return pd.DataFrame(columns=["archive_timestamp", "snapshot_date", "unit_id", "person_name", "worth"])

    worth = pd.to_numeric(chunk["final_worth_billion_usd"], errors="coerce")
    unit_id = chunk["natural_id"].fillna("")
    snapshot_date = chunk["snapshot_date"].fillna("").where(
        chunk["snapshot_date"].fillna("").ne(""),
        chunk["archive_capture_date"].fillna(""),
    )
    out = pd.DataFrame(
        {
            "archive_timestamp": chunk["archive_timestamp"].astype(str),
            "snapshot_date": snapshot_date.astype(str),
            "unit_id": unit_id.astype(str),
            "person_name": chunk["person_name"].astype(str),
            "worth": worth,
        }
    )
    out = out.dropna(subset=["worth"])
    out = out[(out["unit_id"] != "") & out["worth"].gt(0)].copy()
    out = out.loc[~out["person_name"].isin(BAD_FORBES_PERSON_NAMES)].copy()
    out = out.sort_values(["archive_timestamp", "unit_id", "worth"], ascending=[True, True, False])
    return out.drop_duplicates(["archive_timestamp", "unit_id"], keep="first")


def read_forbes_archive_subset(path: Path, timestamps: list[str]) -> pd.DataFrame:
    keep = set(timestamps)
    chunks = []
    for chunk in pd.read_csv(path, usecols=FORBES_USECOLS, dtype=str, chunksize=500_000):
        chunk = clean_forbes_archive_chunk(chunk, keep)
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        raise ValueError(f"No Forbes archive rows found in {path} for selected timestamps.")
    d = pd.concat(chunks, ignore_index=True)
    d = d[d["archive_timestamp"].isin(keep)].copy()
    d["archive_timestamp"] = pd.Categorical(d["archive_timestamp"], categories=timestamps, ordered=True)
    d = d.rename(columns={"worth": "final_worth_billion_usd"})
    d = d.sort_values(["archive_timestamp", "unit_id", "final_worth_billion_usd"], ascending=[True, True, False])
    d = d.drop_duplicates(["archive_timestamp", "unit_id"], keep="first")
    return d


def forbes_level_panel(path: Path, timestamps: list[str]) -> tuple[pd.DataFrame, pd.Series, float]:
    """Balance wealth on observed archive dates, bridging short personal gaps."""
    max_gap = DAILY_RETURN_MAX_GAP if forbes_archive_frequency(path) == "daily" else 0
    if len(timestamps) <= 500:
        d = read_forbes_archive_subset(path, timestamps)
        date_by_ts = (
            d[["archive_timestamp", "snapshot_date"]]
            .drop_duplicates()
            .sort_values("archive_timestamp")
            .set_index("archive_timestamp")["snapshot_date"]
        )
        time_index = pd.to_datetime(date_by_ts.reindex(timestamps).to_numpy())
        if time_index.isna().any():
            raise ValueError("Forbes archive selected timestamps have missing dates.")
        levels = (
            d.pivot(index="unit_id", columns="archive_timestamp", values="final_worth_billion_usd")
            .reindex(columns=timestamps)
        )
        levels = fill_short_internal_return_gaps(levels, max_gap, carry_forward=True).dropna(axis=0)
        levels.columns = pd.Index(time_index.date, name="date")
        labels = (
            d[d["archive_timestamp"].eq(timestamps[0])]
            .drop_duplicates("unit_id")
            .set_index("unit_id")["person_name"]
            .astype(str)
            .reindex(levels.index)
        )
        last_total = float(d[d["archive_timestamp"].eq(timestamps[-1])]["final_worth_billion_usd"].sum())
        coverage = float(levels.iloc[:, -1].sum() / last_total) if last_total > 0 else np.nan
        return levels, labels, coverage

    keep = set(timestamps)
    first_ts = timestamps[0]
    last_ts = timestamps[-1]
    last_units: set[str] = set()
    labels_by_unit: dict[str, str] = {}
    date_by_ts: dict[str, str] = {}
    last_total = 0.0

    for chunk in pd.read_csv(path, usecols=FORBES_USECOLS, dtype=str, chunksize=500_000):
        c = clean_forbes_archive_chunk(chunk, keep)
        if c.empty:
            continue
        date_by_ts.update(c.drop_duplicates("archive_timestamp").set_index("archive_timestamp")["snapshot_date"].to_dict())
        first = c[c["archive_timestamp"].eq(first_ts)]
        labels_by_unit.update(first.drop_duplicates("unit_id").set_index("unit_id")["person_name"].astype(str).to_dict())
        last = c.loc[c["archive_timestamp"].eq(last_ts)]
        last_units.update(last.unit_id)
        last_total += float(last.worth.sum())

    candidate_units = set(labels_by_unit) & last_units
    if not candidate_units:
        raise ValueError("Forbes full-window balanced panel has zero billionaires.")

    chunks = []
    for chunk in pd.read_csv(path, usecols=FORBES_USECOLS, dtype=str, chunksize=500_000):
        c = clean_forbes_archive_chunk(chunk, keep)
        c = c[c["unit_id"].isin(candidate_units)]
        if not c.empty:
            chunks.append(c[["archive_timestamp", "unit_id", "worth"]])
    if not chunks:
        raise ValueError("No Forbes rows found for the balanced full-window unit set.")

    d = pd.concat(chunks, ignore_index=True)
    d["archive_timestamp"] = pd.Categorical(d["archive_timestamp"], categories=timestamps, ordered=True)
    levels = d.pivot(index="unit_id", columns="archive_timestamp", values="worth").reindex(columns=timestamps)
    time_index = pd.to_datetime([date_by_ts.get(ts, "") for ts in timestamps])
    if time_index.isna().any():
        raise ValueError("Forbes archive selected timestamps have missing dates.")
    levels = fill_short_internal_return_gaps(levels, max_gap, carry_forward=True).dropna(axis=0)
    levels.columns = pd.Index(time_index.date, name="date")
    labels = pd.Series(labels_by_unit, dtype=str).reindex(levels.index)
    coverage = float(levels.iloc[:, -1].sum() / last_total) if last_total > 0 else np.nan
    return levels, labels, coverage


def load_forbes_billionaire_panels(
    t_values: list[int] | None,
    archive_csv: Path,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> list[SourcePanel]:
    if not archive_csv.exists():
        raise FileNotFoundError(
            f"Missing {archive_csv}. Build it with "
            "`python3 data/forbes_billionaires/build_forbes_billionaires_archive.py --frequency daily --no-raw`."
        )
    if t_values is not None and any(t <= 1 for t in t_values):
        raise ValueError("--forbes-t values must be greater than 1.")

    frequency = forbes_archive_frequency(archive_csv)
    if t_values is None:
        timestamps = forbes_archive_timestamps(archive_csv, level_count=None)
        windows = [("full", timestamps)]
        print(
            f"[forbes_billionaires] reading {archive_csv.name}: "
            f"full archive window with {len(timestamps):,} level snapshots",
            flush=True,
        )
    else:
        max_level_count = max(t_values) + 1
        timestamps = forbes_archive_timestamps(archive_csv, level_count=max_level_count)
        windows = [(f"T{t}", timestamps[-(t + 1) :]) for t in sorted(dict.fromkeys(t_values))]
        print(
            f"[forbes_billionaires] reading {archive_csv.name}: "
            f"last {max_level_count} level snapshots for T={sorted(t_values)}",
            flush=True,
        )

    panels: list[SourcePanel] = []
    for window_name, ts in windows:
        cache_path = source_panel_cache_path(
            "forbes_billionaire_wealth",
            {
                "person_id": "natural_id",
                "archive": str(Path(archive_csv).resolve()),
                "frequency": frequency,
                "window": window_name,
                "n_timestamps": len(ts),
                "first_timestamp": ts[0],
                "last_timestamp": ts[-1],
                "weight_timing": WEIGHT_TIMING,
                "internal_gap_max": DAILY_RETURN_MAX_GAP if frequency == "daily" else 0,
            },
            [archive_csv],
        )
        cached = load_source_panel_cache(cache_path, use_cache, refresh_cache)
        if cached is not None:
            panels.append(cached)
            continue

        levels, labels, coverage = forbes_level_panel(archive_csv, ts)
        t = levels.shape[1] - 1
        if levels.empty:
            raise ValueError(f"Forbes archive {window_name} has no balanced billionaires.")

        y = np.log(levels).diff(axis=1).iloc[:, 1:]
        if levels.columns[0] >= y.columns[0]:
            raise ValueError("Forbes wealth weights must precede the first wealth-growth observation.")
        y = standardize_panel(y)
        y = y.sub(y.mean().mean())
        sizes = levels.iloc[:, 0].astype(float)
        panel = SourcePanel(
            name=f"forbes_billionaire_wealth_{frequency}_{window_name}",
            y=y,
            sizes=sizes,
            labels=labels,
            start_date=y.columns[0],
            last_date=y.columns[-1],
            frequency=frequency,
            size_name="net worth at snapshot preceding first growth (t-1; USD billions)",
            value_name=f"{frequency} log net-worth growth",
            coverage=coverage,
            notes=(
                f"Forbes real-time billionaire wealth via Internet Archive; "
                f"balanced {window_name} window with {t + 1} level snapshots and T={t} log-growth observations; first snapshot supplies t-1 weights; "
                f"individual gaps up to {DAILY_RETURN_MAX_GAP if frequency == 'daily' else 0} observed archive dates are bridged "
                "by carrying wealth forward (zero growth), retaining the catch-up change on reappearance; "
                "longer and endpoint gaps exclude the person. Dates with no valid capture are omitted, "
                "so calendar spacing can be irregular. Growth winsorized at 1st/99th pct."
            ),
            weight_timing=WEIGHT_TIMING,
            weight_date=levels.columns[0],
        )
        panels.append(panel)
        save_source_panel_cache(cache_path, panel, use_cache)
        print(
            f"[forbes_billionaires {window_name}] balanced N={levels.shape[0]:,}, "
            f"T={t:,}, dates={levels.columns[0]} to {levels.columns[-1]}, coverage={coverage:.3f}",
            flush=True,
        )
    return panels
