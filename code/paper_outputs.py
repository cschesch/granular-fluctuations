"""Validate, collect, and sync the external artifacts used by the paper."""

import argparse
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

from project_paths import CODE_DIR, ROOT


MANUSCRIPT = ROOT / "overleaf" / "main.tex"
OUTPUT = ROOT / "output"
INTERMEDIATES = CODE_DIR / ".cache" / "nonpaper_outputs"

# This list also lets the replication checkout run without the manuscript.
EXHIBITS = {
    "atalay_elasticity_profile.pdf",
    "atalay_elasticity_surface.png",
    "bea_granular_ceiling_figure1.pdf",
    "compustat_industry_dependence_ci.pdf",
    "factor_count_overview.pdf",
    "figure1.pdf",
    "figure1_hmgp.pdf",
    "firm_granular_volatility.pdf",
    "lpt_compustat_qtr.pdf",
    "lpt_foerster_net_M10.pdf",
    "mc_new_bias.pdf",
    "mc_new_compustat_intervals.pdf",
    "mc_new_coverage.pdf",
    "mc_new_delta.pdf",
    "mc_new_designs.pdf",
    "mc_new_limit_law.pdf",
    "mc_new_tightness.pdf",
    "msweep_combined.pdf",
    "sales_factor_averaging.pdf",
    "table_applications_homoskedastic.tex",
    "table_applications_vertical.tex",
    "table_foerster_matched_tabular.tex",
    "table_grouped_giv_mc_combined_tabular.tex",
    "table_grouping_schemes_mc_new_tabular.tex",
    "table_mc_new_tabular.tex",
    "table_sales_covariance_budget_tabular.tex",
    "table_stella_input_moments.tex",
    "welburn_input_moments.tex",
}

# The manuscript's generated inputs are literal paths under output/.
INCLUDE = re.compile(
    r"\\(?:input|include|includegraphics)\b\s*(?:\[[^\]]*\]\s*)?"
    r"\{output/([^{}]+)\}"
)


def paper_files(manuscript: Path = MANUSCRIPT) -> set[str]:
    source = re.sub(r"(?<!\\)%[^\n]*", "", manuscript.read_text())
    names = set(INCLUDE.findall(source))
    if not names or any(Path(name).name != name or not Path(name).suffix for name in names):
        raise ValueError("Expected literal filenames with extensions under output/ in the manuscript")
    return names


def check_outputs(names: set[str], since: Path | None = None) -> None:
    invalid = sorted(
        name for name in names
        if not (OUTPUT / name).is_file() or (OUTPUT / name).stat().st_size == 0
    )
    if invalid:
        raise FileNotFoundError("Missing or empty paper outputs: " + ", ".join(invalid))
    if since is not None:
        started = since.stat().st_mtime_ns
        stale = sorted(name for name in names if (OUTPUT / name).stat().st_mtime_ns < started)
        if stale:
            raise RuntimeError("Paper outputs were not refreshed in this run: " + ", ".join(stale))


def collect_outputs(names: set[str]) -> None:
    """Keep only paper artifacts in output/ and retain other generated files in the cache."""
    INTERMEDIATES.mkdir(parents=True, exist_ok=True)
    for item in OUTPUT.iterdir():
        if item.name in names:
            continue
        destination = INTERMEDIATES / item.name
        if item.is_dir() and not item.is_symlink() and destination.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
            shutil.rmtree(item)
        else:
            item.replace(destination)


def sync_outputs(manuscript: Path, names: set[str]) -> None:
    """Replace the manuscript's output directory after preparing every artifact."""
    parent = manuscript.resolve().parent
    destination = parent / "output"
    if destination.is_symlink() or destination.resolve() == OUTPUT.resolve():
        raise ValueError("Expected a separate, real manuscript output directory")
    with TemporaryDirectory(prefix=".paper-output-", dir=parent) as temporary:
        staging = Path(temporary) / "new"
        backup = Path(temporary) / "previous"
        staging.mkdir()
        for name in sorted(names):
            shutil.copy2(OUTPUT / name, staging / name)
        if paper_files(manuscript) != names:
            raise RuntimeError("The manuscript's output references changed during staging")
        try:
            if destination.exists():
                destination.rename(backup)
            staging.rename(destination)
        except BaseException:
            if backup.exists() and not destination.exists():
                backup.rename(destination)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "collect", "sync"))
    parser.add_argument("--manuscript", type=Path, default=MANUSCRIPT)
    parser.add_argument("--since", type=Path, help="Require outputs written since this marker.")
    args = parser.parse_args()
    names = EXHIBITS
    if args.manuscript.is_file() or args.action == "sync":
        draft_names = paper_files(args.manuscript)
        if draft_names != names:
            raise ValueError(f"Update EXHIBITS for the draft's changed files: {sorted(draft_names ^ names)}")
    check_outputs(names, args.since)
    if args.action == "collect":
        collect_outputs(names)
    elif args.action == "sync":
        sync_outputs(args.manuscript, names)
    print(f"{args.action}: {len(names)} paper artifacts")


if __name__ == "__main__":
    main()
