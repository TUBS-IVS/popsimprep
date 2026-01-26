#!/usr/bin/env python3
"""
Batch runner for PopulationSim across multiple RegioStar folders.

Features:
- Discovers and runs all popsim_regiostar_* folders
- Optionally splits large runs (by 100m cell count) into sub-runs by 1km cells
- Parallel execution of multiple PopulationSim runs (default: 3 workers)
- Auto-resume: skips folders that already have output files
- Merges results with verification that 100m cells are unique across runs
- Produces combined final_expanded_household_ids_combined.csv

Safety improvements:
- Deterministic splits with split_manifest.json
- Will NOT overwrite existing splits unless --rebuild-splits is provided
- Detects split drift and errors out instead of silently changing inputs

Logging improvements:
- Thread-safe logging with timestamps
- Progress logging includes ETA and estimated finish time

Usage:
    uv run batch_run_popsim.py [--max-cells 3000] [--parallel 3] [--dry-run] [--merge-only]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

# Thread-safe print lock
print_lock = threading.Lock()


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str, level: str = "INFO"):
    """Thread-safe logging with timestamp."""
    with print_lock:
        print(f"[{now_str()}] [{level}] {message}", flush=True)


def is_completed(folder: Path) -> bool:
    """Check if a folder already has completed output."""
    output_file = folder / "output" / "final_expanded_household_ids.csv"
    return output_file.exists()


def discover_popsim_folders(base_dir: Path) -> list[Path]:
    """Find all popsim_regiostar_* folders (excluding split folders)."""
    folders = sorted(base_dir.glob("popsim_regiostar_*"))
    return [f for f in folders if f.is_dir() and "_split" not in f.name]


def read_geo(folder: Path) -> pd.DataFrame:
    """Read geo_cross_walk with stable dtypes to ensure deterministic sorting."""
    geo_file = folder / "data" / "geo_cross_walk.csv"
    if not geo_file.exists():
        raise FileNotFoundError(f"geo_cross_walk.csv not found in {folder}")
    return pd.read_csv(
        geo_file,
        dtype={"ZENSUS100m": "string", "ZENSUS1km": "string"},
    )


def count_100m_cells(folder: Path) -> int:
    """Count the number of 100m cells in a popsim folder."""
    df = read_geo(folder)
    return df["ZENSUS100m"].nunique()


def get_1km_cell_groups(folder: Path) -> dict[str, list[str]]:
    """Get mapping of 1km cells to their 100m cells (deterministic ordering enforced later)."""
    df = read_geo(folder)
    groups = {}
    for km_cell, group in df.groupby("ZENSUS1km", sort=True):
        groups[str(km_cell)] = group["ZENSUS100m"].astype("string").unique().tolist()
    return groups


def split_manifest_for_partition(
    source_folder_name: str,
    max_cells: int,
    split_index: int,
    km_cells: list[str],
    cells_100m: set[str],
) -> dict:
    """Create a deterministic manifest for a split."""
    # Sort for stability
    km_cells_sorted = sorted([str(x) for x in km_cells])
    cells_100m_sorted = sorted([str(x) for x in cells_100m])

    return {
        "schema_version": 1,
        "created_at": now_str(),
        "source_folder": source_folder_name,
        "max_cells": int(max_cells),
        "split_index": int(split_index),
        "num_1km_cells": len(km_cells_sorted),
        "num_100m_cells": len(cells_100m_sorted),
        "km_cells": km_cells_sorted,
        # Store a hash for fast compare, and optionally the full list for audit/debug
        "cells_100m_sha1": sha1_of_list(cells_100m_sorted),
        # Comment out if you want smaller manifests:
        # "cells_100m": cells_100m_sorted,
    }


def sha1_of_list(items: list[str]) -> str:
    import hashlib
    h = hashlib.sha1()
    for s in items:
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def load_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def manifests_equivalent(a: dict, b: dict) -> bool:
    """Compare relevant fields to decide if an existing split matches the desired split."""
    if not a or not b:
        return False
    keys = [
        "schema_version",
        "source_folder",
        "max_cells",
        "split_index",
        "num_1km_cells",
        "num_100m_cells",
        "km_cells",
        "cells_100m_sha1",
    ]
    return all(a.get(k) == b.get(k) for k in keys)


def split_folder_by_1km(folder: Path, max_cells: int, rebuild_splits: bool) -> list[Path]:
    """
    Split a popsim folder into multiple sub-folders if it exceeds max_cells.

    Safe & deterministic:
    - Creates split_manifest.json in each split folder
    - Reuses existing splits if manifest matches
    - Errors if existing split manifest differs, unless rebuild_splits=True (then deletes/recreates)
    """
    num_cells = count_100m_cells(folder)

    if num_cells <= max_cells:
        log(f"{folder.name}: {num_cells} cells (no split needed)", "SPLIT")
        return [folder]

    log(f"{folder.name}: {num_cells} cells (splitting into sub-runs...)", "SPLIT")

    km_groups = get_1km_cell_groups(folder)

    # Load all data files
    geo_df = read_geo(folder)
    seed_hh = pd.read_csv(folder / "data" / "seed_households.csv")
    seed_persons = pd.read_csv(folder / "data" / "seed_persons.csv")

    # Load control files
    control_100m = pd.read_csv(folder / "data" / "control_totals_ZENSUS100m.csv")
    control_1km = pd.read_csv(folder / "data" / "control_totals_ZENSUS1km.csv")
    control_staat = pd.read_csv(folder / "data" / "control_totals_STAAT.csv")
    control_welt = pd.read_csv(folder / "data" / "control_totals_WELT.csv")

    # Partition 1km cells into groups that don't exceed max_cells (greedy, deterministic order)
    partitions: list[list[str]] = []
    current_partition: list[str] = []
    current_count = 0

    for km_cell in sorted(km_groups.keys()):
        cells_100m = km_groups[km_cell]
        cell_count = len(cells_100m)

        if current_count + cell_count > max_cells and current_partition:
            partitions.append(current_partition)
            current_partition = []
            current_count = 0

        current_partition.append(km_cell)
        current_count += cell_count

    if current_partition:
        partitions.append(current_partition)

    log(f"-> Creating/using {len(partitions)} sub-folders", "SPLIT")

    split_folders: list[Path] = []
    for i, km_cells in enumerate(partitions, 1):
        split_name = f"{folder.name}_split{i:02d}"
        split_folder = folder.parent / split_name
        manifest_path = split_folder / "split_manifest.json"

        # Determine 100m cells for this partition
        cells_100m: set[str] = set()
        for km_cell in km_cells:
            cells_100m.update(km_groups[km_cell])

        desired_manifest = split_manifest_for_partition(
            source_folder_name=folder.name,
            max_cells=max_cells,
            split_index=i,
            km_cells=km_cells,
            cells_100m=cells_100m,
        )

        if split_folder.exists():
            existing_manifest = load_manifest(manifest_path)
            if manifests_equivalent(existing_manifest, desired_manifest):
                log(f"{split_name}: exists and matches manifest -> reusing", "SPLIT")
                split_folders.append(split_folder)
                continue

            # Split exists but doesn't match desired content
            if not rebuild_splits:
                raise RuntimeError(
                    f"Split folder '{split_name}' already exists but does NOT match the expected split.\n"
                    f"To rebuild splits, re-run with --rebuild-splits.\n"
                    f"(Refusing to overwrite to avoid stale outputs / inconsistent inputs.)"
                )

            log(f"{split_name}: exists but differs -> rebuilding (deleting old split)", "SPLIT")
            shutil.rmtree(split_folder)

        # Create folder structure fresh
        split_folder.mkdir(exist_ok=False)
        (split_folder / "configs").mkdir(exist_ok=True)
        (split_folder / "data").mkdir(exist_ok=True)
        (split_folder / "output").mkdir(exist_ok=True)

        # Copy configs (overwrites not applicable because folder is fresh)
        for config_file in (folder / "configs").glob("*"):
            shutil.copy(config_file, split_folder / "configs" / config_file.name)

        # Filter and save geo_cross_walk
        geo_split = geo_df[geo_df["ZENSUS100m"].astype("string").isin(list(cells_100m))]
        geo_split.to_csv(split_folder / "data" / "geo_cross_walk.csv", index=False)

        # Seed data is the same (all households available for sampling)
        seed_hh.to_csv(split_folder / "data" / "seed_households.csv", index=False)
        seed_persons.to_csv(split_folder / "data" / "seed_persons.csv", index=False)

        # Filter control files
        control_100m_split = control_100m[control_100m["ZENSUS100m"].astype("string").isin(list(cells_100m))]
        control_100m_split.to_csv(split_folder / "data" / "control_totals_ZENSUS100m.csv", index=False)

        control_1km_split = control_1km[control_1km["ZENSUS1km"].astype("string").isin(list(km_cells))]
        control_1km_split.to_csv(split_folder / "data" / "control_totals_ZENSUS1km.csv", index=False)

        # Upper-level controls copied as-is
        control_staat.to_csv(split_folder / "data" / "control_totals_STAAT.csv", index=False)
        control_welt.to_csv(split_folder / "data" / "control_totals_WELT.csv", index=False)

        # Write manifest last (so partial splits are less likely to look "valid")
        manifest_path.write_text(json.dumps(desired_manifest, indent=2), encoding="utf-8")

        log(
            f"{split_name}: {len(km_cells)} 1km cells, {len(cells_100m)} 100m cells (manifest written)",
            "SPLIT",
        )
        split_folders.append(split_folder)

    return split_folders


def run_popsim(folder: Path, dry_run: bool = False) -> tuple[Path, bool, str, float]:
    """
    Run PopulationSim for a single folder.

    Returns: (folder, success, message, duration_seconds)
    """
    start = time.monotonic()

    if is_completed(folder):
        log(f"SKIP {folder.name} - output already exists (resume)", "SKIP")
        return (folder, True, "skipped (already completed)", time.monotonic() - start)

    log(f"START {folder.name}", "RUN")

    if dry_run:
        log(f"DRY RUN {folder.name} - would execute populationsim", "DRY")
        return (folder, True, "dry run", time.monotonic() - start)

    try:
        result = subprocess.run(
            ["uv", "run", "populationsim", "-w", str(folder)],
            cwd=folder.parent,
            capture_output=True,
            text=True,
            timeout=3600
        )

        if result.returncode != 0:
            log(f"FAIL {folder.name} - exit code {result.returncode}", "ERROR")
            return (folder, False, f"failed with code {result.returncode}", time.monotonic() - start)

        output_file = folder / "output" / "final_expanded_household_ids.csv"
        if not output_file.exists():
            log(f"FAIL {folder.name} - no output file created", "ERROR")
            return (folder, False, "no output file created", time.monotonic() - start)

        log(f"DONE {folder.name}", "OK")
        return (folder, True, "completed", time.monotonic() - start)

    except subprocess.TimeoutExpired:
        log(f"FAIL {folder.name} - timeout after 1 hour", "ERROR")
        return (folder, False, "timeout", time.monotonic() - start)
    except Exception as e:
        log(f"FAIL {folder.name} - {e}", "ERROR")
        return (folder, False, str(e), time.monotonic() - start)


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds + 0.5), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def run_popsim_parallel(
    folders: list[Path],
    num_workers: int,
    dry_run: bool = False
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Run PopulationSim for multiple folders in parallel.

    Returns: (successful, failed, skipped)
    """
    successful: list[Path] = []
    failed: list[Path] = []
    skipped: list[Path] = []

    total = len(folders)
    completed_count = 0

    # Throughput-based ETA (REAL runs only, skips/dry excluded)
    t0 = time.monotonic()
    real_done = 0

    log(f"Starting parallel execution with {num_workers} workers", "INFO")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_folder = {
            executor.submit(run_popsim, folder, dry_run): folder
            for folder in folders
        }

        for future in as_completed(future_to_folder):
            folder, success, message, dur = future.result()
            completed_count += 1

            # Classify result
            is_skipped = "skipped" in message
            is_dry = (message == "dry run")
            is_real = (not is_skipped) and (not is_dry)

            if is_skipped:
                skipped.append(folder)
            elif success:
                successful.append(folder)
            else:
                failed.append(folder)

            if is_real:
                real_done += 1

            elapsed = time.monotonic() - t0
            remaining = total - completed_count

            # Throughput: seconds per REAL run (already reflects parallelism)
            if real_done > 0:
                sec_per_run = elapsed / real_done
                eta_seconds = remaining * sec_per_run
                rate_str = f"{sec_per_run/60:.1f} min/run"
            else:
                eta_seconds = 0.0
                rate_str = "n/a"

            est_finish = datetime.now().timestamp() + eta_seconds
            est_finish_str = datetime.fromtimestamp(est_finish).strftime("%H:%M:%S")

            log(
                f"Progress: {completed_count}/{total} "
                f"({len(successful)} done, {len(skipped)} skipped, {len(failed)} failed) | "
                f"elapsed {fmt_duration(elapsed)} | "
                f"rate {rate_str} | "
                f"ETA {fmt_duration(eta_seconds)} (≈ {est_finish_str})",
                "PROGRESS"
            )

    return successful, failed, skipped




def verify_unique_cells(dfs: list[tuple[str, pd.DataFrame]]) -> tuple[bool, list[str]]:
    """Verify that 100m cells are unique across all DataFrames."""
    all_cells = {}
    duplicates = []

    for name, df in dfs:
        cells = df["ZENSUS100m"].astype("string").unique()
        for cell in cells:
            if cell in all_cells:
                duplicates.append(f"{cell} (in {all_cells[cell]} and {name})")
            else:
                all_cells[cell] = name

    return len(duplicates) == 0, duplicates


def merge_results(folders: list[Path], output_dir: Path) -> pd.DataFrame:
    print("\n" + "="*60)
    print("Merging results...")
    print("="*60)

    results = []
    for folder in folders:
        output_file = folder / "output" / "final_expanded_household_ids.csv"
        if not output_file.exists():
            print(f"  WARNING: Skipping {folder.name} (no output file)")
            continue

        df = pd.read_csv(output_file, dtype={"ZENSUS100m": "string", "ZENSUS1km": "string"})
        df["source_folder"] = folder.name
        results.append((folder.name, df))
        print(f"  Loaded {folder.name}: {len(df)} expanded households, {df['ZENSUS100m'].nunique()} cells")

    if not results:
        raise ValueError("No results to merge!")

    is_valid, duplicates = verify_unique_cells(results)
    if not is_valid:
        print(f"\n  ERROR: Found {len(duplicates)} duplicate 100m cells across runs!")
        for dup in duplicates[:10]:
            print(f"    - {dup}")
        if len(duplicates) > 10:
            print(f"    ... and {len(duplicates) - 10} more")
        raise ValueError("Duplicate cells found - cannot merge!")

    print(f"\n  VERIFIED: All 100m cells are unique across runs")

    combined = pd.concat([df for _, df in results], ignore_index=True)
    final_df = combined.drop(columns=["source_folder"])

    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "final_expanded_household_ids_combined.csv"
    final_df.to_csv(output_file, index=False)

    print(f"\n  Combined results saved to: {output_file}")
    print(f"  Total expanded households: {len(final_df)}")
    print(f"  Total 100m cells: {final_df['ZENSUS100m'].nunique()}")
    print(f"  Total 1km cells: {final_df['ZENSUS1km'].nunique()}")

    return final_df


def main():
    parser = argparse.ArgumentParser(description="Batch runner for PopulationSim")
    parser.add_argument("--max-cells", type=int, default=3000,
                        help="Maximum 100m cells per run before splitting (default: 3000)")
    parser.add_argument("--parallel", "-p", type=int, default=3,
                        help="Number of parallel PopulationSim runs (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without running PopulationSim")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge existing results, don't run PopulationSim")
    parser.add_argument("--no-split", action="store_true",
                        help="Don't split large folders, run as-is")
    parser.add_argument("--rebuild-splits", action="store_true",
                        help="Delete and recreate existing split folders if manifests differ")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(),
                        help="Base directory containing popsim_regiostar_* folders")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for combined results (default: base_dir/popsim_combined)")

    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    output_dir = args.output_dir or (base_dir / "popsim_combined")

    print("="*60)
    print("PopulationSim Batch Runner")
    print("="*60)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max cells per run: {args.max_cells}")
    print(f"Parallel workers: {args.parallel}")
    print(f"Rebuild splits: {args.rebuild_splits}")

    folders = discover_popsim_folders(base_dir)
    print(f"\nDiscovered {len(folders)} popsim folders:")
    for f in folders:
        print(f"  - {f.name}")

    if not folders:
        print("No popsim_regiostar_* folders found!")
        return 1

    folders_to_run: list[Path] = []

    print(f"\nAnalyzing folder sizes (max {args.max_cells} cells)...")
    for folder in folders:
        if args.no_split:
            num_cells = count_100m_cells(folder)
            print(f"  {folder.name}: {num_cells} cells")
            folders_to_run.append(folder)
        else:
            split_folders = split_folder_by_1km(folder, args.max_cells, args.rebuild_splits)
            folders_to_run.extend(split_folders)

    already_completed = sum(1 for f in folders_to_run if is_completed(f))
    print(f"\nTotal runs: {len(folders_to_run)}")
    print(f"  Already completed: {already_completed}")
    print(f"  Remaining: {len(folders_to_run) - already_completed}")

    if args.merge_only:
        print("\n--merge-only specified, skipping PopulationSim runs")
    else:
        print("\n" + "="*60)
        print(f"Running PopulationSim ({args.parallel} parallel workers)...")
        print("="*60)

        successful, failed, skipped = run_popsim_parallel(
            folders_to_run, args.parallel, args.dry_run
        )

        print(f"\n\nRun Summary:")
        print(f"  Completed: {len(successful)}")
        print(f"  Skipped (already done): {len(skipped)}")
        print(f"  Failed: {len(failed)}")

        if failed:
            print("\nFailed runs:")
            for f in failed:
                print(f"  - {f.name}")

        if args.dry_run:
            print("\n[DRY RUN] No actual runs performed")
            return 0

    try:
        merge_results(folders_to_run, output_dir)
    except ValueError as e:
        print(f"\nMerge failed: {e}")
        return 1

    print("\n" + "="*60)
    print("Batch run complete!")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
