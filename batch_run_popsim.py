#!/usr/bin/env python3
"""
Batch runner for PopulationSim across multiple RegioStar folders.

Features:
- Discovers and runs all popsim_regiostar_* folders
- Optionally splits large runs (by 100m cell count) into sub-runs by 1km cells
- Merges results with verification that 100m cells are unique across runs
- Produces combined final_expanded_household_ids.csv

Usage:
    uv run batch_run_popsim.py [--max-cells 3000] [--dry-run] [--merge-only]
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def discover_popsim_folders(base_dir: Path) -> list[Path]:
    """Find all popsim_regiostar_* folders."""
    folders = sorted(base_dir.glob("popsim_regiostar_*"))
    # Filter to only directories, exclude any that are sub-splits (contain underscore after number)
    return [f for f in folders if f.is_dir() and not f.name.split("_")[-1].startswith("split")]


def count_100m_cells(folder: Path) -> int:
    """Count the number of 100m cells in a popsim folder."""
    geo_file = folder / "data" / "geo_cross_walk.csv"
    if not geo_file.exists():
        raise FileNotFoundError(f"geo_cross_walk.csv not found in {folder}")

    df = pd.read_csv(geo_file)
    return df["ZENSUS100m"].nunique()


def get_1km_cell_groups(folder: Path) -> dict[str, list[str]]:
    """Get mapping of 1km cells to their 100m cells."""
    geo_file = folder / "data" / "geo_cross_walk.csv"
    df = pd.read_csv(geo_file)

    groups = {}
    for km_cell, group in df.groupby("ZENSUS1km"):
        groups[km_cell] = group["ZENSUS100m"].unique().tolist()

    return groups


def split_folder_by_1km(folder: Path, max_cells: int) -> list[Path]:
    """
    Split a popsim folder into multiple sub-folders if it exceeds max_cells.

    Splits along 1km cell boundaries to maintain geographic hierarchy.
    Returns list of folders to run (original if no split needed, or sub-folders).
    """
    num_cells = count_100m_cells(folder)

    if num_cells <= max_cells:
        print(f"  {folder.name}: {num_cells} cells (no split needed)")
        return [folder]

    print(f"  {folder.name}: {num_cells} cells (splitting into sub-runs...)")

    # Get 1km cell groups
    km_groups = get_1km_cell_groups(folder)

    # Load all data files
    geo_df = pd.read_csv(folder / "data" / "geo_cross_walk.csv")
    seed_hh = pd.read_csv(folder / "data" / "seed_households.csv")
    seed_persons = pd.read_csv(folder / "data" / "seed_persons.csv")

    # Load control files
    control_100m = pd.read_csv(folder / "data" / "control_totals_ZENSUS100m.csv")
    control_1km = pd.read_csv(folder / "data" / "control_totals_ZENSUS1km.csv")
    control_staat = pd.read_csv(folder / "data" / "control_totals_STAAT.csv")
    control_welt = pd.read_csv(folder / "data" / "control_totals_WELT.csv")

    # Partition 1km cells into groups that don't exceed max_cells
    partitions = []
    current_partition = []
    current_count = 0

    for km_cell, cells_100m in sorted(km_groups.items()):
        cell_count = len(cells_100m)

        if current_count + cell_count > max_cells and current_partition:
            partitions.append(current_partition)
            current_partition = []
            current_count = 0

        current_partition.append(km_cell)
        current_count += cell_count

    if current_partition:
        partitions.append(current_partition)

    print(f"    -> Creating {len(partitions)} sub-folders")

    # Create sub-folders
    split_folders = []
    for i, km_cells in enumerate(partitions, 1):
        split_name = f"{folder.name}_split{i:02d}"
        split_folder = folder.parent / split_name

        # Get 100m cells for this partition
        cells_100m = set()
        for km_cell in km_cells:
            cells_100m.update(km_groups[km_cell])

        # Create folder structure
        split_folder.mkdir(exist_ok=True)
        (split_folder / "configs").mkdir(exist_ok=True)
        (split_folder / "data").mkdir(exist_ok=True)
        (split_folder / "output").mkdir(exist_ok=True)

        # Copy configs
        for config_file in (folder / "configs").glob("*"):
            shutil.copy(config_file, split_folder / "configs" / config_file.name)

        # Filter and save geo_cross_walk
        geo_split = geo_df[geo_df["ZENSUS100m"].isin(cells_100m)]
        geo_split.to_csv(split_folder / "data" / "geo_cross_walk.csv", index=False)

        # Seed data is the same (all households available for sampling)
        seed_hh.to_csv(split_folder / "data" / "seed_households.csv", index=False)
        seed_persons.to_csv(split_folder / "data" / "seed_persons.csv", index=False)

        # Filter control files
        control_100m_split = control_100m[control_100m["ZENSUS100m"].isin(cells_100m)]
        control_100m_split.to_csv(split_folder / "data" / "control_totals_ZENSUS100m.csv", index=False)

        control_1km_split = control_1km[control_1km["ZENSUS1km"].isin(km_cells)]
        control_1km_split.to_csv(split_folder / "data" / "control_totals_ZENSUS1km.csv", index=False)

        # STAAT and WELT controls need to be recalculated (sum of sub-area)
        # For now, copy as-is (they're upper-level controls)
        control_staat.to_csv(split_folder / "data" / "control_totals_STAAT.csv", index=False)
        control_welt.to_csv(split_folder / "data" / "control_totals_WELT.csv", index=False)

        num_100m = len(cells_100m)
        num_1km = len(km_cells)
        print(f"    Split {i}: {num_1km} 1km cells, {num_100m} 100m cells")

        split_folders.append(split_folder)

    return split_folders


def run_popsim(folder: Path, dry_run: bool = False) -> bool:
    """Run PopulationSim for a single folder."""
    print(f"\nRunning PopulationSim: {folder.name}")

    if dry_run:
        print("  [DRY RUN] Would execute: uv run populationsim -w", folder)
        return True

    try:
        result = subprocess.run(
            ["uv", "run", "populationsim", "-w", str(folder)],
            cwd=folder.parent,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per run
        )

        if result.returncode != 0:
            print(f"  ERROR: PopulationSim failed with code {result.returncode}")
            print(f"  stderr: {result.stderr[:500]}")
            return False

        # Check if output was created
        output_file = folder / "output" / "final_expanded_household_ids.csv"
        if not output_file.exists():
            print(f"  ERROR: Output file not created")
            return False

        print(f"  SUCCESS: Output created")
        return True

    except subprocess.TimeoutExpired:
        print(f"  ERROR: Timeout after 1 hour")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def verify_unique_cells(dfs: list[tuple[str, pd.DataFrame]]) -> tuple[bool, list[str]]:
    """
    Verify that 100m cells are unique across all DataFrames.

    Returns (is_valid, list of duplicate cells).
    """
    all_cells = {}
    duplicates = []

    for name, df in dfs:
        cells = df["ZENSUS100m"].unique()
        for cell in cells:
            if cell in all_cells:
                duplicates.append(f"{cell} (in {all_cells[cell]} and {name})")
            else:
                all_cells[cell] = name

    return len(duplicates) == 0, duplicates


def merge_results(folders: list[Path], output_dir: Path) -> pd.DataFrame:
    """
    Merge final_expanded_household_ids.csv from all folders.

    Verifies that 100m cells are unique across runs.
    """
    print("\n" + "="*60)
    print("Merging results...")
    print("="*60)

    results = []
    for folder in folders:
        output_file = folder / "output" / "final_expanded_household_ids.csv"
        if not output_file.exists():
            print(f"  WARNING: Skipping {folder.name} (no output file)")
            continue

        df = pd.read_csv(output_file)
        df["source_folder"] = folder.name
        results.append((folder.name, df))
        print(f"  Loaded {folder.name}: {len(df)} expanded households, {df['ZENSUS100m'].nunique()} cells")

    if not results:
        raise ValueError("No results to merge!")

    # Verify unique cells
    is_valid, duplicates = verify_unique_cells(results)

    if not is_valid:
        print(f"\n  ERROR: Found {len(duplicates)} duplicate 100m cells across runs!")
        for dup in duplicates[:10]:
            print(f"    - {dup}")
        if len(duplicates) > 10:
            print(f"    ... and {len(duplicates) - 10} more")
        raise ValueError("Duplicate cells found - cannot merge!")

    print(f"\n  VERIFIED: All 100m cells are unique across runs")

    # Concatenate results
    combined = pd.concat([df for _, df in results], ignore_index=True)

    # Remove source_folder column for final output
    final_df = combined.drop(columns=["source_folder"])

    # Save combined results
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
    parser.add_argument(
        "--max-cells", type=int, default=3000,
        help="Maximum 100m cells per run before splitting (default: 3000)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without running PopulationSim"
    )
    parser.add_argument(
        "--merge-only", action="store_true",
        help="Only merge existing results, don't run PopulationSim"
    )
    parser.add_argument(
        "--no-split", action="store_true",
        help="Don't split large folders, run as-is"
    )
    parser.add_argument(
        "--base-dir", type=Path, default=Path.cwd(),
        help="Base directory containing popsim_regiostar_* folders"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for combined results (default: base_dir/popsim_combined)"
    )

    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    output_dir = args.output_dir or (base_dir / "popsim_combined")

    print("="*60)
    print("PopulationSim Batch Runner")
    print("="*60)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max cells per run: {args.max_cells}")

    # Discover folders
    folders = discover_popsim_folders(base_dir)
    print(f"\nDiscovered {len(folders)} popsim folders:")
    for f in folders:
        print(f"  - {f.name}")

    if not folders:
        print("No popsim_regiostar_* folders found!")
        return 1

    # Determine folders to run (with optional splitting)
    folders_to_run = []

    print(f"\nAnalyzing folder sizes (max {args.max_cells} cells)...")
    for folder in folders:
        if args.no_split:
            num_cells = count_100m_cells(folder)
            print(f"  {folder.name}: {num_cells} cells")
            folders_to_run.append(folder)
        else:
            split_folders = split_folder_by_1km(folder, args.max_cells)
            folders_to_run.extend(split_folders)

    print(f"\nTotal runs to execute: {len(folders_to_run)}")

    if args.merge_only:
        print("\n--merge-only specified, skipping PopulationSim runs")
    else:
        # Run PopulationSim for each folder
        print("\n" + "="*60)
        print("Running PopulationSim...")
        print("="*60)

        successful = []
        failed = []

        for folder in folders_to_run:
            if run_popsim(folder, args.dry_run):
                successful.append(folder)
            else:
                failed.append(folder)

        print(f"\n\nRun Summary:")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")

        if failed:
            print("\nFailed runs:")
            for f in failed:
                print(f"  - {f.name}")

        if args.dry_run:
            print("\n[DRY RUN] No actual runs performed")
            return 0

    # Merge results
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
