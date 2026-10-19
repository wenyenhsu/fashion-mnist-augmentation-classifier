"""Model selection script based on validation accuracy across seeds 17 and 29."""

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import shutil
import sys

from src.utils import compute_file_sha256


def _invalidate_and_record_incomplete(
    output_dir: Path,
    results_dir: Path,
    model_dir: Path,
    summary_path: Path,
    selection_path: Path,
    record: dict,
) -> None:
    """Record incomplete selection and remove stale downstream artifacts."""
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    # Invalidate copied checkpoint so it cannot be silently reused
    dest_checkpoint = model_dir / "selected_state.pt"
    if dest_checkpoint.is_file():
        dest_checkpoint.unlink()

    # Invalidate stale final_test row in summary.csv if present
    if summary_path.is_file():
        valid_rows = []
        with open(summary_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("run_id") != "final_test":
                    valid_rows.append(row)
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "run_id",
                "condition",
                "seed",
                "epochs",
                "evaluation_split",
                "n_examples",
                "loss",
                "accuracy",
                "status",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in valid_rows:
                writer.writerow(r)


def select_model(output_dir: Path) -> None:
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "summary.csv"

    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    selection_path = results_dir / "selection.json"

    required_runs = ["baseline_s17", "baseline_s29", "flip_s17", "flip_s29"]
    timestamp = datetime.now(timezone.utc).isoformat()

    if not summary_path.is_file():
        record = {
            "status": "incomplete",
            "missing_run_ids": required_runs,
            "error": "results/summary.csv not found",
            "timestamp": timestamp,
        }
        _invalidate_and_record_incomplete(
            output_dir=output_dir,
            results_dir=results_dir,
            model_dir=model_dir,
            summary_path=summary_path,
            selection_path=selection_path,
            record=record,
        )
        print("Selection incomplete: summary.csv not found.", file=sys.stderr)
        return

    # Read summary rows
    run_records: dict[str, dict] = {}
    with open(summary_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_records[row["run_id"]] = row

    missing_runs = []
    for r in required_runs:
        if r not in run_records or run_records[r].get("status") != "completed":
            missing_runs.append(r)

    if missing_runs:
        record = {
            "status": "incomplete",
            "missing_run_ids": missing_runs,
            "error": f"Comparison incomplete. Missing or non-completed runs: {', '.join(missing_runs)}",
            "timestamp": timestamp,
        }
        _invalidate_and_record_incomplete(
            output_dir=output_dir,
            results_dir=results_dir,
            model_dir=model_dir,
            summary_path=summary_path,
            selection_path=selection_path,
            record=record,
        )
        print(f"Selection incomplete: missing runs {missing_runs}.", file=sys.stderr)
        return

    # Compute validation means
    base_17_acc = Decimal(run_records["baseline_s17"]["accuracy"])
    base_29_acc = Decimal(run_records["baseline_s29"]["accuracy"])
    flip_17_acc = Decimal(run_records["flip_s17"]["accuracy"])
    flip_29_acc = Decimal(run_records["flip_s29"]["accuracy"])

    baseline_mean = (base_17_acc + base_29_acc) / 2
    flip_mean = (flip_17_acc + flip_29_acc) / 2

    # Decision rule: higher mean validation accuracy; tie breaks to baseline
    if flip_mean > baseline_mean:
        selected_condition = "flip"
    else:
        selected_condition = "baseline"

    selected_seed = 17
    selected_run_id = f"{selected_condition}_s17"
    source_checkpoint = results_dir / selected_run_id / "checkpoint.pt"

    if not source_checkpoint.is_file():
        record = {
            "status": "incomplete",
            "missing_run_ids": [selected_run_id],
            "error": f"Selected checkpoint not found: {source_checkpoint}",
            "timestamp": timestamp,
        }
        _invalidate_and_record_incomplete(
            output_dir=output_dir,
            results_dir=results_dir,
            model_dir=model_dir,
            summary_path=summary_path,
            selection_path=selection_path,
            record=record,
        )
        print(f"Error: Selected checkpoint {source_checkpoint} does not exist.", file=sys.stderr)
        return

    dest_checkpoint = model_dir / "selected_state.pt"
    shutil.copy2(source_checkpoint, dest_checkpoint)
    checkpoint_sha = compute_file_sha256(dest_checkpoint)

    selection_record = {
        "status": "completed",
        "validation_means": {
            "baseline": float(round(baseline_mean, 6)),
            "flip": float(round(flip_mean, 6)),
        },
        "selection_rule": "mean validation accuracy across seeds 17 and 29; tie breaks to baseline; fixed seed 17 checkpoint",
        "selected_condition": selected_condition,
        "selected_seed": selected_seed,
        "selected_run_id": selected_run_id,
        "selected_checkpoint_sha256": checkpoint_sha,
        "timestamp": timestamp,
    }

    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection_record, f, indent=2)

    print(
        f"Model selection complete: Selected {selected_condition} "
        f"(baseline mean: {baseline_mean:.4f}, flip mean: {flip_mean:.4f}). "
        f"Copied {selected_run_id} checkpoint to {dest_checkpoint} (sha: {checkpoint_sha[:8]}...)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select model based on validation performance.")
    parser.add_argument("--output-dir", type=str, required=True, help="Submission root directory")
    args = parser.parse_args()

    select_model(Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
