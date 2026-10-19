"""Final test evaluation script for selected model checkpoint."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn

from src.dataset import FashionMNISTSubset, create_dataloader, get_transforms
from src.models import SmallCNN
from src.training import evaluate
from src.utils import (
    compute_actual_id_digest,
    compute_file_sha256,
    export_predictions,
    save_config,
    update_summary,
)


def evaluate_test(
    data_dir: Path,
    output_dir: Path,
    manifest_dir: Path,
    batch_size: int = 128,
    overwrite: bool = False,
) -> None:
    torch.set_num_threads(2)
    device = torch.device("cpu")

    results_dir = output_dir / "results"
    final_test_dir = results_dir / "final_test"
    final_test_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "summary.csv"

    # Setup checks: failures before evaluation phase are explicit terminal errors
    selection_path = results_dir / "selection.json"
    if not selection_path.is_file():
        print(f"Error: selection record not found at {selection_path}. Run select_model.py first.", file=sys.stderr)
        sys.exit(1)

    with open(selection_path, "r", encoding="utf-8") as f:
        selection_record = json.load(f)

    if selection_record.get("status") != "completed":
        print(f"Error: selection is incomplete: {selection_record.get('error')}", file=sys.stderr)
        sys.exit(1)

    selected_run = selection_record["selected_run_id"]
    selected_condition = selection_record["selected_condition"]
    selected_seed = selection_record["selected_seed"]

    selected_checkpoint_path = output_dir / "model" / "selected_state.pt"
    if not selected_checkpoint_path.is_file():
        print(f"Error: Selected checkpoint not found at {selected_checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    # Verify checkpoint hash matches selection record BEFORE evaluation
    expected_sha = selection_record.get("selected_checkpoint_sha256")
    checkpoint_sha = compute_file_sha256(selected_checkpoint_path)
    if expected_sha is not None and checkpoint_sha != expected_sha:
        print(
            f"Error: Selected checkpoint hash mismatch. selection.json recorded {expected_sha}, "
            f"but {selected_checkpoint_path} has {checkpoint_sha}. "
            f"Model weights were modified or replaced after model selection.",
            file=sys.stderr,
        )
        sys.exit(1)

    test_manifest_path = manifest_dir / "test.csv"
    if not test_manifest_path.is_file():
        print(f"Error: Test manifest not found at {test_manifest_path}", file=sys.stderr)
        sys.exit(1)

    test_manifest_sha = compute_file_sha256(test_manifest_path)

    test_pred_path = final_test_dir / "predictions.csv"
    if test_pred_path.is_file():
        if not overwrite:
            print(
                f"Error: Final test predictions already exist at {test_pred_path}. "
                f"Provide a fresh output directory or pass --overwrite.",
                file=sys.stderr,
            )
            sys.exit(1)
        test_pred_path.unlink()

    # Load test dataset with eval transform (no augmentation)
    _, eval_transform = get_transforms(condition="baseline", seed=17)
    test_dataset = FashionMNISTSubset(manifest_path=test_manifest_path, data_dir=data_dir, transform=eval_transform)
    test_loader = create_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model state
    model = SmallCNN().to(device)
    state_dict = torch.load(selected_checkpoint_path, weights_only=True, map_location="cpu")
    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    try:
        test_loss, test_acc, test_preds = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )
        elapsed = time.time() - start_time

        # Export test predictions
        export_predictions(
            output_path=test_pred_path,
            run_id="final_test",
            course_split="test",
            predictions=test_preds,
        )

        actual_sids = [
            p["sample_id"] if isinstance(p, dict) else p[0]
            for p in test_preds
        ]
        eval_id_digest = compute_actual_id_digest(actual_sids)

        config_data = {
            "run_id": "final_test",
            "selected_run": selected_run,
            "selected_checkpoint_sha256": checkpoint_sha,
            "test_manifest_sha256": test_manifest_sha,
            "evaluation_source_partition": "test",
            "evaluation_ids_sha256": eval_id_digest,
            "evaluation_count": len(test_preds),
            "device": "cpu",
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "torchvision_version": sys.modules["torchvision"].__version__,
            "elapsed_seconds": round(elapsed, 4),
            "status": "completed",
            "error": None,
        }
        save_config(final_test_dir / "config.json", config_data)

        update_summary(summary_path, {
            "run_id": "final_test",
            "condition": selected_condition,
            "seed": selected_seed,
            "epochs": "",
            "evaluation_split": "test",
            "n_examples": len(test_preds),
            "loss": test_loss,
            "accuracy": test_acc,
            "status": "completed",
        })

        print(
            f"Final test evaluation completed in {elapsed:.2f}s: "
            f"Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f} on {len(test_preds)} test samples."
        )

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Final test evaluation failed: {e}", file=sys.stderr)
        test_pred_path.unlink(missing_ok=True)
        config_data = {
            "run_id": "final_test",
            "selected_run": selected_run,
            "selected_checkpoint_sha256": checkpoint_sha,
            "test_manifest_sha256": test_manifest_sha,
            "evaluation_source_partition": "test",
            "evaluation_ids_sha256": compute_actual_id_digest([]),
            "evaluation_count": 0,
            "device": "cpu",
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "torchvision_version": sys.modules["torchvision"].__version__,
            "elapsed_seconds": round(elapsed, 4),
            "status": "failed",
            "error": str(e),
        }
        save_config(final_test_dir / "config.json", config_data)
        update_summary(summary_path, {
            "run_id": "final_test",
            "condition": selected_condition,
            "seed": selected_seed,
            "epochs": "",
            "evaluation_split": "test",
            "n_examples": 0,
            "loss": "",
            "accuracy": "",
            "status": "failed",
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate selected model on final test set.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to FashionMNIST data directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Submission root directory")
    parser.add_argument("--manifest-dir", type=str, default=None, help="Directory containing test manifest")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for test evaluation")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing test predictions")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_dir = Path(args.manifest_dir).resolve() if args.manifest_dir else Path(__file__).resolve().parent / "manifests"

    evaluate_test(
        data_dir=data_dir,
        output_dir=output_dir,
        manifest_dir=manifest_dir,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
