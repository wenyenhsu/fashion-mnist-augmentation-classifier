"""Run a controlled training and evaluation experiment on FashionMNIST subset."""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import FashionMNISTSubset, create_dataloader, get_transforms
from src.models import SmallCNN
from src.training import evaluate, train_one_epoch
from src.utils import (
    compute_actual_id_digest,
    compute_file_sha256,
    compute_source_sha256,
    export_history,
    export_predictions,
    save_config,
    update_summary,
)


def run_experiment(
    condition: str,
    seed: int,
    data_dir: Path,
    output_dir: Path,
    manifest_dir: Path,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 0.001,
    overwrite: bool = False,
) -> None:
    # Limit execution to 2 CPU threads
    torch.set_num_threads(2)
    device = torch.device("cpu")

    run_id = f"{condition}_s{seed}"
    results_dir = output_dir / "results"
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "summary.csv"

    start_time = time.time()

    # Load manifests and verify existence
    train_manifest = manifest_dir / "train.csv"
    val_manifest = manifest_dir / "validation.csv"
    train_manifest_sha = compute_file_sha256(train_manifest)
    val_manifest_sha = compute_file_sha256(val_manifest)

    # Compute source code hash mapping including executed entry script
    package_root = Path(__file__).resolve().parent
    source_sha_map = compute_source_sha256(package_root, entry_scripts=["run_experiment.py"])

    # Transforms: augmentation applied only to training images in 'flip' condition
    train_transform, eval_transform = get_transforms(condition=condition, seed=seed)

    # Datasets
    train_dataset = FashionMNISTSubset(manifest_path=train_manifest, data_dir=data_dir, transform=train_transform)
    train_eval_dataset = FashionMNISTSubset(manifest_path=train_manifest, data_dir=data_dir, transform=eval_transform)
    val_dataset = FashionMNISTSubset(manifest_path=val_manifest, data_dir=data_dir, transform=eval_transform)

    # Data loaders
    # Use deterministic generator for reproducible sample shuffling
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True, seed=seed)
    train_eval_loader = create_dataloader(train_eval_dataset, batch_size=batch_size, shuffle=False)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model initialization seeded deterministically
    torch.manual_seed(seed)
    model = SmallCNN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: list[dict] = []
    checkpoint_path = run_dir / "checkpoint.pt"
    val_pred_path = run_dir / "validation_predictions.csv"

    if checkpoint_path.is_file() or val_pred_path.is_file():
        if not overwrite:
            raise RuntimeError(
                f"Run directory '{run_dir}' already contains completed artifacts. "
                f"Provide a fresh --output-dir or pass --overwrite."
            )
        checkpoint_path.unlink(missing_ok=True)
        val_pred_path.unlink(missing_ok=True)

    try:
        val_loss, val_acc, val_preds = 0.0, 0.0, []
        for epoch in range(1, epochs + 1):
            train_loss, train_acc, train_count = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )

            # Evaluate model performance
            val_loss, val_acc, val_preds = evaluate(
                model=model,
                dataloader=train_eval_loader,
                criterion=criterion,
                device=device,
            )

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "validation_loss": val_loss,
                "validation_accuracy": val_acc,
                "train_count": train_count,
                "validation_count": len(val_preds),
            })

            print(
                f"[{run_id}] Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

        elapsed = time.time() - start_time

        # Save final epoch checkpoint (plain state_dict only)
        torch.save(model.state_dict(), checkpoint_path)
        checkpoint_sha = compute_file_sha256(checkpoint_path)

        # Export history
        export_history(run_dir / "history.csv", history)

        # Export actual evaluated prediction records
        export_predictions(
            output_path=run_dir / "validation_predictions.csv",
            run_id=run_id,
            course_split="validation",
            predictions=val_preds,
        )

        actual_sids = [
            p["sample_id"] if isinstance(p, dict) else p[0]
            for p in val_preds
        ]
        eval_id_digest = compute_actual_id_digest(actual_sids)

        config_data = {
            "run_id": run_id,
            "condition": condition,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "Adam",
            "learning_rate": lr,
            "architecture": "SmallCNN",
            "augmentation": "random_horizontal_flip(p=0.5)" if condition == "flip" else "none",
            "preprocessing": "to_tensor,scale_[0,1]",
            "train_manifest_sha256": train_manifest_sha,
            "validation_manifest_sha256": val_manifest_sha,
            "source_sha256": source_sha_map,
            "checkpoint_sha256": checkpoint_sha,
            "evaluation_source_partition": "train",
            "evaluation_ids_sha256": eval_id_digest,
            "evaluation_count": len(val_preds),
            "device": "cpu",
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "torchvision_version": sys.modules["torchvision"].__version__,
            "elapsed_seconds": round(elapsed, 4),
            "status": "completed",
            "error": None,
        }
        save_config(run_dir / "config.json", config_data)

        # Update summary row
        update_summary(summary_path, {
            "run_id": run_id,
            "condition": condition,
            "seed": seed,
            "epochs": epochs,
            "evaluation_split": "validation",
            "n_examples": len(val_preds),
            "loss": val_loss,
            "accuracy": val_acc,
            "status": "completed",
        })
        print(f"Run {run_id} completed in {elapsed:.2f}s.")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Run {run_id} failed: {e}", file=sys.stderr)
        # Ensure partial checkpoint or predictions from this failed run do not linger
        checkpoint_path.unlink(missing_ok=True)
        (run_dir / "validation_predictions.csv").unlink(missing_ok=True)
        if history:
            export_history(run_dir / "history.csv", history)

        config_data = {
            "run_id": run_id,
            "condition": condition,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "Adam",
            "learning_rate": lr,
            "architecture": "SmallCNN",
            "augmentation": "random_horizontal_flip(p=0.5)" if condition == "flip" else "none",
            "preprocessing": "to_tensor,scale_[0,1]",
            "train_manifest_sha256": train_manifest_sha,
            "validation_manifest_sha256": val_manifest_sha,
            "source_sha256": source_sha_map,
            "checkpoint_sha256": None,
            "evaluation_source_partition": "train",
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
        save_config(run_dir / "config.json", config_data)

        update_summary(summary_path, {
            "run_id": run_id,
            "condition": condition,
            "seed": seed,
            "epochs": epochs,
            "evaluation_split": "validation",
            "n_examples": 0,
            "loss": "",
            "accuracy": "",
            "status": "failed",
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled FashionMNIST experiment.")
    parser.add_argument("--condition", choices=["baseline", "flip"], required=True, help="Augmentation condition")
    parser.add_argument("--seed", type=int, required=True, help="Random seed (17 or 29)")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to FashionMNIST data directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Root directory for submission output")
    parser.add_argument("--manifest-dir", type=str, default=None, help="Path to directory containing manifest CSVs")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default 5)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default 128)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default 0.001)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run artifacts")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_dir = Path(args.manifest_dir).resolve() if args.manifest_dir else Path(__file__).resolve().parent / "manifests"

    run_experiment(
        condition=args.condition,
        seed=args.seed,
        data_dir=data_dir,
        output_dir=output_dir,
        manifest_dir=manifest_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
