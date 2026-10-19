"""Download and verify FashionMNIST dataset for CSCI 566 assignment."""

import argparse
import sys
from pathlib import Path
from torchvision.datasets import FashionMNIST

from src.dataset import FashionMNISTSubset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify FashionMNIST dataset.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory to store FashionMNIST data")
    parser.add_argument("--manifest-dir", type=str, default=None, help="Directory containing train/validation/test manifests")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    manifest_dir = Path(args.manifest_dir).resolve() if args.manifest_dir else Path(__file__).resolve().parent / "manifests"

    print(f"Checking FashionMNIST data in: {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download if not present
    FashionMNIST(root=str(data_dir), train=True, download=True)
    FashionMNIST(root=str(data_dir), train=False, download=True)

    # Verify manifests
    manifest_files = {
        "train": manifest_dir / "train.csv",
        "validation": manifest_dir / "validation.csv",
        "test": manifest_dir / "test.csv",
    }

    for name, path in manifest_files.items():
        if not path.is_file():
            print(f"Error: Missing manifest {path}", file=sys.stderr)
            sys.exit(1)
        subset = FashionMNISTSubset(manifest_path=path, data_dir=data_dir)
        print(f"Verified {name} split: {len(subset)} examples successfully loaded.")

    print("Data preparation and verification complete.")


if __name__ == "__main__":
    main()
