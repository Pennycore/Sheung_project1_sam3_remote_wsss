from __future__ import annotations

from pathlib import Path


def prepare_training_output(
    output_dir: Path,
    experiment_name: str,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    checkpoint_dir = output_dir / "checkpoints"
    log_path = output_dir / "train_log.jsonl"
    managed_paths = [
        log_path,
        checkpoint_dir / "best.pt",
        checkpoint_dir / "last.pt",
    ]
    existing = [path for path in managed_paths if path.exists()]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"{experiment_name} output already contains training artifacts: "
            f"{paths}. Use a new --output-dir, or pass --overwrite-output "
            "only when replacement is intentional."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            path.unlink()
    log_path.write_text("", encoding="utf-8")
    return checkpoint_dir, log_path
