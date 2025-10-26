from __future__ import annotations
import argparse
import os
import sys
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging
import shutil

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

## Helpers Functions
from app.utils import (
    download_gcs_folder,
    upload_folder_to_gcs,
)


# -----------------------------
# Train
# -----------------------------

def _find_latest_run_dir(task: str = "detect") -> Optional[Path]:
    """Find the most recent run directory under runs/{task}/train* or runs/{task}/train*/weights."""
    runs_root = Path("runs")
    task_dir = runs_root / task
    if not task_dir.exists():
        return None
    candidates = list(task_dir.glob("train*"))
    if not candidates:
        # maybe tuning or custom names
        candidates = list(task_dir.iterdir())
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def _collect_weights_from_run(run_dir: Path) -> Dict[str, Path]:
    """Return dict with keys 'best' and 'last' if found inside run_dir or run_dir/weights."""
    candidates = {}
    possible_paths = [run_dir, run_dir / "weights"]
    for base in possible_paths:
        if not base.exists():
            continue
        for name in ["best.pt", "last.pt"]:
            p = base / name
            if p.exists():
                candidates[name.split(".")[0]] = p
    # also check for any .pt in base
    if not candidates:
        for base in possible_paths:
            if not base.exists():
                continue
            pts = list(base.glob("*.pt"))
            for p in pts:
                if "best" in p.name:
                    candidates.setdefault("best", p)
                elif "last" in p.name:
                    candidates.setdefault("last", p)
                else:
                    # keep a fallback
                    candidates.setdefault("other", p)
    return candidates


def train(
    data: str,
    model: str = "yolo11n-cls.pt",
    task: str = "classify",
    epochs: int = 10,
    device: str = "cpu",
    batch: int = 64,
    imgsz: int = 224,
    workers: int = 4,
    save: bool = True,
    pretrained: Optional[str] = None,
    # MLflow / GCS options
    save_to_gs: Optional[str] = None,
    **kwargs,
):
    """Train the model.

    - data: path to data yaml, a local folder, or a GCS path starting with gs://
    - model: pretrained weights or model name
    - task: 'classify' or 'detect'

    Optional MLflow integration:
      -- if save_to_mlflow True (or mlflow_experiment provided) the script will try to log artifacts to MLflow.
      -- use mlflow_uri to point to a remote tracking server.

    Optional GCS upload:
      -- save_to_gs: e.g. gs://my-bucket/runs/exp1 will upload the run directory after training.
    """
    _local_tmp_dir = None

    # If provided a GCS path, download locally first
    if isinstance(data, str) and data.startswith("gs://"):
        logger.info("Detected gs:// path, downloading dataset to local temp dir...")
        _local_tmp_dir = download_gcs_folder(data)
        data = str(Path(_local_tmp_dir).resolve())

    logger.info("Starting training: model=%s task=%s data=%s", model, task, data)
    y = YOLO(model)
    y.train(
        data=data,
        epochs=epochs,
        device=device,
        batch=batch,
        imgsz=imgsz,
        workers=workers,
        save=save,
        task=task,
        **kwargs,
    )

    # After training, find latest run directory and artifacts
    run_dir = _find_latest_run_dir(task=task)
    if run_dir:
        logger.info("Detected run directory: %s", run_dir)
        weights = _collect_weights_from_run(run_dir)
        logger.info("Collected weights: %s", weights)

        # upload to GCS if requested
        if save_to_gs:
            try:
                upload_folder_to_gcs(str(run_dir), save_to_gs)
                logger.info("Uploaded run directory to %s", save_to_gs)
            except Exception:
                logger.exception("Failed to upload run dir to GCS")

    # cleanup local temp downloaded dataset
    if _local_tmp_dir:
        try:
            shutil.rmtree(_local_tmp_dir)
        except Exception:
            logger.warning("Failed to remove temp dir: %s", _local_tmp_dir)


# -----------------------------
# Tuning
# -----------------------------

def default_search_space() -> Dict[str, Any]:
    return {
        "lr0": (1e-5, 1e-1),
        "lrf": (0.01, 1.0),
        "momentum": (0.6, 0.98),
        "weight_decay": (0.0, 0.001),
        "warmup_epochs": (0.0, 5.0),
        "warmup_momentum": (0.0, 0.95),
        # detection-specific losses
        "box": (0.02, 0.2),
        "cls": (0.2, 4.0),
        # data augmentation ranges
        "hsv_h": (0.0, 0.1),
        "hsv_s": (0.0, 0.9),
        "hsv_v": (0.0, 0.9),
        "degrees": (0.0, 45.0),
        "translate": (0.0, 0.9),
        "scale": (0.0, 0.9),
        "shear": (0.0, 10.0),
        "perspective": (0.0, 0.001),
        "flipud": (0.0, 1.0),
        "fliplr": (0.0, 1.0),
        "mosaic": (0.0, 1.0),
        "mixup": (0.0, 1.0),
        "copy_paste": (0.0, 1.0),
    }


def tune(
    data: str,
    model: str = "yolo11n.pt",
    task: str = "detect",
    epochs: int = 30,
    iterations: int = 300,
    optimizer: str = "SGD",
    space: Optional[Dict[str, Any]] = None,
    plots: bool = False,
    save: bool = True,
    val: bool = True,
    resume: bool = False,
    use_ray: bool = False,
    name: Optional[str] = None,
    save_to_gs: Optional[str] = None,
    **kwargs,
):
    """Run hyperparameter tuning using Ultralytics' model.tune()

    - data: yaml path or a folder
    - model: base model checkpoints
    - task: 'detect' or 'classify'
    - iterations: number of tuning iterations
    """
    _local_tmp_dir = None
    if isinstance(data, str) and data.startswith("gs://"):
        logger.info("Detected gs:// path for tuning, downloading...")
        _local_tmp_dir = download_gcs_folder(data)
        data = str(Path(_local_tmp_dir).resolve())

    search_space = space or default_search_space()
    y = YOLO(model)
    logger.info("Starting tuning: model=%s data=%s iterations=%s", model, data, iterations)

    results = y.tune(
        data=data,
        epochs=epochs,
        iterations=iterations,
        optimizer=optimizer,
        space=search_space,
        plots=plots,
        save=save,
        val=val,
        resume=resume,
        use_ray=use_ray,
        name=name,
        **kwargs,
    )

    logger.info("Tuning finished. Results: %s", results)

    # cleanup
    if _local_tmp_dir:
        try:
            shutil.rmtree(_local_tmp_dir)
        except Exception:
            logger.warning("Failed to remove temp dir: %s", _local_tmp_dir)

    return results


# -----------------------------
# Evaluate & Predict
# -----------------------------

def evaluate(weights: str, data: Optional[str] = None, task: str = "detect") -> None:
    y = YOLO(weights)
    logger.info("Evaluating %s on %s", weights, data)
    ret = y.val(data=data, task=task)
    logger.info("Eval results: %s", ret)
    return ret


def predict(weights: str, source: str, imgsz: int = 640, conf: float = 0.25, save: bool = False) -> None:
    y = YOLO(weights)
    logger.info("Running predict: weights=%s source=%s", weights, source)
    results = y.predict(source=source, imgsz=imgsz, conf=conf, save=save)
    return results


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train / Tune / Eval / Predict with Ultralytics YOLO")
    sub = p.add_subparsers(dest="cmd")

    # train
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--model", default="yolo11n-cls.pt")
    t.add_argument("--task", default="classify", choices=["classify", "detect"])
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--device", default="cpu")
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--imgsz", type=int, default=224)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--save-gs", default=None, help="GCS path to upload run outputs, e.g. gs://bucket/runs/exp1")


    # tune
    tt = sub.add_parser("tune")
    tt.add_argument("--data", required=True)
    tt.add_argument("--model", default="yolo11n.pt")
    tt.add_argument("--task", default="detect", choices=["detect", "classify"])
    tt.add_argument("--epochs", type=int, default=30)
    tt.add_argument("--iterations", type=int, default=300)
    tt.add_argument("--optimizer", default="SGD")
    tt.add_argument("--use_ray", action="store_true")
    tt.add_argument("--resume", action="store_true")
    tt.add_argument("--name", default=None)
    tt.add_argument("--save-gs", default=None, help="GCS path to upload tune outputs")


    # eval
    e = sub.add_parser("eval")
    e.add_argument("--weights", required=True)
    e.add_argument("--data", default=None)
    e.add_argument("--task", default="detect")

    # predict
    p2 = sub.add_parser("predict")
    p2.add_argument("--weights", required=True)
    p2.add_argument("--source", required=True)
    p2.add_argument("--imgsz", type=int, default=640)
    p2.add_argument("--conf", type=float, default=0.25)
    p2.add_argument("--save", action="store_true")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.cmd == "train":
        train(
            data=args.data,
            model=args.model,
            task=args.task,
            epochs=args.epochs,
            device=args.device,
            batch=args.batch,
            imgsz=args.imgsz,
            workers=args.workers,
            save_to_gs=args.save_gs,
        )
    elif args.cmd == "tune":
        tune(
            data=args.data,
            model=args.model,
            task=args.task,
            epochs=args.epochs,
            iterations=args.iterations,
            optimizer=args.optimizer,
            use_ray=args.use_ray,
            resume=args.resume,
            name=args.name,
            save_to_gs=args.save_gs,

        )
    elif args.cmd == "eval":
        evaluate(weights=args.weights, data=args.data, task=args.task)
    elif args.cmd == "predict":
        predict(weights=args.weights, source=args.source, imgsz=args.imgsz, conf=args.conf, save=args.save)
    else:
        print("No command provided. Use train | tune | eval | predict")


if __name__ == "__main__":
    main()
