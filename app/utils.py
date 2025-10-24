
from __future__ import annotations
import os
import random
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import yaml
import logging
import requests
# Optional GCS client (used as fallback if gsutil is not available)
try:
    from google.cloud import storage  # type: ignore
    HAVE_GCS_CLIENT = True
except Exception:
    storage = None  # type: ignore
    HAVE_GCS_CLIENT = False

logger = logging.getLogger(__name__)


def ensure_dir(p: str) -> None:
    """Ensure directory exists (mkdir -p equivalent)."""
    Path(p).mkdir(parents=True, exist_ok=True)


def generate_classification_data_yaml(root: str, out: str = "data_class.yaml") -> str:
    """Generate a classification-style data yaml for Ultralytics.

    Expects `root/train` and `root/val` directories with class subfolders.
    Returns the path to the written yaml.
    """
    root = Path(root)
    train = str((root / "train").resolve())
    val = str((root / "val").resolve())

    # collect class names from train folder subdirs
    classes = [p.name for p in sorted((root / "train").iterdir()) if p.is_dir()]
    nc = len(classes)
    data = {
        "train": train,
        "val": val,
        "nc": nc,
        "names": classes,
    }
    with open(out, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    logger.info("Generated classification data yaml: %s", out)
    return out


def generate_detection_data_yaml(root: str, out: str = "data_det.yaml") -> str:
    """Generate a detection-style data yaml that expects YOLO-format labels in same folders.

    Expects `root/images/train`, `root/images/val` or `root/train`/`root/val` with image/label pairs.
    Adjust this helper if your structure is different.
    """
    root = Path(root)
    # Try common patterns
    if (root / "images").exists():
        train = str((root / "images" / "train").resolve())
        val = str((root / "images" / "val").resolve())
    else:
        train = str((root / "train").resolve())
        val = str((root / "val").resolve())

    # try to get class names from a classes.txt or names file
    names_file = root / "names.txt"
    names = None
    if names_file.exists():
        names = [l.strip() for l in names_file.read_text().splitlines() if l.strip()]
    nc = len(names) if names else None

    data = {"train": train, "val": val}
    if nc:
        data.update({"nc": nc, "names": names})

    with open(out, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    logger.info("Generated detection data yaml: %s", out)
    return out


# ============================================================
# 1️⃣ Convert Food-101 to YOLO Classification Format
# ============================================================


def convert_food101_to_yolo(base_dir: str, output_subdir: str = "food101_yolo") -> Path:
    """
    Converts the original Food-101 dataset into YOLO classification format.
    Creates a folder with train/val folders each containing class subfolders.

    Expects the standard Food-101 structure:
      base_dir/images/<class>/<image>.jpg
      base_dir/meta/train.txt
      base_dir/meta/test.txt
      base_dir/meta/classes.txt

    Returns the Path to the created dataset folder.
    """
    base_dir = Path(base_dir)
    images_dir = base_dir / "images"
    meta_dir = base_dir / "meta"
    output_dir = base_dir / output_subdir

    train_file = meta_dir / "train.txt"
    test_file = meta_dir / "test.txt"
    classes_file = meta_dir / "classes.txt"

    if not train_file.exists() or not test_file.exists() or not classes_file.exists():
        raise FileNotFoundError("Food-101 meta files not found under: {}".format(meta_dir))

    # === CREATE OUTPUT FOLDERS ===
    for split_file, split_name in [(train_file, "train"), (test_file, "val")]:
        split_dir = output_dir / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

        with open(split_file) as f:
            lines = f.read().strip().splitlines()

        for line in lines:
            class_name, img_name = line.split("/")
            src = images_dir / f"{class_name}/{img_name}.jpg"
            if not src.exists():
                logger.warning("Missing source image: %s", src)
                continue
            dst_dir = split_dir / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst_dir / f"{img_name}.jpg")

    logger.info("Converted dataset to YOLO format at: %s", output_dir)

    # === READ CLASSES ===
    with open(classes_file) as f:
        class_names = [line.strip() for line in f if line.strip()]

    # === CREATE YAML FILE ===
    data_yaml = {
        "path": str(output_dir),
        "train": "train",
        "val": "val",
        "names": {i: name for i, name in enumerate(class_names)}
    }

    yaml_path = base_dir / f"{output_subdir}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    logger.info("Created YAML file: %s", yaml_path)
    return output_dir


# ============================================================
# 2️⃣ Create a Smaller Subset of the YOLO Dataset
# ============================================================


def make_small_subset(full_dataset_path: str, output_path: str, num_images_per_class: int = 100, seed: int = 42) -> Path:
    """
    Creates a smaller subset of a YOLO-style dataset for faster training.

    full_dataset_path: path to the dataset root (containing `train/` and `val/` subfolders)
    output_path: destination folder where the smaller dataset will be written
    """
    random.seed(seed)
    full_dataset = Path(full_dataset_path)
    output_dir = Path(output_path)

    for split in ["train", "val"]:
        src_split = full_dataset / split
        dst_split = output_dir / split
        if dst_split.exists():
            shutil.rmtree(dst_split)
        dst_split.mkdir(parents=True, exist_ok=True)

        classes = [d for d in src_split.iterdir() if d.is_dir()]
        for cls in classes:
            images = list(cls.glob("*.jpg"))
            random.shuffle(images)
            subset = images[:num_images_per_class]

            dst_cls = dst_split / cls.name
            dst_cls.mkdir(parents=True, exist_ok=True)

            for img_path in subset:
                shutil.copy(img_path, dst_cls / img_path.name)

    logger.info("Created small subset (%d per class) at: %s", num_images_per_class, output_dir)
    return output_dir


# ============================================================
# GCS helpers (gsutil or google-cloud-storage fallback)
# ============================================================

def _run_cmd(cmd: list) -> Tuple[int, str, str]:
    """Run subprocess command, return (returncode, stdout, stderr)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err


def gsutil_available() -> bool:
    rc, _, _ = _run_cmd(["which", "gsutil"])
    return rc == 0


def download_gcs_folder(gs_path: str, dst: Optional[str] = None, use_gsutil_first: bool = True) -> str:
    """
    Download a GCS folder (gs://bucket/path) to a local folder.
    If dst is None, creates a temporary directory and returns its path.
    Tries gsutil -m cp -r for speed; falls back to google-cloud-storage client.
    """
    assert gs_path.startswith("gs://"), "gs_path must start with gs://"
    if dst is None:
        dst = tempfile.mkdtemp(prefix="gcs_download_")
    dst = str(Path(dst).resolve())

    if use_gsutil_first and gsutil_available():
        cmd = ["gsutil", "-m", "cp", "-r", gs_path.rstrip("/") + "/*", dst]
        rc, out, err = _run_cmd(cmd)
        if rc == 0:
            logger.info("Downloaded GCS folder via gsutil to %s", dst)
            return dst
        else:
            logger.warning("gsutil failed (%s). Falling back to python client. stderr: %s", rc, err)

    if not HAVE_GCS_CLIENT:
        raise RuntimeError("gsutil not available and google-cloud-storage is not installed")

    # Fallback using google-cloud-storage client
    # gs_path format: gs://bucket_name/optional/path
    parts = gs_path[5:].split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)

    count = 0
    for blob in blobs:
        # skip "directory" placeholders
        if blob.name.endswith("/"):
            continue
        rel_path = blob.name[len(prefix):] if prefix else blob.name
        local_path = Path(dst) / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        count += 1

    logger.info("Downloaded %d files from gs://%s/%s to %s via python client", count, bucket_name, prefix, dst)
    return dst


def upload_folder_to_gcs(local_path: str, gs_dest: str, use_gsutil_first: bool = True) -> None:
    """
    Upload a local folder to a GCS destination like gs://bucket/runs/
    """
    assert gs_dest.startswith("gs://"), "gs_dest must start with gs://"
    if use_gsutil_first and gsutil_available():
        cmd = ["gsutil", "-m", "cp", "-r", str(local_path).rstrip("/") + "/*", gs_dest]
        rc, out, err = _run_cmd(cmd)
        if rc == 0:
            logger.info("Uploaded %s to %s via gsutil", local_path, gs_dest)
            return
        else:
            logger.warning("gsutil upload failed (%s). Falling back to python client. stderr: %s", rc, err)

    if not HAVE_GCS_CLIENT:
        raise RuntimeError("gsutil not available and google-cloud-storage is not installed")

    # fallback: use google-cloud-storage client (single-threaded)
    parts = gs_dest[5:].split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    uploaded = 0
    for root, _, files in os.walk(local_path):
        for fname in files:
            full = Path(root) / fname
            rel = os.path.relpath(full, local_path)
            blob_name = prefix + rel.replace("\\", "/")
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(full))
            uploaded += 1

    logger.info("Uploaded %d files to gs://%s/%s via python client", uploaded, bucket_name, prefix)


# ============================================================
# Nutrition & unit helpers
# ============================================================

def lb_to_kg(weight_lb: float) -> float:
    """Convert pounds to kilograms."""
    return weight_lb * 0.453592


def inch_to_cm(height_inch: float) -> float:
    """Convert inches to centimeters."""
    return height_inch * 2.54


def calculate_calories(
    weight_kg: float,
    height_cm: float,
    age: int,
    gender: str,
    activity_level: str = 'sedentary',
    goal: str = 'maintain',
) -> int:
    """
    Calculate daily calorie needs based on user data using Mifflin-St Jeor equation.

    Parameters:
    - weight_kg: Weight in kilograms
    - height_cm: Height in centimeters
    - age: Age in years
    - gender: 'male' or 'female'
    - activity_level: Activity level ['sedentary', 'light', 'moderate', 'active', 'very_active']
    - goal: Weight goal ['maintain', 'lose', 'gain']

    Returns:
    - calories: Approximate daily calorie intake (rounded integer)
    """
    if gender.lower() == 'male':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

    activity_factors: Dict[str, float] = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'very_active': 1.9,
    }

    calories = bmr * activity_factors.get(activity_level, 1.2)

    if goal.lower() == 'lose':
        calories -= 500
    elif goal.lower() == 'gain':
        calories += 500

    return int(round(calories))


API_KEY = "mGUx0gjzsaCgT8ITDOEARUi1gzxKdMPY1yD90ZHk"

def get_nutrition(food_name):
    """
    Fetch nutrition information for a given food name from USDA API.

    Parameters:
    - food_name (str): Name of the food to search for.

    Returns:
    - dict: Contains calories, protein, fat, and carbs.
    """
    # Build the search URL
    search_url = f"https://api.nal.usda.gov/fdc/v1/foods/search?api_key={API_KEY}&query={food_name}"

    # Send GET request to USDA API
    res = requests.get(search_url).json()

    # Pick the first result from the search
    if not res.get('foods'):
        # Return zeros if no result found
        return {'calories': 0, 'protein': 0, 'fat': 0, 'carbs': 0}

    food = res['foods'][0]

    # Extract nutrients into a dictionary
    nutrients = {n['nutrientName']: n['value'] for n in food['foodNutrients']}

    # Map nutrients to simplified names
    cal = nutrients.get('Energy', 0)
    protein = nutrients.get('Protein', 0)
    fat = nutrients.get('Total lipid (fat)', 0)
    carbs = nutrients.get('Carbohydrate, by difference', 0)

    return {'calories': cal, 'protein': protein, 'fat': fat, 'carbs': carbs}

# ============================================================
# ✅ Example usage (only runs when executed directly)
# ============================================================
if __name__ == "__main__":
    BASE = os.environ.get("FOOD101_BASE", "/Users/muhannad/code/zMuh/FoodVision/raw_data/food-101")

'''    # Step 1: Convert original Food-101 to YOLO format
    yolo_dataset = convert_food101_to_yolo(BASE)

    # Step 2: Create a small 100-image-per-class subset
    make_small_subset(
        full_dataset_path=yolo_dataset,
        output_path=os.path.join(BASE, "food101_yolo_small"),
        num_images_per_class=100,
    )

    # Example: calories
    print("example: 180 lb -> kg:", lb_to_kg(180))
    print("example: 70 inch -> cm:", inch_to_cm(70))
    print("example calories:", calculate_calories(weight_kg=lb_to_kg(180), height_cm=inch_to_cm(70), age=30, gender='male', activity_level='moderate', goal='maintain'))'''
