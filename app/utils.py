import os
import random
import shutil
from pathlib import Path
import yaml


# ============================================================
# 1️⃣ Convert Food-101 to YOLO Classification Format
# ============================================================

def convert_food101_to_yolo(base_dir: str):
    """
    Converts the original Food-101 dataset into YOLO classification format.
    Creates a folder with train/val folders each containing class subfolders.
    """
    base_dir = Path(base_dir)
    images_dir = base_dir / "images"
    meta_dir = base_dir / "meta"
    output_dir = base_dir / "food101_yolo"

    train_file = meta_dir / "train.txt"
    test_file = meta_dir / "test.txt"
    classes_file = meta_dir / "classes.txt"

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
            dst_dir = split_dir / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst_dir / f"{img_name}.jpg")

    print(f"✅ Converted dataset to YOLO format at: {output_dir}")

    # === READ CLASSES ===
    with open(classes_file) as f:
        class_names = [line.strip() for line in f]

    # === CREATE YAML FILE ===
    data_yaml = {
        "path": str(output_dir),
        "train": "train",
        "val": "val",
        "names": {i: name for i, name in enumerate(class_names)}
    }

    yaml_path = base_dir / "food101_yolo.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    print(f"✅ Created YAML file: {yaml_path}")
    return output_dir


# ============================================================
# 2️⃣ Create a Smaller Subset of the YOLO Dataset
# ============================================================

def make_small_subset(full_dataset_path: str, output_path: str, num_images_per_class: int = 100, seed: int = 42):
    """
    Creates a smaller subset of a YOLO-style dataset for faster training.
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

    print(f"✅ Created small subset ({num_images_per_class} per class) at: {output_dir}")
    return output_dir


# ============================================================
# ✅ Example usage (only runs when executed directly)
# ============================================================

if __name__ == "__main__":
    BASE = "/Users/muhannad/code/zMuh/FoodVision/raw_data/food-101"

    # Step 1: Convert original Food-101 to YOLO format
    yolo_dataset = convert_food101_to_yolo(BASE)

    # Step 2: Create a small 100-image-per-class subset
    make_small_subset(
        full_dataset_path=yolo_dataset,
        output_path=os.path.join(BASE, "food101_yolo_small"),
        num_images_per_class=100
    )
