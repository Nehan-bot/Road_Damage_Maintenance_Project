from pathlib import Path
from datetime import datetime
import argparse
import zipfile
import math
import shutil

PROJECT_DATA = Path(
    r"data\raw\road_images\RDD2022_India_4Class_Corrected_Balanced600"
)
DEFAULT_ZIP = Path(
    r"experiments\manual_review\review_batch_01_cvat_export.zip"
)
BACKUP_ROOT = Path("experiments/manual_review/label_backups")

CLASS_NAMES = {
    0: "longitudinal crack",
    1: "transverse crack",
    2: "alligator crack",
    3: "pothole",
}


def get_zip_member(zf, suffix):
    matches = [n for n in zf.namelist() if n.replace("\\", "/").endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one ZIP member ending with {suffix}; found {matches[:5]}")
    return matches[0]


def parse_yolo_label(text, member_name):
    valid_lines = []
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{member_name}:{line_no}: expected 5 fields")
        try:
            class_id = int(parts[0])
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            raise ValueError(f"{member_name}:{line_no}: nonnumeric row")
        if class_id not in CLASS_NAMES:
            raise ValueError(f"{member_name}:{line_no}: unknown class id {class_id}")
        if not all(math.isfinite(v) for v in vals):
            raise ValueError(f"{member_name}:{line_no}: non-finite coordinate")
        x, y, w, h = vals
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"{member_name}:{line_no}: box outside normalized range")
        valid_lines.append(
            f"{class_id} " + " ".join(f"{v:.6f}" for v in vals)
        )
    return valid_lines


def main():
    parser = argparse.ArgumentParser(
        description="Safely merge CVAT-reviewed labels for the 125-image batch into project train labels."
    )
    parser.add_argument("--zip", default=str(DEFAULT_ZIP), help="Path to CVAT export ZIP")
    parser.add_argument("--apply", action="store_true", help="Write corrected labels; default is dry-run")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise FileNotFoundError(
            f"CVAT export ZIP not found: {zip_path}\n"
            f"Save your downloaded export there or pass --zip <path>."
        )

    if not PROJECT_DATA.exists():
        raise FileNotFoundError(f"Project dataset folder not found: {PROJECT_DATA}")

    with zipfile.ZipFile(zip_path) as zf:
        names = [n.replace("\\", "/") for n in zf.namelist()]
        train_member = get_zip_member(zf, "train.txt")
        train_text = zf.read(train_member).decode("utf-8-sig")
        frame_paths = [line.strip() for line in train_text.splitlines() if line.strip()]

        stems = [Path(p.replace("\\", "/")).stem for p in frame_paths]
        if len(stems) != 125:
            raise RuntimeError(f"Expected 125 reviewed frames, found {len(stems)}")
        if len(set(stems)) != len(stems):
            raise RuntimeError("Duplicate frame stems found in train.txt")

        label_members = {}
        for member in names:
            if member.lower().endswith(".txt") and "/labels/" in f"/{member}":
                label_members[Path(member).stem.lower()] = member

        reviewed = {}
        missing_export_labels = []
        for stem in stems:
            member = label_members.get(stem.lower())
            if member is None:
                # CVAT YOLO exports commonly omit a TXT for a frame with no boxes.
                reviewed[stem] = []
                missing_export_labels.append(stem)
            else:
                reviewed[stem] = parse_yolo_label(
                    zf.read(member).decode("utf-8-sig"), member
                )

    train_images = PROJECT_DATA / "train" / "images"
    train_labels = PROJECT_DATA / "train" / "labels"

    # Resolve target image by stem and confirm exact target label location.
    target_pairs = []
    for stem in stems:
        candidates = [
            p for p in train_images.iterdir()
            if p.is_file() and p.stem.lower() == stem.lower()
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one project train image for {stem}; found {len(candidates)}"
            )
        image_path = candidates[0]
        label_path = train_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Original train label missing: {label_path}")
        target_pairs.append((stem, image_path, label_path))

    # Count boxes in reviewed labels for a preview.
    class_counts = {c: 0 for c in CLASS_NAMES}
    for rows in reviewed.values():
        for row in rows:
            class_counts[int(row.split()[0])] += 1

    print("CVAT export review:")
    print(f"  Frames listed: {len(stems)}")
    print(f"  Label TXT files in export: {len(stems) - len(missing_export_labels)}")
    print(f"  Frames without TXT (treated as empty): {len(missing_export_labels)}")
    print(f"  Reviewed boxes: {sum(class_counts.values())}")
    for cid, name in CLASS_NAMES.items():
        print(f"    Class {cid} {name}: {class_counts[cid]}")
    print(f"  Project train images matched: {len(target_pairs)}")

    if not args.apply:
        print("\nDRY RUN ONLY. No project labels were changed.")
        print("If the counts and class mapping look correct, run with --apply.")
        return

    backup_dir = BACKUP_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=False)

    # Back up every original label before replacing any.
    for _, _, label_path in target_pairs:
        shutil.copy2(label_path, backup_dir / label_path.name)

    # Replace labels only for these 125 manually reviewed training images.
    for stem, _, label_path in target_pairs:
        rows = reviewed[stem]
        content = "\n".join(rows)
        if content:
            content += "\n"
        label_path.write_text(content, encoding="utf-8")

    print("\nMERGE COMPLETE.")
    print(f"Updated labels: {len(target_pairs)}")
    print(f"Backup of original labels: {backup_dir.resolve()}")
    print("No images or val/test labels were changed.")


if __name__ == "__main__":
    main()
