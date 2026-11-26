
from pathlib import Path

ORIGINAL = Path(
    r"C:\College\M.tech\Datasets\RDD2022\RDD2022_India_4Class\train\labels"
)

def check_boxes(label_dir):

    total = 0
    tiny = 0
    moderate = 0
    large = 0

    for label in label_dir.glob("*.txt"):

        for line_number, line in enumerate(
            label.read_text(encoding="utf-8").splitlines(), start=1
        ):

            parts = line.split()

            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                continue

            overflow = max(
                0,
                -(x - w / 2),
                (x + w / 2) - 1,
                -(y - h / 2),
                (y + h / 2) - 1,
            )

            if overflow > 0:
                total += 1

                if overflow <= 0.00001:
                    tiny += 1
                elif overflow <= 0.005:
                    moderate += 1
                else:
                    large += 1

                print(
                    label.name,
                    "line:", line_number,
                    "class:", class_id,
                    "overflow:", overflow
                )

    print("\n========== ORIGINAL DATASET OVERFLOW ==========")
    print("Total:", total)
    print("Tiny:", tiny)
    print("Moderate:", moderate)
    print("Large:", large)


if __name__ == "__main__":
    check_boxes(ORIGINAL)
