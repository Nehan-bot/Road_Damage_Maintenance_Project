
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
import math

csv_path = Path(
    "data/processed/priority/manual_review_sample.csv"
)

image_dir = Path(
    "data/processed/priority/review_images"
)

output_file = Path(
    "data/processed/priority/review_contact_sheet.jpg"
)

df = pd.read_csv(csv_path)

thumb_width = 240
thumb_height = 160
cell_width = 260
cell_height = 205
columns = 5

rows = math.ceil(len(df) / columns)

sheet = Image.new(
    "RGB",
    (columns * cell_width, rows * cell_height),
    "white"
)

draw = ImageDraw.Draw(sheet)

for index, row in df.iterrows():

    image_path = image_dir / row["image_name"]

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((thumb_width, thumb_height))

    x = (index % columns) * cell_width
    y = (index // columns) * cell_height

    image_x = x + (cell_width - image.width) // 2

    sheet.paste(image, (image_x, y))

    label = (
        f'{row["image_name"]}\n'
        f'Priority: {row["priority_label"]}'
    )

    draw.text(
        (x + 5, y + 165),
        label,
        fill="black"
    )

sheet.save(output_file, quality=90)

print("Contact sheet created.")
print("Images:", len(df))
print("Saved:", output_file)
