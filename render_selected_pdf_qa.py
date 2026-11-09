"""Render selected generated PDFs into contact sheets for visual inspection."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw


def make_contact_sheet(images: list[Image.Image], output: Path) -> None:
    width = 360
    gap = 16
    label_height = 26
    scaled = []
    for image in images:
        height = round(image.height * width / image.width)
        scaled.append(image.resize((width, height), Image.Resampling.LANCZOS))
    columns = 4
    rows = math.ceil(len(scaled) / columns)
    cell_height = max(image.height for image in scaled) + label_height
    sheet = Image.new(
        "RGB",
        (columns * width + (columns + 1) * gap, rows * cell_height + (rows + 1) * gap),
        "#d9dee3",
    )
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(scaled):
        row = index // columns
        column = index % columns
        x = gap + column * (width + gap)
        y = gap + row * (cell_height + gap)
        draw.rectangle([x, y, x + width, y + cell_height], fill="#ffffff")
        sheet.paste(image, (x, y + label_height))
        draw.text((x + 8, y + 6), f"page {index + 1}", fill="#263241")
    sheet.save(output)


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: render_selected_pdf_qa.py PDF OUTPUT_DIR")
    pdf_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(pdf_path)
    rendered = []
    for page in document:
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        rendered.append(image)
    for start in range(0, len(rendered), 8):
        make_contact_sheet(
            rendered[start : start + 8],
            output_dir / f"contact-{start // 8 + 1:02d}.png",
        )
    print(
        {
            "pdf": str(pdf_path),
            "pages": len(document),
            "output_dir": str(output_dir),
        }
    )
    document.close()


if __name__ == "__main__":
    main()

