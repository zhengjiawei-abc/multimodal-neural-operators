"""Render generated PDFs and create contact sheets for visual inspection."""

from __future__ import annotations

import math
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "pdf"
QA_DIR = ROOT / ".qa"


def make_contact_sheet(
    page_images: list[Path],
    output: Path,
    columns: int = 4,
) -> None:
    thumbnails = [Image.open(path).convert("RGB") for path in page_images]
    width = 360
    scaled = []
    for image in thumbnails:
        height = round(image.height * width / image.width)
        scaled.append(image.resize((width, height), Image.Resampling.LANCZOS))

    gap = 16
    label_height = 26
    cell_height = max(image.height for image in scaled) + label_height
    rows = math.ceil(len(scaled) / columns)
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
        draw.rectangle(
            [x, y, x + width, y + cell_height],
            fill="#ffffff",
        )
        sheet.paste(image, (x, y + label_height))
        draw.text((x + 8, y + 6), f"page {index + 1}", fill="#263241")

    sheet.save(output)


def render_pdf(pdf_path: Path) -> dict[str, object]:
    output_dir = QA_DIR / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(pdf_path)
    page_images: list[Path] = []
    text_chars = 0
    blank_pages: list[int] = []

    for index, page in enumerate(document):
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
        image_path = output_dir / f"page-{index + 1:02d}.png"
        pixmap.save(image_path)
        page_images.append(image_path)

        text = page.get_text("text")
        text_chars += len(text.strip())
        if index > 0 and len(text.strip()) < 20:
            blank_pages.append(index + 1)

    for start in range(0, len(page_images), 8):
        make_contact_sheet(
            page_images[start : start + 8],
            output_dir / f"contact-{start // 8 + 1:02d}.png",
        )

    result = {
        "pdf": str(pdf_path),
        "pages": len(document),
        "text_chars": text_chars,
        "blank_pages": blank_pages,
        "output_dir": str(output_dir),
    }
    document.close()
    return result


def main() -> None:
    results = [render_pdf(path) for path in sorted(PDF_DIR.glob("*.pdf"))]
    for result in results:
        print(result)


if __name__ == "__main__":
    main()

