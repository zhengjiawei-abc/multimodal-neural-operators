"""Validate source links, generated PDFs, and rendered-text sanity."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf


ROOT = Path(__file__).resolve().parent.parent
PUBLICATION = Path(__file__).resolve().parent

MARKDOWN_FILES = [
    PUBLICATION / "multimodal_neural_operator_paper_en.md",
    PUBLICATION / "multimodal_neural_operator_paper_zh.md",
    PUBLICATION / "multimodal_neural_operator_technical_document_en.md",
    PUBLICATION / "multimodal_neural_operator_technical_document_zh.md",
]

PDF_FILES = [
    PUBLICATION / "pdf" / "Multimodal_Neural_Operator_Paper_EN.pdf",
    PUBLICATION / "pdf" / "Multimodal_Neural_Operator_Paper_ZH.pdf",
    PUBLICATION / "pdf" / "Multimodal_Neural_Operator_Technical_Document_EN.pdf",
    PUBLICATION / "pdf" / "Multimodal_Neural_Operator_Technical_Document_ZH.pdf",
]


def validate_markdown(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("# "):
        raise ValueError(f"Missing level-one title: {path}")
    if "ZQXMATH" in text:
        raise ValueError(f"Unrestored math token: {path}")

    image_refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    missing = []
    for reference in image_refs:
        target = (ROOT / reference).resolve()
        if not target.exists():
            missing.append(reference)
    if missing:
        raise FileNotFoundError(f"Missing image references in {path}: {missing}")

    return {
        "path": str(path),
        "characters": len(text),
        "images": len(image_refs),
        "level_two_sections": len(re.findall(r"(?m)^##\s+", text)),
    }


def validate_pdf(path: Path) -> dict[str, object]:
    document = pymupdf.open(path)
    extracted = "\n".join(page.get_text("text") for page in document)
    metadata = document.metadata
    result = {
        "path": str(path),
        "pages": len(document),
        "characters": len(extracted.strip()),
        "title": metadata.get("title", ""),
        "replacement_characters": extracted.count("\ufffd"),
    }
    document.close()
    if result["pages"] < 5:
        raise ValueError(f"Unexpectedly short PDF: {path}")
    if result["characters"] < 2000:
        raise ValueError(f"PDF has too little extracted text: {path}")
    if result["replacement_characters"]:
        raise ValueError(f"PDF text contains replacement characters: {path}")
    return result


def main() -> None:
    for path in MARKDOWN_FILES:
        print(validate_markdown(path))
    for path in PDF_FILES:
        print(validate_pdf(path))


if __name__ == "__main__":
    main()

