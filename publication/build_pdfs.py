"""Build the four publication PDFs with the existing document pipeline."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parent.parent
PUBLICATION = ROOT / "publication"
BUILDER_PATH = ROOT / "tmp" / "pdfs" / "build_all_documents_pdf.py"


def load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "multimodal_document_builder",
        BUILDER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load document builder: {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    builder = load_builder()
    builder.TEMP_DIR = PUBLICATION / ".build"

    original_build_html = builder.build_html

    def build_html_with_base(*args, **kwargs):
        html = original_build_html(*args, **kwargs)
        base = ROOT.resolve().as_uri() + "/"
        return html.replace("<head>", f'<head><base href="{base}">', 1)

    builder.build_html = build_html_with_base

    jobs = [
        (
            PUBLICATION / "multimodal_neural_operator_paper_en.md",
            PUBLICATION / "pdf" / "Multimodal_Neural_Operator_Paper_EN.pdf",
            "en",
        ),
        (
            PUBLICATION / "multimodal_neural_operator_paper_zh.md",
            PUBLICATION / "pdf" / "Multimodal_Neural_Operator_Paper_ZH.pdf",
            "zh",
        ),
        (
            PUBLICATION / "multimodal_neural_operator_technical_document_en.md",
            PUBLICATION / "pdf" / "Multimodal_Neural_Operator_Technical_Document_EN.pdf",
            "en",
        ),
        (
            PUBLICATION / "multimodal_neural_operator_technical_document_zh.md",
            PUBLICATION / "pdf" / "Multimodal_Neural_Operator_Technical_Document_ZH.pdf",
            "zh",
        ),
    ]

    results = []
    for source, output, language in jobs:
        if not source.exists():
            raise FileNotFoundError(source)
        summary = builder.build_markdown(source, output, language)
        results.append(
            {
                "source": str(source),
                "output": str(output),
                "language": language,
                **summary,
            }
        )

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

