"""Build a two-column paper PDF in the style of a NeurIPS manuscript."""

from __future__ import annotations

import html
import importlib.util
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright


PUBLICATION = Path(__file__).resolve().parent
ROOT = PUBLICATION.parent
SOURCE = PUBLICATION / "multimodal_neural_operator_paper_zh.md"
OUTPUT = PUBLICATION / "pdf" / "Multimodal_Neural_Operator_Paper_AttentionStyle_ZH.pdf"
BUILD = PUBLICATION / ".attention_build"


def load_renderer():
    spec = importlib.util.spec_from_file_location(
        "document_renderer",
        ROOT / "build_multimodal_pdf_set.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the document renderer.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def markdown_engine():
    return markdown.Markdown(
        extensions=[
            "tables",
            "fenced_code",
            "codehilite",
            "sane_lists",
            "attr_list",
        ],
        extension_configs={"codehilite": {"guess_lang": False, "noclasses": False}},
        output_format="html5",
    )


def render_segment(segment: str, renderer) -> tuple[str, bool]:
    protected, replacements = renderer.protect_math(segment)
    engine = markdown_engine()
    rendered = renderer.restore_math(engine.convert(protected), replacements)
    return rendered, bool(replacements)


def build_html(
    title: str,
    metadata: dict[str, str],
    abstract: str,
    body: str,
    has_math: bool,
) -> str:
    safe_title = html.escape(title)
    english_title = html.escape(
        metadata.get("English title", "Query-Resolved Multimodal Neural Operators")
    )
    authors = metadata.get("Authors", "作者信息待补充")
    affiliations = metadata.get("Affiliations", "单位信息待补充")
    corresponding = metadata.get("Corresponding author", "通讯作者信息待补充")
    date = metadata.get("Date", "2026 年 11 月 9 日")

    math_script = ""
    if has_math:
        math_script = """
<script>
window.MathJax = {
  tex: {inlineMath: [["\\\\(", "\\\\)"], ["$", "$"]],
        displayMath: [["\\\\[", "\\\\]"], ["$$", "$$"]]},
  svg: {fontCache: "global"},
  startup: {typeset: false}
};
window.__mathjaxLoaded = false;
window.__mathjaxFailed = false;
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
  onload="window.__mathjaxLoaded = true"
  onerror="window.__mathjaxFailed = true"></script>
"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{safe_title}</title>
<meta name="author" content="{html.escape(authors)}">
<base href="{ROOT.resolve().as_uri()}/">
<style>
@font-face {{
  font-family: "PaperSans";
  src: url("file:///C:/Windows/Fonts/NotoSansSC-VF.ttf") format("truetype");
  font-weight: 100 900;
}}
@font-face {{
  font-family: "PaperSerif";
  src: url("file:///C:/Windows/Fonts/NotoSerifSC-VF.ttf") format("truetype");
  font-weight: 100 900;
}}
@page {{ size: A4; }}
* {{ box-sizing: border-box; }}
html {{ color-scheme: light; }}
body {{
  margin: 0;
  color: #111;
  background: #fff;
  font-family: "PaperSerif", "SimSun", serif;
  font-size: 8.7pt;
  line-height: 1.56;
}}
.masthead {{
  text-align: center;
  padding: 0 8mm 5mm;
  border-bottom: .25mm solid #b8b8b8;
  margin-bottom: 4mm;
}}
.kicker {{
  margin: 0 0 2.5mm;
  color: #555;
  font-family: "PaperSans", "Microsoft YaHei", sans-serif;
  font-size: 7pt;
  letter-spacing: .14em;
  text-transform: uppercase;
}}
h1 {{
  margin: 0 auto 2.5mm;
  max-width: 150mm;
  font-family: "PaperSerif", "SimSun", serif;
  font-size: 20pt;
  line-height: 1.18;
  font-weight: 700;
}}
.english-title {{
  margin: 0 auto 3.5mm;
  max-width: 155mm;
  color: #333;
  font-family: "PaperSerif", "Times New Roman", serif;
  font-size: 10.3pt;
  line-height: 1.25;
}}
.authors {{
  margin: 0 0 1.2mm;
  font-family: "PaperSans", "Microsoft YaHei", sans-serif;
  font-size: 9.2pt;
  font-weight: 650;
}}
.affiliations, .corresponding {{
  margin: .7mm auto;
  color: #444;
  font-family: "PaperSans", "Microsoft YaHei", sans-serif;
  font-size: 7.1pt;
  line-height: 1.35;
}}
.abstract {{
  margin: 0 0 4mm;
  padding: 3.6mm 8mm 3.2mm;
  background: #f4f4f4;
  border-top: .35mm solid #222;
  border-bottom: .2mm solid #999;
}}
.abstract h2 {{
  margin: 0 0 1.5mm;
  border: 0;
  padding: 0;
  font-family: "PaperSans", "Microsoft YaHei", sans-serif;
  font-size: 9.8pt;
  text-align: center;
  letter-spacing: .05em;
}}
.abstract p {{
  margin: 0;
  font-size: 8.5pt;
  line-height: 1.52;
  text-align: justify;
}}
.columns {{
  column-count: 2;
  column-gap: 7.2mm;
  column-rule: .15mm solid #ddd;
}}
h2, h3, h4 {{
  break-after: avoid;
  page-break-after: avoid;
}}
h2 {{
  margin: 4.2mm 0 1.5mm;
  padding-top: 1.2mm;
  border-top: .35mm solid #222;
  font-family: "PaperSans", "Microsoft YaHei", sans-serif;
  font-size: 11.2pt;
  line-height: 1.22;
}}
h3 {{
  margin: 3.2mm 0 1.1mm;
  font-family: "PaperSans", "Microsoft YaHei", sans-serif;
  font-size: 9.4pt;
  line-height: 1.25;
}}
h4 {{
  margin: 2.5mm 0 .9mm;
  font-family: "PaperSans", "Microsoft YaHei", sans-serif;
  font-size: 8.9pt;
  line-height: 1.25;
}}
p {{
  margin: 0 0 2.2mm;
  text-align: justify;
  orphans: 2;
  widows: 2;
}}
ul, ol {{
  margin: 1.4mm 0 2.4mm;
  padding-left: 5mm;
}}
li {{ margin: .55mm 0; }}
strong {{ font-weight: 700; }}
code {{
  font-family: "Cascadia Code", Consolas, monospace;
  font-size: 7.1pt;
  overflow-wrap: anywhere;
}}
pre {{
  margin: 2mm 0 2.8mm;
  padding: 2.2mm;
  border-left: .6mm solid #777;
  background: #f6f6f6;
  font-family: "Cascadia Code", Consolas, monospace;
  font-size: 6.5pt;
  line-height: 1.35;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  break-inside: avoid;
}}
pre code {{ font-size: inherit; }}
img {{
  display: block;
  width: 100%;
  max-width: 100%;
  max-height: 100mm;
  margin: 0 auto;
  break-inside: avoid;
}}
p:has(> img.wide-figure) {{
  column-span: all;
  margin: 1.8mm 0 .7mm;
  text-align: center;
  break-inside: avoid;
}}
p:has(> img.wide-figure) img {{
  width: auto;
  max-height: 82mm;
}}
p:has(> img.wide-figure) + p {{
  column-span: all;
  margin: 0 0 3.2mm;
  color: #333;
  font-family: "PaperSans", "Microsoft YaHei", sans-serif;
  font-size: 7.1pt;
  line-height: 1.4;
  text-align: center;
}}
mjx-container[display="true"] {{
  margin: 2mm 0 !important;
  font-size: 82% !important;
}}
table {{
  width: 100%;
  margin: 1.8mm 0 2.7mm;
  border-collapse: collapse;
  table-layout: fixed;
  font-family: "PaperSans", "Microsoft YaHei", sans-serif;
  font-size: 6.55pt;
  line-height: 1.25;
  break-inside: avoid;
}}
thead {{ display: table-header-group; }}
tr {{ break-inside: avoid; }}
th {{
  padding: 1.1mm .8mm;
  border-top: .35mm solid #222;
  border-bottom: .25mm solid #222;
  font-weight: 700;
  text-align: left;
}}
td {{
  padding: .9mm .8mm;
  border-bottom: .12mm solid #ccc;
  vertical-align: top;
  overflow-wrap: anywhere;
}}
tbody tr:last-child td {{ border-bottom: .35mm solid #222; }}
.references p {{
  margin-bottom: 1.35mm;
  font-size: 7.1pt;
  line-height: 1.38;
  text-align: left;
}}
.references p:first-child {{ text-indent: 0; }}
</style>
{math_script}
</head>
<body>
<header class="masthead">
  <p class="kicker">Preprint · Scientific Machine Learning · Operator Learning</p>
  <h1>{safe_title}</h1>
  <p class="english-title">{english_title}</p>
  <p class="authors">{authors}</p>
  <p class="affiliations">{affiliations}</p>
  <p class="corresponding">{html.escape(corresponding)} · {html.escape(date)}</p>
</header>
<section class="abstract">{abstract}</section>
<main class="columns">{body}</main>
</body>
</html>
"""


def main() -> None:
    renderer = load_renderer()
    source_text = renderer.strip_emoji(SOURCE.read_text(encoding="utf-8"))
    title, metadata, body = renderer.extract_document(source_text)

    marker = "\n## 1 引言"
    if marker not in body:
        raise ValueError("Could not locate the main body marker '## 1 引言'.")
    abstract_source, main_source = body.split(marker, 1)
    main_source = "## 1 引言" + main_source
    abstract, abstract_math = render_segment(abstract_source, renderer)
    main_html, main_math = render_segment(main_source, renderer)
    rendered = build_html(
        title,
        metadata,
        abstract,
        main_html,
        abstract_math or main_math,
    )

    BUILD.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    html_path = BUILD / "attention_style_paper_zh.html"
    html_path.write_text(rendered, encoding="utf-8")

    header = """
<div style="width:100%;font-size:6.5px;color:#666;font-family:Arial,sans-serif;
padding:0 14mm;display:flex;justify-content:space-between;">
<span>Query-Resolved Multimodal Neural Operators</span>
<span>Preprint · 2026</span>
</div>
"""
    footer = """
<div style="width:100%;font-size:7px;color:#555;font-family:Arial,sans-serif;
text-align:center;">
<span class="pageNumber"></span> / <span class="totalPages"></span>
</div>
"""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--allow-file-access-from-files", "--font-render-hinting=none"],
        )
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle", timeout=120_000)
        page.wait_for_function("document.fonts.status === 'loaded'", timeout=60_000)
        if abstract_math or main_math:
            page.wait_for_function(
                "window.__mathjaxLoaded === true || window.__mathjaxFailed === true",
                timeout=60_000,
            )
            if page.evaluate("window.__mathjaxFailed"):
                raise RuntimeError("MathJax failed to load.")
            page.evaluate(
                """async () => {
                    await MathJax.startup.promise;
                    await MathJax.typesetPromise();
                    await document.fonts.ready;
                }"""
            )
        page.emulate_media(media="print")
        page.pdf(
            path=str(OUTPUT),
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template=header,
            footer_template=footer,
            margin={
                "top": "16mm",
                "right": "15mm",
                "bottom": "14mm",
                "left": "15mm",
            },
            tagged=True,
            outline=True,
        )
        browser.close()
    print(OUTPUT)


if __name__ == "__main__":
    main()

