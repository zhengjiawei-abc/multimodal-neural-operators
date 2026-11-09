"""Build the single-column Chinese methodology PDF."""

from __future__ import annotations

import html
import importlib.util
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright


PUBLICATION = Path(__file__).resolve().parent
ROOT = PUBLICATION.parent
SOURCE = PUBLICATION / "MNO_Burgers_Methodology_Chinese_V2.0.md"
OUTPUT = PUBLICATION / "pdf" / "MNO_Burgers_Methodology_Chinese_V2.0.pdf"
BUILD = PUBLICATION / ".methodology_build"


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
    rendered = renderer.restore_math(markdown_engine().convert(protected), replacements)
    return rendered, bool(replacements)


def build_html(title: str, metadata: dict[str, str], body: str, has_math: bool) -> str:
    safe_title = html.escape(title)
    english_title = html.escape(
        metadata.get("English title", "A Data-Augmented Physics-Informed Multimodal Neural Operator")
    )
    subtitle = html.escape(metadata.get("Subtitle", "研究方法和实验结果"))
    version = html.escape(metadata.get("Version", "V2.0"))

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
<base href="{ROOT.resolve().as_uri()}/">
<style>
@font-face {{
  font-family: "PaperSerif";
  src: url("file:///C:/Windows/Fonts/NotoSerifSC-VF.ttf") format("truetype");
  font-weight: 100 900;
}}
@font-face {{
  font-family: "PaperSans";
  src: url("file:///C:/Windows/Fonts/NotoSansSC-VF.ttf") format("truetype");
  font-weight: 100 900;
}}
@page {{ size: A4; }}
* {{ box-sizing: border-box; }}
html {{ color-scheme: light; }}
body {{
  margin: 0;
  color: #191919;
  background: white;
  font-family: "PaperSerif", "SimSun", serif;
  font-size: 9.5pt;
  line-height: 1.67;
}}
.masthead {{
  text-align: center;
  margin: 0 0 5mm;
}}
.title {{
  margin: 0 auto 2.4mm;
  max-width: 160mm;
  font-size: 20pt;
  line-height: 1.28;
  font-weight: 700;
}}
.english-title {{
  margin: 0 auto 1.8mm;
  max-width: 160mm;
  font-family: "PaperSerif", "Times New Roman", serif;
  font-size: 10.4pt;
  line-height: 1.3;
}}
.subtitle {{
  margin: 0 auto 1.1mm;
  color: #333;
  font-family: "PaperSans", "Microsoft YaHei", sans-serif;
  font-size: 9.4pt;
}}
.version {{
  margin: 0;
  color: #666;
  font-family: "PaperSans", "Microsoft YaHei", sans-serif;
  font-size: 8pt;
}}
.rule {{
  height: 0;
  margin: 4.2mm 0 4.5mm;
  border-top: .35mm solid #222;
}}
h2 {{
  margin: 5mm 0 2mm;
  font-family: "PaperSans", "Microsoft YaHei", sans-serif;
  font-size: 13pt;
  line-height: 1.28;
  font-weight: 700;
  break-after: avoid;
}}
h3 {{
  margin: 3.5mm 0 1.3mm;
  font-family: "PaperSans", "Microsoft YaHei", sans-serif;
  font-size: 10.4pt;
  line-height: 1.3;
  font-weight: 700;
  break-after: avoid;
}}
p {{
  margin: 0 0 2.3mm;
  text-align: justify;
  text-indent: 2em;
  orphans: 2;
  widows: 2;
}}
.abstract p,
.masthead p,
li p,
td p,
.caption {{
  text-indent: 0;
}}
.abstract {{
  padding: 0 3mm;
}}
.abstract h2 {{
  display: inline;
  margin: 0 1em 0 0;
  font-size: 9.6pt;
}}
.abstract p {{
  display: inline;
}}
.keywords {{
  margin: 1.8mm 0 3mm;
  padding: 0 3mm;
  font-size: 9.2pt;
}}
ul, ol {{
  margin: 1.4mm 0 2.4mm;
  padding-left: 8mm;
}}
li {{ margin: 1mm 0; text-align: justify; }}
strong {{ font-weight: 700; }}
code {{
  font-family: "Cascadia Code", Consolas, monospace;
  font-size: 8pt;
  overflow-wrap: anywhere;
}}
pre {{
  margin: 2mm 0 3mm;
  padding: 2.8mm;
  border: .2mm solid #ccc;
  border-left: .8mm solid #555;
  background: #f7f7f7;
  font-family: "Cascadia Code", Consolas, monospace;
  font-size: 7.3pt;
  line-height: 1.4;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  break-inside: avoid;
}}
img {{
  display: block;
  width: 100%;
  max-width: 100%;
  max-height: 128mm;
  object-fit: contain;
  margin: 2.2mm auto 1.6mm;
  break-inside: avoid;
}}
mjx-container[display="true"] {{
  margin: 2.5mm 0 !important;
  font-size: 88% !important;
}}
table {{
  width: 100%;
  margin: 2.4mm 0 3mm;
  border-collapse: collapse;
  font-family: "PaperSans", "Microsoft YaHei", sans-serif;
  font-size: 7.8pt;
  line-height: 1.35;
  break-inside: avoid;
}}
thead {{ display: table-header-group; }}
tr {{ break-inside: avoid; }}
th {{
  padding: 1.3mm 1.2mm;
  border-top: .35mm solid #222;
  border-bottom: .25mm solid #222;
  text-align: left;
  font-weight: 700;
}}
td {{
  padding: 1.1mm 1.2mm;
  border-bottom: .12mm solid #ccc;
  vertical-align: top;
}}
tbody tr:last-child td {{ border-bottom: .35mm solid #222; }}
.caption {{
  margin: 0 0 3.3mm;
  color: #333;
  font-family: "PaperSans", "Microsoft YaHei", sans-serif;
  font-size: 8.2pt;
  line-height: 1.45;
  text-align: center;
}}
</style>
{math_script}
</head>
<body>
<header class="masthead">
  <h1 class="title">{safe_title}</h1>
  <p class="english-title">{english_title}</p>
  <p class="subtitle">{subtitle}</p>
  <p class="version">{version}</p>
</header>
<div class="rule"></div>
{body}
</body>
</html>
"""


def main() -> None:
    renderer = load_renderer()
    source_text = renderer.strip_emoji(SOURCE.read_text(encoding="utf-8"))
    title, metadata, body = renderer.extract_document(source_text)
    rendered, has_math = render_segment(body, renderer)
    rendered = rendered.replace(
        '<h2>摘要</h2>',
        '<section class="abstract"><h2>摘要</h2>',
        1,
    )
    rendered = rendered.replace(
        '<p><strong>关键词</strong>',
        '</section><p class="keywords"><strong>关键词</strong>',
        1,
    )
    rendered = rendered.replace(
        '<p><strong>关键词：</strong>',
        '</section><p class="keywords"><strong>关键词：</strong>',
        1,
    )
    rendered = rendered.replace(
        '<p><strong>关键词</strong></p>',
        '</section><p class="keywords"><strong>关键词</strong></p>',
        1,
    )
    rendered = rendered.replace(
        '<h2>1 研究背景</h2>',
        '<h2>1 研究背景</h2>',
        1,
    )
    html_text = build_html(title, metadata, rendered, has_math)

    BUILD.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    html_path = BUILD / "methodology_v2.html"
    html_path.write_text(html_text, encoding="utf-8")

    header = """
<div style="width:100%;font-size:6.4px;color:#666;font-family:Arial,sans-serif;
padding:0 16mm;display:flex;justify-content:space-between;">
<span>Burgers 方程多模态神经算子：研究方法和实验结果</span>
<span>V2.0</span>
</div>
"""
    footer = """
<div style="width:100%;font-size:7px;color:#555;font-family:Arial,sans-serif;
text-align:center;">
<span class="pageNumber"></span>
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
        if has_math:
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
                "top": "17mm",
                "right": "18mm",
                "bottom": "15mm",
                "left": "18mm",
            },
            tagged=True,
            outline=True,
        )
        browser.close()
    print(OUTPUT)


if __name__ == "__main__":
    main()

