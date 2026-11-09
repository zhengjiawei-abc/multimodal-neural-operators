"""Build the multimodal-neural-operator document set as print-ready PDFs."""

from __future__ import annotations

import html
import io
import json
import re
from pathlib import Path

import markdown
import pymupdf
from markdown.extensions.toc import slugify_unicode
from playwright.sync_api import sync_playwright
from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import Color, HexColor
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / ".pdf_build"
MATH_INLINE = "ZQXMATHINLINE"
MATH_BLOCK = "ZQXMATHBLOCK"


def extract_document(source: str) -> tuple[str, dict[str, str], str]:
    lines = source.splitlines()
    title_index = next(
        (index for index, line in enumerate(lines) if re.match(r"^#\s+\S", line)),
        None,
    )
    if title_index is None:
        raise ValueError("Markdown document has no level-one title")
    title = re.sub(r"^#\s+", "", lines[title_index]).strip()
    metadata: dict[str, str] = {}
    divider = None
    for index in range(title_index + 1, len(lines)):
        if lines[index].startswith("## "):
            break
        if lines[index].strip() == "---":
            divider = index
            break
    body_start = title_index + 1
    if divider is not None:
        pattern = re.compile(r"^\*\*(.+?)\*\*\s*:\s*(.*?)\s*$")
        for line in lines[title_index + 1 : divider]:
            match = pattern.match(line)
            if match:
                metadata[match.group(1).strip()] = match.group(2).strip()
        body_start = divider + 1
    return title, metadata, "\n".join(lines[body_start:])


def strip_emoji(text: str) -> str:
    pattern = re.compile(
        "["
        "\U0001F300-\U0001FAFF"
        "\u2600-\u27BF"
        "\uFE0F"
        "\u200D"
        "]+"
    )
    return pattern.sub("", text)


def protect_math(body: str) -> tuple[str, list[tuple[str, str]]]:
    replacements: list[tuple[str, str]] = []

    def protect_segment(segment: str) -> str:
        def block(match: re.Match[str]) -> str:
            token = f"{MATH_BLOCK}{len(replacements):04d}"
            replacements.append((token, match.group(1)))
            return token

        def inline(match: re.Match[str]) -> str:
            token = f"{MATH_INLINE}{len(replacements):04d}"
            replacements.append((token, match.group(1)))
            return token

        segment = re.sub(
            r"(?s)(?<!\\)\$\$(.+?)(?<!\\)\$\$",
            block,
            segment,
        )
        return re.sub(
            r"(?<!\\)\$(?!\$)([^$\n]+?)(?<!\\)\$",
            inline,
            segment,
        )

    fence_pattern = re.compile(r"(?ms)^```[^\n]*\n.*?^```[ \t]*$")
    cursor = 0
    parts: list[str] = []
    for match in fence_pattern.finditer(body):
        parts.append(protect_segment(body[cursor : match.start()]))
        parts.append(match.group(0))
        cursor = match.end()
    parts.append(protect_segment(body[cursor:]))
    return "".join(parts), replacements


def restore_math(rendered: str, replacements: list[tuple[str, str]]) -> str:
    for token, expression in replacements:
        if token.startswith(MATH_BLOCK):
            rendered = rendered.replace(token, "\\[" + expression + "\\]")
        else:
            rendered = rendered.replace(token, "\\(" + expression + "\\)")
    return rendered


def metadata_value(
    metadata: dict[str, str],
    *keys: str,
    default: str = "",
) -> str:
    for key in keys:
        value = metadata.get(key)
        if value:
            return value
    return default


def build_html(
    title: str,
    metadata: dict[str, str],
    toc: str,
    body: str,
    has_math: bool,
    language: str,
) -> str:
    labels = {
        "en": {
            "series": "RESEARCH DOCUMENT",
            "date": "DATE",
            "author": "AUTHOR",
            "project": "PROJECT",
            "project_name": "Multimodal Neural Operator Study",
            "contents": "CONTENTS",
            "toc": "Contents",
            "subtitle": "Academic and technical manuscript",
            "footer": "MULTIMODAL NEURAL OPERATOR",
        },
        "zh": {
            "series": "研究文档",
            "date": "日期",
            "author": "作者",
            "project": "项目",
            "project_name": "多模态神经算子研究",
            "contents": "目录",
            "toc": "目录",
            "subtitle": "学术与技术文档",
            "footer": "多模态神经算子",
        },
    }[language]
    safe_title = html.escape(title)
    document_type = metadata_value(
        metadata,
        "Document type",
        "文档类型",
        default=labels["series"],
    )
    running_title = metadata_value(
        metadata,
        "Running Title",
        "副标题",
        default=labels["subtitle"],
    )
    date = metadata_value(
        metadata,
        "Date",
        "日期",
        default=("9 November 2026" if language == "en" else "2026 年 11 月 9 日"),
    )
    author = metadata_value(
        metadata,
        "Author",
        "作者",
        default=("Project team" if language == "en" else "项目组"),
    )
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
<html lang="{'en' if language == 'en' else 'zh-CN'}">
<head>
<meta charset="utf-8">
<meta name="author" content="{html.escape(author)}">
<meta name="description" content="{safe_title}">
<title>{safe_title}</title>
<base href="{ROOT.resolve().as_uri()}/">
<style>
@font-face {{
  font-family: "DocumentSans";
  src: url("file:///C:/Windows/Fonts/NotoSansSC-VF.ttf") format("truetype");
  font-weight: 100 900;
}}
@font-face {{
  font-family: "DocumentSerif";
  src: url("file:///C:/Windows/Fonts/NotoSerifSC-VF.ttf") format("truetype");
  font-weight: 100 900;
}}
@page {{ size: A4; margin: 22mm 19mm 20mm 19mm; }}
@page :first {{ margin: 0; }}
* {{ box-sizing: border-box; }}
html {{ color-scheme: light; }}
body {{
  margin: 0;
  color: #202124;
  background: #fff;
  font-family: "DocumentSerif", "SimSun", serif;
  font-size: 10.4pt;
  line-height: 1.78;
}}
p {{ margin: 0 0 7pt; text-align: justify; orphans: 3; widows: 3; }}
strong {{ color: #111; font-weight: 700; }}
a {{ color: #355f73; text-decoration: none; }}
ul, ol {{ margin: 6pt 0 10pt; padding-left: 20pt; }}
li {{ margin: 2.8pt 0; }}
hr {{ display: none; }}
.cover {{
  width: 210mm;
  height: 297mm;
  padding: 26mm 24mm 22mm;
  background: #fff;
  color: #202124;
  break-after: page;
  display: flex;
  flex-direction: column;
}}
.cover-header {{
  display: flex;
  justify-content: space-between;
  padding-top: 3.5mm;
  border-top: 1mm solid #202124;
  color: #61666c;
  font-family: "DocumentSans", "Microsoft YaHei", sans-serif;
  font-size: 7.7pt;
  letter-spacing: .12em;
}}
.cover-main {{ margin-top: 52mm; max-width: 160mm; }}
.cover-overline {{
  margin: 0 0 8mm;
  color: #7b3341;
  font-family: "DocumentSans", "Microsoft YaHei", sans-serif;
  font-size: 8pt;
  font-weight: 700;
  letter-spacing: .14em;
  text-align: left;
}}
.cover h1 {{
  margin: 0;
  color: #17191c;
  font-family: "DocumentSerif", "SimSun", serif;
  font-size: 30pt;
  font-weight: 650;
  line-height: 1.26;
  text-align: left;
}}
.cover-subtitle {{
  margin: 9mm 0 0;
  padding-left: 5mm;
  border-left: 1.1mm solid #7b3341;
  color: #4c5157;
  font-family: "DocumentSans", "Microsoft YaHei", sans-serif;
  font-size: 14pt;
  font-weight: 400;
  line-height: 1.55;
  text-align: left;
}}
.cover-rule {{
  height: 0;
  margin: 13mm 0 5mm;
  border-top: .35mm solid #c8ccd0;
}}
.cover-summary {{
  margin: 0;
  color: #666b71;
  font-family: "DocumentSans", "Microsoft YaHei", sans-serif;
  font-size: 8.4pt;
  letter-spacing: .06em;
  line-height: 1.7;
  text-align: left;
}}
.cover-footer {{
  margin-top: auto;
  display: grid;
  grid-template-columns: 1.35fr 1fr;
  gap: 18mm;
  align-items: end;
  padding-top: 5mm;
  border-top: .35mm solid #c8ccd0;
}}
.cover-project span, .cover-meta span {{
  display: block;
  margin-bottom: 2mm;
  color: #7a7f84;
  font-family: "DocumentSans", "Microsoft YaHei", sans-serif;
  font-size: 7.3pt;
  letter-spacing: .08em;
}}
.cover-project strong, .cover-meta strong {{
  display: block;
  color: #22262a;
  font-family: "DocumentSans", "Microsoft YaHei", sans-serif;
  font-size: 9.5pt;
  font-weight: 500;
}}
.toc-shell {{
  min-height: 240mm;
  break-after: page;
  counter-reset: tocitem;
}}
.toc-head {{
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 12mm;
  padding-bottom: 3.5mm;
  border-bottom: .35mm solid #202124;
}}
.section-kicker {{
  margin: 0;
  color: #7b3341;
  font-family: "DocumentSans", "Microsoft YaHei", sans-serif;
  font-size: 7.8pt;
  font-weight: 700;
  letter-spacing: .13em;
}}
.toc-head h2 {{
  margin: 0;
  border: 0;
  padding: 0;
  font-size: 23pt;
}}
.toc-shell ul {{ list-style: none; margin: 0; padding: 0; }}
.toc-shell > ul > li {{
  counter-increment: tocitem;
  padding: 3.3mm 0;
  border-bottom: .22mm solid #d9dce0;
}}
.toc-shell > ul > li > a::before {{
  content: counter(tocitem, decimal-leading-zero);
  display: inline-block;
  width: 11mm;
  color: #7b3341;
  font-family: "DocumentSans", "Microsoft YaHei", sans-serif;
  font-size: 8pt;
  letter-spacing: .08em;
}}
.toc-shell ul ul {{ display: none; }}
.toc-shell a {{ color: #202124; font-weight: 600; }}
.content > h2:first-child {{ margin-top: 0; }}
h1, h2, h3, h4 {{ break-after: avoid; }}
h1 {{
  margin: 0 0 8mm;
  color: #17191c;
  font-family: "DocumentSerif", "SimSun", serif;
  font-size: 23pt;
  line-height: 1.3;
}}
h2 {{
  margin: 13mm 0 6mm;
  color: #17191c;
  font-family: "DocumentSerif", "SimSun", serif;
  font-size: 18pt;
  font-weight: 650;
  line-height: 1.35;
}}
h3 {{
  margin: 8mm 0 3.5mm;
  padding-left: 3mm;
  border-left: .8mm solid #7b3341;
  color: #23272b;
  font-family: "DocumentSans", "Microsoft YaHei", sans-serif;
  font-size: 12pt;
  font-weight: 650;
  line-height: 1.45;
}}
h4 {{
  margin: 6mm 0 2.5mm;
  color: #3f4449;
  font-family: "DocumentSans", "Microsoft YaHei", sans-serif;
  font-size: 10.6pt;
}}
code {{
  padding: .6pt 2.2pt;
  border: .18mm solid #d7dadd;
  border-radius: 1.5pt;
  background: #f7f7f8;
  color: #6f2a37;
  font-family: "Cascadia Code", Consolas, monospace;
  font-size: 8.3pt;
  overflow-wrap: anywhere;
}}
pre {{
  margin: 4mm 0 6mm;
  padding: 4mm 4.5mm;
  border: .22mm solid #d7dadd;
  border-left: 1mm solid #8a8f94;
  background: #f8f8f9;
  color: #25292d;
  font-family: "Cascadia Code", Consolas, monospace;
  font-size: 7.9pt;
  line-height: 1.52;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  break-inside: avoid;
}}
pre code {{ padding: 0; border: 0; background: transparent; color: inherit; }}
.codehilite {{ margin: 4mm 0 6mm; break-inside: avoid; }}
.codehilite pre {{ margin: 0; }}
table {{
  width: 100%;
  margin: 4mm 0 6mm;
  border-collapse: collapse;
  table-layout: fixed;
  font-family: "DocumentSans", "Microsoft YaHei", sans-serif;
  font-size: 8.3pt;
  line-height: 1.5;
}}
thead {{ display: table-header-group; }}
tr {{ break-inside: avoid; }}
th {{
  padding: 2.2mm 2mm;
  color: #202124;
  background: #e9ecee;
  border: .2mm solid #c7ccd1;
  font-weight: 700;
  text-align: left;
  vertical-align: middle;
}}
td {{
  padding: 2mm;
  border: .2mm solid #d8dcdf;
  vertical-align: top;
  overflow-wrap: anywhere;
}}
tbody tr:nth-child(even) td {{ background: #fafafa; }}
blockquote {{
  margin: 4.5mm 0;
  padding: 3.5mm 4.5mm;
  border-left: .8mm solid #8d9195;
  background: #f8f8f8;
  color: #484d52;
}}
img {{ display: block; max-width: 100%; margin: 5mm auto; }}
mjx-container[display="true"] {{ margin: 4mm 0 !important; }}
</style>
{math_script}
</head>
<body>
<section class="cover">
  <header class="cover-header">
    <div>{labels['footer']}</div>
    <div>{html.escape(document_type)}</div>
  </header>
  <main class="cover-main">
    <p class="cover-overline">{labels['series']}</p>
    <h1>{safe_title}</h1>
    <p class="cover-subtitle">{html.escape(running_title)}</p>
    <div class="cover-rule"></div>
    <p class="cover-summary">BURGERS EQUATION · OPERATOR LEARNING · REPRODUCIBILITY</p>
  </main>
  <footer class="cover-footer">
    <div class="cover-project">
      <span>{labels['project']}</span>
      <strong>{labels['project_name']}</strong>
    </div>
    <div class="cover-meta">
      <div><span>{labels['date']}</span><strong>{html.escape(date)}</strong></div>
      <div><span>{labels['author']}</span><strong>{html.escape(author)}</strong></div>
    </div>
  </footer>
</section>
<nav class="toc-shell">
  <div class="toc-head">
    <p class="section-kicker">{labels['contents']}</p>
    <h2>{labels['toc']}</h2>
  </div>
  {toc}
</nav>
<main class="content">{body}</main>
</body>
</html>
"""


def render_html(html_path: Path, raw_pdf: Path, has_math: bool) -> None:
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
                raise RuntimeError("MathJax failed to load")
            page.evaluate(
                """async () => {
                    await MathJax.startup.promise;
                    await MathJax.typesetPromise();
                    await document.fonts.ready;
                }"""
            )
        page.wait_for_timeout(300)
        page.emulate_media(media="print")
        page.pdf(
            path=str(raw_pdf),
            format="A4",
            print_background=True,
            prefer_css_page_size=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            tagged=True,
            outline=True,
        )
        browser.close()


def register_cjk_font() -> str:
    candidates = [
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
    ]
    for index, path in enumerate(candidates):
        if not path.exists():
            continue
        name = f"RootDocumentOverlay{index}"
        try:
            pdfmetrics.registerFont(TTFont(name, str(path)))
            return name
        except Exception:
            continue
    raise RuntimeError("No usable CJK font found")


def make_overlay(page_count: int, title: str, language: str) -> bytes:
    font = register_cjk_font()
    stream = io.BytesIO()
    overlay = canvas.Canvas(stream, pagesize=(210 * mm, 297 * mm))
    for page_number in range(2, page_count + 1):
        overlay.setFillColor(HexColor("#6b7075"))
        overlay.setFont(font, 7.1)
        overlay.drawString(19 * mm, 287 * mm, title[:66])
        overlay.drawRightString(
            191 * mm,
            287 * mm,
            "RESEARCH DOCUMENT" if language == "en" else "研究文档",
        )
        overlay.setStrokeColor(Color(0.74, 0.76, 0.78, alpha=0.8))
        overlay.setLineWidth(0.3)
        overlay.line(19 * mm, 283.5 * mm, 191 * mm, 283.5 * mm)
        overlay.line(19 * mm, 13.5 * mm, 191 * mm, 13.5 * mm)
        overlay.setFillColor(HexColor("#71767b"))
        overlay.setFont(font, 7.1)
        overlay.drawString(
            19 * mm,
            8.8 * mm,
            "MULTIMODAL NEURAL OPERATOR"
            if language == "en"
            else "多模态神经算子研究",
        )
        overlay.drawRightString(
            191 * mm,
            8.8 * mm,
            f"Page {page_number} / {page_count}"
            if language == "en"
            else f"第 {page_number} 页 / 共 {page_count} 页",
        )
        overlay.showPage()
    overlay.save()
    return stream.getvalue()


def finalize_pdf(
    raw_pdf: Path,
    output: Path,
    title: str,
    language: str,
) -> dict[str, object]:
    reader = PdfReader(str(raw_pdf))
    page_count = len(reader.pages)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    overlay = PdfReader(io.BytesIO(make_overlay(page_count, title, language)))
    for index, page in enumerate(writer.pages):
        if index > 0:
            page.merge_page(overlay.pages[index - 1])
    writer.add_metadata(
        {
            "/Title": title,
            "/Author": "Multimodal Neural Operator Project",
            "/Subject": "Multimodal neural operator documentation",
            "/Creator": "Markdown + Chromium + ReportLab",
        }
    )
    writer.page_mode = "/UseOutlines"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        writer.write(handle)
    check = pymupdf.open(str(output))
    result = {
        "pages": page_count,
        "bytes": output.stat().st_size,
        "bookmarks": len(check.get_toc(simple=True)),
    }
    check.close()
    return result


def build_markdown(source: Path, output: Path, language: str) -> dict[str, object]:
    text = strip_emoji(source.read_text(encoding="utf-8"))
    title, metadata, body = extract_document(text)
    body, replacements = protect_math(body)
    engine = markdown.Markdown(
        extensions=[
            "tables",
            "fenced_code",
            "codehilite",
            "toc",
            "sane_lists",
            "attr_list",
        ],
        extension_configs={
            "toc": {"toc_depth": "2-2", "slugify": slugify_unicode},
            "codehilite": {"guess_lang": False, "noclasses": False},
        },
        output_format="html5",
    )
    rendered = restore_math(engine.convert(body), replacements)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    html_path = BUILD_DIR / f"{source.stem}_{language}.html"
    raw_pdf = BUILD_DIR / f"{source.stem}_{language}_raw.pdf"
    html_path.write_text(
        build_html(title, metadata, engine.toc, rendered, bool(replacements), language),
        encoding="utf-8",
    )
    render_html(html_path, raw_pdf, bool(replacements))
    result = finalize_pdf(raw_pdf, output, title, language)
    return {
        "title": title,
        "language": language,
        "source": str(source),
        "output": str(output),
        **result,
    }


def main() -> None:
    jobs = [
        (ROOT / "PAPER_READY_DATA.md", ROOT / "01_论文核心数据与对比.pdf", "zh"),
        (ROOT / "PAPER_VENUE_ANALYSIS.md", ROOT / "02_期刊投稿分析_CS方向.pdf", "zh"),
        (ROOT / "CS_VENUE_STRATEGY.md", ROOT / "03_CS投稿策略_JMLR优先.pdf", "zh"),
        (ROOT / "WORKSHOP_PAPER_TEMPLATE.md", ROOT / "04_NeurIPS_Workshop论文模板.pdf", "zh"),
        (ROOT / "CODE_NAVIGATION.md", ROOT / "05_多模态神经算子代码导航.pdf", "zh"),
        (ROOT / "EXPERIMENTS_CHECKPOINTS.md", ROOT / "06_完整实验Checkpoints汇总.pdf", "zh"),
        (ROOT / "FINAL_PROJECT_SUMMARY.md", ROOT / "08_项目最终总结_发表级.pdf", "zh"),
        (
            ROOT / "publication" / "multimodal_neural_operator_paper_en.md",
            ROOT / "publication" / "pdf" / "Multimodal_Neural_Operator_Paper_EN.pdf",
            "en",
        ),
        (
            ROOT / "publication" / "multimodal_neural_operator_paper_zh.md",
            ROOT / "publication" / "pdf" / "Multimodal_Neural_Operator_Paper_ZH.pdf",
            "zh",
        ),
        (
            ROOT
            / "publication"
            / "multimodal_neural_operator_technical_document_en.md",
            ROOT
            / "publication"
            / "pdf"
            / "Multimodal_Neural_Operator_Technical_Document_EN.pdf",
            "en",
        ),
        (
            ROOT
            / "publication"
            / "multimodal_neural_operator_technical_document_zh.md",
            ROOT
            / "publication"
            / "pdf"
            / "Multimodal_Neural_Operator_Technical_Document_ZH.pdf",
            "zh",
        ),
    ]
    results = []
    for source, output, language in jobs:
        if not source.exists():
            raise FileNotFoundError(source)
        results.append(build_markdown(source, output, language))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

