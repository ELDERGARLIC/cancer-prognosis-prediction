"""Task 1: Assemble the six chapter files into thesis_full.md + thesis_full.pdf.

Order: Title page → Table of contents → Abstract → Introduction → Methods →
Results → Discussion → Conclusion.

Title page metadata: title (placeholder), author (Mahdi Sarhangi from
pyproject.toml), supervisor (Doç. Dr. Özgür Gümüş, confirmed in conversation),
institution (Ege University), date (placeholder, 2026).

Outputs:
  - results/thesis/thesis_full.md  (markdown for ongoing edits)
  - results/thesis/thesis_full.pdf (typeset PDF for supervisor read)
  - intermediate: results/thesis/thesis_full.html (pandoc → HTML → weasyprint)

Pipeline: pandoc converts each chapter md → HTML chunk; we concatenate
with title page + TOC + chapter wrappers; weasyprint renders to PDF.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THESIS = ROOT / "results" / "thesis"

# Chapter files in submission order (excluding outlines)
CHAPTERS = [
    ("Abstract", THESIS / "abstract.md"),
    ("Chapter 1. Introduction", THESIS / "introduction.md"),
    ("Chapter 2. Methods", THESIS / "methods.md"),
    ("Chapter 3. Results", THESIS / "results.md"),
    ("Chapter 4. Discussion", THESIS / "discussion.md"),
    ("Chapter 5. Conclusion", THESIS / "conclusion.md"),
]

OUT_MD = THESIS / "thesis_full.md"
OUT_HTML = THESIS / "thesis_full.html"
OUT_PDF = THESIS / "thesis_full.pdf"

# ---- Title page metadata ----
TITLE = "A Leakage-Corrected, Externally-Validated Graph Neural Network for Breast Cancer Prognosis"
SUBTITLE = ("Per-fold leakage correction and paired-bootstrap on identical "
            "external patients as a methodological framework for GNN claims "
            "on TCGA-BRCA")
AUTHOR = "Mahdi Sarhangi"
SUPERVISOR = "Doç. Dr. Özgür Gümüş"
INSTITUTION = "Ege University"
DEPARTMENT = "Department of Computer Engineering"
DEGREE = "Master of Science"
DATE = "2026"  # placeholder — student to update at submission time


def title_page() -> str:
    """Markdown for the title page. Pandoc/weasyprint will style it."""
    return f"""<div class="title-page">

# {TITLE}

## {SUBTITLE}

**{AUTHOR}**

A thesis submitted in partial fulfilment of the requirements for the degree of

**{DEGREE}**

Supervisor: {SUPERVISOR}

{DEPARTMENT}

{INSTITUTION}

{DATE}

</div>

\\newpage

"""


def build_toc(chapters: list) -> str:
    """Generate a manual table of contents from the chapter files.

    Reads each chapter for ## headers (sub-section titles), produces:
        Chapter N. Title
            N.1 Subsection title
            N.2 ...
    """
    lines = ["# Table of Contents\n"]
    for ch_title, ch_path in chapters:
        lines.append(f"\n**{ch_title}**\n")
        text = ch_path.read_text()
        # Find ## headers
        sections = re.findall(r"^##\s+(.+)$", text, flags=re.M)
        for s in sections:
            # Strip leading numbers like "1. " or "5.1 " for cleaner display
            lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;{s}\n")
    lines.append("\n\\newpage\n")
    return "\n".join(lines)


def chapter_with_pagebreak(title: str, md_path: Path) -> str:
    """Read a chapter file, prepend a page break."""
    text = md_path.read_text()
    return f"\n\\newpage\n\n{text}\n"


def assemble_md() -> str:
    """Concatenate title page + TOC + 6 chapters into a single markdown doc."""
    parts = [title_page(), build_toc(CHAPTERS)]
    for ch_title, ch_path in CHAPTERS:
        parts.append(chapter_with_pagebreak(ch_title, ch_path))
    return "".join(parts)


def md_to_html_pandoc(md_path: Path, html_path: Path) -> None:
    """Convert markdown to standalone HTML via pandoc, with embedded CSS."""
    css = r"""
    @page {
        size: A4;
        margin: 2.5cm 2cm 2.5cm 2cm;
        @bottom-center { content: counter(page); font-size: 9pt; }
    }
    body {
        font-family: 'Times New Roman', Georgia, serif;
        font-size: 11pt;
        line-height: 1.55;
        color: #111;
        max-width: none;
    }
    h1 {
        font-size: 22pt;
        margin-top: 1.5em;
        margin-bottom: 0.6em;
        page-break-before: always;
    }
    h1:first-of-type { page-break-before: avoid; }
    h2 {
        font-size: 14pt;
        margin-top: 1.5em;
        margin-bottom: 0.4em;
    }
    h3 {
        font-size: 12pt;
        margin-top: 1.2em;
        margin-bottom: 0.3em;
        font-style: italic;
    }
    p { text-align: justify; margin: 0 0 0.6em 0; }
    .title-page {
        text-align: center;
        page-break-after: always;
        padding-top: 8em;
    }
    .title-page h1 { font-size: 20pt; page-break-before: avoid; }
    .title-page h2 { font-size: 13pt; font-weight: normal; font-style: italic; margin-bottom: 4em; }
    .title-page p { text-align: center; margin: 0.5em 0; }
    table {
        border-collapse: collapse;
        margin: 1em auto;
        font-size: 9.5pt;
        max-width: 100%;
    }
    th, td { border: 1px solid #888; padding: 0.4em 0.6em; text-align: left; }
    th { background: #e8e8e8; font-weight: 600; }
    code {
        font-family: 'Menlo', 'Consolas', monospace;
        font-size: 9.5pt;
        background: #f4f4f4;
        padding: 0.1em 0.3em;
        border-radius: 3px;
    }
    pre code { display: block; padding: 0.6em; }
    img { max-width: 95%; display: block; margin: 1em auto; }
    blockquote {
        border-left: 3px solid #888;
        margin-left: 0;
        padding-left: 1em;
        color: #444;
        font-style: italic;
    }
    """
    cmd = [
        "pandoc",
        str(md_path),
        "-f", "markdown+raw_html+pipe_tables+grid_tables+raw_tex",
        "-t", "html5",
        "-s",
        "--metadata", f"title={TITLE}",
        "-o", str(html_path),
        "--css=/dev/stdin",
    ]
    # Pandoc --css=/dev/stdin trick: easier to embed CSS via -H header file
    css_file = html_path.parent / "_thesis_style.css"
    css_file.write_text(css)
    cmd = [
        "pandoc",
        str(md_path),
        "-f", "markdown+raw_html+pipe_tables+grid_tables+raw_tex",
        "-t", "html5",
        "-s",
        # `pagetitle` only sets <title> tag; it doesn't auto-inject a heading.
        # Avoids the duplicate-title issue from `--metadata title=...`.
        "--metadata", f"pagetitle={TITLE}",
        "-c", str(css_file),
        "-o", str(html_path),
    ]
    print(f"  pandoc: pandoc ... -o {html_path.name}")
    subprocess.run(cmd, check=True)


def html_to_pdf_weasyprint(html_path: Path, pdf_path: Path) -> None:
    """Render HTML to PDF via WeasyPrint."""
    print(f"  weasyprint → {pdf_path}")
    from weasyprint import HTML  # noqa: E402
    # base_url = html's own directory so relative paths like ../figures/ resolve
    HTML(filename=str(html_path), base_url=str(html_path.parent)).write_pdf(str(pdf_path))


def main():
    print("==> Assembling thesis_full.md")
    full_md = assemble_md()
    OUT_MD.write_text(full_md)
    print(f"    wrote {OUT_MD} ({len(full_md):,} chars)")

    # Word count summary
    body_words = len(re.sub(r"<[^>]+>|\\\w+", "", full_md).split())
    print(f"    rough word count (incl. tables): {body_words:,}")

    print("==> Generating HTML via pandoc")
    md_to_html_pandoc(OUT_MD, OUT_HTML)
    print(f"    wrote {OUT_HTML} ({OUT_HTML.stat().st_size:,} bytes)")

    print("==> Generating PDF via WeasyPrint")
    html_to_pdf_weasyprint(OUT_HTML, OUT_PDF)
    print(f"    wrote {OUT_PDF} ({OUT_PDF.stat().st_size:,} bytes)")

    print("\n==> Done.")
    print(f"    thesis_full.md  : {OUT_MD}")
    print(f"    thesis_full.html: {OUT_HTML}")
    print(f"    thesis_full.pdf : {OUT_PDF}")


if __name__ == "__main__":
    main()
