#!/usr/bin/env python3
"""
docs/_build_pdf.py
------------------
Convert ``docs/report.md`` to ``docs/report.pdf`` using WeasyPrint.

Usage
-----
    /opt/homebrew/bin/python3.10 docs/_build_pdf.py

Why a dedicated script?
    WeasyPrint gives us first-class CSS paged-media support (@page,
    @bottom-center for page numbers), which is what the project
    brief requires. Wrapping the conversion in a script keeps the
    style, margins, and font fully reproducible from the repo.
"""

from __future__ import annotations

from pathlib import Path

import markdown
from weasyprint import CSS, HTML

ROOT = Path(__file__).resolve().parent
REPORT_MD = ROOT / "report.md"
REPORT_PDF = ROOT / "report.pdf"

# ---------------------------------------------------------------------------
# 1. Markdown -> HTML body
# ---------------------------------------------------------------------------
md_text = REPORT_MD.read_text(encoding="utf-8")
html_body = markdown.markdown(
    md_text,
    extensions=[
        "extra",        # tables, fenced code, def lists, abbreviations
        "tables",       # explicit (belt and braces)
        "fenced_code",  # explicit
        "sane_lists",   # nicer list rendering
        "smarty",       # smart quotes / dashes for polish
    ],
    output_format="html5",
)

# ---------------------------------------------------------------------------
# 2. Full HTML document wrapper
# ---------------------------------------------------------------------------
# Charset declared so WeasyPrint does not fall back to a different
# encoding when reading the in-memory string.
html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Product Review Intelligence — Final Report</title>
</head>
<body>
{html_body}
</body>
</html>
"""

# ---------------------------------------------------------------------------
# 3. Stylesheet — owns margins, font, and page numbers
# ---------------------------------------------------------------------------
pdf_css = CSS(string="""
/* Page setup: A4, 1.8 cm margins on every side, centred page number.
   (Pulled in from 2.5cm to meet the 8-12 page target.) */
@page {
    size: A4;
    margin: 1.8cm;

    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-family: Georgia, "Times New Roman", serif;
        font-size: 10pt;
        color: #555;
        padding-top: 0.6cm;
    }
}

/* Body typography: 10.5pt serif, tight line-height, justified. */
html, body {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 10.5pt;
    line-height: 1.3;
    color: #1d1d1f;
    margin: 0;
    padding: 0;
}
p {
    margin: 0 0 0.35em 0;
    text-align: justify;
    hyphens: auto;
}

/* Headings — tightened top margins to control page count. */
h1 {
    font-size: 19pt;
    line-height: 1.15;
    margin: 14pt 0 0.2em 0;
    border-bottom: 2px solid #222;
    padding-bottom: 0.1em;
    page-break-after: avoid;
}
h2 {
    font-size: 14pt;
    margin: 11pt 0 0.2em 0;
    border-bottom: 1px solid #bbb;
    padding-bottom: 0.05em;
    page-break-after: avoid;
    page-break-before: auto;
}
h3 {
    font-size: 12pt;
    margin: 8pt 0 0.15em 0;
    page-break-after: avoid;
}
h4 {
    font-size: 11pt;
    margin: 6pt 0 0.15em 0;
    font-style: italic;
    page-break-after: avoid;
}

/* Lists */
ul, ol { margin: 0.2em 0 0.5em 1.4em; padding: 0; }
li { margin-bottom: 0.1em; }

/* Inline code and fenced blocks */
code {
    font-family: "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
    font-size: 9.5pt;
    background: #f4f4f6;
    padding: 0.05em 0.3em;
    border-radius: 3px;
}
pre {
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 9pt;
    line-height: 1.3;
    background: #f4f4f6;
    padding: 0.5em 0.8em;
    border: 1px solid #e2e2e2;
    border-radius: 4px;
    overflow-x: auto;
    page-break-inside: avoid;
    white-space: pre;
    margin: 0.4em 0;
}
pre code {
    background: transparent;
    padding: 0;
    font-size: inherit;
}

/* Tables */
table {
    border-collapse: collapse;
    margin: 0.4em 0 0.6em 0;
    width: 100%;
    font-size: 10pt;
    page-break-inside: avoid;
}
th, td {
    border: 1px solid #bfbfbf;
    padding: 0.25em 0.45em;
    text-align: left;
    vertical-align: top;
}
th {
    background: #ececec;
    font-weight: 600;
}

/* Block quotes and rules */
blockquote {
    border-left: 3px solid #bfbfbf;
    margin: 0.4em 0;
    padding: 0.05em 0 0.05em 0.9em;
    color: #555;
    font-style: italic;
}
hr {
    border: none;
    border-top: 1px solid #c8c8c8;
    margin: 0.6em 0;
}

/* Links stay muted — this is an academic report, not a web page. */
a, a:visited { color: #1d1d1f; text-decoration: underline; }

/* Title block: first h1 on its own page feels too heavy; leave it inline. */
""")

# ---------------------------------------------------------------------------
# 4. Render to PDF
# ---------------------------------------------------------------------------
HTML(string=html_doc, base_url=str(ROOT)).write_pdf(
    target=str(REPORT_PDF),
    stylesheets=[pdf_css],
)

size_kb = REPORT_PDF.stat().st_size / 1024
print(f"Wrote {REPORT_PDF}  ({size_kb:.1f} KB)")
