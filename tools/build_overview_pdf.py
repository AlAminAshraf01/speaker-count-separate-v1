#!/usr/bin/env python3
"""Print ``OVERVIEW.md`` to ``OVERVIEW.pdf``, with every figure inside the file.

A Markdown file does not contain its images, only paths to them
(``report/figures/fig1_architecture.png``). On GitHub, or anywhere the repo folder travels
intact, those paths resolve. Sent on its own -- an attachment, a chat message -- the .md
arrives as text with eight broken image links. A PDF carries the figures inside it, so it
survives being sent anywhere.

No third-party dependency. The page is printed by Microsoft Edge (present on every Windows
install) or Chrome, headless. The Markdown is read by the small converter below, which covers
what OVERVIEW.md uses -- headings, paragraphs, lists, tables, a blockquote, fenced code,
images -- and refuses to print if an image is missing, rather than producing a PDF with a
hole in it.

Usage::

    python tools/build_overview_pdf.py                  # OVERVIEW.md -> OVERVIEW.pdf
    python tools/build_overview_pdf.py --browser PATH   # if Edge/Chrome is somewhere unusual

The PDF does not update itself: re-run this after editing OVERVIEW.md.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import html
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "OVERVIEW.md"
TITLE = "Speaker counting and separation · team guide"
REPO_URL = "https://github.com/AlAminAshraf01/speaker-count-separate-v1"

BROWSERS = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)
BROWSER_NAMES = ("msedge", "microsoft-edge", "google-chrome", "chromium", "chromium-browser")

CSS = """
@page {
  size: A4;
  margin: 16mm 17mm 17mm;
  @bottom-left  { content: "@TITLE@"; font: 7.5pt "Segoe UI", sans-serif; color: #8c8b87; }
  @bottom-right { content: counter(page) " / " counter(pages);
                  font: 7.5pt "Segoe UI", sans-serif; color: #8c8b87; }
}
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; background: #fff; color: #1b1b1a;
       font: 10.4pt/1.55 "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
h1 { font-size: 20pt; line-height: 1.2; font-weight: 700; margin: 0 0 3pt; }
.meta { font-size: 8.8pt; color: #52514e; margin: 0 0 12pt; }
.meta a { color: #2a78d6; text-decoration: none; }
h2 { font-size: 14.5pt; line-height: 1.25; font-weight: 600; margin: 0 0 7pt; break-after: avoid; }
h3 { font-size: 11.5pt; font-weight: 600; margin: 12pt 0 5pt; break-after: avoid; }
p { margin: 0 0 7pt; orphans: 3; widows: 3; }
strong { font-weight: 600; }
a { color: #2a78d6; }
hr { border: 0; border-top: 0.75pt solid #dcdad4; margin: 14pt 0 13pt; }
code { font-family: Consolas, "Cascadia Mono", Menlo, monospace; font-size: 0.88em;
       background: #f2f1ed; border-radius: 3px; padding: 0.08em 0.3em; }
pre { font-family: Consolas, "Cascadia Mono", Menlo, monospace; font-size: 8.4pt;
      line-height: 1.17; background: #f5f4f0; border-radius: 5px; padding: 8pt 10pt;
      margin: 3pt 0 9pt; white-space: pre; break-inside: avoid; }
pre code { font-size: inherit; background: none; padding: 0; border-radius: 0; }
figure { margin: 6pt 0 10pt; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; max-height: 76mm; }
.keep { break-inside: avoid; }
table { border-collapse: collapse; margin: 3pt 0 10pt; font-size: 9.6pt; line-height: 1.4;
        break-inside: avoid; border-top: 1pt solid #1b1b1a; border-bottom: 1pt solid #1b1b1a; }
th, td { text-align: left; vertical-align: top; padding: 3.5pt 14pt 3.5pt 0; }
th:last-child, td:last-child { padding-right: 0; }
th { font-weight: 600; border-bottom: 0.6pt solid #1b1b1a; }
tbody tr + tr td { border-top: 0.5pt solid #e2e0da; }
ul, ol { margin: 0 0 8pt; padding-left: 16pt; }
li { margin: 0 0 3pt; }
blockquote { margin: 4pt 0 8pt; padding: 8pt 11pt; background: #f5f4f0;
             border-left: 2.5pt solid #2a78d6; break-inside: avoid; }
blockquote p:last-child { margin-bottom: 0; }
"""

BLOCK_START = re.compile(r"(#{1,6}\s|```|\||>)")
LIST_ITEM = re.compile(r"([-*]|\d+\.)\s+")
IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")


def _stash(spans: list[str], fragment: str) -> str:
    """Park finished HTML behind a placeholder so later passes cannot touch it."""
    spans.append(fragment)
    return f"\x00{len(spans) - 1}\x00"


def _cells(row: str) -> list[str]:
    row = row.strip()
    row = row[1:] if row.startswith("|") else row
    row = row[:-1] if row.endswith("|") and not row.endswith("\\|") else row
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", row)]


class Converter:
    """Just enough Markdown for OVERVIEW.md. Images are inlined as data URIs."""

    def __init__(self, base: Path):
        self.base = base
        self.embedded: list[str] = []
        self.missing: list[str] = []

    def image(self, alt: str, src: str) -> str:
        path = self.base / src
        if not path.is_file():
            self.missing.append(src)
            return ""
        self.embedded.append(src)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<img src="data:{mime};base64,{data}" alt="{html.escape(alt)}">'

    def inline(self, text: str) -> str:
        spans: list[str] = []
        text = re.sub(r"`([^`]+)`", lambda m: _stash(
            spans, "<code>" + html.escape(m.group(1), quote=False) + "</code>"), text)
        text = IMAGE.sub(lambda m: _stash(spans, self.image(m.group(1), m.group(2))), text)
        text = html.escape(text, quote=False)
        text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text, flags=re.S)
        text = re.sub(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])", r"<em>\1</em>", text,
                      flags=re.S)
        return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], text)

    def table(self, rows: list[str]) -> str:
        if len(rows) < 2 or not all(re.fullmatch(r":?-+:?", c) for c in _cells(rows[1])):
            raise ValueError(f"table without a |---| rule under its header: {rows[0]!r}")
        head, body = _cells(rows[0]), [_cells(r) for r in rows[2:]]
        # "| | |" is how a table without a header is written; print no empty header row
        thead = ("<thead><tr>" + "".join(f"<th>{self.inline(c)}</th>" for c in head)
                 + "</tr></thead>") if any(head) else ""
        tbody = "".join("<tr>" + "".join(f"<td>{self.inline(c)}</td>" for c in r) + "</tr>"
                        for r in body)
        return f"<table>{thead}<tbody>{tbody}</tbody></table>"

    def blocks(self, lines: list[str]) -> str:
        out, i = [], 0
        while i < len(lines):
            s = lines[i].strip()
            if not s:
                i += 1
            elif s.startswith("```"):
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith("```"):
                    j += 1
                code = html.escape("\n".join(lines[i + 1:j]), quote=False)
                out.append(f"<pre><code>{code}</code></pre>")
                i = j + 1
            elif m := re.match(r"(#{1,6})\s+(.*)", s):
                level = len(m.group(1))
                out.append(f"<h{level}>{self.inline(m.group(2))}</h{level}>")
                i += 1
            elif re.fullmatch(r"-{3,}|\*{3,}|_{3,}", s):
                out.append("<hr>")
                i += 1
            elif s.startswith("|"):
                j = i
                while j < len(lines) and lines[j].strip().startswith("|"):
                    j += 1
                out.append(self.table(lines[i:j]))
                i = j
            elif s.startswith(">"):
                j = i
                while j < len(lines) and lines[j].strip().startswith(">"):
                    j += 1
                inner = [re.sub(r"^\s*>\s?", "", line) for line in lines[i:j]]
                out.append(f"<blockquote>{self.blocks(inner)}</blockquote>")
                i = j
            elif m := LIST_ITEM.match(s):
                tag = "ol" if m.group(1)[0].isdigit() else "ul"
                items, j = [], i
                while j < len(lines):
                    line = lines[j]
                    if LIST_ITEM.match(line):
                        items.append(LIST_ITEM.sub("", line, count=1))
                    elif line.strip() and line[0] in " \t" and items:
                        items[-1] += " " + line.strip()      # continuation of the item
                    else:
                        break
                    j += 1
                out.append(f"<{tag}>" + "".join(f"<li>{self.inline(t)}</li>" for t in items)
                           + f"</{tag}>")
                i = j
            elif IMAGE.fullmatch(s):
                fig = f"<figure>{self.inline(s)}</figure>"
                if out and out[-1].startswith("<p>"):
                    # the sentence introducing a figure moves to the next page with it
                    fig = f'<div class="keep">{out.pop()}{fig}</div>'
                out.append(fig)
                i += 1
            else:
                j = i + 1
                while j < len(lines) and lines[j].strip() and not BLOCK_START.match(
                        lines[j].strip()):
                    j += 1
                out.append(f"<p>{self.inline(' '.join(x.strip() for x in lines[i:j]))}</p>")
                i = j
        return "\n".join(out)


def find_browser(explicit: str | None) -> str | None:
    if explicit:
        return explicit if Path(explicit).is_file() else None
    for path in BROWSERS:
        if Path(path).is_file():
            return path
    for name in BROWSER_NAMES:
        if found := shutil.which(name):
            return found
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(SRC.with_suffix(".pdf")))
    ap.add_argument("--browser", default=None, help="path to msedge or chrome")
    args = ap.parse_args()
    out = Path(args.out).resolve()

    browser = find_browser(args.browser)
    if browser is None:
        print("No Edge or Chrome found. Pass its path with --browser.", file=sys.stderr)
        return 2

    text = re.sub(r"<!--.*?-->", "", SRC.read_text(encoding="utf-8"), flags=re.S)
    conv = Converter(SRC.parent)
    body = conv.blocks(text.splitlines())
    if conv.missing:
        print("FAILED: these images do not exist, so the PDF would have holes in it:",
              file=sys.stderr)
        for src in conv.missing:
            print(f"  {src}", file=sys.stderr)
        return 2

    meta = (f'<p class="meta">EEE 402 · Group 10 · <a href="{REPO_URL}">'
            f'{REPO_URL.split("://", 1)[1]}</a> · printed {datetime.date.today():%d %b %Y}</p>')
    body = body.replace("</h1>", "</h1>\n" + meta, 1)
    css = CSS.replace("@TITLE@", TITLE)
    page = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f"<title>{html.escape(TITLE)}</title><style>{css}</style>"
            f"</head><body>\n{body}\n</body></html>\n")

    # Printed next to the target and moved into place only once complete, so a failed
    # print leaves the previous PDF alone instead of a truncated one in its place.
    partial = out.with_name(out.name + ".partial")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        page_path = Path(tmp) / "overview.html"
        page_path.write_text(page, encoding="utf-8")
        cmd = [browser, "--headless", "--disable-gpu", "--no-first-run",
               "--no-default-browser-check", "--disable-extensions",
               f"--user-data-dir={Path(tmp) / 'profile'}",   # never touch the real profile
               "--no-pdf-header-footer",                     # else each page shows file://
               f"--print-to-pdf={partial}", page_path.as_uri()]
        partial.unlink(missing_ok=True)
        res = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace",
                             timeout=180)
    if not partial.is_file() or partial.stat().st_size < 10_000:
        print(f"FAILED: {browser} did not write a PDF (exit {res.returncode}).\n"
              f"{res.stderr[-2000:]}", file=sys.stderr)
        return 2
    try:
        os.replace(partial, out)
    except PermissionError:
        print(f"FAILED: {out.name} is open in another program. Close it and re-run.",
              file=sys.stderr)
        return 2

    print(f"  {len(conv.embedded)} images embedded   {out.stat().st_size / 1e6:.2f} MB  -> {out}")
    try:
        from pypdf import PdfReader
    except ImportError:
        return 0
    reader = PdfReader(str(out))
    inside = sum(len(p.images) for p in reader.pages)
    print(f"  {len(reader.pages)} pages, {inside} images found inside the PDF")
    if inside < len(conv.embedded):
        print(f"FAILED: {len(conv.embedded)} images went in, {inside} came out.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
