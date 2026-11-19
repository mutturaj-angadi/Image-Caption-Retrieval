"""Render reports/final_report.md to reports/final_report.pdf (ReportLab, A4, compact two-page layout).

    python scripts/build_report.py
    python scripts/build_report.py --pages 2          # warn if the PDF does not have exactly this many pages

The results table is taken from reports/metrics/summary.json (written by make_report_assets.py). Missing
results are rendered as "pending". Any remaining "[TO COMPLETE ...]" markers are listed so you can fill them in.
"""
import _bootstrap  # noqa: F401

import argparse
import html
import re
import subprocess
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from scripts.make_report_assets import MODELS, markdown_table  # noqa: E402  (scripts is a namespace package via sys.path)
from src.utils.common import load_json, resolve_path

PLACEHOLDER = re.compile(r"\[TO COMPLETE[^\]]*\]")


def inline(text: str) -> str:
    """Markdown inline subset -> ReportLab mini-HTML."""
    out = html.escape(text, quote=False)
    for tag in ("sub", "super"):  # whitelisted markup (Helvetica has no unicode sub/superscript glyphs)
        out = out.replace(f"&lt;{tag}&gt;", f"<{tag}>").replace(f"&lt;/{tag}&gt;", f"</{tag}>")
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", out)
    out = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', out)
    return out


def results_block(reports: Path) -> tuple[str, str]:
    """Return (markdown table, note) for the Results section based on real result files only."""
    summary = reports / "metrics" / "summary.json"
    if summary.is_file():
        data = load_json(summary)
        results = data["results"]
        measured = [r for r in results.values() if r["status"] == "measured"]
        table = markdown_table(results)
        if measured:
            subs = any(r.get("subset") for r in measured)
            note = (f"Text→image (T2I) and image→text (I2T) recall in %, mean rank lower is better; split: {data['split']}, "
                    f"protocol: {data['protocol']}, {measured[0]['num_images']} images / {measured[0]['num_captions']} captions"
                    f"{' (development SUBSET)' if subs else ''}. Source: reports/metrics/*.json.")
            return table, note
    pending = {name: {"label": label, "status": "pending"} for name, label in MODELS}
    return markdown_table(pending), ("**No experiments have been run yet, so all results are pending.** Run the pipeline "
                                     "(README, Route B) and `scripts/make_report_assets.py`, then rebuild this report.")


def figure_flowables(reports: Path) -> list:
    """Embed generated figures (reports/figures/*.png) when they exist; nothing is drawn otherwise."""
    from reportlab.platypus import Image as RLImage

    figs = [reports / "figures" / n for n in ("model_comparison.png", "gru_glove_loss.png") if (reports / "figures" / n).is_file()]
    if not figs:
        return []
    row = []
    for f in figs:
        from PIL import Image as PILImage

        w, h = PILImage.open(f).size
        width = 8.4 * cm
        row.append(RLImage(str(f), width=width, height=width * h / w))
    return [Table([row], hAlign="CENTER"), Spacer(1, 4)]


def render(md_path: Path, pdf_path: Path, reports: Path) -> list[str]:
    text = md_path.read_text(encoding="utf-8")
    table_md, note = results_block(reports)
    text = text.replace("{{RESULTS_TABLE}}", table_md).replace("{{RESULTS_NOTE}}", note)
    todo = PLACEHOLDER.findall(text)

    base = ParagraphStyle("base", fontName="Helvetica", fontSize=9.4, leading=12.0, alignment=TA_JUSTIFY, spaceAfter=3)
    h1 = ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=14.5, leading=17.5, alignment=0, spaceAfter=3)
    h2 = ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=10.6, leading=13, alignment=0,
                        spaceBefore=6, spaceAfter=2, keepWithNext=1, textColor=colors.HexColor("#312e81"))
    cell = ParagraphStyle("cell", parent=base, fontSize=7.6, leading=9.2, alignment=0, spaceAfter=0)
    head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)

    story: list = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
        elif line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), h1))
            i += 1
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), h2))
            i += 1
        elif line.strip() == "{{FIGURES}}":
            story += figure_flowables(reports)
            i += 1
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in block if not re.match(r"^\|\s*-", r)]
            data = [[Paragraph(inline(c), head if ri == 0 else cell) for c in row] for ri, row in enumerate(rows)]
            ncol = len(rows[0])
            first = 4.1 * cm
            rest = (A4[0] - 3.2 * cm - first) / max(ncol - 1, 1)
            t = Table(data, colWidths=[first] + [rest] * (ncol - 1), repeatRows=1)
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
                                   ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c7c7d6")),
                                   ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                                   ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4fb")])]))
            story += [t, Spacer(1, 3)]
        else:
            para = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|")):
                para.append(lines[i].rstrip())
                i += 1
            story.append(Paragraph(inline(" ".join(para)), base))

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.3 * cm,
                            bottomMargin=1.2 * cm, title="Neural Caption-Image Retrieval - Mini-Project Report",
                            author="[team]")
    doc.build(story)
    return todo


def page_count(pdf: Path) -> int | None:
    try:
        out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
        return int(re.search(r"Pages:\s+(\d+)", out).group(1))  # type: ignore[union-attr]
    except (OSError, subprocess.CalledProcessError, AttributeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md", default="reports/final_report.md")
    ap.add_argument("--out", default="reports/final_report.pdf")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--pages", type=int, default=2)
    args = ap.parse_args()
    pdf = resolve_path(args.out)
    todo = render(resolve_path(args.md), pdf, resolve_path(args.reports_dir))
    n = page_count(pdf)
    print(f"wrote {pdf} ({pdf.stat().st_size / 1024:.0f} KB), pages: {n if n is not None else 'unknown (pdfinfo not found)'}")
    if n is not None and n != args.pages:
        print(f"WARNING: expected {args.pages} pages but got {n}; trim/extend the text or adjust font sizes.", file=sys.stderr)
    if todo:
        print(f"{len(todo)} placeholder(s) still to complete:")
        for t in todo:
            print("  -", t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
