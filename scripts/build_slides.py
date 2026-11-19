"""Generate the 10-slide presentation (presentation/NeuralSeek.pptx) and its outline (slides_outline.md).

    python scripts/build_slides.py

Both files are produced from the single SLIDES list below, so they cannot drift apart. The Results slide
uses reports/metrics/summary.json when real results exist and otherwise says "pending" -- nothing is invented.
Edit the TEAM line and any text, then re-run.
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

from scripts.make_report_assets import MODELS
from src.utils.common import load_json, resolve_path

TEAM = "[TO COMPLETE: team names / SRNs]  ·  UE24CS352A Machine Learning Mini-Project  ·  PES University"
INDIGO, DARK, MUTED, LIGHT = RGBColor(0x31, 0x2E, 0x81), RGBColor(0x1E, 0x29, 0x3B), RGBColor(0x64, 0x74, 0x8B), RGBColor(0xEE, 0xF2, 0xFF)

SLIDES = [
    {"title": "NeuralSeek: Neural Caption–Image Retrieval", "kind": "title",
     "bullets": ["Text-to-image search with deep learning", "Joint image–caption embedding space · MS-COCO"],
     "visual": "Screenshot of the Search page showing a query and its top results.",
     "notes": "Introduce the team and the one-line idea: you type a sentence and the system finds matching images without any tags. Mention the reference paper and that the project is inspired by it.",
     "explain": "We built a search engine where you describe a picture in words and it finds it."},
    {"title": "Problem statement and motivation",
     "bullets": ["Users describe images in natural language, not keywords", "Keyword/caption matching fails on paraphrases and uncaptioned images",
                 "Goal: rank images by semantic match to a sentence", "Needs language understanding + visual understanding in one space"],
     "visual": "Side-by-side: keyword search missing 'puppy' vs. embedding search finding 'dog' images.",
     "notes": "Give the motivating example ('a dog running through snow'). Explain that exact word matching misses 'puppy' or 'snowy field', while embeddings capture meaning.",
     "explain": "Words and pictures live in different worlds; we teach a model to put them in the same place so closeness means 'matches'."},
    {"title": "Objectives and challenges",
     "bullets": ["Reproducible pipeline: data → features → models → index → app", "Baseline (linear regression) vs GRU vs LSTM, with GloVe",
                 "Evaluate with Recall@K and mean rank, no test-set leakage", "Challenges: modality gap, ambiguity, bias, limited compute"],
     "visual": "Roadmap graphic with the five pipeline stages.",
     "notes": "Say what success means: a working app plus honest evaluation. Mention compute limits (CPU-only possible, VGG-19 features cached).",
     "explain": "We compare a simple method with neural ones, measure fairly, and ship a working app."},
    {"title": "Dataset and preprocessing",
     "bullets": ["MS-COCO 2014: 5 human captions per image", "Karpathy split: 113,287 train / 5,000 val / 5,000 test (full run)",
                 "[TO COMPLETE: data actually used – full split or labelled subset]", "Cleaning, tokenisation, train-only vocabulary, <unk> handling",
                 "Validation: missing/corrupt/duplicate images, leakage check"],
     "visual": "Bar chart of split sizes and a caption-length histogram (from data/processed/stats.json).",
     "notes": "State exactly which data you used. Emphasise the vocabulary is built from training captions only and that every image appears in exactly one split.",
     "explain": "Each photo has five descriptions written by people; we clean them and make sure test photos never leak into training."},
    {"title": "System architecture", "kind": "architecture",
     "bullets": ["Frozen VGG-19 fc7 (4,096-d) → linear → unit vector", "GloVe → GRU/LSTM → linear → unit vector",
                 "Cosine similarity in a 1,024-d space; exact index (FAISS optional)", "FastAPI service + React UI"],
     "visual": "The diagram on this slide (replace with a polished version if you like).",
     "notes": "Walk left to right: image branch, text branch, shared space, index, API, UI. Stress that evaluation, indexing and search share the same encoder code.",
     "explain": "Two encoders translate pictures and sentences into the same kind of vector; the nearest vectors are the best matches."},
    {"title": "Model methodology",
     "bullets": ["Baseline: mean GloVe vector → ridge regression → predicted fc7", "GRU/LSTM: word order matters; final hidden state is the sentence vector",
                 "Margin ranking loss: matching pair must beat mismatches by α = 0.2", "Adam, early stopping on validation recall, best checkpoint kept"],
     "visual": "Equation of the ranking loss and a small similarity-matrix sketch (diagonal = positives).",
     "notes": "Explain positives (image with its own caption) and negatives (other pairs in the batch). Mention that captions of the same image are masked out of the negatives.",
     "explain": "Training pulls a photo toward its own caption and pushes it away from other captions."},
    {"title": "Implementation and user interface",
     "bullets": ["Python/PyTorch library, YAML configs, seeded runs, checkpoints + metadata", "50+ automated tests; synthetic smoke mode runs without COCO",
                 "FastAPI: /api/search, /api/models, /api/evaluation, validated errors", "React + TypeScript: Search, Models, Evaluation, About"],
     "visual": "Screenshots: Search results grid, Models page, Evaluation page.",
     "notes": "Briefly show the repo structure and the UI. Mention validation, graceful errors and that images are served only for known ids.",
     "explain": "It is a full application, not just a notebook: tested code, an API and a web interface."},
    {"title": "Experimental results", "kind": "results",
     "bullets": [], "visual": "Model-comparison bar chart and loss curves generated by scripts/make_report_assets.py.",
     "notes": "Read the numbers from the table only. Compare with the paper's reference values (GRU+GloVe R@1 37.0 %, R@10 86.8 %) as context, not as a target, and explain differences (subset, torchvision weights, protocol).",
     "explain": "Recall@K asks: is the right image in the top K? Higher is better; mean rank lower is better."},
    {"title": "Live demonstration and failure cases",
     "bullets": ["Query: a dog running through snow → ranked images + scores", "Switch model: baseline vs GRU on the same sentence",
                 "Evaluation page: real metrics", "Failure case: [TO COMPLETE: query and why it fails, e.g. counting/negation]"],
     "visual": "Screenshot of a failure case with the explanation.",
     "notes": "Follow docs/demo_script.md (4 minutes). Show one failure honestly; link it to a limitation.",
     "explain": "Showing where it fails teaches as much as showing where it works."},
    {"title": "Conclusion, limitations and future work",
     "bullets": ["Complete tested pipeline from data to web app", "[TO COMPLETE: key result sentence from your measured numbers]",
                 "Limits: no attention/context, OOV words ignored, dataset bias", "Future: dynamic attention, context-aware queries, fine-tuned CNN, CLIP comparison"],
     "visual": "Roadmap of future work.",
     "notes": "Summarise what was built and what was measured. Be explicit that results are not a reproduction of the paper. End with the future work from the paper.",
     "explain": "It works, it has known weaknesses, and we know how to improve it."},
]


# ----------------------------------------------------------------------------- outline
def write_outline(path: Path) -> None:
    lines = ["# Slide outline (generated by scripts/build_slides.py)\n", f"{TEAM}\n"]
    for n, s in enumerate(SLIDES, 1):
        lines.append(f"\n## Slide {n} — {s['title']}\n")
        for b in s["bullets"]:
            lines.append(f"- {b}")
        if s.get("kind") == "results":
            lines.append("- Comparison table from `reports/metrics/summary.md` (pending until experiments are run)")
        lines += [f"\n**Recommended visual:** {s['visual']}", f"\n**Speaker notes:** {s['notes']}", f"\n**Student-friendly explanation:** {s['explain']}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- pptx helpers
def textbox(slide, x, y, w, h, text, size=18, bold=False, color=DARK, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.text = text
    for p in tf.paragraphs:
        p.alignment = align
        for r in p.runs:
            r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, "Calibri"
    return tb


def bullets_box(slide, x, y, w, h, items, size=20):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "•  " + item
        p.space_after = Pt(9)
        for r in p.runs:
            r.font.size, r.font.color.rgb, r.font.name = Pt(size), DARK, "Calibri"
    return tb


def box(slide, x, y, w, h, label, fill=LIGHT, size=13, line=INDIGO):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = line
    tf = shp.text_frame
    tf.word_wrap = True
    tf.text = label
    for p in tf.paragraphs:
        p.alignment = PP_ALIGN.CENTER
        for r in p.runs:
            r.font.size, r.font.color.rgb, r.font.name = Pt(size), DARK, "Calibri"
    return shp


def arrow(slide, x1, y1, x2, y2):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = INDIGO
    c.line.width = Pt(2)
    ln = c.line._get_or_add_ln()
    from lxml import etree

    tail = etree.SubElement(ln, "{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd")
    tail.set("type", "triangle")


def base_slide(prs, title, number):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.05))
    band.fill.solid()
    band.fill.fore_color.rgb = INDIGO
    band.line.fill.background()
    textbox(slide, Inches(0.5), Inches(0.18), Inches(11.5), Inches(0.8), title, 30, True, RGBColor(255, 255, 255))
    textbox(slide, Inches(12.2), Inches(7.05), Inches(0.9), Inches(0.35), str(number), 12, False, MUTED, PP_ALIGN.RIGHT)
    return slide


def visual_hint(slide, text):
    textbox(slide, Inches(0.5), Inches(6.85), Inches(11.5), Inches(0.5), "Suggested visual: " + text, 12, False, MUTED)


def architecture(slide):
    y1, y2, w, h = Inches(2.1), Inches(3.55), Inches(2.05), Inches(0.85)
    box(slide, Inches(0.5), y1, w, h, "Image\n(COCO photo)")
    box(slide, Inches(2.85), y1, w, h, "VGG-19 fc7\n4,096-d (frozen)")
    box(slide, Inches(5.2), y1, w, h, "Linear → unit vector\n1,024-d")
    box(slide, Inches(0.5), y2, w, h, "Query sentence")
    box(slide, Inches(2.85), y2, w, h, "GloVe → GRU / LSTM")
    box(slide, Inches(5.2), y2, w, h, "Linear → unit vector\n1,024-d")
    box(slide, Inches(7.6), Inches(2.82), Inches(2.0), Inches(0.9), "Shared space\ncosine similarity", fill=RGBColor(0xE0, 0xE7, 0xFF))
    box(slide, Inches(10.0), Inches(2.82), Inches(2.0), Inches(0.9), "Index → top-K\n(FastAPI → React)", fill=RGBColor(0xDC, 0xFC, 0xE7), line=RGBColor(0x16, 0xA3, 0x4A))
    for x in (Inches(2.55), Inches(4.9)):
        arrow(slide, x, y1 + h // 2, x + Inches(0.3), y1 + h // 2)
        arrow(slide, x, y2 + h // 2, x + Inches(0.3), y2 + h // 2)
    arrow(slide, Inches(7.25), y1 + h // 2, Inches(7.6), Inches(3.05))
    arrow(slide, Inches(7.25), y2 + h // 2, Inches(7.6), Inches(3.5))
    arrow(slide, Inches(9.6), Inches(3.27), Inches(10.0), Inches(3.27))
    textbox(slide, Inches(0.5), Inches(4.9), Inches(11.5), Inches(1.6),
            "Training: margin ranking loss pulls each image toward its own caption and away from other captions in the batch.\n"
            "Inference: the query is encoded with the same text encoder, then the nearest image embeddings are returned.", 16)


def results(slide, reports: Path):
    summary = reports / "metrics" / "summary.json"
    rows = []
    measured_any = False
    if summary.is_file():
        data = load_json(summary)["results"]
        for name, label in MODELS:
            e = data.get(name, {})
            if e.get("status") == "measured":
                m = e["metrics"]
                measured_any = True
                rows.append([label, f"{m['t2i_R@1']:.1f}", f"{m['t2i_R@5']:.1f}", f"{m['t2i_R@10']:.1f}", f"{m['t2i_mean_rank']:.1f}"])
            else:
                rows.append([label] + ["pending"] * 4)
    else:
        rows = [[label] + ["pending"] * 4 for _, label in MODELS]
    table = slide.shapes.add_table(len(rows) + 1, 5, Inches(0.6), Inches(1.5), Inches(8.2), Inches(2.6)).table
    for c, h in enumerate(["Model", "T2I R@1 (%)", "R@5 (%)", "R@10 (%)", "Mean rank"]):
        table.cell(0, c).text = h
    for r, row in enumerate(rows, 1):
        for c, v in enumerate(row):
            table.cell(r, c).text = v
    table.columns[0].width = Inches(3.4)
    for c in range(1, 5):
        table.columns[c].width = Inches(1.2)
    for r in range(len(rows) + 1):
        for c in range(5):
            for p in table.cell(r, c).text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(15)
    fig = reports / "figures" / "model_comparison.png"
    if fig.is_file():
        slide.shapes.add_picture(str(fig), Inches(9.0), Inches(1.5), width=Inches(4.0))
    textbox(slide, Inches(0.6), Inches(4.7), Inches(12), Inches(2.0),
            ("Measured on the held-out split; text→image retrieval (caption = query). Source: reports/metrics/*.json."
             if measured_any else
             "NO EXPERIMENTS HAVE BEEN RUN YET – every value above is pending. Run the pipeline, then python scripts/make_report_assets.py "
             "and rebuild these slides.") +
            "\nReference only (paper, own setup): GRU + GloVe Recall@1 37.0 %, Recall@10 86.8 % – not reproduced or claimed here.", 16)


def build(out: Path, reports: Path) -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    for n, s in enumerate(SLIDES, 1):
        kind = s.get("kind")
        if kind == "title":
            slide = prs.slides.add_slide(prs.slide_layouts[6])
            bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
            bg.fill.solid()
            bg.fill.fore_color.rgb = INDIGO
            bg.line.fill.background()
            textbox(slide, Inches(0.9), Inches(2.0), Inches(11.5), Inches(1.6), s["title"], 44, True, RGBColor(255, 255, 255))
            textbox(slide, Inches(0.9), Inches(3.8), Inches(11.5), Inches(1.2), "\n".join(s["bullets"]), 24, False, RGBColor(0xE0, 0xE7, 0xFF))
            textbox(slide, Inches(0.9), Inches(6.3), Inches(11.5), Inches(0.6), TEAM, 14, False, RGBColor(0xC7, 0xD2, 0xFE))
        else:
            slide = base_slide(prs, s["title"], n)
            if kind == "architecture":
                architecture(slide)
            elif kind == "results":
                results(slide, reports)
            else:
                bullets_box(slide, Inches(0.7), Inches(1.5), Inches(11.8), Inches(5.0), s["bullets"])
            visual_hint(slide, s["visual"])
        slide.notes_slide.notes_text_frame.text = f"{s['notes']}\n\nIn simple words: {s['explain']}"
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="presentation/NeuralSeek.pptx")
    ap.add_argument("--outline", default="presentation/slides_outline.md")
    ap.add_argument("--reports-dir", default="reports")
    args = ap.parse_args()
    write_outline(resolve_path(args.outline))
    build(resolve_path(args.out), resolve_path(args.reports_dir))
    print(f"wrote {resolve_path(args.out)} and {resolve_path(args.outline)} ({len(SLIDES)} slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
