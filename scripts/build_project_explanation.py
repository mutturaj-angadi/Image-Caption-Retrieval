"""Build a concise, presentation-friendly NeuralSeek project explanation PDF."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "project_explanation.pdf"
NAVY = colors.HexColor("#172554")
BLUE = colors.HexColor("#2563EB")
PALE = colors.HexColor("#EFF6FF")
INK = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#64748B")
GRID = colors.HexColor("#CBD5E1")


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=18*mm,
                            leftMargin=18*mm, topMargin=19*mm, bottomMargin=18*mm,
                            title="NeuralSeek Project Explanation",
                            author="NeuralSeek project team")
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("TitleCustom", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=27, leading=32, textColor=NAVY, alignment=TA_CENTER,
                                spaceAfter=7*mm),
        "subtitle": ParagraphStyle("SubtitleCustom", parent=base["Normal"], fontSize=11,
                                    leading=16, textColor=MUTED, alignment=TA_CENTER,
                                    spaceAfter=8*mm),
        "h1": ParagraphStyle("H1Custom", parent=base["Heading1"], fontName="Helvetica-Bold",
                              fontSize=17, leading=21, textColor=NAVY, spaceBefore=3*mm,
                              spaceAfter=3*mm, keepWithNext=True),
        "h2": ParagraphStyle("H2Custom", parent=base["Heading2"], fontName="Helvetica-Bold",
                              fontSize=11.5, leading=15, textColor=BLUE, spaceBefore=2*mm,
                              spaceAfter=1.5*mm, keepWithNext=True),
        "body": ParagraphStyle("BodyCustom", parent=base["BodyText"], fontName="Helvetica",
                                fontSize=9.2, leading=13.5, textColor=INK, spaceAfter=2.4*mm),
        "small": ParagraphStyle("SmallCustom", parent=base["BodyText"], fontSize=8,
                                 leading=10.5, textColor=MUTED, spaceAfter=1.5*mm),
        "cell": ParagraphStyle("CellCustom", parent=base["BodyText"], fontSize=8,
                                leading=10.5, textColor=INK),
        "cell_head": ParagraphStyle("CellHeadCustom", parent=base["BodyText"], fontName="Helvetica-Bold",
                                     fontSize=8, leading=10, textColor=colors.white),
        "callout": ParagraphStyle("CalloutCustom", parent=base["BodyText"], fontName="Helvetica-Bold",
                                   fontSize=10, leading=14, textColor=NAVY, alignment=TA_CENTER),
        "code": ParagraphStyle("CodeCustom", parent=base["Code"], fontName="Courier",
                                fontSize=7.3, leading=10, textColor=INK, backColor=PALE,
                                borderColor=GRID, borderWidth=.4, borderPadding=6,
                                spaceBefore=1*mm, spaceAfter=3*mm),
    }

    def p(text: str, style: str = "body") -> Paragraph:
        return Paragraph(text, styles[style])

    def table(rows: list[list[str]], widths: list[float], header: bool = True) -> Table:
        converted = [[p(x, "cell_head" if header and i == 0 else "cell") for x in row]
                     for i, row in enumerate(rows)]
        t = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
        commands = [("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), .35, GRID),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]
        if header:
            commands.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
            if len(rows) > 1:
                commands.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]))
        t.setStyle(TableStyle(commands))
        return t

    story = [Spacer(1, 13*mm), p("NeuralSeek", "title"),
             p("Text-to-image retrieval with shared visual and language representations",
               "subtitle")]
    callout = Table([[p("Type a sentence. NeuralSeek ranks the indexed photos that best match it.", "callout")]],
                    colWidths=[174*mm])
    callout.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,-1), PALE),
                                 ("BOX", (0,0),(-1,-1), .8, BLUE),
                                 ("LEFTPADDING",(0,0),(-1,-1),12),
                                 ("RIGHTPADDING",(0,0),(-1,-1),12),
                                 ("TOPPADDING",(0,0),(-1,-1),11),
                                 ("BOTTOMPADDING",(0,0),(-1,-1),11)]))
    story += [callout, Spacer(1, 5*mm), p("1. Project in a minute", "h1"),
              p("NeuralSeek is a machine-learning image search application. It represents image content and captions as vectors, then ranks images by cosine similarity to a text query. The repository contains data preparation, model training, evaluation, a retrieval API, and a browser interface."),
              p("Why this approach?", "h2"),
              p("A keyword search needs the query words to match image tags. NeuralSeek instead learns or estimates a visual vector from the sentence and compares it with image vectors, allowing semantic matching across the caption and image collection."),
              p("End-to-end flow", "h2"),
              table([["Stage", "What happens"],
                     ["Prepare", "Read MS-COCO captions and splits; validate images; build a training-only vocabulary."],
                     ["Represent", "Extract a 4,096-value VGG-19 fc7 feature for each image; encode captions with GloVe and a baseline, GRU, or LSTM."],
                     ["Learn and evaluate", "Train the neural encoders with a bidirectional margin-ranking loss; choose checkpoints on validation data; report Recall@K and rank."],
                     ["Serve", "Index normalized image vectors; FastAPI encodes a query, scores the index, and returns ranked image IDs and captions."],
                     ["Show", "A React and TypeScript interface calls the API and displays results and evaluation summaries."]],
                    [30*mm, 144*mm]),
              p("A useful way to describe the product: it is an image retrieval system, not an image caption generator. It searches a fixed image collection; it does not create new captions or images.", "small"),
              PageBreak(), p("2. Data and model design", "h1"),
              p("Dataset and preprocessing", "h2"),
              p("The pilot uses MS-COCO 2014 with the Karpathy split. Each image has five captions. The prepared run used 10,000 training images, 100 validation images, and 100 test images. Captions are lower-cased and tokenized; vocabulary entries come from training captions only. VGG-19 ImageNet weights produce cached fc7 image features."),
              p("Three model variants", "h2"),
              table([["Model", "Text representation", "Image representation"],
                     ["Linear baseline", "Mean of 300-dimensional GloVe word vectors, mapped by ridge-regularized linear regression.", "4,096-dimensional VGG feature in the same retrieval space."],
                     ["GRU + GloVe", "GloVe token embeddings → GRU → projection → L2-normalized vector.", "VGG feature → learned projection → L2-normalized vector."],
                     ["LSTM + GloVe", "GloVe token embeddings → LSTM → projection → L2-normalized vector.", "VGG feature → learned projection → L2-normalized vector."]],
                    [27*mm, 80*mm, 67*mm]),
              p("How neural training works", "h2"),
              p("For each matched image-caption pair, the model increases the cosine score of the match and decreases the scores of mismatched pairs in the same batch. The bidirectional margin loss compares image-to-text and text-to-image relationships. Captions belonging to the same image are not treated as negatives."),
              p("Evaluation terms", "h2"),
              table([["Metric", "Plain explanation"],
                     ["Recall@K", "The percentage of queries whose correct target appears among the top K results. Higher is better."],
                     ["Mean rank", "The average position of the correct target. Lower is better."],
                     ["Cosine similarity", "A ranking score between vectors. It is not a probability or confidence percentage."]],
                    [35*mm, 139*mm]),
              PageBreak(), p("3. Pilot results and how to explain them", "h1"),
              p("The following values are from the reported 100-image test split using the full protocol. The neural runs used one epoch, 256-dimensional embeddings, and CPU. The baseline tuned its ridge strength on validation data."),
              table([["Model", "Text→image R@1", "R@5", "R@10", "Mean rank", "Image→text R@1"],
                     ["Baseline", "41.6", "82.8", "92.6", "3.62", "62.0"],
                     ["GRU + GloVe", "49.8", "89.6", "96.6", "2.70", "59.0"],
                     ["LSTM + GloVe", "49.6", "87.0", "96.2", "2.86", "60.0"]],
                    [33*mm, 30*mm, 18*mm, 19*mm, 27*mm, 39*mm]),
              p("What the pilot suggests", "h2"),
              p("The configured validation selection metric is text-to-image R@1 + R@5 + R@10. Its scores were 227.4 for the baseline, 233.4 for the GRU, and 233.2 for the LSTM; the GRU was selected, though the GRU–LSTM difference is only 0.2 points on 100 validation images. On test, the GRU had the highest text-to-image Recall@K in this run. The neural models trained for only one epoch, so these results do not establish their best achievable performance. Test results are for final reporting, not tuning."),
              p("How to state the limitation", "h2"),
              p("“This is an end-to-end pilot on a 10,000/100/100 image subset. The neural models ran for one epoch with 256-dimensional embeddings on CPU. The 100-image test set is small, so the measurements are preliminary and should not be compared directly with full-scale results from the reference paper.”"),
              p("The pilot is useful for demonstrating that data preparation, feature extraction, model training, retrieval indexing, and evaluation work together. It is not a full reproduction or a statistically strong benchmark."),
              PageBreak(), p("4. Application demo", "h1"),
              p("The backend loads available model checkpoints and their compatible indexes at startup. The search endpoint encodes the query, retrieves nearest image vectors, and returns ranked results. The UI also has Models and Evaluation views. Baseline, GRU, and LSTM indexes were built for this run."),
              p("Run locally from Command Prompt", "h2"),
              p("First, set the repository’s .env paths to the real subset (not data/smoke): NCIR_ARTIFACTS_DIR=artifacts; NCIR_METADATA_PATH=data/processed/metadata.json; NCIR_IMAGES_ROOT=data/raw/coco; NCIR_REPORTS_DIR=reports; NCIR_STATS_PATH=data/processed/stats.json. Then use two terminals:")]
    story.append(p("Terminal 1 — backend:\ncd /d \"C:\\SEM 5\\ML Lab Submission\\Image Caption\\neural-caption-image-retrieval\"\n.venv\\Scripts\\activate.bat\npython -m uvicorn backend.main:app --reload --port 8000\n\nTerminal 2 — frontend:\ncd /d \"C:\\SEM 5\\ML Lab Submission\\Image Caption\\neural-caption-image-retrieval\\frontend\"\nnpm run dev\n\nOpen http://localhost:5173", "code"))
    story += [p("Suggested 60-second walkthrough", "h2"),
              table([["Step", "Say / show"],
                     ["Search", "Enter a simple query such as ‘a dog’. Explain that the list is ranked by cosine similarity."],
                     ["Models", "Show which checkpoints and indexes are available. The current pilot includes a baseline and two one-epoch neural models."],
                     ["Evaluation", "Show Recall@K, explain the 10,000/100/100 subset, and state that the result is preliminary."],
                     ["Close", "Mention that longer training, larger evaluation splits, and qualitative error analysis are the next research steps."]],
                    [25*mm, 149*mm]),
              p("Before presenting", "h2"),
              p("Verify /health returns status ‘ok’; make sure the app is using data/processed/metadata.json and artifacts/, not data/smoke; try a query and confirm image thumbnails load. All three model indexes were built for the reported run."),
              p("Project limits", "h2"),
              p("The system searches indexed images only. It has no attention or multi-sentence context; rare or unknown words, counting, negation, and spatial relations can be difficult. VGG-19 features and training settings differ from the original paper. All pilot metrics must remain labelled with their subset and training settings."),
              Spacer(1, 3*mm),
              p("Project explanation prepared 26 October 2026. Pilot measurements and run settings in this document are transcribed from the user-provided console output; they are not a new experiment run by this PDF generator.", "small")]

    def page(canvas, document):
        canvas.saveState()
        w, h = A4
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(.5)
        canvas.line(18*mm, h-13*mm, w-18*mm, h-13*mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(18*mm, h-10*mm, "NEURALSEEK  /  PROJECT EXPLANATION")
        canvas.drawRightString(w-18*mm, 10*mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=page, onLaterPages=page)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    build()
