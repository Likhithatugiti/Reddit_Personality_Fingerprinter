"""
src/utils/report_generator.py

Generates a one-page PDF personality report using reportlab.
Returns the path to the saved PDF.
"""

import logging
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

from config import OUTPUT_DIR, TRAIT_NAMES, TRAITS
from src.models.predictor import PersonalityProfile

logger = logging.getLogger(__name__)

# Trait descriptions (short blurb shown in the report)
TRAIT_DESCRIPTIONS = {
    "O": "Reflects curiosity, creativity, and openness to new ideas and experiences.",
    "C": "Reflects organisation, dependability, self-discipline, and goal-directed behaviour.",
    "E": "Reflects sociability, assertiveness, positive emotionality, and talkativeness.",
    "A": "Reflects cooperativeness, empathy, trust, and a desire to help others.",
    "N": "Reflects emotional instability, anxiety, moodiness, and stress reactivity.",
}

SCORE_LABELS = {
    (1.0, 2.5): "Very Low",
    (2.5, 3.5): "Low",
    (3.5, 4.5): "Moderate",
    (4.5, 5.5): "High",
    (5.5, 7.1): "Very High",
}


def _score_label(score: float) -> str:
    for (lo, hi), label in SCORE_LABELS.items():
        if lo <= score < hi:
            return label
    return "Moderate"


def generate_pdf(profile: PersonalityProfile, output_dir: Path = OUTPUT_DIR) -> Path:
    """
    Build a one-page PDF report for the given PersonalityProfile.

    Returns
    -------
    Path — location of the saved PDF file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename    = f"{profile.username}_personality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    output_path = output_dir / filename

    doc    = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()
    story  = []

    # ── Title ──────────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.HexColor("#3F51B5"),
        spaceAfter=4,
    )
    story.append(Paragraph("Reddit Personality Fingerprinter", title_style))

    sub_style = ParagraphStyle(
        "Sub",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#666666"),
        spaceAfter=12,
    )
    story.append(Paragraph(
        f"Big-Five Profile for u/{profile.username} "
        f"| Generated {datetime.now().strftime('%B %d, %Y')}",
        sub_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#3F51B5")))
    story.append(Spacer(1, 0.4*cm))

    # ── Metadata ───────────────────────────────────────────────────────────
    meta_data = [
        ["Comments analysed", str(profile.comment_count)],
        ["Total words",        str(profile.word_count)],
        ["Reliability",        "Low ⚠" if profile.low_data_warning else "Acceptable ✓"],
    ]
    meta_table = Table(meta_data, colWidths=[5*cm, 6*cm])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE",     (0, 0), (-1, -1), 10),
        ("TEXTCOLOR",    (0, 0), (0, -1),  colors.HexColor("#555555")),
        ("TEXTCOLOR",    (1, 0), (1, -1),  colors.black),
        ("FONTNAME",     (0, 0), (0, -1),  "Helvetica-Bold"),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.5*cm))

    # ── Score table ────────────────────────────────────────────────────────
    story.append(Paragraph("Trait Scores", styles["Heading2"]))
    story.append(Spacer(1, 0.2*cm))

    header = ["Trait", "Score", "Level", "Description"]
    rows   = [header]
    for trait in TRAITS:
        score = profile.scores.get(trait, 4.0)
        rows.append([
            TRAIT_NAMES[trait],
            f"{score:.2f} / 7.00",
            _score_label(score),
            TRAIT_DESCRIPTIONS[trait],
        ])

    score_table = Table(rows, colWidths=[3.5*cm, 2.5*cm, 2.5*cm, 8.5*cm])
    score_table.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#3F51B5")),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  10),
        # Data rows
        ("FONTSIZE",     (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F5F5F5"), colors.white]),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.5*cm))

    # ── Disclaimer ─────────────────────────────────────────────────────────
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#999999"),
        spaceAfter=0,
    )
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#DDDDDD")))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "⚠ Disclaimer: These scores are probabilistic estimates derived from public text data "
        "and are NOT clinical assessments. Do not use for hiring, screening, or any consequential "
        "decision-making. Methodology based on Harrison et al. (2019), SMJ.",
        disclaimer_style,
    ))

    doc.build(story)
    logger.info("PDF report saved to %s", output_path)
    return output_path
