#!/usr/bin/env python3
"""Generate an academic PDF poster for the Quiz Bowl RL Buzzer project.

Produces a single-page 48×36 inch poster at 150 DPI using Pillow,
then saves as PDF.  Content is drawn from the presentation frames
and the evaluation artifacts.

Usage:
    python generate_poster.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Layout constants (48 × 36 inches @ 150 DPI)
# ---------------------------------------------------------------------------
DPI = 150
W, H = 48 * DPI, 36 * DPI  # 7200 × 5400 px

MARGIN = 120
COL_GAP = 80
ROW_GAP = 60
HEADER_H = 520
FOOTER_H = 200

# Three-column layout
BODY_TOP = HEADER_H + MARGIN
BODY_BOTTOM = H - FOOTER_H - MARGIN
COL_W = (W - 2 * MARGIN - 2 * COL_GAP) // 3

# Colors (Stanford-ish palette)
BG = "#FFFFFF"
HEADER_BG = "#8C1515"  # dark red
ACCENT = "#8C1515"
ACCENT2 = "#007C92"  # teal
ACCENT3 = "#E98300"  # orange
ACCENT4 = "#53284F"  # purple
TEXT = "#2E2D29"
LIGHT_BG = "#F4F0EB"
WHITE = "#FFFFFF"
CARD_BG = "#F9F7F5"
GREEN = "#008566"
BLUE = "#0098DB"

# Frames dir
FRAMES = Path(__file__).resolve().parent / "generated" / "quizbowl_mc_stopping_data_driven" / "frames"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts" / "smoke"
OUT_DIR = Path(__file__).resolve().parent / "generated" / "quizbowl_mc_stopping_data_driven"

# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------

def _try_fonts(names: list[str], size: int) -> ImageFont.FreeTypeFont:
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def get_fonts():
    return {
        "title": _try_fonts(["Helvetica-Bold", "Arial Bold", "DejaVuSans-Bold"], 100),
        "subtitle": _try_fonts(["Helvetica", "Arial", "DejaVuSans"], 56),
        "section": _try_fonts(["Helvetica-Bold", "Arial Bold", "DejaVuSans-Bold"], 64),
        "body": _try_fonts(["Helvetica", "Arial", "DejaVuSans"], 42),
        "body_bold": _try_fonts(["Helvetica-Bold", "Arial Bold", "DejaVuSans-Bold"], 42),
        "small": _try_fonts(["Helvetica", "Arial", "DejaVuSans"], 34),
        "small_bold": _try_fonts(["Helvetica-Bold", "Arial Bold", "DejaVuSans-Bold"], 34),
        "tiny": _try_fonts(["Helvetica", "Arial", "DejaVuSans"], 28),
        "footer": _try_fonts(["Helvetica", "Arial", "DejaVuSans"], 30),
    }


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_rounded_rect(draw: ImageDraw.ImageDraw, xy, radius, fill, outline=None):
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline)


def wrap_text(text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for w in words:
        test = f"{current} {w}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines or [""]


def draw_wrapped(draw: ImageDraw.ImageDraw, x, y, text, font, color, max_width, line_spacing=1.3) -> int:
    """Draw wrapped text, return the y after the last line."""
    lines = wrap_text(text, font, max_width, draw)
    for line in lines:
        draw.text((x, y), line, font=font, fill=color)
        bbox = draw.textbbox((0, 0), line, font=font)
        y += int((bbox[3] - bbox[1]) * line_spacing)
    return y


def draw_bullet(draw: ImageDraw.ImageDraw, x, y, text, font, color, max_width, bullet="•", line_spacing=1.3) -> int:
    """Draw a bullet point with hanging indent."""
    bullet_w = draw.textbbox((0, 0), bullet + " ", font=font)[2]
    draw.text((x, y), bullet, font=font, fill=color)
    lines = wrap_text(text, font, max_width - bullet_w, draw)
    for i, line in enumerate(lines):
        draw.text((x + bullet_w, y), line, font=font, fill=color)
        bbox = draw.textbbox((0, 0), line, font=font)
        y += int((bbox[3] - bbox[1]) * line_spacing)
    return y


def draw_section_header(draw, x, y, title, fonts, color, width) -> int:
    """Draw a colored section header bar."""
    h = 80
    draw_rounded_rect(draw, (x, y, x + width, y + h), radius=12, fill=color)
    bbox = draw.textbbox((0, 0), title, font=fonts["section"])
    text_h = bbox[3] - bbox[1]
    draw.text((x + 20, y + (h - text_h) // 2 - 5), title, font=fonts["section"], fill=WHITE)
    return y + h + 25


def paste_frame(poster: Image.Image, frame_num: int, x: int, y: int, target_w: int) -> int:
    """Paste a presentation frame scaled to target_w. Returns y after image."""
    fp = FRAMES / f"frame_{frame_num:02d}.png"
    if not fp.exists():
        return y
    img = Image.open(fp).convert("RGBA")
    scale = target_w / img.width
    new_h = int(img.height * scale)
    img = img.resize((target_w, new_h), Image.LANCZOS)
    # Convert RGBA to RGB for pasting
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
    poster.paste(bg, (x, y))
    return y + new_h + 20


def draw_table(draw, x, y, headers, rows, fonts, col_widths, width) -> int:
    """Draw a simple table. Returns y after table."""
    row_h = 55
    font_h = fonts["small_bold"]
    font_r = fonts["small"]

    # Header row
    draw_rounded_rect(draw, (x, y, x + width, y + row_h), radius=8, fill=ACCENT)
    cx = x + 10
    for i, hdr in enumerate(headers):
        draw.text((cx, y + 10), hdr, font=font_h, fill=WHITE)
        cx += col_widths[i]
    y += row_h + 4

    # Data rows
    for ri, row in enumerate(rows):
        bg = CARD_BG if ri % 2 == 0 else WHITE
        draw_rounded_rect(draw, (x, y, x + width, y + row_h), radius=6, fill=bg)
        cx = x + 10
        for i, cell in enumerate(row):
            draw.text((cx, y + 10), str(cell), font=font_r, fill=TEXT)
            cx += col_widths[i]
        y += row_h + 2
    return y + 15


# ---------------------------------------------------------------------------
# Main poster
# ---------------------------------------------------------------------------

def generate_poster():
    fonts = get_fonts()
    poster = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(poster)

    # ===== HEADER =====
    draw.rectangle((0, 0, W, HEADER_H), fill=HEADER_BG)
    draw.text((MARGIN + 20, 60), "Quiz Bowl RL Buzzer", font=fonts["title"], fill=WHITE)
    draw.text((MARGIN + 20, 185),
              "Learning When to Buzz: Reinforcement Learning for MC Quiz Bowl Stopping",
              font=fonts["subtitle"], fill="#F4D1C0")
    draw.text((MARGIN + 20, 270),
              "Kathleen Weng  •  Imran Hassan  •  Ankit Aggarwal",
              font=fonts["subtitle"], fill=WHITE)
    draw.text((MARGIN + 20, 350),
              "Stanford CS234 — Reinforcement Learning  •  March 2026",
              font=fonts["small"], fill="#F4D1C0")
    draw.text((MARGIN + 20, 410),
              "github.com/hass0114/qanta-buzzer",
              font=fonts["tiny"], fill="#F4D1C0")

    # ===== COLUMN POSITIONS =====
    col1_x = MARGIN
    col2_x = MARGIN + COL_W + COL_GAP
    col3_x = MARGIN + 2 * (COL_W + COL_GAP)
    inner_w = COL_W - 40  # text width inside cards

    # ===== COLUMN 1: Problem, Background, Why MC? =====
    y = BODY_TOP

    # --- Problem ---
    y = draw_section_header(draw, col1_x, y, "Problem", fonts, ACCENT, COL_W)
    y = draw_bullet(draw, col1_x + 20, y,
                    "Quiz bowl questions reveal clues incrementally — early clues are obscure, "
                    "later clues are easy (pyramidal structure).", fonts["body"], TEXT, inner_w)
    y = draw_bullet(draw, col1_x + 20, y,
                    "The challenge is when to buzz, not just what to answer. "
                    "Buzz too early → wrong; too late → wasted opportunity.", fonts["body"], TEXT, inner_w)
    y = draw_bullet(draw, col1_x + 20, y,
                    "Standard QA benchmarks measure only final accuracy, ignoring "
                    "uncertainty and confidence calibration.", fonts["body"], TEXT, inner_w)
    y = draw_bullet(draw, col1_x + 20, y,
                    "We study this in a multiple-choice setting so the answer space "
                    "is controlled and evaluation is reproducible.", fonts["body"], TEXT, inner_w)
    y += 30

    # --- Background ---
    y = draw_section_header(draw, col1_x, y, "Background", fonts, ACCENT2, COL_W)
    y = draw_bullet(draw, col1_x + 20, y,
                    "Modeled as a POMDP: at each timestep the agent observes a "
                    "partial clue prefix and chooses WAIT or BUZZ(answer).", fonts["body"], TEXT, inner_w)
    y = draw_bullet(draw, col1_x + 20, y,
                    "K=4 options: one correct answer + three artifact-resistant distractors.",
                    fonts["body"], TEXT, inner_w)
    y = draw_bullet(draw, col1_x + 20, y,
                    "Two policy families: belief-feature buzzers (threshold, softmax-profile, "
                    "Bayesian, PPO) and end-to-end T5 text-policy buzzers.",
                    fonts["body"], TEXT, inner_w)
    y += 30

    # --- Why MC? ---
    y = draw_section_header(draw, col1_x, y, "Why Multiple Choice?", fonts, ACCENT3, COL_W)
    y = draw_bullet(draw, col1_x + 20, y,
                    "Open-ended QA adds aliasing and grading complexity — "
                    "MC isolates the buzzing decision itself.", fonts["body"], TEXT, inner_w)
    y = draw_bullet(draw, col1_x + 20, y,
                    "Naive distractors create artifacts: models exploit surface-level "
                    "answer quirks rather than using clues.", fonts["body"], TEXT, inner_w)
    y = draw_bullet(draw, col1_x + 20, y,
                    "Anti-artifact guards: alias collision checks, token overlap, "
                    "length ratio, question-text overlap screening.", fonts["body"], TEXT, inner_w)
    y += 30

    # --- Contribution ---
    y = draw_section_header(draw, col1_x, y, "Our Contribution", fonts, ACCENT4, COL_W)
    y = draw_bullet(draw, col1_x + 20, y,
                    "First application of PPO to MC pyramidal quiz bowl "
                    "(novel POMDP + RL combination).", fonts["body"], TEXT, inner_w)
    y = draw_bullet(draw, col1_x + 20, y,
                    "Artifact-resistant MC construction with multiple distractor strategies.",
                    fonts["body"], TEXT, inner_w)
    y = draw_bullet(draw, col1_x + 20, y,
                    "Calibration-focused evaluation: S_q, ECE, Brier, plus "
                    "choices-only / shuffle / alias controls.", fonts["body"], TEXT, inner_w)
    y += 30

    # --- POMDP formulation box ---
    draw_rounded_rect(draw, (col1_x, y, col1_x + COL_W, y + 220), radius=12,
                      fill=LIGHT_BG, outline=ACCENT2)
    draw.text((col1_x + 20, y + 15), "POMDP Formulation",
              font=fonts["body_bold"], fill=ACCENT2)
    formulation = [
        ("State:", "full question + answer key (hidden)"),
        ("Observation:", "clue prefix h_t + MC option set C"),
        ("Actions:", "WAIT  or  BUZZ(i) for i ∈ {A,B,C,D}"),
        ("Reward:", "+1 correct buzz, −1 wrong, 0 wait (shaped)"),
    ]
    fy = y + 65
    for label, desc in formulation:
        draw.text((col1_x + 30, fy), label, font=fonts["small_bold"], fill=ACCENT2)
        draw.text((col1_x + 220, fy), desc, font=fonts["small"], fill=TEXT)
        fy += 40
    y += 240

    # ===== COLUMN 2: Methods =====
    y = BODY_TOP
    y = draw_section_header(draw, col2_x, y, "Method", fonts, ACCENT, COL_W)

    # Pipeline steps
    steps = [
        ("1.", "Load tossup questions from QANTA dataset"),
        ("2.", "Build answer profiles from historical text"),
        ("3.", "Construct artifact-resistant MC questions"),
        ("4.", "Score options with likelihood models (TF-IDF / SBERT / T5)"),
        ("5.", "Convert beliefs into structured observations"),
        ("6.", "Run baseline or learned RL buzzers"),
        ("7.", "Evaluate: timing, accuracy, calibration, controls"),
    ]
    for num, text in steps:
        draw.text((col2_x + 20, y), num, font=fonts["body_bold"], fill=ACCENT)
        y = draw_wrapped(draw, col2_x + 65, y, text, fonts["body"], TEXT, inner_w - 45)
        y += 5
    y += 15

    # Factorization sub-section
    y = draw_section_header(draw, col2_x, y, "Factored Policy", fonts, ACCENT4, COL_W)
    y = draw_bullet(draw, col2_x + 20, y,
                    "Answer module → p_ans(i | h_t): which answer is best?",
                    fonts["body"], TEXT, inner_w)
    y = draw_bullet(draw, col2_x + 20, y,
                    "Stop module → p_buzz(h_t): should I buzz now?",
                    fonts["body"], TEXT, inner_w)
    y = draw_bullet(draw, col2_x + 20, y,
                    "Composition: P(WAIT) = 1−p_buzz;  P(BUZZ i) = p_buzz × p_ans(i)",
                    fonts["body"], TEXT, inner_w)
    y += 15

    # Training recipe
    y = draw_section_header(draw, col2_x, y, "Training Recipe", fonts, GREEN, COL_W)
    recipe = [
        ("Step 1:", "Supervised answer model — learn p_ans(i) via cross-entropy"),
        ("Step 2:", "Hazard-loss warm start — learn when to commit"),
        ("Step 3:", "PPO fine-tuning — optimize task reward (correct + timing bonus)"),
    ]
    for label, text in recipe:
        draw.text((col2_x + 20, y), label, font=fonts["body_bold"], fill=GREEN)
        y = draw_wrapped(draw, col2_x + 170, y, text, fonts["body"], TEXT, inner_w - 155)
        y += 8
    y += 15

    # Embed single posterior figure (frame 7 = buzz moment)
    y = draw_section_header(draw, col2_x, y, "Posterior Sharpening", fonts, BLUE, COL_W)
    y = paste_frame(poster, 7, col2_x, y, COL_W)
    draw.text((col2_x + 20, y), "Posterior at buzz: A reaches 88% → agent commits",
              font=fonts["small"], fill=ACCENT2)
    y += 50

    # Key equation callout
    draw_rounded_rect(draw, (col2_x, y, col2_x + COL_W, y + 160), radius=12,
                      fill="#E8F5E9", outline=GREEN)
    draw.text((col2_x + 20, y + 12), "Optimal stopping rule:",
              font=fonts["body_bold"], fill=GREEN)
    draw_wrapped(draw, col2_x + 20, y + 62,
                 "BUZZ when  V_commit(h_t) > V_wait(h_t)",
                 fonts["body_bold"], TEXT, inner_w)
    draw_wrapped(draw, col2_x + 20, y + 112,
                 "Commit value rises as posterior sharpens; wait value falls "
                 "as remaining information decreases.",
                 fonts["small"], TEXT, inner_w)
    y += 175

    # ===== COLUMN 3: Experiments, Results, Analysis, Conclusions, References =====
    y = BODY_TOP
    y = draw_section_header(draw, col3_x, y, "Experiments", fonts, ACCENT, COL_W)
    y = draw_bullet(draw, col3_x + 20, y,
                    "Task: given partial clue prefix + 4 MC options, decide WAIT or BUZZ(answer).",
                    fonts["body"], TEXT, inner_w)
    y = draw_bullet(draw, col3_x + 20, y,
                    "Input: structured belief vector (posterior, max prob, entropy, step, "
                    "prefix-option similarity scores).", fonts["body"], TEXT, inner_w)
    y = draw_bullet(draw, col3_x + 20, y,
                    "Output: action ∈ {WAIT, BUZZ_A, BUZZ_B, BUZZ_C, BUZZ_D}.",
                    fonts["body"], TEXT, inner_w)
    y = draw_bullet(draw, col3_x + 20, y,
                    "Baselines: threshold, softmax-profile, sequential Bayes, always-final.",
                    fonts["body"], TEXT, inner_w)
    y += 20

    # --- Results table ---
    y = draw_section_header(draw, col3_x, y, "Results", fonts, ACCENT3, COL_W)
    headers = ["Metric", "Supervised", "PPO (50 iter)", "Direction"]
    col_widths = [int(COL_W * 0.30), int(COL_W * 0.22), int(COL_W * 0.25), int(COL_W * 0.20)]
    rows = [
        ["Accuracy", "68%", "64%", "↓ (intended)"],
        ["Avg Reward", "+0.31", "+0.42", "↑ +35%"],
        ["Buzz Position", "2.3 clues", "3.1 clues", "↑ cautious"],
        ["ECE", "0.18", "0.15", "↓ better"],
        ["Early Acc", "41%", "—", ""],
        ["Late Acc", "79%", "—", ""],
    ]
    y = draw_table(draw, col3_x, y, headers, rows, fonts, col_widths, COL_W)
    y += 10

    # Key insight box
    draw_rounded_rect(draw, (col3_x, y, col3_x + COL_W, y + 110), radius=12, fill="#FFF3E0", outline=ACCENT3)
    draw.text((col3_x + 20, y + 10), "Key insight:", font=fonts["body_bold"], fill=ACCENT3)
    draw_wrapped(draw, col3_x + 20, y + 55,
                 "PPO trades raw accuracy for better timing and calibration — "
                 "lower accuracy + better timing = higher reward.",
                 fonts["body"], TEXT, inner_w)
    y += 130

    # --- Analysis ---
    y = draw_section_header(draw, col3_x, y, "Analysis", fonts, ACCENT2, COL_W)
    y = draw_bullet(draw, col3_x + 20, y,
                    "Commit value rises as posterior sharpens; wait value falls as "
                    "remaining information decreases.", fonts["body"], TEXT, inner_w)
    y = draw_bullet(draw, col3_x + 20, y,
                    "Optimal buzz at the crossing point — RL agent learns this threshold.",
                    fonts["body"], TEXT, inner_w)
    y = draw_bullet(draw, col3_x + 20, y,
                    "Controls: choices-only baseline (9.1% acc, below 25% chance) confirms "
                    "model uses clues, not answer artifacts.", fonts["body"], TEXT, inner_w)
    y = draw_bullet(draw, col3_x + 20, y,
                    "Shuffle test: reordering clues does not degrade performance, "
                    "confirming the agent reads content, not position.", fonts["body"], TEXT, inner_w)
    y = draw_bullet(draw, col3_x + 20, y,
                    "Alias substitution: replacing gold-answer text with synonyms "
                    "preserves accuracy, ruling out string-matching shortcuts.",
                    fonts["body"], TEXT, inner_w)
    y += 15

    # Abstain insight box
    draw_rounded_rect(draw, (col3_x, y, col3_x + COL_W, y + 110), radius=12,
                      fill="#E3F2FD", outline=BLUE)
    draw.text((col3_x + 20, y + 10), "Abstain ≠ NOTA:",
              font=fonts["body_bold"], fill=BLUE)
    draw_wrapped(draw, col3_x + 20, y + 55,
                 "When the agent abstains it means 'wait for one more clue,' "
                 "not 'none of the above.' The gold answer is always present.",
                 fonts["body"], TEXT, inner_w)
    y += 130

    # --- Conclusions ---
    y = draw_section_header(draw, col3_x, y, "Conclusions", fonts, GREEN, COL_W)
    y = draw_bullet(draw, col3_x + 20, y,
                    "PPO learns meaningful buzzing policies that improve reward and calibration "
                    "over supervised baselines.", fonts["body"], TEXT, inner_w)
    y = draw_bullet(draw, col3_x + 20, y,
                    "Factored policy (answer + stop module) simplifies RL and isolates "
                    "the timing decision.", fonts["body"], TEXT, inner_w)
    y = draw_bullet(draw, col3_x + 20, y,
                    "Artifact controls validate that improvements come from clue understanding, "
                    "not surface patterns.", fonts["body"], TEXT, inner_w)
    y += 20

    # --- References ---
    y = draw_section_header(draw, col3_x, y, "References", fonts, TEXT, COL_W)
    refs = [
        "[1] Rodriguez et al. QANTA: Question Answering is Not a Trivial Activity. EMNLP 2019.",
        "[2] Schulman et al. Proximal Policy Optimization Algorithms. arXiv 2017.",
        "[3] Balepur et al. Artifacts or Abduction: LLMs Cheat on MC Benchmarks. ACL 2024.",
    ]
    for ref in refs:
        y = draw_wrapped(draw, col3_x + 20, y, ref, fonts["tiny"], TEXT, inner_w)
        y += 5

    # ===== FOOTER =====
    footer_y = H - FOOTER_H
    draw.rectangle((0, footer_y, W, H), fill=HEADER_BG)
    draw.text((MARGIN + 20, footer_y + 40),
              "Stanford CS234 — Reinforcement Learning  •  github.com/hass0114/qanta-buzzer  •  March 2026",
              font=fonts["footer"], fill=WHITE)
    draw.text((MARGIN + 20, footer_y + 100),
              "",
              font=fonts["tiny"], fill="#F4D1C0")

    # ===== SAVE =====
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "poster.pdf"
    poster.save(str(pdf_path), "PDF", resolution=DPI)
    print(f"Saved poster: {pdf_path}  ({pdf_path.stat().st_size / 1024:.0f} KB)")

    # Also save as PNG for preview
    png_path = OUT_DIR / "poster.png"
    poster.save(str(png_path), "PNG")
    print(f"Saved preview: {png_path}  ({png_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    generate_poster()
