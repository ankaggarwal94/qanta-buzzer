#!/usr/bin/env python3
"""Generate a course-compliant project poster for the Quiz Bowl buzzer.

The CS234 poster guidance recommends a roughly 20"x30" landscape poster with:
- large fonts
- short bullets instead of paragraphs
- a visual-first layout
- a concise subset of the full report

This script generates a single-page 30"x20" landscape poster at 150 DPI and
reuses a few rendered slide frames as figures so the result stays visual.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Layout constants (30 x 20 inches @ 150 DPI)
# ---------------------------------------------------------------------------
DPI = 150
W, H = 30 * DPI, 20 * DPI  # 4500 x 3000 px

MARGIN = 72
COL_GAP = 54
HEADER_H = 280
FOOTER_H = 84

BODY_TOP = HEADER_H + MARGIN
BODY_BOTTOM = H - FOOTER_H - MARGIN
COL_W = (W - 2 * MARGIN - 2 * COL_GAP) // 3

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
BG = "#F7F5F2"
WHITE = "#FFFFFF"
TEXT = "#2E2D29"
TEXT_SOFT = "#5F5A55"
BORDER = "#D8D2CB"

CARD_BG = "#FCFBF8"
HEADER_BG = "#8C1515"
ACCENT = "#8C1515"
ACCENT2 = "#007C92"
ACCENT3 = "#E98300"
ACCENT4 = "#53284F"
GREEN = "#008566"
BLUE = "#345BD3"
BLUE_SOFT = "#E5EDFF"
RED_SOFT = "#FBE7E3"
GOLD_SOFT = "#FFF2D8"
MINT_SOFT = "#E8F7F1"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
FRAMES = ROOT / "generated" / "quizbowl_mc_stopping_data_driven" / "frames"
REPORT_PATH = ROOT / "artifacts" / "smoke" / "evaluation_report.json"
OUT_DIR = ROOT / "generated" / "quizbowl_mc_stopping_data_driven"


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------
def _try_fonts(names: List[str], size: int):
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def get_fonts():
    return {
        "title": _try_fonts(["Helvetica-Bold", "Arial Bold", "DejaVuSans-Bold"], 84),
        "subtitle": _try_fonts(["Helvetica", "Arial", "DejaVuSans"], 36),
        "author": _try_fonts(["Helvetica", "Arial", "DejaVuSans"], 28),
        "section": _try_fonts(["Helvetica-Bold", "Arial Bold", "DejaVuSans-Bold"], 34),
        "body": _try_fonts(["Helvetica", "Arial", "DejaVuSans"], 27),
        "body_bold": _try_fonts(["Helvetica-Bold", "Arial Bold", "DejaVuSans-Bold"], 27),
        "small": _try_fonts(["Helvetica", "Arial", "DejaVuSans"], 21),
        "small_bold": _try_fonts(["Helvetica-Bold", "Arial Bold", "DejaVuSans-Bold"], 21),
        "tiny": _try_fonts(["Helvetica", "Arial", "DejaVuSans"], 17),
        "footer": _try_fonts(["Helvetica", "Arial", "DejaVuSans"], 18),
    }


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_eval_report() -> Dict:
    if not REPORT_PATH.exists():
        return {}
    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def best_baseline_result(report: Dict) -> Tuple[str, Dict]:
    summary = report.get("baseline_summary", {})
    best_label = "baseline"
    best_metrics: Dict = {}
    best_sq = float("-inf")

    for method, payload in summary.items():
        if isinstance(payload, dict) and payload and all(isinstance(v, dict) for v in payload.values()):
            for threshold, metrics in payload.items():
                sq = metrics.get("mean_sq")
                if isinstance(sq, (int, float)) and sq > best_sq:
                    best_sq = float(sq)
                    best_label = f"{method} @ {threshold}"
                    best_metrics = metrics
        elif isinstance(payload, dict):
            sq = payload.get("mean_sq")
            if isinstance(sq, (int, float)) and sq > best_sq:
                best_sq = float(sq)
                best_label = method
                best_metrics = payload

    return best_label, best_metrics


def fmt_pct(value) -> str:
    try:
        return f"{100 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def fmt_num(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def rounded(draw: ImageDraw.ImageDraw, xy, radius: int, fill: str, outline: str | None = None, width: int = 1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def measure(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        if measure(draw, test, font)[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font,
    fill: str,
    max_width: int,
    line_spacing: float = 1.25,
) -> int:
    for line in wrap_text(draw, text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        _, height = measure(draw, line, font)
        y += int(height * line_spacing)
    return y


def draw_bullet(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font,
    fill: str,
    max_width: int,
    bullet: str = "•",
) -> int:
    bullet_width = measure(draw, bullet + " ", font)[0]
    draw.text((x, y), bullet, font=font, fill=fill)
    lines = wrap_text(draw, text, font, max_width - bullet_width)
    for line in lines:
        draw.text((x + bullet_width, y), line, font=font, fill=fill)
        _, height = measure(draw, line, font)
        y += int(height * 1.25)
    return y + 4


def draw_section_header(draw: ImageDraw.ImageDraw, x: int, y: int, title: str, width: int, color: str, fonts) -> int:
    rounded(draw, (x, y, x + width, y + 54), 12, color)
    draw.text((x + 18, y + 8), title, font=fonts["section"], fill=WHITE)
    return y + 72


def draw_card(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, height: int, title: str, color: str, fonts):
    rounded(draw, (x, y, x + width, y + height), 18, CARD_BG, outline=BORDER, width=2)
    draw.text((x + 18, y + 16), title, font=fonts["small_bold"], fill=color)


def paste_frame(poster: Image.Image, frame_num: int, x: int, y: int, target_w: int) -> int:
    frame_path = FRAMES / f"frame_{frame_num:02d}.png"
    if not frame_path.exists():
        return y
    image = Image.open(frame_path).convert("RGBA")
    scale = target_w / image.width
    resized = image.resize((target_w, int(image.height * scale)), Image.LANCZOS)
    bg = Image.new("RGB", resized.size, WHITE)
    bg.paste(resized, mask=resized.split()[3])
    poster.paste(bg, (x, y))
    return y + resized.height


def scaled_frame_height(frame_num: int, target_w: int) -> int:
    frame_path = FRAMES / f"frame_{frame_num:02d}.png"
    if not frame_path.exists():
        return 0
    with Image.open(frame_path) as image:
        return int(image.height * (target_w / image.width))


def draw_metric_chart(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    baseline_label: str,
    baseline_metrics: Dict,
    ppo_metrics: Dict,
    fonts,
) -> None:
    rounded(draw, (x, y, x + width, y + height), 18, CARD_BG, outline=BORDER, width=2)
    draw.text((x + 18, y + 14), "Smoke comparison", font=fonts["small_bold"], fill=ACCENT3)
    draw.text((x + 18, y + 48), baseline_label, font=fonts["tiny"], fill=BLUE)
    draw.text((x + 220, y + 48), "PPO checkpoint", font=fonts["tiny"], fill=ACCENT4)
    draw.text((x + width - 130, y + 48), "n = 44", font=fonts["tiny"], fill=TEXT_SOFT)

    metrics = [
        ("mean S_q", baseline_metrics.get("mean_sq"), ppo_metrics.get("mean_sq"), "higher is better"),
        ("buzz accuracy", baseline_metrics.get("buzz_accuracy"), ppo_metrics.get("buzz_accuracy"), "higher is better"),
        ("ECE", baseline_metrics.get("ece"), ppo_metrics.get("ece"), "lower is better"),
    ]

    row_y = y + 90
    label_w = 210
    bar_w = width - label_w - 140
    for label, base_val, ppo_val, note in metrics:
        base_num = float(base_val or 0.0)
        ppo_num = float(ppo_val or 0.0)
        max_num = max(base_num, ppo_num, 1e-6)

        draw.text((x + 18, row_y), label, font=fonts["small_bold"], fill=TEXT)
        draw.text((x + width - 160, row_y), note, font=fonts["tiny"], fill=TEXT_SOFT)

        top = row_y + 34
        left = x + label_w
        rounded(draw, (left, top, left + bar_w, top + 22), 8, "#EEE9E2")
        rounded(draw, (left, top + 32, left + bar_w, top + 54), 8, "#EEE9E2")

        base_len = int(bar_w * (base_num / max_num))
        ppo_len = int(bar_w * (ppo_num / max_num))
        rounded(draw, (left, top, left + max(base_len, 8), top + 22), 8, BLUE_SOFT, outline=BLUE, width=2)
        rounded(draw, (left, top + 32, left + max(ppo_len, 8), top + 54), 8, RED_SOFT, outline=ACCENT4, width=2)

        draw.text((left + 10, top + 1), fmt_num(base_num), font=fonts["tiny"], fill=BLUE)
        draw.text((left + 10, top + 33), fmt_num(ppo_num), font=fonts["tiny"], fill=ACCENT4)
        row_y += 98

    draw.text(
        (x + 18, y + height - 34),
        "Use the smoke slice as a sanity check, not a final leaderboard.",
        font=fonts["tiny"],
        fill=TEXT_SOFT,
    )


def draw_footer(draw: ImageDraw.ImageDraw, fonts) -> None:
    y = H - FOOTER_H
    draw.rectangle((0, y, W, H), fill=HEADER_BG)
    draw.text(
        (MARGIN, y + 28),
        "Stanford CS234 • Reinforcement Learning • github.com/hass0114/qanta-buzzer • March 2026",
        font=fonts["footer"],
        fill=WHITE,
    )


# ---------------------------------------------------------------------------
# Main poster
# ---------------------------------------------------------------------------
def generate_poster() -> None:
    report = load_eval_report()
    baseline_label, baseline_metrics = best_baseline_result(report)
    full_eval = report.get("full_eval", {})
    ppo_metrics = report.get("ppo_summary", {})
    controls = report.get("controls", {})

    fonts = get_fonts()
    poster = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(poster)

    # Header
    draw.rectangle((0, 0, W, HEADER_H), fill=HEADER_BG)
    draw.text((MARGIN, 48), "Quiz Bowl RL Buzzer", font=fonts["title"], fill=WHITE)
    draw.text(
        (MARGIN, 140),
        "Learning when to buzz in multiple-choice pyramidal quiz bowl",
        font=fonts["subtitle"],
        fill="#F4D1C0",
    )
    draw.text(
        (MARGIN, 192),
        "Kathleen Weng  •  Imran Hassan  •  Ankit Aggarwal",
        font=fonts["author"],
        fill=WHITE,
    )
    draw.text(
        (MARGIN, 228),
        "Poster uses a concise subset of the report plus visuals from the slide deck.",
        font=fonts["small"],
        fill="#F4D1C0",
    )
    draw.text((W - 860, 228), "github.com/hass0114/qanta-buzzer", font=fonts["small"], fill="#F4D1C0")

    col1_x = MARGIN
    col2_x = MARGIN + COL_W + COL_GAP
    col3_x = MARGIN + 2 * (COL_W + COL_GAP)
    inner_w = COL_W - 36

    # Column 1: problem / setup
    y1 = BODY_TOP
    y1 = draw_section_header(draw, col1_x, y1, "Problem + Setup", COL_W, ACCENT, fonts)
    y1 = draw_bullet(
        draw,
        col1_x + 8,
        y1,
        "Quiz bowl reveals clues incrementally, so the agent must decide when to act, not only what to answer.",
        fonts["body"],
        TEXT,
        inner_w,
    )
    y1 = draw_bullet(
        draw,
        col1_x + 8,
        y1,
        "We study a controlled multiple-choice version to isolate the stopping decision and make evaluation reproducible.",
        fonts["body"],
        TEXT,
        inner_w,
    )
    y1 = draw_bullet(
        draw,
        col1_x + 8,
        y1,
        "At each prefix the policy chooses WAIT or BUZZ(answer) under rewards that trade correctness against timing.",
        fonts["body"],
        TEXT,
        inner_w,
    )
    y1 += 10

    draw_card(draw, col1_x, y1, COL_W, 258, "Why this setting works", ACCENT2, fonts)
    box_y = y1 + 56
    box_x = col1_x + 18
    box_w = COL_W - 36
    box_y = draw_bullet(
        draw,
        box_x,
        box_y,
        "Open-ended QA mixes buzzing with aliasing and grading noise. MC keeps the gold answer in the set and focuses the task.",
        fonts["small"],
        TEXT,
        box_w,
    )
    box_y = draw_bullet(
        draw,
        box_x,
        box_y,
        "Distractors are screened for alias collisions, question overlap, and easy surface artifacts.",
        fonts["small"],
        TEXT,
        box_w,
    )
    box_y = draw_bullet(
        draw,
        box_x,
        box_y,
        "We report system score, buzz accuracy, timing, and control tests instead of raw accuracy alone.",
        fonts["small"],
        TEXT,
        box_w,
    )
    y1 += 286

    draw_card(draw, col1_x, y1, COL_W, 240, "POMDP view", ACCENT4, fonts)
    draw.text((col1_x + 18, y1 + 60), "Observation", font=fonts["small_bold"], fill=ACCENT4)
    draw.text((col1_x + 180, y1 + 60), "clue prefix h_t + answer set C", font=fonts["small"], fill=TEXT)
    draw.text((col1_x + 18, y1 + 102), "Action", font=fonts["small_bold"], fill=ACCENT4)
    draw.text((col1_x + 180, y1 + 102), "WAIT or BUZZ(i), i in {A, B, C, D}", font=fonts["small"], fill=TEXT)
    draw.text((col1_x + 18, y1 + 144), "Policy", font=fonts["small_bold"], fill=ACCENT4)
    draw.text((col1_x + 180, y1 + 144), "p(WAIT) = 1 - p_buzz;  p(BUZZ i) = p_buzz * p_ans(i)", font=fonts["small"], fill=TEXT)
    draw.text((col1_x + 18, y1 + 186), "Goal", font=fonts["small_bold"], fill=ACCENT4)
    draw.text((col1_x + 180, y1 + 186), "stop when committing beats waiting", font=fonts["small"], fill=TEXT)
    y1 += 270

    draw_card(draw, col1_x, y1, COL_W, 214, "Contributions", GREEN, fonts)
    cy = y1 + 56
    cy = draw_bullet(
        draw,
        col1_x + 18,
        cy,
        "An MC quiz-bowl environment that turns buzzing into a clean sequential decision problem.",
        fonts["small"],
        TEXT,
        box_w,
    )
    cy = draw_bullet(
        draw,
        col1_x + 18,
        cy,
        "A factored policy view: one module answers, another decides when the evidence is strong enough to buzz.",
        fonts["small"],
        TEXT,
        box_w,
    )
    draw_bullet(
        draw,
        col1_x + 18,
        cy,
        "Control evaluations that test whether the agent uses clue content instead of answer-set artifacts.",
        fonts["small"],
        TEXT,
        box_w,
    )
    y1 += 244

    example_w = COL_W - 40
    example_h = scaled_frame_height(9, example_w)
    example_card_h = 56 + example_h + 74
    draw_card(draw, col1_x, y1, COL_W, example_card_h, "Running example", BLUE, fonts)
    frame_y = y1 + 56
    frame_y = paste_frame(poster, 9, col1_x + 20, frame_y, example_w)
    draw.text(
        (col1_x + 20, frame_y + 8),
        "The Markov tossup shows how repeated cues collapse the posterior toward one answer.",
        font=fonts["small"],
        fill=TEXT_SOFT,
    )

    # Column 2: method + visuals
    y2 = BODY_TOP
    y2 = draw_section_header(draw, col2_x, y2, "Method", COL_W, ACCENT2, fonts)
    y2 = draw_bullet(
        draw,
        col2_x + 8,
        y2,
        "Build MC prefixes from QANTA tossups and score each option with profile-based likelihood models.",
        fonts["body"],
        TEXT,
        inner_w,
    )
    y2 = draw_bullet(
        draw,
        col2_x + 8,
        y2,
        "Convert beliefs into structured observations: posterior mass, entropy, step index, and similarity features.",
        fonts["body"],
        TEXT,
        inner_w,
    )
    y2 = draw_bullet(
        draw,
        col2_x + 8,
        y2,
        "Warm-start with supervised answer learning, then fine-tune a stopping policy with PPO.",
        fonts["body"],
        TEXT,
        inner_w,
    )
    y2 += 14

    figure_w = COL_W - 40
    pipeline_h = scaled_frame_height(31, figure_w)
    pipeline_card_h = 56 + pipeline_h + 70
    draw_card(draw, col2_x, y2, COL_W, pipeline_card_h, "Training pipeline", BLUE, fonts)
    frame_y = y2 + 56
    frame_y = paste_frame(poster, 31, col2_x + 20, frame_y, figure_w)
    draw.text(
        (col2_x + 20, frame_y + 10),
        "Data prep, baseline sweeps, and PPO fine-tuning are implemented as separate stages.",
        font=fonts["small"],
        fill=TEXT_SOFT,
    )
    y2 += pipeline_card_h + 18

    decision_h = scaled_frame_height(13, figure_w)
    decision_card_h = 56 + decision_h + 104
    draw_card(draw, col2_x, y2, COL_W, decision_card_h, "Decision dynamics", BLUE, fonts)
    frame_y = y2 + 56
    frame_y = paste_frame(poster, 13, col2_x + 20, frame_y, figure_w)
    draw.text(
        (col2_x + 20, frame_y + 8),
        "As more clues arrive, the posterior sharpens and the stop decision becomes easier.",
        font=fonts["small"],
        fill=TEXT_SOFT,
    )
    draw.text(
        (col2_x + 20, frame_y + 40),
        "Optimal rule: buzz once the value of committing exceeds the value of waiting.",
        font=fonts["small_bold"],
        fill=GREEN,
    )

    # Column 3: experiments / results / analysis
    y3 = BODY_TOP
    y3 = draw_section_header(draw, col3_x, y3, "Experiments + Results", COL_W, ACCENT3, fonts)
    y3 = draw_bullet(
        draw,
        col3_x + 8,
        y3,
        "Task: given a clue prefix and four answer options, choose WAIT or BUZZ(answer).",
        fonts["body"],
        TEXT,
        inner_w,
    )
    y3 = draw_bullet(
        draw,
        col3_x + 8,
        y3,
        f"Smoke evaluation snapshot: best baseline is {baseline_label}, while the PPO checkpoint tests whether RL changes timing behavior.",
        fonts["body"],
        TEXT,
        inner_w,
    )
    y3 += 12

    draw_metric_chart(draw, col3_x, y3, COL_W, 410, baseline_label, baseline_metrics, ppo_metrics, fonts)
    y3 += 440

    draw_card(draw, col3_x, y3, COL_W, 260, "Analysis + controls", ACCENT2, fonts)
    ay = y3 + 56
    ay = draw_bullet(
        draw,
        col3_x + 18,
        ay,
        f"Choices-only accuracy is {fmt_pct(controls.get('choices_only', {}).get('accuracy'))}, below the 25% chance level.",
        fonts["small"],
        TEXT,
        inner_w,
    )
    ay = draw_bullet(
        draw,
        col3_x + 18,
        ay,
        f"Shuffle leaves mean S_q at {fmt_num(controls.get('shuffle', {}).get('mean_sq'))}, so content still matters after clue reordering.",
        fonts["small"],
        TEXT,
        inner_w,
    )
    draw_bullet(
        draw,
        col3_x + 18,
        ay,
        f"Alias substitution produces mean S_q {fmt_num(controls.get('alias_substitution', {}).get('mean_sq'))}; this path is still a limited control because alias lookup coverage is sparse.",
        fonts["small"],
        TEXT,
        inner_w,
    )
    y3 += 292

    draw_card(draw, col3_x, y3, COL_W, 202, "Conclusions", GREEN, fonts)
    cy = y3 + 56
    cy = draw_bullet(
        draw,
        col3_x + 18,
        cy,
        "The project reframes quiz-bowl buzzing as an optimal stopping problem with explicit uncertainty.",
        fonts["small"],
        TEXT,
        inner_w,
    )
    cy = draw_bullet(
        draw,
        col3_x + 18,
        cy,
        "The factored policy gives a clean way to separate answering from timing.",
        fonts["small"],
        TEXT,
        inner_w,
    )
    draw_bullet(
        draw,
        col3_x + 18,
        cy,
        f"Current smoke full-eval snapshot: mean S_q {fmt_num(full_eval.get('mean_sq'))}, buzz accuracy {fmt_pct(full_eval.get('buzz_accuracy'))}, mean step {fmt_num(full_eval.get('mean_buzz_step'), 2)}.",
        fonts["small"],
        TEXT,
        inner_w,
    )
    y3 += 232

    draw_card(draw, col3_x, y3, COL_W, 154, "References", ACCENT4, fonts)
    refs = [
        "Rodriguez et al. (2019) QANTA and incremental QA.",
        "Schulman et al. (2017) Proximal Policy Optimization.",
        "Boyd-Graber and Daume (2012/2013) Bayesian quiz-bowl buzzing.",
    ]
    ry = y3 + 52
    for ref in refs:
        ry = draw_bullet(draw, col3_x + 18, ry, ref, fonts["tiny"], TEXT, inner_w, bullet="-")

    draw_footer(draw, fonts)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "poster.pdf"
    png_path = OUT_DIR / "poster.png"
    poster.save(str(pdf_path), "PDF", resolution=DPI)
    poster.save(str(png_path), "PNG")

    print(f"Saved poster: {pdf_path} ({pdf_path.stat().st_size / 1024:.0f} KB)")
    print(f"Saved preview: {png_path} ({png_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    generate_poster()
