#!/usr/bin/env python3
"""Generate a readable CS234 poster for the Quiz Bowl buzzer project.

Three-column landscape layout (30 x 20 in @ 150 dpi) with cross-platform
font loading and generous minimum font sizes for poster-session readability.
All content from the original poster is preserved.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Canvas: 30 x 20 inches at 150 dpi
# ---------------------------------------------------------------------------
DPI = 150
W, H = 30 * DPI, 20 * DPI  # 4500 x 3000 px
MARGIN = 70
COL_GAP = 40
CARD_GAP = 28
HEADER_H = 340
FOOTER_H = 70

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BG = "#F6F3EE"
WHITE = "#FFFFFF"
PANEL = WHITE
PANEL_SOFT = "#FBF9F5"
TEXT = "#1F2933"
TEXT_SOFT = "#5F6B76"
BORDER = "#D7D1C8"
STANFORD_RED = "#8C1515"
NAVY = "#2F4EA1"
NAVY_DARK = "#243B7A"
BLUE = "#3B63D0"
BLUE_SOFT = "#E8EEFF"
PURPLE = "#7A57E2"
PURPLE_SOFT = "#EFE7FF"
GREEN = "#1E8E5A"
GREEN_SOFT = "#E7F7EE"
ORANGE = "#D8881F"
ORANGE_SOFT = "#FFF1DD"
GOLD = "#B57900"
GOLD_SOFT = "#FFF4DA"
RED_SOFT = "#FCE6E3"
GRID = "#E8E3DB"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "artifacts" / "smoke" / "evaluation_report.json"
OUT_DIR = ROOT / "generated" / "quizbowl_mc_stopping_data_driven"

# ---------------------------------------------------------------------------
# Running-example content (from generate_presentation.py / frame deck)
# ---------------------------------------------------------------------------
EXAMPLE_SOURCE = "Example tossup: QBReader 2022 ARCADIA 08/11"
EXAMPLE_CHOICES = [
    ("A", "Andrey Andreyevich Markov"),
    ("B", "Leonhard Euler"),
    ("C", "Carl Friedrich Gauss"),
    ("D", "Augustin-Louis Cauchy"),
]
TOSSUP_SENTENCES = [
    [
        ("Given a set of nodes ", False),
        ("named for this person", True),
        (", a node is conditionally independent from the rest of "
         "a Bayesian network.", False),
    ],
    [
        ("The Baum-Welch algorithm is used to train a type of model ", False),
        ("named for this person", True),
        (" that is used for multiple sequence alignment.", False),
    ],
    [
        ('The initial "burn-in" states of a process ', False),
        ("named for this person", True),
        (" are discarded in methods like Gibbs sampling.", False),
    ],
    [
        ("The Metropolis-Hastings algorithm approximates an unknown "
         "distribution as the (*) stationary distribution of a process ", False),
        ("named for this person", True),
        (".", False),
    ],
    [
        ("Monte Carlo methods often use processes ", False),
        ("named for this person", True),
        (" that have stochastic transition matrices.", False),
    ],
    [
        ('Dynamic programming is used to decode "hidden" models ', False),
        ("named for this person", True),
        (".", False),
    ],
    [
        ("The next state of a random process ", False),
        ("named for this person", True),
        (" depends only on the current state.", False),
    ],
    [
        ("For 10 points, what Russian mathematician names a type of ", False),
        ('memoryless "chain?"', True),
    ],
]
TOSSUP_ANSWER = "Andrey Andreyevich Markov"
EXAMPLE_PREFIXES = [
    "blankets + HMM",
    "burn-in / Gibbs",
    "MC / stochastic",
    "memoryless prop.",
    'giveaway: "chain"',
]
EXAMPLE_POSTERIORS = [
    [0.42, 0.24, 0.21, 0.13],
    [0.56, 0.19, 0.15, 0.10],
    [0.68, 0.14, 0.11, 0.07],
    [0.82, 0.08, 0.06, 0.04],
    [0.91, 0.04, 0.03, 0.02],
]
EXAMPLE_DECISIONS = ["WAIT", "WAIT", "WAIT", "BUZZ", "BUZZ"]
ACT_NOW_VALUES = [0.18, 0.38, 0.49, 0.63, 0.74]
WAIT_VALUES = [0.64, 0.58, 0.52, 0.40, 0.28]

# ---------------------------------------------------------------------------
# Cross-platform font loading (macOS + Linux)
# ---------------------------------------------------------------------------
_FONT_PATHS = {
    "regular": [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ],
    "bold": [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ],
}
_TTC_FALLBACKS = [
    ("/System/Library/Fonts/Helvetica.ttc", 0, 1),
    ("/System/Library/Fonts/HelveticaNeue.ttc", 0, 1),
]

_font_warning_printed = False


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    global _font_warning_printed
    key = "bold" if bold else "regular"
    for p in _FONT_PATHS[key]:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    for ttc_path, reg_idx, bold_idx in _TTC_FALLBACKS:
        if Path(ttc_path).exists():
            try:
                return ImageFont.truetype(
                    ttc_path, size=size, index=bold_idx if bold else reg_idx
                )
            except Exception:
                continue
    if not _font_warning_printed:
        print("WARNING: No system TrueType font found — falling back to PIL default")
        _font_warning_printed = True
    return ImageFont.load_default()


FONTS = {
    "title": get_font(140, bold=True),
    "subtitle": get_font(56),
    "authors": get_font(44),
    "section": get_font(48, bold=True),
    "body": get_font(40),
    "body_bold": get_font(40, bold=True),
    "small": get_font(36),
    "small_bold": get_font(36, bold=True),
    "detail": get_font(32),
    "detail_bold": get_font(32, bold=True),
    "caption": get_font(28),
    "caption_bold": get_font(28, bold=True),
}

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def measure(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(
    draw: ImageDraw.ImageDraw, text: str, font, max_width: int
) -> List[str]:
    if not text:
        return [""]
    lines: List[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            lines.append("")
            continue
        words = para.split()
        cur = words[0]
        for word in words[1:]:
            trial = cur + " " + word
            if measure(draw, trial, font)[0] <= max_width:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


def fit_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: Tuple[int, int, int, int],
    *,
    max_size: int,
    min_size: int,
    bold: bool = False,
    line_gap: int = 6,
):
    x0, y0, x1, y1 = box
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    for size in range(max_size, min_size - 1, -1):
        font = get_font(size, bold=bold)
        lines = wrap_text(draw, text, font, bw)
        _, line_h = measure(draw, "Ag", font)
        total_h = len(lines) * line_h + max(0, len(lines) - 1) * line_gap
        if total_h <= bh and all(measure(draw, ln, font)[0] <= bw for ln in lines):
            return font, lines, total_h
    font = get_font(min_size, bold=bold)
    lines = wrap_text(draw, text, font, bw)
    _, line_h = measure(draw, "Ag", font)
    total_h = len(lines) * line_h + max(0, len(lines) - 1) * line_gap
    return font, lines, total_h


def draw_text_fit(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    *,
    fill=TEXT,
    max_size: int = 36,
    min_size: int = 24,
    bold: bool = False,
    align: str = "left",
    valign: str = "top",
    line_gap: int = 6,
):
    x0, y0, x1, y1 = box
    font, lines, total_h = fit_wrapped_text(
        draw, text, box,
        max_size=max_size, min_size=min_size, bold=bold, line_gap=line_gap,
    )
    _, line_h = measure(draw, "Ag", font)
    if valign == "center":
        y = y0 + max(0, ((y1 - y0) - total_h) // 2)
    else:
        y = y0
    for line in lines:
        lw, _ = measure(draw, line, font)
        if align == "center":
            x = x0 + max(0, ((x1 - x0) - lw) // 2)
        elif align == "right":
            x = x1 - lw
        else:
            x = x0
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h + line_gap
    return y


def draw_bullets(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    items: Sequence[str],
    *,
    font=None,
    fill=TEXT,
    bullet_fill=STANFORD_RED,
    bullet_radius: int = 7,
    gap_after: int = 12,
) -> int:
    if font is None:
        font = FONTS["body"]
    x0, y0, x1, y1 = box
    bullet_pad = 30
    y = y0
    for item in items:
        lines = wrap_text(draw, item, font, (x1 - x0) - bullet_pad)
        _, line_h = measure(draw, "Ag", font)
        cy = y + line_h // 2 + 2
        draw.ellipse(
            (x0, cy - bullet_radius, x0 + 2 * bullet_radius, cy + bullet_radius),
            fill=bullet_fill,
        )
        for line in lines:
            draw.text((x0 + bullet_pad, y), line, font=font, fill=fill)
            y += line_h + 4
        y += gap_after
        if y > y1:
            break
    return y


# ---------------------------------------------------------------------------
# Primitive drawing helpers
# ---------------------------------------------------------------------------

CARD_HEADER_H = 76


def rounded(draw: ImageDraw.ImageDraw, xy, fill, outline=None, width=1, radius=22):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_card(
    draw: ImageDraw.ImageDraw,
    rect,
    title: str,
    accent: str,
    *,
    fill=PANEL,
    title_bg=None,
):
    x0, y0, x1, _y1 = rect
    rounded(draw, rect, fill=fill, outline=BORDER, width=2, radius=26)
    tb = title_bg or accent
    rounded(
        draw, (x0, y0, x1, y0 + CARD_HEADER_H),
        fill=tb, outline=tb, width=1, radius=24,
    )
    draw.rectangle(
        (x0 + 2, y0 + CARD_HEADER_H - 12, x1 - 2, y0 + CARD_HEADER_H), fill=tb,
    )
    draw.text((x0 + 22, y0 + 14), title, font=FONTS["section"], fill=WHITE)


def draw_subcard(
    draw: ImageDraw.ImageDraw, rect, title: str, color: str, fill=PANEL_SOFT,
):
    x0, y0, _x1, _y1 = rect
    rounded(draw, rect, fill=fill, outline=BORDER, width=2, radius=18)
    draw.text((x0 + 18, y0 + 12), title, font=FONTS["small_bold"], fill=color)


def draw_chip(
    draw: ImageDraw.ImageDraw, rect, text: str, fill, outline, text_fill, *, bold=False,
):
    rounded(draw, rect, fill=fill, outline=outline, width=2, radius=16)
    draw_text_fit(
        draw, (rect[0] + 8, rect[1] + 6, rect[2] - 8, rect[3] - 6),
        text, fill=text_fill, max_size=30, min_size=20, bold=bold,
        align="center", valign="center", line_gap=3,
    )


def arrow(
    draw: ImageDraw.ImageDraw,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: str,
    width: int = 4,
):
    draw.line([start, end], fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if dx == 0 and dy == 0:
        return
    ang = math.atan2(dy, dx)
    ah = 16
    left = (end[0] - ah * math.cos(ang - 0.5), end[1] - ah * math.sin(ang - 0.5))
    right = (end[0] - ah * math.cos(ang + 0.5), end[1] - ah * math.sin(ang + 0.5))
    draw.polygon([end, left, right], fill=color)


def fmt_pct(v) -> str:
    try:
        return f"{100 * float(v):.1f}%"
    except Exception:
        return "n/a"


def fmt_num(v, digits: int = 3) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return "n/a"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_report() -> Dict:
    if not REPORT_PATH.exists():
        return {}
    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def baseline_points(report: Dict) -> List[Dict]:
    out: List[Dict] = []
    summary = report.get("baseline_summary", {})
    for method, payload in summary.items():
        if isinstance(payload, dict) and payload and all(
            isinstance(v, dict) for v in payload.values()
        ):
            for threshold, metrics in payload.items():
                out.append({
                    "label": f"{method}@{threshold}",
                    "family": method,
                    "threshold": threshold,
                    "mean_sq": float(metrics.get("mean_sq", 0.0)),
                    "mean_buzz_step": float(metrics.get("mean_buzz_step", 0.0)),
                    "buzz_accuracy": float(metrics.get("buzz_accuracy", 0.0)),
                    "mean_reward_like": float(metrics.get("mean_reward_like", 0.0)),
                })
        elif isinstance(payload, dict):
            out.append({
                "label": method,
                "family": method,
                "threshold": None,
                "mean_sq": float(payload.get("mean_sq", 0.0)),
                "mean_buzz_step": float(payload.get("mean_buzz_step", 0.0)),
                "buzz_accuracy": float(payload.get("buzz_accuracy", 0.0)),
                "mean_reward_like": float(payload.get("mean_reward_like", 0.0)),
            })
    return out


def pick_points(report: Dict) -> Dict[str, Dict]:
    points = baseline_points(report)
    seq05 = max(
        (
            p for p in points
            if p["family"] == "sequential_bayes" and p["threshold"] == "0.5"
        ),
        key=lambda x: x["mean_sq"],
        default=None,
    )
    always_final = next(
        (p for p in points if p["family"] == "always_final"), None,
    )
    ppo = report.get("ppo_summary", {})
    return {
        "sequential_bayes_05": seq05,
        "always_final": always_final,
        "ppo": {
            "label": "PPO smoke",
            "family": "ppo",
            "mean_sq": float(ppo.get("mean_sq", 0.0)),
            "mean_buzz_step": float(ppo.get("mean_buzz_step", 0.0)),
            "buzz_accuracy": float(ppo.get("buzz_accuracy", 0.0)),
            "mean_reward_like": float(ppo.get("mean_reward_like", 0.0)),
            "ece": float(ppo.get("ece", 0.0)),
            "brier": float(ppo.get("brier", 0.0)),
        },
    }


# ---------------------------------------------------------------------------
# Column 1 panels
# ---------------------------------------------------------------------------


def draw_problem_card(draw: ImageDraw.ImageDraw, rect):
    x0, y0, x1, y1 = rect
    draw_card(draw, rect, "Problem + design choice", STANFORD_RED)
    bx0, bx1 = x0 + 24, x1 - 24
    p = CARD_HEADER_H + 16

    y = draw_bullets(
        draw, (bx0, y0 + p, bx1, y0 + p + 300),
        [
            "Clues arrive incrementally, so the system must decide "
            "when to buzz — timing matters beyond answer selection.",
            "Early buzzes risk mistakes; late buzzes waste strategic value.",
            "A fixed 4-choice setting makes the stopping problem reproducible.",
        ],
        font=FONTS["small"], fill=TEXT,
        bullet_fill=STANFORD_RED, bullet_radius=7, gap_after=14,
    )

    py0 = y + 8
    py1 = py0 + 190
    draw_subcard(draw, (bx0, py0, bx1, py1), "POMDP at a glance", NAVY)
    fy = py0 + 52
    for label, value in [
        ("Observation", "prefix h_t + choice set C"),
        ("Actions", "WAIT or BUZZ(i)"),
        ("Goal", "buzz when acting beats waiting"),
    ]:
        draw_text_fit(
            draw, (bx0 + 20, fy, bx0 + 240, fy + 38),
            label, max_size=30, min_size=24, bold=True, fill=NAVY,
        )
        draw_text_fit(
            draw, (bx0 + 248, fy, bx1 - 20, fy + 38),
            value, max_size=30, min_size=24, fill=TEXT,
        )
        fy += 44

    my0 = py1 + 14
    my1 = my0 + 270
    draw_subcard(draw, (bx0, my0, bx1, my1), "Why multiple choice?", GREEN)
    draw_bullets(
        draw, (bx0 + 18, my0 + 52, bx1 - 18, my1 - 60),
        [
            "Removes aliasing and grading noise.",
            "Keeps the gold answer inside the action space.",
            "Allows cleaner baseline and control comparisons.",
            "Uses explicit anti-artifact distractor checks.",
        ],
        font=FONTS["detail"], fill=TEXT,
        bullet_fill=GREEN, bullet_radius=6, gap_after=10,
    )

    chip_y = my1 - 50
    chips = ["alias collisions", "token overlap", "length ratio", "question overlap"]
    cw_chip = (bx1 - bx0 - 18 - 3 * 10) // 4
    for i, label in enumerate(chips):
        cx = bx0 + 18 + i * (cw_chip + 10)
        draw_chip(
            draw, (cx, chip_y, cx + cw_chip, chip_y + 42),
            label, GREEN_SOFT, GREEN, GREEN,
        )

    mg0 = my1 + 14
    mg1 = y1 - 24
    draw_subcard(draw, (bx0, mg0, bx1, mg1), "Metrics glossary", ORANGE)
    metrics_defs = [
        ("S_q", "QANTA system score; rewards early correct buzzes"),
        ("acc", "buzz accuracy — fraction of buzzes on the gold answer"),
        ("step", "mean buzz step — average prefix index at buzz time"),
        ("rew", "reward-like — composite PPO training signal"),
    ]
    row_area = mg1 - mg0 - 44
    row_step = row_area // len(metrics_defs)
    gy = mg0 + 44
    for mlabel, mdesc in metrics_defs:
        draw_text_fit(
            draw, (bx0 + 20, gy, bx0 + 100, gy + row_step - 4),
            mlabel, max_size=28, min_size=18, bold=True, fill=ORANGE,
        )
        draw_text_fit(
            draw, (bx0 + 108, gy, bx1 - 20, gy + row_step - 4),
            mdesc, max_size=26, min_size=16, fill=TEXT,
        )
        gy += row_step


def draw_method_card(draw: ImageDraw.ImageDraw, rect):
    x0, y0, x1, _y1 = rect
    draw_card(draw, rect, "Method", NAVY)
    bx0, bx1 = x0 + 24, x1 - 24
    p = CARD_HEADER_H + 12

    draw_text_fit(
        draw, (bx0, y0 + p, bx1, y0 + p + 70),
        "Factor answer quality and stop timing into separate "
        "components, then combine into WAIT / BUZZ(i).",
        max_size=32, min_size=24, fill=TEXT_SOFT,
    )

    dy = y0 + p + 80
    cx = (bx0 + bx1) // 2
    bw = min(640, bx1 - bx0 - 40)

    ib = (cx - bw // 2, dy, cx + bw // 2, dy + 100)
    rounded(draw, ib, BLUE_SOFT, outline=BLUE, width=3, radius=18)
    draw_text_fit(
        draw, (ib[0] + 14, ib[1] + 8, ib[2] - 14, ib[1] + 44),
        "Incremental clue prefix h_t",
        max_size=30, min_size=22, bold=True, fill=NAVY,
    )
    draw_text_fit(
        draw, (ib[0] + 14, ib[1] + 48, ib[2] - 14, ib[3] - 8),
        "Observed text + 4-choice answer set C",
        max_size=26, min_size=20, fill=TEXT_SOFT,
    )

    split_y = dy + 110
    left_cx = cx - 170
    right_cx = cx + 170
    arrow(draw, (cx, dy + 100), (left_cx, split_y + 24), BLUE, width=4)
    arrow(draw, (cx, dy + 100), (right_cx, split_y + 24), ORANGE, width=4)

    abw = 290
    ab = (left_cx - abw // 2, split_y + 24, left_cx + abw // 2, split_y + 124)
    rounded(draw, ab, PURPLE_SOFT, outline=PURPLE, width=3, radius=16)
    draw_text_fit(
        draw, (ab[0] + 10, ab[1] + 6, ab[2] - 10, ab[1] + 38),
        "Answer model", max_size=28, min_size=20, bold=True, fill=PURPLE,
    )
    draw_text_fit(
        draw, (ab[0] + 10, ab[1] + 40, ab[2] - 10, ab[3] - 6),
        "Outputs p_ans(i | h_t) over A/B/C/D",
        max_size=24, min_size=18, fill=TEXT_SOFT,
    )

    sb = (right_cx - abw // 2, split_y + 24, right_cx + abw // 2, split_y + 124)
    rounded(draw, sb, GOLD_SOFT, outline=ORANGE, width=3, radius=16)
    draw_text_fit(
        draw, (sb[0] + 10, sb[1] + 6, sb[2] - 10, sb[1] + 38),
        "Stop head", max_size=28, min_size=20, bold=True, fill=ORANGE,
    )
    draw_text_fit(
        draw, (sb[0] + 10, sb[1] + 40, sb[2] - 10, sb[3] - 6),
        "Uses posterior / entropy / step features",
        max_size=24, min_size=18, fill=TEXT_SOFT,
    )

    merge_y = split_y + 134
    arrow(draw, (left_cx, split_y + 124), (cx, merge_y + 24), PURPLE, width=4)
    arrow(draw, (right_cx, split_y + 124), (cx, merge_y + 24), ORANGE, width=4)

    ob = (cx - bw // 2, merge_y + 24, cx + bw // 2, merge_y + 124)
    rounded(draw, ob, PANEL_SOFT, outline=GREEN, width=3, radius=18)
    draw_text_fit(
        draw, (ob[0] + 14, ob[1] + 8, ob[2] - 14, ob[1] + 40),
        "Flattened action distribution",
        max_size=28, min_size=22, bold=True, fill=GREEN,
    )
    draw_text_fit(
        draw, (ob[0] + 14, ob[1] + 44, ob[2] - 14, ob[3] - 8),
        'P("WAIT") = 1\u2212p_buzz   P("BUZZ i") = p_buzz \u00d7 p_ans(i)',
        max_size=24, min_size=18, fill=TEXT, align="center",
    )

    strip_y = merge_y + 140
    rounded(
        draw, (bx0, strip_y, bx1, strip_y + 82),
        "#F8F7F4", outline=BORDER, width=2, radius=14,
    )
    steps = [
        ("1", "Build MC prefixes", BLUE_SOFT, BLUE),
        ("2", "Run baselines", PURPLE_SOFT, PURPLE),
        ("3", "Warm-start stop", GOLD_SOFT, ORANGE),
        ("4", "PPO fine-tune", GREEN_SOFT, GREEN),
    ]
    sw = ((bx1 - bx0) - 40 - 3 * 12) // 4
    sx = bx0 + 20
    for i, (num, label, sfill, soutline) in enumerate(steps):
        bx = sx + i * (sw + 12)
        rounded(
            draw, (bx, strip_y + 12, bx + sw, strip_y + 70),
            sfill, outline=soutline, width=2, radius=12,
        )
        draw_chip(
            draw, (bx + 8, strip_y + 20, bx + 46, strip_y + 58),
            num, soutline, soutline, WHITE, bold=True,
        )
        draw_text_fit(
            draw, (bx + 54, strip_y + 20, bx + sw - 8, strip_y + 62),
            label, max_size=24, min_size=18, fill=TEXT, valign="center",
        )


def draw_conclusions_card(draw: ImageDraw.ImageDraw, rect):
    x0, y0, x1, y1 = rect
    draw_card(draw, rect, "Conclusions", STANFORD_RED)
    draw_bullets(
        draw, (x0 + 24, y0 + CARD_HEADER_H + 16, x1 - 24, y1 - 24),
        [
            "With a fixed answer set, buzzing becomes an "
            "optimal-stopping problem over growing evidence.",
            "The Markov example shows posterior mass concentrating "
            "before the final giveaway.",
        ],
        font=FONTS["small"], fill=TEXT,
        bullet_fill=STANFORD_RED, bullet_radius=7, gap_after=14,
    )


def draw_references_card(draw: ImageDraw.ImageDraw, rect):
    x0, y0, x1, y1 = rect
    draw_card(draw, rect, "References", GREEN)
    refs = [
        "Rodriguez et al. (2019). Quizbowl and incremental QA.",
        "QANTA (2024). System score S_q for buzzing.",
        "Sung et al. (2025). ADVSCORE.",
    ]
    y = draw_bullets(
        draw, (x0 + 24, y0 + CARD_HEADER_H + 16, x1 - 24, y1 - 64),
        refs, font=FONTS["detail"], fill=TEXT,
        bullet_fill=GREEN, bullet_radius=5, gap_after=8,
    )
    draw_text_fit(
        draw, (x0 + 24, y - 4, x1 - 24, y1 - 24),
        "Repo: github.com/hass0114/qanta-buzzer",
        max_size=28, min_size=22, fill=TEXT_SOFT,
    )


# ---------------------------------------------------------------------------
# Column 2: running example
# ---------------------------------------------------------------------------


def draw_highlighted_clue(
    draw: ImageDraw.ImageDraw,
    rect,
    left_text: str,
    highlight: str,
    right_text: str,
    *,
    fill=PANEL_SOFT,
    outline=BORDER,
    hl_fill=BLUE_SOFT,
    hl_text=NAVY_DARK,
):
    x0, y0, x1, y1 = rect
    rounded(draw, rect, fill=fill, outline=outline, width=2, radius=14)
    avail = (x1 - x0) - 32
    font = None
    for size in [32, 30, 28, 26, 24]:
        f = get_font(size)
        total = sum(
            measure(draw, t, f)[0] for t in [left_text, highlight, right_text]
        ) + 30
        if total <= avail:
            font = f
            break
    if font is None:
        font = get_font(24)

    _, fh = measure(draw, "Ag", font)
    tx = x0 + 16
    ty = y0 + ((y1 - y0) - fh) // 2
    draw.text((tx, ty), left_text, font=font, fill=TEXT)
    left_w = measure(draw, left_text, font)[0]
    hx0 = tx + left_w
    hpad = 6
    hl_w, hl_h = measure(draw, highlight, font)
    rounded(
        draw,
        (hx0 - 2, ty - 2, hx0 + hl_w + 2 * hpad, ty + hl_h + 2 * hpad - 2),
        hl_fill, outline=None, radius=10,
    )
    draw.text(
        (hx0 + hpad - 2, ty + hpad - 4), highlight, font=font, fill=hl_text,
    )
    rx = hx0 + hl_w + 2 * hpad + 6
    draw.text((rx, ty), right_text, font=font, fill=TEXT)


def draw_tossup_flow(
    draw: ImageDraw.ImageDraw, bx0: int, cy: int, bx1: int,
) -> int:
    """Render all 8 tossup sentences with run badges and anaphoric highlights.

    Returns the y coordinate below the last rendered line.
    """
    font = get_font(22)
    hl_font = get_font(22, bold=True)
    badge_font = get_font(16, bold=True)
    _, line_h = measure(draw, "Ag", font)
    space_w = measure(draw, " ", font)[0]

    badge_colors = [BLUE, BLUE, BLUE, PURPLE, PURPLE, ORANGE, ORANGE, GOLD]

    for run_idx, segments in enumerate(TOSSUP_SENTENCES):
        run_num = run_idx + 1
        bc = badge_colors[run_idx]
        is_giveaway = run_num == len(TOSSUP_SENTENCES)

        badge_r = 14
        badge_x = bx0 + badge_r + 2
        first_line_y = cy

        tokens: list = []
        for text, is_hl in segments:
            for w in text.split():
                if w:
                    tokens.append((w, is_hl))

        text_x0 = bx0 + badge_r * 2 + 14
        cx = text_x0

        for word, is_hl in tokens:
            f = hl_font if is_hl else font
            hl_bg = (
                GOLD_SOFT if (is_giveaway and is_hl)
                else (BLUE_SOFT if is_hl else None)
            )
            hl_color = (
                GOLD if (is_giveaway and is_hl)
                else (NAVY_DARK if is_hl else TEXT)
            )

            w = measure(draw, word, f)[0]

            if word and word[0] in ',.;:?!\u201d' and cx > text_x0:
                cx -= space_w

            if cx + w > bx1 and cx > text_x0:
                cx = text_x0
                cy += line_h + 2

            if hl_bg:
                rounded(
                    draw,
                    (int(cx - 2), cy - 1, int(cx + w + 2), cy + line_h + 1),
                    hl_bg, outline=None, radius=5,
                )

            draw.text((cx, cy), word, font=f, fill=hl_color)
            cx += w + space_w

        badge_cy = first_line_y + line_h // 2
        draw.ellipse(
            (badge_x - badge_r, badge_cy - badge_r,
             badge_x + badge_r, badge_cy + badge_r),
            fill=bc,
        )
        bw_b, bh_b = measure(draw, str(run_num), badge_font)
        draw.text(
            (badge_x - bw_b // 2, badge_cy - bh_b // 2),
            str(run_num), font=badge_font, fill=WHITE,
        )

        cy += line_h + 8

    draw_text_fit(
        draw, (bx0, cy + 2, bx1, cy + 30),
        f"ANSWER: {TOSSUP_ANSWER}",
        max_size=24, min_size=18, bold=True, fill=STANFORD_RED,
    )
    cy += 34
    return cy


def draw_example_card(draw: ImageDraw.ImageDraw, rect):
    x0, y0, x1, y1 = rect
    draw_card(draw, rect, "Key example: the Markov tossup", BLUE)

    bx0, bx1 = x0 + 22, x1 - 22
    cw = bx1 - bx0
    sec_gap = 18

    cy = y0 + CARD_HEADER_H + 12

    draw_text_fit(
        draw, (bx0, cy, bx1, cy + 28),
        EXAMPLE_SOURCE, max_size=24, min_size=18, fill=TEXT_SOFT,
    )
    cy += 30
    draw_text_fit(
        draw, (bx0, cy, bx1, cy + 36),
        "Full tossup \u2014 pyramidal clue progression (8 runs)",
        max_size=32, min_size=24, bold=True, fill=NAVY,
    )
    cy += 40

    cy = draw_tossup_flow(draw, bx0, cy, bx1)
    cy += sec_gap

    exp_h = 72
    rounded(
        draw, (bx0, cy, bx1, cy + exp_h),
        "#F8F7F4", outline=BORDER, width=2, radius=14,
    )
    draw_text_fit(
        draw, (bx0 + 14, cy + 6, bx1 - 14, cy + exp_h - 6),
        'Each sentence repeats the oblique referent "named for this person"; '
        'the final giveaway says "memoryless chain." '
        "Under MC framing, the core challenge shifts to timing the buzz.",
        max_size=22, min_size=16, fill=TEXT_SOFT, valign="center",
    )
    cy += exp_h + sec_gap

    ans_h = 100
    draw_subcard(draw, (bx0, cy, bx1, cy + ans_h), "Fixed answer set", NAVY)
    grid_x = bx0 + 18
    grid_y = cy + 46
    gbw = (cw - 56 - 3 * 12) // 4
    gbh = 40
    option_styles = [
        (BLUE_SOFT, BLUE), (PANEL_SOFT, BORDER),
        (PANEL_SOFT, BORDER), (PANEL_SOFT, BORDER),
    ]
    for idx, (label, text) in enumerate(EXAMPLE_CHOICES):
        gx = grid_x + idx * (gbw + 12)
        gy = grid_y
        gfill, goutline = option_styles[idx]
        rounded(
            draw, (gx, gy, gx + gbw, gy + gbh),
            fill=gfill, outline=goutline,
            width=3 if idx == 0 else 2, radius=14,
        )
        draw_chip(
            draw, (gx + 6, gy + 6, gx + 34, gy + gbh - 6),
            label, "#EDE7FF", goutline if idx != 0 else BLUE, NAVY, bold=True,
        )
        draw_text_fit(
            draw, (gx + 40, gy + 2, gx + gbw - 6, gy + gbh - 2),
            text, max_size=20, min_size=14, fill=TEXT, valign="center",
        )
    cy += ans_h + sec_gap

    remaining = y1 - cy - 24
    post_h = int(remaining * 0.55)
    chart_h = remaining - post_h - sec_gap

    poy0 = cy
    poy1 = poy0 + post_h
    draw_subcard(
        draw, (bx0, poy0, bx1, poy1),
        "Posterior over A/B/C/D as clues arrive", PURPLE,
    )
    colors = [BLUE, ORANGE, PURPLE, "#8A94A6"]
    bar_x0 = bx0 + 240
    bar_x1 = bx1 - 130
    n_rows = len(EXAMPLE_POSTERIORS)
    row_area = poy1 - poy0 - 100
    row_step = row_area // n_rows
    bar_h = min(36, row_step // 2)
    row_y = poy0 + 58
    run_labels = [
        "t\u22642", "t\u22643", "t\u22645", "t\u22647", "t=8",
    ]
    for i, probs in enumerate(EXAMPLE_POSTERIORS):
        draw_text_fit(
            draw, (bx0 + 16, row_y, bar_x0 - 12, row_y + 30),
            run_labels[i], max_size=28, min_size=20, bold=True, fill=TEXT,
        )
        draw_text_fit(
            draw, (bx0 + 16, row_y + 30, bar_x0 - 12, row_y + 56),
            EXAMPLE_PREFIXES[i], max_size=24, min_size=16, fill=TEXT_SOFT,
        )
        bar_top = row_y + 10
        rounded(
            draw, (bar_x0, bar_top, bar_x1, bar_top + bar_h),
            GRID, outline=None, radius=12,
        )
        seg = bar_x0
        total_w = bar_x1 - bar_x0
        for p_val, c in zip(probs, colors):
            seg_w = total_w * p_val
            rounded(
                draw, (seg, bar_top, seg + seg_w, bar_top + bar_h),
                c, outline=None, radius=12,
            )
            seg += seg_w
        draw_text_fit(
            draw, (bar_x1 + 10, bar_top, bx1 - 130, bar_top + bar_h),
            f"{probs[0] * 100:.0f}%",
            max_size=26, min_size=18, fill=TEXT_SOFT, valign="center",
        )
        dec = EXAMPLE_DECISIONS[i]
        dfill = ORANGE_SOFT if dec == "WAIT" else GREEN_SOFT
        dout = ORANGE if dec == "WAIT" else GREEN
        draw_chip(
            draw, (bx1 - 122, row_y + 4, bx1 - 18, row_y + 48),
            dec, dfill, dout, dout, bold=True,
        )
        row_y += row_step

    lx = bx0 + 20
    for lbl, c in zip(["A", "B", "C", "D"], colors):
        rounded(
            draw, (lx, poy1 - 38, lx + 22, poy1 - 16),
            c, outline=None, radius=6,
        )
        draw.text(
            (lx + 30, poy1 - 40), lbl, font=FONTS["caption_bold"], fill=TEXT_SOFT,
        )
        lx += 80
    cy += post_h + sec_gap

    vy0 = cy
    vy1 = vy0 + chart_h
    draw_subcard(draw, (bx0, vy0, bx1, vy1), "Act now vs. wait value", GREEN)

    chart_x0 = bx0 + 80
    chart_x1 = bx1 - 60
    chart_y0 = vy0 + 66
    chart_y1 = vy1 - 50

    draw.line((chart_x0, chart_y1, chart_x1, chart_y1), fill=BORDER, width=3)
    draw.line((chart_x0, chart_y0, chart_x0, chart_y1), fill=BORDER, width=3)
    for t in range(5):
        x = chart_x0 + t * (chart_x1 - chart_x0) / 4
        draw.line((x, chart_y1, x, chart_y1 + 8), fill=BORDER, width=2)
        draw_text_fit(
            draw, (int(x - 30), chart_y1 + 10, int(x + 30), chart_y1 + 34),
            f"{t + 1}", max_size=22, min_size=16, fill=TEXT_SOFT, align="center",
        )
    for yv in [0.2, 0.4, 0.6, 0.8]:
        y = chart_y1 - yv * (chart_y1 - chart_y0) / 0.8
        draw.line((chart_x0, y, chart_x1, y), fill=GRID, width=1)
        draw_text_fit(
            draw, (chart_x0 - 52, int(y - 10), chart_x0 - 6, int(y + 10)),
            f"{yv:.1f}", max_size=20, min_size=14, fill=TEXT_SOFT,
            align="right", valign="center",
        )

    def map_pt(i: int, val: float):
        return (
            chart_x0 + i * (chart_x1 - chart_x0) / 4,
            chart_y1 - val * (chart_y1 - chart_y0) / 0.8,
        )

    act_pts = [map_pt(i, v) for i, v in enumerate(ACT_NOW_VALUES)]
    wait_pts = [map_pt(i, v) for i, v in enumerate(WAIT_VALUES)]
    draw.line(act_pts, fill=BLUE, width=5)
    draw.line(wait_pts, fill=ORANGE, width=5)
    for pts, c in [(act_pts, BLUE), (wait_pts, ORANGE)]:
        for px, py in pts:
            draw.ellipse(
                (px - 6, py - 6, px + 6, py + 6),
                fill=PANEL, outline=c, width=3,
            )

    bx_buzz, _ = map_pt(3, 0)
    draw.line((bx_buzz, chart_y0 - 6, bx_buzz, chart_y1), fill=GREEN, width=3)
    draw_chip(
        draw,
        (int(bx_buzz - 50), chart_y0 - 36, int(bx_buzz + 50), chart_y0 - 4),
        "buzz", GREEN_SOFT, GREEN, GREEN, bold=True,
    )

    rounded(
        draw, (chart_x1 - 240, vy0 + 14, chart_x1 - 20, vy0 + 48),
        PANEL_SOFT, outline=BORDER, width=1, radius=12,
    )
    draw.line(
        (chart_x1 - 220, vy0 + 32, chart_x1 - 190, vy0 + 32),
        fill=BLUE, width=5,
    )
    draw.text(
        (chart_x1 - 182, vy0 + 18), "act now",
        font=FONTS["caption_bold"], fill=TEXT_SOFT,
    )
    draw.line(
        (chart_x1 - 112, vy0 + 32, chart_x1 - 82, vy0 + 32),
        fill=ORANGE, width=5,
    )
    draw.text(
        (chart_x1 - 74, vy0 + 18), "wait",
        font=FONTS["caption_bold"], fill=TEXT_SOFT,
    )


# ---------------------------------------------------------------------------
# Column 3 panels
# ---------------------------------------------------------------------------


def draw_scatter_card(draw: ImageDraw.ImageDraw, rect, report: Dict):
    x0, y0, x1, y1 = rect
    draw_card(draw, rect, "Smoke snapshot (sanity-check)", ORANGE)

    bx0, bx1 = x0 + 20, x1 - 20
    p = CARD_HEADER_H + 12
    draw_text_fit(
        draw, (bx0, y0 + p, bx1, y0 + p + 36),
        "n = 44. Smoke-test sanity checks only.",
        max_size=28, min_size=22, fill=TEXT_SOFT,
    )

    points = baseline_points(report)
    plotted = [
        pt for pt in points
        if pt["family"] in {"threshold", "sequential_bayes", "always_final"}
    ]
    ppo = report.get("ppo_summary", {})

    plot_x0 = bx0 + 70
    plot_y0 = y0 + p + 52
    plot_x1 = bx1 - 24
    plot_y1 = plot_y0 + 300
    x_max, y_max = 4.5, 0.44

    def xmap(v: float) -> int:
        return int(plot_x0 + (plot_x1 - plot_x0) * (v / x_max))

    def ymap(v: float) -> int:
        return int(plot_y1 - (plot_y1 - plot_y0) * (v / y_max))

    for tick in [0, 1, 2, 3, 4]:
        x = xmap(tick)
        draw.line((x, plot_y0, x, plot_y1), fill=GRID, width=1)
        draw_text_fit(
            draw, (x - 24, plot_y1 + 6, x + 24, plot_y1 + 28),
            f"{tick:.0f}", max_size=22, min_size=16, fill=TEXT_SOFT, align="center",
        )
    for tick in [0.1, 0.2, 0.3, 0.4]:
        y = ymap(tick)
        draw.line((plot_x0, y, plot_x1, y), fill=GRID, width=1)
        draw_text_fit(
            draw, (plot_x0 - 58, y - 10, plot_x0 - 6, y + 10),
            f"{tick:.1f}", max_size=22, min_size=16, fill=TEXT_SOFT,
            align="right", valign="center",
        )
    draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill=BORDER, width=3)
    draw.line((plot_x0, plot_y0, plot_x0, plot_y1), fill=BORDER, width=3)
    draw_text_fit(
        draw, (plot_x0, plot_y1 + 28, plot_x1, plot_y1 + 52),
        "mean buzz step (later = more evidence)",
        max_size=24, min_size=18, fill=TEXT_SOFT, align="center",
    )
    draw_text_fit(
        draw, (bx0, plot_y0, plot_x0 - 8, plot_y1),
        "mean\nS_q", max_size=24, min_size=18, bold=True, fill=TEXT_SOFT,
        align="center", valign="center",
    )

    family_style = {
        "threshold": (BLUE, True),
        "sequential_bayes": (GREEN, True),
        "always_final": (STANFORD_RED, False),
    }
    families: Dict[str, list] = {}
    for pt in plotted:
        families.setdefault(pt["family"], []).append(pt)

    for family, seq in families.items():
        color, line = family_style[family]
        seq = sorted(seq, key=lambda s: s["mean_buzz_step"])
        pts = [(xmap(s["mean_buzz_step"]), ymap(s["mean_sq"])) for s in seq]
        if line and len(pts) > 1:
            draw.line(pts, fill=color, width=4)
        for (px, py), s in zip(pts, seq):
            r = 10 if family != "always_final" else 12
            draw.ellipse(
                (px - r, py - r, px + r, py + r),
                fill=color, outline=PANEL, width=3,
            )
            if family == "always_final":
                draw_text_fit(
                    draw, (px + 16, py - 14, px + 160, py + 14),
                    "always_final", max_size=20, min_size=16, fill=color,
                )

    ppo_x = xmap(float(ppo.get("mean_buzz_step", 0.0)))
    ppo_y = ymap(float(ppo.get("mean_sq", 0.0)))
    draw.polygon(
        [
            (ppo_x, ppo_y - 13),
            (ppo_x + 13, ppo_y),
            (ppo_x, ppo_y + 13),
            (ppo_x - 13, ppo_y),
        ],
        fill=ORANGE, outline=PANEL,
    )
    draw_text_fit(
        draw, (ppo_x + 16, ppo_y - 16, min(plot_x1, ppo_x + 160), ppo_y + 16),
        "PPO smoke", max_size=22, min_size=16, fill=ORANGE,
    )

    seq05 = next(
        (pt for pt in families.get("sequential_bayes", [])
         if pt["threshold"] == "0.5"),
        None,
    )
    thr05 = next(
        (pt for pt in families.get("threshold", []) if pt["threshold"] == "0.5"),
        None,
    )
    if seq05:
        sx, sy = xmap(seq05["mean_buzz_step"]), ymap(seq05["mean_sq"])
        draw_text_fit(
            draw, (sx + 14, sy - 30, sx + 180, sy - 4),
            "seq_bayes@0.5", max_size=20, min_size=16, fill=GREEN,
        )
    if thr05:
        tx, ty_pt = xmap(thr05["mean_buzz_step"]), ymap(thr05["mean_sq"])
        draw_text_fit(
            draw, (tx + 14, ty_pt + 4, tx + 180, ty_pt + 28),
            "threshold@0.5", max_size=20, min_size=16, fill=BLUE,
        )

    leg_y = plot_y1 + 54
    items = [
        (BLUE, "threshold"), (GREEN, "seq_bayes"),
        (STANFORD_RED, "always_final"), (ORANGE, "ppo"),
    ]
    lx = bx0 + 10
    leg_step = (bx1 - bx0 - 20) // 4
    for c, lbl in items:
        rounded(draw, (lx, leg_y, lx + 16, leg_y + 16), c, outline=None, radius=5)
        draw.text(
            (lx + 24, leg_y - 4), lbl,
            font=FONTS["caption_bold"], fill=TEXT_SOFT,
        )
        lx += leg_step

    note_y = leg_y + 28
    draw_text_fit(
        draw, (bx0, note_y, bx1, note_y + 30),
        "softmax_profile overlaps threshold, omitted for readability.",
        max_size=24, min_size=18, fill=TEXT_SOFT,
    )

    chips_y0 = note_y + 40
    picks = pick_points(report)
    cards_data = [
        ("late baseline", picks["always_final"], STANFORD_RED, RED_SOFT),
        ("best non-final", picks["sequential_bayes_05"], GREEN, GREEN_SOFT),
        ("ppo smoke", picks["ppo"], ORANGE, ORANGE_SOFT),
    ]
    chip_w = (bx1 - bx0 - 2 * 12) // 3
    for i, (title, metrics, color, cfill) in enumerate(cards_data):
        cx0 = bx0 + i * (chip_w + 12)
        cx1 = cx0 + chip_w
        cy1 = y1 - 20
        rounded(
            draw, (cx0, chips_y0, cx1, cy1),
            cfill, outline=color, width=2, radius=14,
        )
        draw_text_fit(
            draw, (cx0 + 10, chips_y0 + 8, cx1 - 10, chips_y0 + 36),
            title, max_size=24, min_size=18, bold=True, fill=color, align="center",
        )
        lines = [
            f"S_q = {fmt_num(metrics.get('mean_sq'))}",
            f"acc = {fmt_pct(metrics.get('buzz_accuracy'))}",
            f"step = {fmt_num(metrics.get('mean_buzz_step'), 2)}",
        ]
        if title == "ppo smoke":
            lines.append(f"rew = {fmt_num(metrics.get('mean_reward_like'))}")
        draw_bullets(
            draw, (cx0 + 12, chips_y0 + 44, cx1 - 12, cy1 - 8),
            lines, font=FONTS["caption"], fill=TEXT,
            bullet_fill=color, bullet_radius=4, gap_after=4,
        )


def draw_controls_card(draw: ImageDraw.ImageDraw, rect, report: Dict):
    x0, y0, x1, y1 = rect
    draw_card(draw, rect, "Results + controls", GREEN)

    bx0, bx1 = x0 + 20, x1 - 20
    p = CARD_HEADER_H + 12

    rounded(
        draw, (bx0, y0 + p, bx1, y0 + p + 90),
        PANEL_SOFT, outline=BORDER, width=2, radius=14,
    )
    draw_text_fit(
        draw, (bx0 + 16, y0 + p + 8, bx1 - 16, y0 + p + 82),
        "Smoke takeaway: always_final best S_q (0.386); "
        "sequential_bayes@0.5 strongest non-final (0.267); "
        "PPO reaches 0.326 but buzzes at step 0.0.",
        max_size=26, min_size=20, fill=TEXT, valign="center",
    )

    controls = report.get("controls", {})
    full = report.get("full_eval", {})
    choices = controls.get("choices_only", {})
    shuffle = controls.get("shuffle", {})
    alias = controls.get("alias_substitution", {})

    card_h = 150
    card_gap = 14
    cards = [
        {
            "title": "Choices only",
            "big": fmt_pct(choices.get("accuracy")),
            "sub": f"chance = {fmt_pct(choices.get('chance'))}  \u2022  "
                   f"n = {fmt_num(choices.get('n_test'), 0)}",
            "note": "Answer options alone do not solve the task.",
            "color": BLUE, "fill": BLUE_SOFT,
        },
        {
            "title": "Shuffle clues",
            "big": f"\u0394S_q = {float(shuffle.get('mean_sq', 0)) - float(full.get('mean_sq', 0)):+.3f}",
            "sub": f"full = {fmt_num(full.get('mean_sq'))}  \u2022  "
                   f"shuffled = {fmt_num(shuffle.get('mean_sq'))}",
            "note": "Very small movement on the smoke slice.",
            "color": ORANGE, "fill": ORANGE_SOFT,
        },
        {
            "title": "Alias substitution",
            "big": f"\u0394S_q = {float(alias.get('mean_sq', 0)) - float(full.get('mean_sq', 0)):+.3f}",
            "sub": f"full = {fmt_num(full.get('mean_sq'))}  \u2022  "
                   f"alias = {fmt_num(alias.get('mean_sq'))}",
            "note": "Essentially unchanged in this smoke report.",
            "color": GREEN, "fill": GREEN_SOFT,
        },
    ]

    cy = y0 + p + 106
    for c in cards:
        rounded(
            draw, (bx0, cy, bx1, cy + card_h),
            c["fill"], outline=c["color"], width=2, radius=14,
        )
        draw_text_fit(
            draw, (bx0 + 14, cy + 10, bx1 - 14, cy + 36),
            c["title"], max_size=26, min_size=20, bold=True, fill=c["color"],
        )
        draw_text_fit(
            draw, (bx0 + 14, cy + 38, bx1 - 14, cy + 74),
            c["big"], max_size=34, min_size=24, bold=True, fill=TEXT,
        )
        draw_text_fit(
            draw, (bx0 + 14, cy + 76, bx1 - 14, cy + 104),
            c["sub"], max_size=24, min_size=18, fill=TEXT_SOFT,
        )
        draw_text_fit(
            draw, (bx0 + 14, cy + 106, bx1 - 14, cy + card_h - 10),
            c["note"], max_size=24, min_size=18, fill=TEXT_SOFT,
        )
        cy += card_h + card_gap

    ry0 = cy + 4
    rounded(
        draw, (bx0, ry0, bx1, ry0 + 100),
        RED_SOFT, outline=STANFORD_RED, width=2, radius=14,
    )
    draw_text_fit(
        draw, (bx0 + 14, ry0 + 8, bx1 - 14, ry0 + 36),
        "Interpret with caution",
        max_size=26, min_size=20, bold=True, fill=STANFORD_RED,
    )
    draw_text_fit(
        draw, (bx0 + 14, ry0 + 40, bx1 - 14, ry0 + 92),
        "Choices-only underperforms, but shuffle and alias barely "
        "move S_q. These controls are suggestive but inconclusive.",
        max_size=24, min_size=18, fill=TEXT, valign="center",
    )

    sy0 = ry0 + 110
    rounded(
        draw, (bx0, sy0, bx1, y1 - 20),
        GOLD_SOFT, outline=GOLD, width=2, radius=14,
    )
    draw_text_fit(
        draw, (bx0 + 14, sy0 + 10, bx1 - 14, y1 - 30),
        "The framing exposes gaps in timing, calibration, and "
        "artifact checking that remain open.",
        max_size=26, min_size=20, fill=TEXT, valign="center",
    )


def draw_limitations_card(draw: ImageDraw.ImageDraw, rect):
    x0, y0, x1, y1 = rect
    draw_card(draw, rect, "Limitations + next checks", NAVY)
    draw_bullets(
        draw, (x0 + 24, y0 + CARD_HEADER_H + 16, x1 - 24, y1 - 24),
        [
            "The smoke slice is small: full_eval uses n = 44 "
            "and choices-only uses n = 11.",
            "PPO still shows degenerate timing, so reward design "
            "needs another pass.",
        ],
        font=FONTS["small"], fill=TEXT,
        bullet_fill=NAVY, bullet_radius=7, gap_after=14,
    )


# ---------------------------------------------------------------------------
# Header & footer
# ---------------------------------------------------------------------------


def draw_header(draw: ImageDraw.ImageDraw):
    draw.rectangle((0, 0, W, 24), fill=STANFORD_RED)

    x0 = MARGIN
    y0 = 48
    draw_text_fit(
        draw, (x0, y0, 2900, y0 + 130),
        "Quiz Bowl RL Buzzer",
        max_size=140, min_size=100, bold=True, fill=TEXT,
    )
    draw_text_fit(
        draw, (x0, y0 + 124, 2700, y0 + 180),
        "Learning when to buzz under incremental clues",
        max_size=56, min_size=40, fill=TEXT_SOFT,
    )
    draw_text_fit(
        draw, (x0, y0 + 186, 2700, y0 + 226),
        "Kathleen Weng  \u2022  Imran Hassan  \u2022  Ankit Aggarwal",
        max_size=42, min_size=30, fill=TEXT_SOFT,
    )
    draw_text_fit(
        draw, (x0, y0 + 230, 2700, y0 + 264),
        "CS234 final project  \u2022  smoke metrics from evaluation_report.json",
        max_size=30, min_size=22, fill=TEXT_SOFT,
    )

    tx0, ty0, tx1, ty1 = 2780, 48, W - MARGIN, 280
    rounded(draw, (tx0, ty0, tx1, ty1), GOLD_SOFT, outline=GOLD, width=2, radius=24)
    draw_text_fit(
        draw, (tx0 + 18, ty0 + 14, tx1 - 18, ty0 + 48),
        "One-sentence takeaway",
        max_size=28, min_size=22, bold=True, fill=GOLD,
    )
    draw_text_fit(
        draw, (tx0 + 18, ty0 + 56, tx1 - 18, ty1 - 14),
        "Buzzing becomes an optimal-stopping problem; on this smoke "
        "slice, simple baselines remain the strongest anchors.",
        max_size=32, min_size=24, fill=TEXT, valign="center",
    )


def draw_footer(draw: ImageDraw.ImageDraw):
    draw.line(
        (MARGIN, H - FOOTER_H + 10, W - MARGIN, H - FOOTER_H + 10),
        fill=BORDER, width=2,
    )
    draw_text_fit(
        draw, (MARGIN, H - FOOTER_H + 18, W - MARGIN, H - 10),
        "Poster: content from the bundled slide deck and smoke report; "
        "all numerical claims from evaluation_report.json; "
        "smoke results are sanity-check snapshots only.",
        max_size=24, min_size=18, fill=TEXT_SOFT,
        align="center", valign="center",
    )


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _stack_cards(
    body_y: int, body_h: int, heights: List[int],
) -> List[Tuple[int, int]]:
    """Return (y_start, y_end) for each card, distributing extra space."""
    n = len(heights)
    total = sum(heights) + (n - 1) * CARD_GAP
    extra = max(0, body_h - total)
    pad = extra // n if n else 0
    adjusted = [h + pad for h in heights]

    y = body_y
    out: List[Tuple[int, int]] = []
    for h in adjusted:
        out.append((y, y + h))
        y += h + CARD_GAP
    return out


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------


def generate_poster() -> None:
    report = load_report()
    poster = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(poster)

    draw_header(draw)
    draw_footer(draw)

    body_y = HEADER_H
    body_h = H - HEADER_H - FOOTER_H - 10
    col_w = (W - 2 * MARGIN - 2 * COL_GAP) // 3

    c1x = MARGIN
    c2x = c1x + col_w + COL_GAP
    c3x = c2x + col_w + COL_GAP

    c1 = _stack_cards(body_y, body_h, [900, 700, 310, 300])
    for (ys, ye), fn in zip(c1, [
        draw_problem_card, draw_method_card,
        draw_conclusions_card, draw_references_card,
    ]):
        fn(draw, (c1x, ys, c1x + col_w, ye))

    draw_example_card(draw, (c2x, body_y, c2x + col_w, body_y + body_h))

    c3 = _stack_cards(body_y, body_h, [770, 1060, 310])
    draw_scatter_card(draw, (c3x, c3[0][0], c3x + col_w, c3[0][1]), report)
    draw_controls_card(draw, (c3x, c3[1][0], c3x + col_w, c3[1][1]), report)
    draw_limitations_card(draw, (c3x, c3[2][0], c3x + col_w, c3[2][1]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "poster.pdf"
    png_path = OUT_DIR / "poster.png"
    poster.save(str(pdf_path), "PDF", resolution=DPI)
    poster.save(str(png_path), "PNG")
    print(f"Saved poster to {pdf_path}")
    print(f"Saved preview to {png_path}")


if __name__ == "__main__":
    generate_poster()
