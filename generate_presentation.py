from __future__ import annotations

import json
import os
import re
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw, ImageFont

# ============================================================
# Paths
# ============================================================
DEFAULT_OUT_DIR = Path("/mnt/data/quizbowl_mc_stopping_data_driven")
FALLBACK_OUT_DIR = (
    Path(__file__).resolve().parent
    / "generated"
    / "quizbowl_mc_stopping_data_driven"
)

env_out_dir = os.getenv("QB_PRESENTATION_OUT_DIR")
if env_out_dir:
    OUT_DIR = Path(env_out_dir).expanduser()
else:
    preferred_parent = DEFAULT_OUT_DIR.parent
    if preferred_parent.exists() and os.access(preferred_parent, os.W_OK):
        OUT_DIR = DEFAULT_OUT_DIR
    else:
        OUT_DIR = FALLBACK_OUT_DIR

OUT_DIR.mkdir(parents=True, exist_ok=True)

GIF_OUT = OUT_DIR / "quizbowl_mc_stopping_data_driven.gif"
CONTACT_OUT = OUT_DIR / "quizbowl_mc_stopping_data_driven_contact.png"
FRAMES_DIR = OUT_DIR / "frames"

# ============================================================
# Canvas + palette
# ============================================================
W, H = 960, 540
BG = "#F3F5F8"
WHITE = "#FFFFFF"

NAVY = "#2E4A9E"
NAVY_DARK = "#243B7A"
TEXT = "#1F2937"
TEXT_SOFT = "#5B6472"
BORDER = "#C8D1E0"
GRID = "#E6EBF3"

BLUE = "#345BD3"
BLUE_SOFT = "#DDE7FF"

PURPLE = "#7A57E2"
PURPLE_SOFT = "#E9E0FF"

GREEN = "#4BB773"
GREEN_SOFT = "#DFF5E7"

ORANGE = "#E9A23B"
ORANGE_SOFT = "#FDEFD9"

RED = "#E76A5E"
RED_SOFT = "#FCE2DE"

# ============================================================
# Running example
# ============================================================
# Source tossup: https://www.qbreader.org/db/tossup/?_id=63ec2f74c5548754cbcb03f4
# Distractors here are a cleaner, presentation-friendly mathematician-only set
# drawn from answers that already exist in the local corpus.
EXAMPLE_TITLE = "Markov tossup"
EXAMPLE_SOURCE = "QBReader • 2022 ARCADIA • packet 08 • tossup 11"
EXAMPLE_TOSSUP_VERBATIM = (
    'Given a set of nodes named for this person, a node is conditionally '
    'independent from the rest of a Bayesian network. The Baum-Welch algorithm '
    'is used to train a type of model named for this person that is used for '
    'multiple sequence alignment. The initial “burn-in” states of a process '
    'named for this person are discarded in methods like Gibbs sampling. The '
    'Metropolis-Hastings algorithm approximates an unknown distribution as the '
    '(*) stationary distribution of a process named for this person. Monte Carlo '
    'methods often use processes named for this person that have stochastic '
    'transition matrices. Dynamic programming is used to decode “hidden” models '
    'named for this person. The next state of a random process named for this '
    'person depends only on the current state. For 10 points, what Russian '
    'mathematician names a type of memoryless “chain?”'
)
EXAMPLE_ANSWERLINE = "ANSWER: Andrey Andreyevich Markov"
EXAMPLE_ACCEPTS = "Accepts: Markov blankets; hidden Markov models; Markov chains"
EXAMPLE_TOSSUP_HIGHLIGHTS = [
    {"phrase": "named for", "fill": ORANGE_SOFT, "text_fill": "#8A5A11"},
    {"phrase": "this person", "fill": BLUE_SOFT, "text_fill": NAVY_DARK},
]
EXAMPLE_OPTIONS = [
    ("A", "Andrey Andreyevich Markov"),
    ("B", "Leonhard Euler"),
    ("C", "Carl Friedrich Gauss"),
    ("D", "Augustin-Louis Cauchy"),
]
EXAMPLE_POSTERIOR_STAGES = [
    {
        "prefix": 'Only the opening referent is visible: "nodes named for this person..."',
        "probs": [0.42, 0.24, 0.21, 0.13],
        "decision": "WAIT",
        "callout": 'Keep "waiting": Markov only leads narrowly over the other mathematicians, so another clue can still change the ranking.',
    },
    {
        "prefix": 'The referent repeats: "a type of model named for this person..."',
        "probs": [0.56, 0.19, 0.15, 0.10],
        "decision": "WAIT",
        "callout": 'Keep "waiting": Markov now leads, but continuation value is still meaningful.',
    },
    {
        "prefix": 'Another repetition appears: "\"burn-in\" states of a process named for this person..."',
        "probs": [0.68, 0.14, 0.11, 0.07],
        "decision": "WAIT",
        "callout": 'Keep "waiting": A is favored, yet one more clue can still sharpen confidence.',
    },
    {
        "prefix": 'The referent stays active: "stationary distribution of a process named for this person..."',
        "probs": [0.82, 0.08, 0.06, 0.04],
        "decision": "BUZZ",
        "callout": '"Buzz" now: the Markov reading is strong enough that waiting has less value.',
    },
    {
        "prefix": 'Finally, "memoryless chain" turns the repeated "this person" referent into a giveaway.',
        "probs": [0.91, 0.04, 0.03, 0.02],
        "decision": "BUZZ",
        "callout": '"Buzz" now: the final clue mostly confirms what the posterior already says.',
    },
]
EXAMPLE_VALUE_NOTES = [
    'step 1 act now = 0.18\nwait = 0.64',
    'step 2 act now = 0.38\nwait = 0.58',
    'step 3 act now = 0.49\nwait = 0.52',
    'step 4 act now = 0.63\nwait = 0.40',
    'step 5 act now = 0.74\nwait = 0.28',
]
EXAMPLE_VALUE_MESSAGES = [
    'On the first Markov clue, another "clue" is still worth much more than committing.',
    'Baum-Welch pushes the example toward Markov, but waiting still wins.',
    'Near the Gibbs / Metropolis region, timing becomes the whole game.',
    'Once the random-process clue arrives, "buzzing" overtakes waiting.',
    'By the "memoryless chain" clue, waiting mostly delays a strong answer.',
]
EXAMPLE_ABSTAIN_MESSAGES = [
    'Even with Markov already listed at A, "ABSTAIN" can still be rational after the earliest clue.',
    '"Waiting" is still about the value of one more clue, not about missing options.',
    'So "ABSTAIN" means continuation value, even when the correct answer has been on the board all along.',
]

# ============================================================
# Fonts
# ============================================================
def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            ]
        )

    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)

    return ImageFont.load_default()

# ============================================================
# Text fitting
# ============================================================
def measure(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]

def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    lines: List[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split()
        current = words[0]
        for word in words[1:]:
            trial = current + " " + word
            tw, _ = measure(draw, trial, font)
            if tw <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines

def fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: Tuple[int, int, int, int],
    *,
    max_size: int,
    min_size: int,
    bold: bool = False,
    line_gap: int = 4,
):
    x0, y0, x1, y1 = box
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    for size in range(max_size, min_size - 1, -1):
        font = load_font(size, bold=bold)
        lines = wrap_text(draw, text, font, bw)
        _, lh = measure(draw, "Ag", font)
        total_h = len(lines) * lh + max(0, len(lines) - 1) * line_gap

        if total_h > bh:
            continue

        too_wide = False
        for line in lines:
            lw, _ = measure(draw, line, font)
            if lw > bw:
                too_wide = True
                break
        if not too_wide:
            return font, lines, total_h

    font = load_font(min_size, bold=bold)
    lines = wrap_text(draw, text, font, bw)
    _, lh = measure(draw, "Ag", font)
    total_h = len(lines) * lh + max(0, len(lines) - 1) * line_gap
    return font, lines, total_h

def draw_text_fit(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    *,
    fill=TEXT,
    max_size=24,
    min_size=12,
    bold=False,
    align="left",
    valign="top",
    line_gap=4,
):
    x0, y0, x1, y1 = box
    font, lines, total_h = fit_text(
        draw,
        text,
        box,
        max_size=max_size,
        min_size=min_size,
        bold=bold,
        line_gap=line_gap,
    )
    _, lh = measure(draw, "Ag", font)

    if valign == "center":
        cy = y0 + max(0, (y1 - y0 - total_h) // 2)
    elif valign == "bottom":
        cy = y1 - total_h
    else:
        cy = y0

    for line in lines:
        lw, _ = measure(draw, line, font)
        if align == "center":
            cx = x0 + max(0, (x1 - x0 - lw) // 2)
        elif align == "right":
            cx = x1 - lw
        else:
            cx = x0
        draw.text((cx, cy), line, font=font, fill=fill)
        cy += lh + line_gap

def _split_highlight_segments(text: str, highlights: List[Dict]) -> List[Tuple[str, Dict | None]]:
    phrase_to_style = {item["phrase"]: item for item in highlights}
    pattern = re.compile(
        "|".join(re.escape(item["phrase"]) for item in sorted(highlights, key=lambda item: len(item["phrase"]), reverse=True))
    )

    segments: List[Tuple[str, Dict | None]] = []
    last = 0
    for match in pattern.finditer(text):
        if match.start() > last:
            segments.append((text[last:match.start()], None))
        segments.append((match.group(0), phrase_to_style[match.group(0)]))
        last = match.end()
    if last < len(text):
        segments.append((text[last:], None))
    return segments

def _layout_highlighted_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    highlights: List[Dict],
    font,
    max_width: int,
) -> List[List[Tuple[str, Dict | None, int]]]:
    raw_tokens: List[Tuple[str, Dict | None]] = []
    for segment, style in _split_highlight_segments(text, highlights):
        if not segment:
            continue
        if style is None:
            raw_tokens.extend((token, None) for token in re.findall(r"\S+\s*", segment))
        else:
            raw_tokens.append((segment, style))

    lines: List[List[Tuple[str, Dict | None, int]]] = []
    current: List[Tuple[str, Dict | None, int]] = []
    current_width = 0

    for token_text, style in raw_tokens:
        display = token_text if current else token_text.lstrip()
        if not display:
            continue
        token_width, _ = measure(draw, display, font)

        if current and current_width + token_width > max_width:
            lines.append(current)
            current = []
            current_width = 0
            display = token_text.lstrip()
            if not display:
                continue
            token_width, _ = measure(draw, display, font)

        current.append((display, style, token_width))
        current_width += token_width

    if current:
        lines.append(current)

    return lines

def draw_highlighted_text_fit(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    highlights: List[Dict],
    *,
    fill=TEXT,
    max_size=16,
    min_size=8,
    line_gap=2,
):
    x0, y0, x1, y1 = box
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    chosen_font = load_font(min_size)
    chosen_lines: List[List[Tuple[str, Dict | None, int]]] = []
    chosen_lh = 0

    for size in range(max_size, min_size - 1, -1):
        font = load_font(size)
        lines = _layout_highlighted_text(draw, text, highlights, font, bw)
        _, lh = measure(draw, "Ag", font)
        total_h = len(lines) * lh + max(0, len(lines) - 1) * line_gap
        if total_h <= bh:
            chosen_font = font
            chosen_lines = lines
            chosen_lh = lh
            break

    if not chosen_lines:
        chosen_lines = _layout_highlighted_text(draw, text, highlights, chosen_font, bw)
        _, chosen_lh = measure(draw, "Ag", chosen_font)

    cy = y0
    for line in chosen_lines:
        cx = x0
        for token_text, style, token_width in line:
            if style is not None:
                rounded(
                    draw,
                    (cx - 2, cy - 1, cx + token_width + 2, cy + chosen_lh - 1),
                    style["fill"],
                    outline=style["fill"],
                    radius=5,
                    width=1,
                )
                draw.text((cx, cy), token_text, font=chosen_font, fill=style["text_fill"])
            else:
                draw.text((cx, cy), token_text, font=chosen_font, fill=fill)
            cx += token_width
        cy += chosen_lh + line_gap

# ============================================================
# Drawing primitives
# ============================================================
def make_canvas() -> Image.Image:
    img = Image.new("RGBA", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 72), fill="#EEF2F8")
    d.line((0, 72, W, 72), fill="#DCE4F0", width=2)
    for y in range(100, H, 80):
        d.line((36, y, W - 36, y), fill=GRID, width=1)
    return img

def rounded(draw: ImageDraw.ImageDraw, box, fill, outline=BORDER, radius=18, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

def card(draw: ImageDraw.ImageDraw, box, title=None, title_color=NAVY, fill=WHITE):
    rounded(draw, box, fill=fill, outline=BORDER, radius=18, width=2)
    if title:
        draw_text_fit(
            draw,
            (box[0] + 18, box[1] + 14, box[2] - 18, box[1] + 46),
            title,
            max_size=22,
            min_size=14,
            bold=True,
            fill=title_color,
        )

def header(draw: ImageDraw.ImageDraw, title: str, subtitle: str | None = None):
    draw_text_fit(
        draw,
        (40, 18, W - 40, 52),
        title,
        max_size=28,
        min_size=16,
        bold=True,
        fill=NAVY,
        align="center",
        valign="center",
    )
    if subtitle:
        draw_text_fit(
            draw,
            (40, 52, W - 40, 70),
            subtitle,
            max_size=14,
            min_size=10,
            fill=TEXT_SOFT,
            align="center",
            valign="center",
        )

def footer(draw: ImageDraw.ImageDraw, txt: str):
    draw_text_fit(
        draw,
        (40, H - 26, W - 40, H - 8),
        txt,
        max_size=12,
        min_size=10,
        fill=TEXT_SOFT,
        align="center",
        valign="center",
    )

def pill(draw: ImageDraw.ImageDraw, box, text, fill, outline=None, text_color=TEXT):
    rounded(draw, box, fill=fill, outline=outline or fill, radius=16, width=2)
    draw_text_fit(
        draw,
        (box[0] + 10, box[1] + 8, box[2] - 10, box[3] - 8),
        text,
        max_size=16,
        min_size=10,
        bold=True,
        fill=text_color,
        align="center",
        valign="center",
    )

def draw_options_grid(draw: ImageDraw.ImageDraw, box, options=None):
    x0, y0, x1, y1 = box
    pad = 14
    gap = 14
    cell_w = (x1 - x0 - 2 * pad - gap) // 2
    cell_h = 84
    labels = options or EXAMPLE_OPTIONS
    positions = [
        (x0 + pad, y0 + pad),
        (x0 + pad + cell_w + gap, y0 + pad),
        (x0 + pad, y0 + pad + cell_h + gap),
        (x0 + pad + cell_w + gap, y0 + pad + cell_h + gap),
    ]

    for idx, ((lab, txt), (cx, cy)) in enumerate(zip(labels, positions)):
        fill = "#EAE6FB" if idx == 0 else "#F7F8FA"
        outline = BLUE if idx == 0 else BORDER
        rounded(draw, (cx, cy, cx + cell_w, cy + cell_h), fill, outline, radius=14, width=2)
        draw.ellipse((cx + 10, cy + 10, cx + 34, cy + 34), fill="#ECEAF7")
        draw_text_fit(draw, (cx + 10, cy + 10, cx + 34, cy + 34), lab, max_size=14, min_size=10, bold=True, align="center", valign="center", fill=NAVY_DARK)
        draw_text_fit(draw, (cx + 42, cy + 18, cx + cell_w - 12, cy + cell_h - 12), txt, max_size=15, min_size=10, valign="center")

def draw_bullets(draw: ImageDraw.ImageDraw, box, items: Iterable[str], bullet_color=BLUE):
    x0, y0, x1, y1 = box
    cy = y0
    for item in items:
        draw.ellipse((x0, cy + 7, x0 + 10, cy + 17), fill=bullet_color)
        draw_text_fit(draw, (x0 + 18, cy, x1, cy + 28), item, max_size=17, min_size=11, fill=TEXT, valign="center")
        cy += 28

def draw_progress_dots(draw: ImageDraw.ImageDraw, box, stage: int, total: int):
    x0, y0, x1, y1 = box
    xs = []
    for i in range(total):
        frac = i / max(1, total - 1)
        xs.append(int(x0 + frac * (x1 - x0)))
    line_y = (y0 + y1) // 2
    draw.line((x0, line_y, x1, line_y), fill=BORDER, width=2)
    for i, x in enumerate(xs):
        fill = BLUE if i <= stage else WHITE
        outline = BLUE if i <= stage else BORDER
        draw.ellipse((x - 5, line_y - 5, x + 5, line_y + 5), fill=fill, outline=outline, width=2)

def draw_posterior_bars(draw: ImageDraw.ImageDraw, box, probs):
    x0, y0, x1, y1 = box
    labels = ["A", "B", "C", "D"]
    colors = [BLUE, ORANGE, PURPLE, "#9CA3AF"]

    bar_y = y1 - 28
    left = x0 + 26
    right = x1 - 10
    usable_w = right - left
    gap = 14
    bw = (usable_w - gap * (len(probs) - 1)) // len(probs)

    for i, p in enumerate(probs):
        bx = left + i * (bw + gap)
        fill_h = int(60 * p) + 8
        by0 = bar_y - fill_h
        by1 = bar_y
        draw.rounded_rectangle((bx, by0, bx + bw, by1), radius=7, fill=colors[i])
        draw_text_fit(draw, (bx, bar_y + 4, bx + bw, bar_y + 24), labels[i], max_size=12, min_size=10, fill=TEXT_SOFT, align="center")
        draw_text_fit(draw, (bx, by0 - 22, bx + bw, by0 - 2), f"{int(round(100*p))}%", max_size=12, min_size=9, fill=TEXT_SOFT, align="center")

def draw_line_chart(draw: ImageDraw.ImageDraw, box, phase: int):
    x0, y0, x1, y1 = box
    draw.line((x0 + 34, y1 - 36, x1 - 18, y1 - 36), fill=TEXT_SOFT, width=2)
    draw.line((x0 + 34, y0 + 18, x0 + 34, y1 - 36), fill=TEXT_SOFT, width=2)
    draw_text_fit(draw, (x0 + 56, y1 - 26, x1 - 24, y1 - 8), "more of the Markov tossup revealed", max_size=11, min_size=9, fill=TEXT_SOFT, align="center")
    draw_text_fit(draw, (x0 - 4, y0 + 22, x0 + 44, y0 + 74), "value", max_size=11, min_size=9, fill=TEXT_SOFT, align="center", valign="center")

    p1 = [
        (x0 + 34, y1 - 60),
        (x0 + 110, y1 - 84),
        (x0 + 190, y1 - 108),
        (x0 + 270, y1 - 132),
        (x0 + 348, y1 - 154),
    ]
    p2 = [
        (x0 + 34, y0 + 44),
        (x0 + 110, y0 + 56),
        (x0 + 190, y0 + 70),
        (x0 + 270, y0 + 88),
        (x0 + 348, y0 + 104),
    ]
    draw.line(p1, fill=BLUE, width=3)
    draw.line(p2, fill=ORANGE, width=3)

    idx = min(phase, 4)
    cx = p1[idx][0]
    draw.line((cx, y0 + 18, cx, y1 - 36), fill="#444444", width=2)

    legend_x = x1 - 140
    rounded(draw, (legend_x, y0 + 24, x1 - 22, y0 + 70), WHITE, outline=BORDER, radius=12, width=2)
    draw.line((legend_x + 12, y0 + 40, legend_x + 34, y0 + 40), fill=BLUE, width=3)
    draw_text_fit(draw, (legend_x + 40, y0 + 28, x1 - 30, y0 + 46), "act now value", max_size=12, min_size=9, fill=TEXT_SOFT)
    draw.line((legend_x + 12, y0 + 56, legend_x + 34, y0 + 56), fill=ORANGE, width=3)
    draw_text_fit(draw, (legend_x + 40, y0 + 44, x1 - 30, y0 + 62), "wait value", max_size=12, min_size=9, fill=TEXT_SOFT)

    notes = EXAMPLE_VALUE_NOTES
    rounded(draw, (x0 + 126, y0 + 52, x0 + 236, y0 + 104), GREEN_SOFT if phase >= 2 else ORANGE_SOFT,
            outline=GREEN if phase >= 2 else ORANGE, radius=12, width=2)
    draw_text_fit(draw, (x0 + 136, y0 + 60, x0 + 226, y0 + 98), notes[idx], max_size=12, min_size=9, fill=TEXT_SOFT, align="center", valign="center")

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

def load_eval_report() -> Dict:
    path = Path(__file__).resolve().parent / "artifacts" / "smoke" / "evaluation_report.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

def best_baseline_result(report: Dict) -> Tuple[str, Dict]:
    baseline_summary = report.get("baseline_summary", {})
    best_label = "n/a"
    best_metrics: Dict = {}
    best_sq = float("-inf")

    for method, payload in baseline_summary.items():
        if isinstance(payload, dict) and payload and all(isinstance(v, dict) for v in payload.values()):
            for threshold, metrics in payload.items():
                sq = metrics.get("mean_sq")
                if isinstance(sq, (float, int)) and sq > best_sq:
                    best_sq = float(sq)
                    best_label = f"{method} @ {threshold}"
                    best_metrics = metrics
        elif isinstance(payload, dict):
            sq = payload.get("mean_sq")
            if isinstance(sq, (float, int)) and sq > best_sq:
                best_sq = float(sq)
                best_label = method
                best_metrics = payload

    return best_label, best_metrics

EVAL_REPORT = load_eval_report()

# ============================================================
# Scene specs
# ============================================================
SCENES: List[Dict] = [
    {
        "kind": "title_slide",
        "footer": "CS234 Final Project",
    },
    {
        "kind": "problem_slide",
        "footer": "Problem",
    },
    {
        "kind": "background_slide",
        "footer": "Background and Setup",
    },
    {
        "kind": "why_mc_slide",
        "footer": "Why Multiple Choice?",
    },
    {
        "kind": "method_slide",
        "footer": "Method Overview",
    },
    {
        "kind": "contribution_slide",
        "footer": "Our Contribution",
    },
    {
        "kind": "expected_results_slide",
        "footer": "Smoke Results Snapshot",
    },
    {
        "kind": "why_ppo_less_accurate",
        "footer": "How To Read PPO Metrics",
    },
    {
        "kind": "intro",
        "footer": "MC Stopping Intuition",
    },
    {
        "kind": "section",
        "title": "Quiz bowl MC buzzing",
        "section_title": "Section 1: the posterior sharpens\nas \"clues\" arrive",
        "accent": BLUE,
        "footer_small": "first: watch uncertainty collapse",
        "footer": "Posterior Sharpening",
    },
    {
        "kind": "posterior",
        "stage": 0,
        "footer": "Posterior Sharpening",
    },
    {
        "kind": "posterior",
        "stage": 1,
        "footer": "Posterior Sharpening",
    },
    {
        "kind": "posterior",
        "stage": 2,
        "footer": "Posterior Sharpening",
    },
    {
        "kind": "posterior",
        "stage": 3,
        "footer": "Posterior Sharpening",
    },
    {
        "kind": "posterior",
        "stage": 4,
        "footer": "Posterior Sharpening",
    },
    {
        "kind": "section",
        "title": "Quiz bowl MC buzzing",
        "section_title": "Section 2: \"buzz\" when acting now\nbeats waiting",
        "accent": ORANGE,
        "footer_small": "next: compare current value to one more clue",
        "footer": "Act Now Vs Wait",
    },
    {
        "kind": "value_chart",
        "stage": 0,
        "footer": "Act Now Vs Wait",
    },
    {
        "kind": "value_chart",
        "stage": 1,
        "footer": "Act Now Vs Wait",
    },
    {
        "kind": "value_chart",
        "stage": 2,
        "footer": "Act Now Vs Wait",
    },
    {
        "kind": "value_chart",
        "stage": 3,
        "footer": "Act Now Vs Wait",
    },
    {
        "kind": "value_chart",
        "stage": 4,
        "footer": "Act Now Vs Wait",
    },
    {
        "kind": "section",
        "title": "Quiz bowl MC buzzing",
        "section_title": "Section 3: \"ABSTAIN\" means wait,\nnot \"None-of-the-Above\"",
        "accent": RED,
        "footer_small": "next: waiting is about value of information",
        "footer": "Abstain Semantics",
    },
    {
        "kind": "abstain",
        "stage": 0,
        "footer": "Abstain Semantics",
    },
    {
        "kind": "abstain",
        "stage": 1,
        "footer": "Abstain Semantics",
    },
    {
        "kind": "abstain",
        "stage": 2,
        "footer": "Abstain Semantics",
    },
    {
        "kind": "section",
        "title": "Quiz bowl MC buzzing",
        "section_title": "Section 4: separate answer quality\nfrom \"buzz\" timing",
        "accent": PURPLE,
        "footer_small": "next: factor the decision into two modules",
        "footer": "Factor Stop And Answer",
    },
    {
        "kind": "factorization",
        "stage": 0,
        "footer": "Factor Stop And Answer",
    },
    {
        "kind": "factorization",
        "stage": 1,
        "footer": "Factor Stop And Answer",
    },
    {
        "kind": "section",
        "title": "Quiz bowl MC buzzing",
        "section_title": "Section 5: one compact mental model",
        "accent": GREEN,
        "footer_small": "finally: compress the whole story",
        "footer": "Training Recipe",
    },
    {
        "kind": "recipe",
        "footer": "Training Recipe",
    },
    {
        "kind": "pipeline",
        "stage": 0,
        "footer": "Training Setup",
    },
    {
        "kind": "pipeline",
        "stage": 1,
        "footer": "Evaluation Setup",
    },
    {
        "kind": "pipeline",
        "stage": 2,
        "footer": "Smoke Results",
    },
    {
        "kind": "evaluation_slide",
        "footer": "Evaluation",
    },
    {
        "kind": "summary",
        "stage": 0,
        "footer": "Summary",
    },
    {
        "kind": "summary",
        "stage": 1,
        "footer": "Summary",
    },
    {
        "kind": "summary",
        "stage": 2,
        "footer": "Summary",
    },
    {
        "kind": "references_slide",
        "footer": "References",
    },
]

assert len(SCENES) == 38

# ============================================================
# Scene renderers
# ============================================================
def render_title_slide(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    draw_text_fit(
        d,
        (80, 100, W - 80, 180),
        "Quiz Bowl RL Buzzer",
        max_size=38,
        min_size=22,
        bold=True,
        fill=NAVY,
        align="center",
        valign="center",
    )
    draw_text_fit(
        d,
        (80, 190, W - 80, 240),
        "Multiple-Choice Strategic Buzzing Under Incremental Clues",
        max_size=22,
        min_size=14,
        fill=TEXT_SOFT,
        align="center",
        valign="center",
    )
    rounded(d, (300, 270, 660, 380), WHITE, outline=BORDER, radius=16, width=2)
    draw_text_fit(
        d,
        (320, 284, 640, 370),
        "CS234 Final Project\n\nKathleen Weng\nImran Hassan\nAnkit Aggarwal",
        max_size=18,
        min_size=12,
        fill=TEXT,
        align="center",
        valign="center",
    )
    draw_text_fit(
        d,
        (80, 400, W - 80, 440),
        "March 2026",
        max_size=16,
        min_size=12,
        fill=TEXT_SOFT,
        align="center",
        valign="center",
    )
    footer(d, spec["footer"])
    return img

def render_problem_slide(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Problem")
    rounded(d, (48, 96, 912, 482), WHITE, outline=BORDER, radius=16, width=2)
    draw_bullets(d, (80, 120, 880, 460), [
        "Quiz bowl questions reveal evidence incrementally",
        "A good system must decide WHEN to buzz, not just WHAT to pick",
        "Buzz too early: higher risk of a wrong answer",
        "Buzz too late: lower strategic value and less chance to beat an opponent",
        "We study this in a multiple-choice setting so the answer space is controlled and evaluation is reproducible",
    ], bullet_color=BLUE)
    footer(d, spec["footer"])
    return img

def render_background_slide(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Background And Setup")
    rounded(d, (48, 96, 460, 482), WHITE, outline=BORDER, radius=16, width=2)
    draw_text_fit(d, (68, 110, 440, 140), "Sequential decision problem", max_size=20, min_size=13, bold=True, fill=NAVY)
    draw_bullets(d, (68, 152, 440, 340), [
        "Model quiz bowl as sequential decision over partial clues",
        "Question prefixes over time",
        "K = 4 answer options",
        "One correct option plus three distractors",
    ], bullet_color=BLUE)
    rounded(d, (500, 96, 912, 482), WHITE, outline=BORDER, radius=16, width=2)
    draw_text_fit(d, (520, 110, 892, 140), "Two policy families", max_size=20, min_size=13, bold=True, fill=PURPLE)
    pill(d, (540, 170, 872, 220), "Belief-feature buzzers", BLUE_SOFT, outline=BLUE, text_color=NAVY)
    draw_bullets(d, (540, 240, 872, 330), [
        "Threshold, softmax-profile, and Bayesian baselines",
        "PPO on structured observations",
    ], bullet_color=BLUE)
    pill(d, (540, 340, 872, 390), "T5 text-policy buzzers", ORANGE_SOFT, outline=ORANGE, text_color=NAVY)
    draw_bullets(d, (540, 410, 872, 470), [
        "End-to-end supervised warm start plus PPO",
    ], bullet_color=ORANGE)
    footer(d, spec["footer"])
    return img

def render_why_mc_slide(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Why Multiple Choice?")
    rounded(d, (48, 96, 500, 340), WHITE, outline=BORDER, radius=16, width=2)
    draw_text_fit(d, (68, 112, 480, 140), "Advantages", max_size=20, min_size=13, bold=True, fill=GREEN)
    draw_bullets(d, (68, 154, 480, 320), [
        "Eliminates aliasing and grading complexity",
        "Isolates the buzzing decision itself",
        "Makes evaluation controlled and reproducible",
    ], bullet_color=GREEN)
    rounded(d, (48, 360, 500, 482), WHITE, outline=BORDER, radius=16, width=2)
    draw_text_fit(d, (68, 374, 480, 400), "Challenge", max_size=20, min_size=13, bold=True, fill=RED)
    draw_bullets(d, (68, 412, 480, 472), [
        "Naive distractors create artifacts",
        "Answer generation quality still matters",
    ], bullet_color=RED)
    rounded(d, (530, 96, 912, 482), BLUE_SOFT, outline=BLUE, radius=16, width=2)
    draw_text_fit(d, (550, 112, 892, 140), "Design goals", max_size=20, min_size=13, bold=True, fill=NAVY)
    draw_bullets(d, (550, 160, 892, 460), [
        "Keep the answer space constrained",
        "Make options hard enough that the agent must use clues",
        "Use anti-artifact guards: alias collision, token overlap, length ratio, and question-text overlap checks",
    ], bullet_color=BLUE)
    footer(d, spec["footer"])
    return img

def render_method_slide(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Method Overview")
    rounded(d, (48, 96, 912, 482), WHITE, outline=BORDER, radius=16, width=2)
    steps = [
        ("1. Load tossups", BLUE_SOFT, BLUE),
        ("2. Build answer profiles", PURPLE_SOFT, PURPLE),
        ("3. Construct MC questions", ORANGE_SOFT, ORANGE),
        ("4. Score with a likelihood model", GREEN_SOFT, GREEN),
        ("5. Convert beliefs into observations", BLUE_SOFT, BLUE),
        ("6. Run buzzer agents", PURPLE_SOFT, PURPLE),
        ("7. Evaluate", GREEN_SOFT, GREEN),
    ]
    y = 116
    for i, (label, bg, outline) in enumerate(steps):
        x0 = 80 if i % 2 == 0 else 480
        rounded(d, (x0, y, x0 + 360, y + 42), bg, outline=outline, radius=12, width=2)
        draw_text_fit(d, (x0 + 14, y + 8, x0 + 346, y + 36), label, max_size=17, min_size=11, bold=True, fill=NAVY, valign="center")
        if i % 2 == 1:
            y += 52
    footer(d, spec["footer"])
    return img

def render_contribution_slide(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Our Contribution")
    rounded(d, (48, 96, 912, 482), WHITE, outline=BORDER, radius=16, width=2)
    draw_text_fit(
        d,
        (68, 112, 892, 144),
        "We study pyramidal quiz bowl under a restricted multiple-choice action space with reinforcement learning.",
        max_size=19,
        min_size=12,
        bold=True,
        fill=NAVY,
    )
    draw_bullets(d, (68, 160, 892, 460), [
        "Multiple-choice POMDP formulation for quiz bowl",
        "PPO to learn when to wait versus buzz and answer",
        "Artifact-resistant MC construction with several distractor strategies",
        "Direct tests of whether learned policies use clues or answer-choice patterns",
        "Calibration-focused evaluation with ECE, Brier, and S_q beyond raw accuracy",
    ], bullet_color=PURPLE)
    footer(d, spec["footer"])
    return img

def render_expected_results_slide(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Smoke Results Snapshot")
    rounded(d, (48, 90, 912, 482), WHITE, outline=BORDER, radius=16, width=2)

    best_name, best_metrics = best_baseline_result(EVAL_REPORT)
    ppo = EVAL_REPORT.get("ppo_summary", {})

    rounded(d, (68, 106, 892, 138), BLUE_SOFT, outline=BLUE, radius=8, width=1)
    draw_text_fit(
        d,
        (84, 112, 876, 132),
        f"Best baseline by mean S_q: {best_name}",
        max_size=14,
        min_size=10,
        bold=True,
        fill=NAVY,
        align="center",
        valign="center",
    )

    col_xs = [68, 338, 578, 770]
    col_ws = [250, 220, 180, 122]
    headers_txt = ["Metric", "Best baseline", "PPO", "Reading"]
    for cx, cw, ht in zip(col_xs, col_ws, headers_txt):
        rounded(d, (cx, 150, cx + cw, 184), NAVY, outline=NAVY, radius=8, width=1)
        draw_text_fit(d, (cx + 8, 154, cx + cw - 8, 180), ht, max_size=15, min_size=10, bold=True, fill=WHITE, align="center", valign="center")

    rows = [
        ("Mean S_q", fmt_num(best_metrics.get("mean_sq")), fmt_num(ppo.get("mean_sq")), "higher better"),
        ("Buzz accuracy", fmt_pct(best_metrics.get("buzz_accuracy")), fmt_pct(ppo.get("buzz_accuracy")), "higher better"),
        ("Mean buzz step", fmt_num(best_metrics.get("mean_buzz_step"), 2), fmt_num(ppo.get("mean_buzz_step"), 2), "timing profile"),
        ("Reward-like", fmt_num(best_metrics.get("mean_reward_like")), fmt_num(ppo.get("mean_reward_like")), "higher better"),
        ("ECE", fmt_num(best_metrics.get("ece")), fmt_num(ppo.get("ece")), "lower better"),
        ("Brier", fmt_num(best_metrics.get("brier")), fmt_num(ppo.get("brier")), "lower better"),
    ]
    for ri, (metric, baseline_val, ppo_val, reading) in enumerate(rows):
        ry = 194 + ri * 42
        bg = "#FAFBFD" if ri % 2 == 0 else WHITE
        for cx, cw, val in zip(col_xs, col_ws, [metric, baseline_val, ppo_val, reading]):
            rounded(d, (cx, ry, cx + cw, ry + 38), bg, outline=BORDER, radius=6, width=1)
            color = TEXT_SOFT if val in {"higher better", "lower better", "timing profile"} else TEXT
            draw_text_fit(d, (cx + 8, ry + 5, cx + cw - 8, ry + 33), val, max_size=14, min_size=9, fill=color, align="center", valign="center")

    rounded(d, (100, 444, 860, 470), ORANGE_SOFT, outline=ORANGE, radius=10, width=1)
    draw_text_fit(
        d,
        (116, 448, 844, 466),
        "Use this as a smoke-test snapshot, not a final leaderboard: compare S_q, calibration, and timing together.",
        max_size=14,
        min_size=10,
        fill=ORANGE,
        align="center",
        valign="center",
    )
    footer(d, spec["footer"])
    return img

def render_why_ppo_less_accurate(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "How To Read PPO Metrics")

    best_name, best_metrics = best_baseline_result(EVAL_REPORT)
    ppo = EVAL_REPORT.get("ppo_summary", {})

    rounded(d, (48, 96, 460, 482), WHITE, outline=BORDER, radius=16, width=2)
    draw_text_fit(d, (68, 112, 440, 140), "Why raw accuracy can dip", max_size=18, min_size=12, bold=True, fill=NAVY)
    draw_bullets(d, (68, 156, 440, 380), [
        "PPO learns a timing policy, not just a final guess",
        "Waiting can avoid low-confidence answers",
        "That may lower raw buzz accuracy on some slices",
        "Reward design and training stability matter a lot",
        "So accuracy alone is not the right summary statistic",
    ], bullet_color=ORANGE)
    rounded(d, (68, 400, 440, 466), ORANGE_SOFT, outline=ORANGE, radius=10, width=1)
    draw_text_fit(
        d,
        (82, 408, 426, 458),
        f"Current smoke: {best_name} buzz acc {fmt_pct(best_metrics.get('buzz_accuracy'))} vs PPO {fmt_pct(ppo.get('buzz_accuracy'))}",
        max_size=13,
        min_size=9,
        fill=ORANGE,
        align="center",
        valign="center",
    )

    rounded(d, (500, 96, 912, 482), WHITE, outline=BORDER, radius=16, width=2)
    draw_text_fit(d, (520, 112, 892, 140), "What to compare alongside it", max_size=18, min_size=12, bold=True, fill=GREEN)
    draw_bullets(d, (520, 156, 892, 360), [
        "Mean S_q or task reward",
        "Mean buzz step / timing profile",
        "Calibration at buzz time",
        "Controls that test clue use versus answer-artifact use",
        "Whether the policy abstains or times out strategically",
    ], bullet_color=GREEN)
    rounded(d, (520, 380, 892, 466), GREEN_SOFT, outline=GREEN, radius=10, width=1)
    draw_text_fit(
        d,
        (534, 388, 878, 458),
        "A PPO buzzer is only better if its timing behavior improves the metrics you actually care about.",
        max_size=14,
        min_size=10,
        fill=GREEN,
        align="center",
        valign="center",
    )

    footer(d, spec["footer"])
    return img

def render_intro(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)

    header(d, "Quiz bowl MC \"buzzing\" is optimal stopping with improving evidence")

    card(d, (48, 96, 458, 486), EXAMPLE_TITLE)
    draw_text_fit(d, (76, 136, 426, 152),
                  "Original tossup (verbatim)", max_size=14, min_size=10, bold=True, fill=TEXT_SOFT)
    draw_text_fit(d, (76, 154, 426, 168),
                  EXAMPLE_SOURCE, max_size=12, min_size=9, fill=TEXT_SOFT)
    draw_highlighted_text_fit(d, (76, 176, 426, 274),
                              EXAMPLE_TOSSUP_VERBATIM,
                              EXAMPLE_TOSSUP_HIGHLIGHTS,
                              max_size=12, min_size=6, fill=TEXT_SOFT, line_gap=0)
    draw_text_fit(d, (76, 282, 426, 296),
                  EXAMPLE_ANSWERLINE,
                  max_size=11, min_size=8, bold=True, fill=NAVY, line_gap=1)
    draw_text_fit(d, (76, 298, 426, 316),
                  EXAMPLE_ACCEPTS,
                  max_size=8, min_size=6, fill=TEXT_SOFT, line_gap=1)
    rounded(d, (86, 326, 430, 376), "#F7F8FA", outline=BORDER, radius=12, width=1)
    draw_text_fit(d, (98, 336, 418, 368),
                  'Clue progression\nEarly clues are obscure, middle clues narrow the field, and the final "memoryless chain" clue is a giveaway.',
                  max_size=13, min_size=9, fill=TEXT_SOFT, line_gap=2)
    rounded(d, (86, 388, 430, 458), WHITE, outline=BLUE, radius=16, width=2)
    draw_text_fit(d, (96, 396, 420, 450),
                  'RL intuition\nThose repeated anaphoric phrases all resolve to Markov, and the strategic decision is when to "buzz."',
                  max_size=20, min_size=12, bold=True, fill=NAVY)

    card(d, (506, 96, 916, 486), "Presentation multiple-choice setting", title_color=PURPLE)
    draw_text_fit(d, (536, 142, 886, 174),
                  'The answer options are fixed from the start, using a cleaner mathematician-only set:',
                  max_size=20, min_size=13, fill=TEXT_SOFT)
    draw_options_grid(d, (536, 184, 884, 360), EXAMPLE_OPTIONS)
    rounded(d, (536, 428, 886, 486), "#EAE6FB", outline=BLUE, radius=16, width=2)
    draw_text_fit(d, (548, 438, 874, 478),
                  'Decision rule\nAt each "prefix," should the policy "WAIT" or "BUZZ" on one listed answer?',
                  max_size=18, min_size=11, bold=True, fill=NAVY)

    footer(d, spec["footer"])
    return img

def render_section(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, spec["title"])
    rounded(d, (255, 170, 705, 332), WHITE, outline=spec["accent"], radius=14, width=2)
    d.line((320, 206, 640, 206), fill=spec["accent"], width=4)
    draw_text_fit(d, (300, 226, 660, 284), spec["section_title"], max_size=26, min_size=15, bold=True, fill=spec["accent"], align="center", valign="center")
    draw_text_fit(d, (292, 286, 668, 312), spec["footer_small"], max_size=12, min_size=10, fill=TEXT_SOFT, align="center")
    return img

def render_posterior(spec: Dict) -> Image.Image:
    stage = spec["stage"]
    stage_data = EXAMPLE_POSTERIOR_STAGES[stage]
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, 'As more "clues" arrive, the Markov posterior collapses onto one option')

    rounded(d, (44, 86, 916, 482), WHITE, outline=BORDER, radius=16, width=2)
    draw_progress_dots(d, (72, 110, 664, 136), stage, 5)
    draw_text_fit(d, (72, 142, 664, 174), "\"prefix t\": only the clues seen so far are visible", max_size=15, min_size=10, fill=TEXT_SOFT)

    rounded(d, (72, 176, 664, 232), WHITE, outline=BORDER, radius=12, width=2)
    draw_text_fit(d, (88, 188, 648, 220), stage_data["prefix"], max_size=17, min_size=11, fill=TEXT)

    draw_posterior_bars(d, (72, 254, 664, 388), stage_data["probs"])

    is_wait = stage_data["decision"] == "WAIT"
    rounded(d, (694, 182, 758, 254), WHITE if is_wait else GREEN_SOFT,
            outline=BORDER if is_wait else GREEN, radius=14, width=2)
    draw_text_fit(d, (700, 197, 752, 240), stage_data["decision"], max_size=18, min_size=12, bold=True, align="center", valign="center", fill=ORANGE if is_wait else GREEN)

    if is_wait:
        rounded(d, (774, 170, 902, 286), ORANGE_SOFT, outline=ORANGE, radius=14, width=2)
        draw_text_fit(d, (786, 182, 890, 274),
                      stage_data["callout"],
                      max_size=17, min_size=10, fill=ORANGE)
    else:
        rounded(d, (774, 170, 902, 286), GREEN_SOFT, outline=GREEN, radius=14, width=2)
        draw_text_fit(d, (786, 182, 890, 274),
                      stage_data["callout"],
                      max_size=17, min_size=10, fill=GREEN)

    rounded(d, (128, 420, 832, 462), "#F7F8FA", outline=BORDER, radius=10, width=1)
    draw_text_fit(d, (142, 430, 818, 454),
                  'With a fixed answer set, the policy is not discovering new options. It is deciding when the current posterior over Markov vs. the three distractors is sharp enough to "act."',
                  max_size=14, min_size=9, fill=TEXT_SOFT, align="center", valign="center")

    footer(d, spec["footer"])
    return img

def render_value_chart(spec: Dict) -> Image.Image:
    stage = spec["stage"]
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, '"Buzz" when acting now is worth more than waiting in the Markov example')

    rounded(d, (50, 92, 910, 470), WHITE, outline=BORDER, radius=16, width=2)
    draw_line_chart(d, (90, 130, 850, 382), stage)

    rounded(d, (160, 408, 800, 448), "#F7F8FA", outline=BORDER, radius=10, width=1)
    draw_text_fit(d, (178, 416, 782, 440), EXAMPLE_VALUE_MESSAGES[stage], max_size=15, min_size=10, fill=TEXT_SOFT, align="center", valign="center")

    footer(d, spec["footer"])
    return img

def render_abstain(spec: Dict) -> Image.Image:
    stage = spec["stage"]
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "\"ABSTAIN\" means wait, not \"none of the above\"")

    rounded(d, (46, 96, 914, 472), WHITE, outline=BORDER, radius=16, width=2)
    card(d, (72, 130, 396, 388), "The correct answer is already in the \"set\"")
    draw_options_grid(d, (92, 176, 376, 312), EXAMPLE_OPTIONS)
    draw_text_fit(d, (92, 326, 376, 374),
                  '"Abstaining" means one more "clue" is worth more than committing, even though Markov is already option A.',
                  max_size=16, min_size=10, fill=TEXT_SOFT, align="center", valign="center")

    card(d, (426, 130, 888, 388), 'What changes over "time" in the Markov example')
    draw_posterior_bars(d, (452, 186, 790, 300), [
        EXAMPLE_POSTERIOR_STAGES[0]["probs"],
        EXAMPLE_POSTERIOR_STAGES[1]["probs"],
        EXAMPLE_POSTERIOR_STAGES[3]["probs"],
    ][stage])

    draw_text_fit(d, (500, 316, 706, 348), "ABSTAIN", max_size=26, min_size=16, bold=True, fill=ORANGE, align="center")
    draw_text_fit(d, (492, 344, 716, 382), "means continue because another \"clue\" still has value", max_size=16, min_size=10, fill=TEXT_SOFT, align="center")

    d.line((760, 334, 856, 370), fill=RED, width=3)
    d.line((760, 370, 856, 334), fill=RED, width=3)
    draw_text_fit(d, (738, 310, 880, 334), "None-of-the-Above", max_size=15, min_size=9, bold=True, fill=RED, align="center")

    rounded(d, (160, 416, 800, 454), "#F7F8FA", outline=BORDER, radius=10, width=1)
    draw_text_fit(d, (176, 424, 784, 446), EXAMPLE_ABSTAIN_MESSAGES[stage], max_size=13, min_size=9, fill=TEXT_SOFT, align="center", valign="center")

    footer(d, spec["footer"])
    return img

def render_factorization(spec: Dict) -> Image.Image:
    stage = spec["stage"]
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Implementation idea: separate answer quality from \"buzz\" timing")

    rounded(d, (46, 96, 914, 472), WHITE, outline=BORDER, radius=16, width=2)

    rounded(d, (84, 162, 214, 214), WHITE, outline=BORDER, radius=12, width=2)
    draw_text_fit(d, (96, 170, 202, 206), "\"prefix h_t\"\n+ option set C", max_size=18, min_size=11, align="center", valign="center")

    rounded(d, (300, 122, 500, 238), BLUE_SOFT if stage >= 0 else WHITE, outline=BLUE, radius=14, width=2)
    draw_text_fit(d, (316, 140, 484, 176), "Answer module", max_size=22, min_size=13, bold=True, align="center")
    draw_text_fit(d, (320, 178, 480, 220), "outputs p_ans(i | h_t)", max_size=20, min_size=12, align="center", valign="center")

    rounded(d, (300, 286, 500, 402), ORANGE_SOFT if stage >= 1 else WHITE, outline=ORANGE, radius=14, width=2)
    draw_text_fit(d, (316, 304, 484, 340), "Stop module", max_size=22, min_size=13, bold=True, align="center")
    draw_text_fit(d, (320, 344, 480, 384), "outputs p_buzz(h_t)", max_size=20, min_size=12, align="center", valign="center")

    d.line((214, 188, 300, 178), fill=BLUE, width=4)
    d.line((214, 188, 300, 344), fill=ORANGE, width=4)

    rounded(d, (608, 180, 866, 344), PURPLE_SOFT if stage >= 1 else WHITE, outline=PURPLE, radius=14, width=2)
    draw_text_fit(d, (624, 198, 850, 234), "Flat action interface", max_size=22, min_size=13, bold=True, align="center")
    draw_text_fit(d, (626, 248, 848, 320),
                  "P(\"WAIT\") = 1 - p_buzz\nP(\"BUZZ i\") = p_buzz * p_ans(i)",
                  max_size=18, min_size=10, align="center", valign="center")

    rounded(d, (158, 424, 804, 462), "#F7F8FA", outline=BORDER, radius=10, width=1)
    draw_text_fit(d, (176, 432, 786, 454),
                  "Same semantics, cleaner code: factor answer choice and \"stop timing\" internally, then flatten only if the RL stack needs it.",
                  max_size=13, min_size=9, fill=TEXT_SOFT, align="center", valign="center")

    footer(d, spec["footer"])
    return img

def render_recipe(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "A practical training recipe")

    rounded(d, (48, 104, 912, 472), WHITE, outline=BORDER, radius=16, width=2)

    steps = [
        ("1. train the answer model", "learn p_ans(i | h_t)", BLUE_SOFT, BLUE),
        ("2. warm-start the stop head", "learn when to \"buzz\"", ORANGE_SOFT, ORANGE),
        ("3. PPO fine-tuning", "optimize task reward", GREEN_SOFT, GREEN),
    ]
    xs = [84, 328, 572]
    for x, (a, b, fill, outline) in zip(xs, steps):
        rounded(d, (x, 202, x + 200, 314), fill, outline=outline, radius=14, width=2)
        draw_text_fit(d, (x + 12, 218, x + 188, 246), a, max_size=20, min_size=12, bold=True, fill=NAVY, align="center")
        draw_text_fit(d, (x + 14, 254, x + 186, 296), b, max_size=18, min_size=11, fill=TEXT_SOFT, align="center", valign="center")

    rounded(d, (154, 394, 806, 438), "#F7F8FA", outline=BORDER, radius=10, width=1)
    draw_text_fit(d, (170, 404, 790, 430),
                  "RL can then focus mostly on \"timing\" instead of learning answer choice and stopping from scratch at the same time.",
                  max_size=15, min_size=10, fill=TEXT_SOFT, align="center", valign="center")
    footer(d, spec["footer"])
    return img

def render_pipeline(spec: Dict) -> Image.Image:
    stage = spec["stage"]
    img = make_canvas()
    d = ImageDraw.Draw(img)

    titles = [
        "Training setup in code",
        "Evaluation protocol",
        "Smoke results snapshot",
    ]
    header(d, titles[stage])
    rounded(d, (48, 98, 912, 474), WHITE, outline=BORDER, radius=16, width=2)

    if stage == 0:
        rounded(d, (74, 138, 886, 230), BLUE_SOFT, outline=BLUE, radius=12, width=2)
        draw_text_fit(
            d,
            (92, 154, 868, 214),
            "scripts/build_mc_dataset.py --smoke -> scripts/run_baselines.py --smoke -> scripts/train_ppo.py --smoke",
            max_size=18,
            min_size=11,
            bold=True,
            fill=NAVY,
            align="center",
            valign="center",
        )

        card(d, (88, 258, 370, 420), "Data + baseline prep", title_color=BLUE)
        draw_bullets(d, (104, 298, 352, 398), [
            "build MC prefixes",
            "set distractor strategy",
            "run baseline sweeps",
        ], bullet_color=BLUE)

        card(d, (396, 258, 678, 420), "RL fine-tune", title_color=ORANGE)
        draw_bullets(d, (412, 298, 660, 398), [
            "initialize policy",
            "optimize timing reward",
            "trade speed versus accuracy",
        ], bullet_color=ORANGE)

        card(d, (704, 258, 886, 420), "Artifacts", title_color=GREEN)
        draw_text_fit(
            d,
            (718, 300, 872, 406),
            "artifacts/smoke/\n- mc_dataset.json\n- baseline_summary.json\n- ppo_runs.json",
            max_size=14,
            min_size=9,
            fill=TEXT_SOFT,
            valign="top",
        )

    elif stage == 1:
        card(d, (88, 136, 472, 320), "Core metrics", title_color=BLUE)
        draw_bullets(d, (106, 176, 454, 294), [
            "mean S_q",
            "buzz accuracy",
            "mean buzz step",
            "ECE and Brier",
        ], bullet_color=BLUE)

        card(d, (488, 136, 872, 320), "Control checks", title_color=PURPLE)
        draw_bullets(d, (506, 176, 854, 294), [
            "choices-only baseline",
            "shuffle test",
            "alias substitution",
            "per-category slices",
        ], bullet_color=PURPLE)

        rounded(d, (88, 344, 872, 430), "#F7F8FA", outline=BORDER, radius=10, width=1)
        draw_text_fit(
            d,
            (106, 360, 854, 416),
            "Evaluation source of truth: artifacts/smoke/evaluation_report.json",
            max_size=18,
            min_size=10,
            fill=TEXT_SOFT,
            align="center",
            valign="center",
        )

    else:
        full_eval = EVAL_REPORT.get("full_eval", {})
        ppo = EVAL_REPORT.get("ppo_summary", {})
        best_name, best_metrics = best_baseline_result(EVAL_REPORT)

        card(d, (88, 136, 472, 424), "Best baseline (mean S_q)", title_color=BLUE)
        draw_text_fit(d, (106, 178, 454, 208), best_name, max_size=20, min_size=11, bold=True, fill=NAVY, align="center")
        draw_bullets(d, (108, 220, 454, 382), [
            f"mean S_q = {fmt_num(best_metrics.get('mean_sq'))}",
            f"buzz acc = {fmt_pct(best_metrics.get('buzz_accuracy'))}",
            f"mean step = {fmt_num(best_metrics.get('mean_buzz_step'), 2)}",
            f"n = {fmt_num(best_metrics.get('n'), 0)}",
        ], bullet_color=BLUE)

        card(d, (488, 136, 872, 424), "PPO smoke", title_color=ORANGE)
        draw_bullets(d, (508, 178, 854, 382), [
            f"mean S_q = {fmt_num(ppo.get('mean_sq'))}",
            f"buzz acc = {fmt_pct(ppo.get('buzz_accuracy'))}",
            f"reward-like = {fmt_num(ppo.get('mean_reward_like'))}",
            f"ECE/Brier = {fmt_num(ppo.get('ece'))}/{fmt_num(ppo.get('brier'))}",
        ], bullet_color=ORANGE)

        rounded(d, (88, 438, 872, 464), BLUE_SOFT, outline=BLUE, radius=10, width=2)
        draw_text_fit(
            d,
            (106, 442, 854, 460),
            f"Full eval: S_q={fmt_num(full_eval.get('mean_sq'))}, accuracy={fmt_pct(full_eval.get('buzz_accuracy'))}, mean step={fmt_num(full_eval.get('mean_buzz_step'), 2)}",
            max_size=13,
            min_size=9,
            fill=NAVY_DARK,
            align="center",
            valign="center",
        )

    footer(d, spec["footer"])
    return img

def render_summary(spec: Dict) -> Image.Image:
    stage = spec["stage"]
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Quiz bowl MC \"buzzing,\" in one picture")

    rounded(d, (48, 98, 912, 474), WHITE, outline=BORDER, radius=16, width=2)
    rounded(d, (74, 144, 650, 418), "#FAFBFD", outline=BORDER, radius=12, width=2)

    bullets = [
        ('1. the Markov "clues" go from Bayesian-network jargon to a giveaway', BLUE),
        ('2. the posterior over the fixed four choices sharpens onto Markov', ORANGE),
        ('3. the key RL decision is when to stop "waiting" and buzz A', PURPLE),
        ('4. "abstain" means another clue is worth it, not "None-of-the-Above"', GREEN),
    ]
    y = 174
    visible = bullets[: stage + 2]
    for i, (txt, color) in enumerate(visible):
        draw_text_fit(d, (96, y, 624, y + 24), txt, max_size=18, min_size=11, bold=(i == 0), fill=color)
        y += 48

    rounded(d, (94, 350, 620, 398), WHITE, outline=BORDER, radius=10, width=1)
    draw_text_fit(d, (108, 360, 606, 388),
                  'Mental model: selective prediction + optimal stopping over a fixed answer "set" that already contains Markov.',
                  max_size=15, min_size=10, fill=TEXT_SOFT, align="center", valign="center")

    rounded(d, (688, 146, 892, 256), "#EEF2F8", outline=BORDER, radius=12, width=2)
    draw_text_fit(d, (702, 160, 878, 188), "For RL people", max_size=18, min_size=12, bold=True, fill=NAVY, align="center")
    draw_text_fit(d, (702, 196, 878, 242),
                  "At each \"prefix,\" compare the value of acting now with the value of one more \"clue.\"",
                  max_size=15, min_size=10, fill=TEXT_SOFT, align="center", valign="center")

    if stage >= 1:
        rounded(d, (688, 282, 892, 388), GREEN_SOFT, outline=GREEN, radius=12, width=2)
        draw_text_fit(d, (702, 296, 878, 374),
                      "If you keep a flat action interface, factor it internally:\nP(\"WAIT\")=1-p_buzz\nP(\"BUZZ i\")=p_buzz * p_ans(i)",
                      max_size=15, min_size=9, fill=GREEN, align="center", valign="center")

    if stage >= 2:
        rounded(d, (86, 434, 860, 464), BLUE_SOFT, outline=BLUE, radius=10, width=2)
        draw_text_fit(d, (100, 440, 846, 458),
                      "Key takeaway: \"pyramidal\" quiz bowl turns multiple-choice QA into an optimal-stopping problem because evidence improves before action.",
                      max_size=13, min_size=9, fill=NAVY_DARK, align="center", valign="center")

    footer(d, spec["footer"])
    return img

def render_evaluation_slide(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Evaluation")
    rounded(d, (48, 96, 460, 482), WHITE, outline=BORDER, radius=16, width=2)
    draw_text_fit(d, (68, 112, 440, 140), "Metrics", max_size=20, min_size=13, bold=True, fill=NAVY)
    draw_bullets(d, (68, 154, 440, 400), [
        "S_q (QANTA system score)",
        "Buzz accuracy",
        "Mean buzz step",
        "Calibration-at-buzz / ECE / Brier",
        "Per-category accuracy",
    ], bullet_color=BLUE)
    rounded(d, (500, 96, 912, 482), WHITE, outline=BORDER, radius=16, width=2)
    draw_text_fit(d, (520, 112, 892, 140), "Controls", max_size=20, min_size=13, bold=True, fill=RED)
    draw_bullets(d, (520, 154, 892, 400), [
        "Choices-only baseline (no question text)",
        "Shuffle clue order",
        "Alias substitution",
        "Distractor difficulty variation",
    ], bullet_color=RED)
    rounded(d, (120, 440, 840, 476), "#F7F8FA", outline=BORDER, radius=10, width=1)
    draw_text_fit(
        d,
        (136, 448, 824, 470),
        "A buzzer can look strong while exploiting answer artifacts instead of clue content.",
        max_size=14,
        min_size=10,
        fill=TEXT_SOFT,
        align="center",
        valign="center",
    )
    footer(d, spec["footer"])
    return img

def render_references_slide(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "References")
    rounded(d, (48, 96, 912, 482), WHITE, outline=BORDER, radius=16, width=2)
    refs = [
        "Rodriguez et al. (2019) — Quizbowl: The case for incremental QA",
        "Schulman et al. — Proximal Policy Optimization (PPO)",
        "Raffel et al. — T5: Exploring the Limits of Transfer Learning",
        "Boyd-Graber and Daume (2013) — Bayesian thinking on your feet",
        "Boyd-Graber and Borschinger (2020) — What QA can learn from trivia nerds",
        "Balepur et al. (2025) — Test-time reasoners are strategic MC test-takers",
        "Kalai et al. (2025) — Why language models hallucinate",
        "UMD / QANTA (2024) — S_q evaluation metric",
    ]
    draw_bullets(d, (80, 120, 880, 460), refs, bullet_color=NAVY)
    footer(d, spec["footer"])
    return img

# ============================================================
# Dispatcher
# ============================================================
RENDERERS = {
    "title_slide": render_title_slide,
    "problem_slide": render_problem_slide,
    "background_slide": render_background_slide,
    "why_mc_slide": render_why_mc_slide,
    "method_slide": render_method_slide,
    "contribution_slide": render_contribution_slide,
    "expected_results_slide": render_expected_results_slide,
    "why_ppo_less_accurate": render_why_ppo_less_accurate,
    "intro": render_intro,
    "section": render_section,
    "posterior": render_posterior,
    "value_chart": render_value_chart,
    "abstain": render_abstain,
    "factorization": render_factorization,
    "recipe": render_recipe,
    "pipeline": render_pipeline,
    "evaluation_slide": render_evaluation_slide,
    "summary": render_summary,
    "references_slide": render_references_slide,
}

# ============================================================
# Build frames
# ============================================================
frames: List[Image.Image] = []
for spec in SCENES:
    frames.append(RENDERERS[spec["kind"]](spec))

assert len(frames) == len(SCENES)

# ============================================================
# Durations
# ============================================================
durations = []
for spec in SCENES:
    kind = spec["kind"]
    stage = spec.get("stage")
    if kind == "title_slide":
        durations.append(1000)
    elif kind in {"section", "references_slide"}:
        durations.append(900)
    elif kind in {"problem_slide", "background_slide", "why_mc_slide", "method_slide", "contribution_slide", "expected_results_slide", "why_ppo_less_accurate", "evaluation_slide"}:
        durations.append(800)
    elif kind == "summary" and stage == 2:
        durations.append(850)
    else:
        durations.append(450)

# ============================================================
# Save frames
# ============================================================
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
for idx, fr in enumerate(frames, start=1):
    fr.save(FRAMES_DIR / f"frame_{idx:02d}.png")

# ============================================================
# Save GIF
# ============================================================
pal_frames = [fr.convert("P", palette=Image.Palette.ADAPTIVE) for fr in frames]
pal_frames[0].save(
    GIF_OUT,
    save_all=True,
    append_images=pal_frames[1:],
    duration=durations,
    loop=0,
    optimize=False,
    disposal=2,
)

# ============================================================
# Save contact sheet
# ============================================================
def make_contact(frames: List[Image.Image], cols: int = 3):
    thumb_w, thumb_h = 248, 139
    cell_w, cell_h = 260, 185
    rows = ceil(len(frames) / cols)

    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "#ECECEC")
    draw = ImageDraw.Draw(sheet)
    label_font = load_font(16, bold=False)

    for idx, fr in enumerate(frames):
        row = idx // cols
        col = idx % cols
        x0 = col * cell_w
        y0 = row * cell_h

        thumb = fr.copy().convert("RGB")
        thumb.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        tx = x0 + (cell_w - thumb.width) // 2
        ty = y0 + 6
        sheet.paste(thumb, (tx, ty))

        label = f"Frame {idx + 1}"
        draw.text((x0 + 6, y0 + 150), label, font=label_font, fill="#111111")

    return sheet

contact = make_contact(frames, cols=3)
contact.save(CONTACT_OUT)

print(f"Saved GIF to: {GIF_OUT}")
print(f"Saved contact sheet to: {CONTACT_OUT}")
print(f"Saved frames to: {FRAMES_DIR}")
