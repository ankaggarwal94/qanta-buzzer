from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

# ============================================================
# Paths
# ============================================================
OUT_DIR = Path("/mnt/data/quizbowl_mc_stopping_data_driven")
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

def draw_options_grid(draw: ImageDraw.ImageDraw, box):
    x0, y0, x1, y1 = box
    pad = 14
    gap = 14
    cell_w = (x1 - x0 - 2 * pad - gap) // 2
    cell_h = 84
    labels = [("A", "A: option A"), ("B", "B: option B"), ("C", "C: option C"), ("D", "D: option D")]
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
        draw_text_fit(draw, (cx + 42, cy + 18, cx + cell_w - 12, cy + cell_h - 12), txt, max_size=16, min_size=11, valign="center")

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
    draw_text_fit(draw, (legend_x + 40, y0 + 28, x1 - 30, y0 + 46), "commit now value", max_size=12, min_size=9, fill=TEXT_SOFT)
    draw.line((legend_x + 12, y0 + 56, legend_x + 34, y0 + 56), fill=ORANGE, width=3)
    draw_text_fit(draw, (legend_x + 40, y0 + 44, x1 - 30, y0 + 62), "wait value", max_size=12, min_size=9, fill=TEXT_SOFT)

    notes = [
        "step 1 commit = 0.10\nwait = 0.64",
        "step 2 commit = 0.28\nwait = 0.58",
        "step 3 commit = 0.46\nwait = 0.50",
        "step 4 commit = 0.62\nwait = 0.40",
        "step 5 commit = 0.72\nwait = 0.30",
    ]
    rounded(draw, (x0 + 126, y0 + 52, x0 + 236, y0 + 104), GREEN_SOFT if phase >= 2 else ORANGE_SOFT,
            outline=GREEN if phase >= 2 else ORANGE, radius=12, width=2)
    draw_text_fit(draw, (x0 + 136, y0 + 60, x0 + 226, y0 + 98), notes[idx], max_size=12, min_size=9, fill=TEXT_SOFT, align="center", valign="center")

# ============================================================
# Scene specs
# ============================================================
SCENES: List[Dict] = [
    {
        "kind": "intro",
        "footer": "Frame 1 • setup",
    },
    {
        "kind": "section",
        "title": "Quiz bowl MC stopping",
        "section_title": "Section 1: posterior sharpens as\nclues arrive",
        "accent": BLUE,
        "footer_small": "next: see the same question as an RL stopping problem",
    },
    {
        "kind": "posterior",
        "stage": 0,
        "footer": "Frame 3 • posterior sharpening",
    },
    {
        "kind": "posterior",
        "stage": 1,
        "footer": "Frame 4 • posterior sharpening",
    },
    {
        "kind": "posterior",
        "stage": 2,
        "footer": "Frame 5 • posterior sharpening",
    },
    {
        "kind": "posterior",
        "stage": 3,
        "footer": "Frame 6 • posterior sharpening",
    },
    {
        "kind": "posterior",
        "stage": 4,
        "footer": "Frame 7 • posterior sharpening",
    },
    {
        "kind": "section",
        "title": "Quiz bowl MC stopping",
        "section_title": "Section 2: stop when commit value\nbeats waiting",
        "accent": ORANGE,
        "footer_small": "next: see the same question as an RL stopping problem",
    },
    {
        "kind": "value_chart",
        "stage": 0,
        "footer": "Frame 9 • stop when commit beats wait",
    },
    {
        "kind": "value_chart",
        "stage": 1,
        "footer": "Frame 10 • stop when commit beats wait",
    },
    {
        "kind": "value_chart",
        "stage": 2,
        "footer": "Frame 11 • stop when commit beats wait",
    },
    {
        "kind": "value_chart",
        "stage": 3,
        "footer": "Frame 12 • stop when commit beats wait",
    },
    {
        "kind": "value_chart",
        "stage": 4,
        "footer": "Frame 13 • stop when commit beats wait",
    },
    {
        "kind": "section",
        "title": "Quiz bowl MC stopping",
        "section_title": "Section 3: abstain means continue,\nnot NOTA",
        "accent": RED,
        "footer_small": "next: see the same question as an RL stopping problem",
    },
    {
        "kind": "abstain",
        "stage": 0,
        "footer": "Frame 15 • abstain semantics",
    },
    {
        "kind": "abstain",
        "stage": 1,
        "footer": "Frame 16 • abstain semantics",
    },
    {
        "kind": "abstain",
        "stage": 2,
        "footer": "Frame 17 • abstain semantics",
    },
    {
        "kind": "section",
        "title": "Quiz bowl MC stopping",
        "section_title": "Section 4: factor answer quality and\nstop timing",
        "accent": PURPLE,
        "footer_small": "next: see the same question as an RL stopping problem",
    },
    {
        "kind": "factorization",
        "stage": 0,
        "footer": "Frame 19 • factor stop and answer",
    },
    {
        "kind": "factorization",
        "stage": 1,
        "footer": "Frame 20 • factor stop and answer",
    },
    {
        "kind": "section",
        "title": "Quiz bowl MC stopping",
        "section_title": "Section 5: one clean mental model",
        "accent": GREEN,
        "footer_small": "next: see the same question as an RL stopping problem",
    },
    {
        "kind": "recipe",
        "footer": "Frame 22 • practical recipe",
    },
    {
        "kind": "summary",
        "stage": 0,
        "footer": "Frame 23 • final summary",
    },
    {
        "kind": "summary",
        "stage": 1,
        "footer": "Frame 24 • final summary",
    },
    {
        "kind": "summary",
        "stage": 2,
        "footer": "Frame 25 • final summary",
    },
]

assert len(SCENES) == 25

# ============================================================
# Scene renderers
# ============================================================
def render_intro(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)

    header(d, "Quiz bowl MC stopping = optimal stopping with improving evidence")

    card(d, (48, 96, 458, 486), "Pyramidal tossup")
    draw_text_fit(d, (76, 142, 426, 202),
                  "A question is read clue-by-clue. Early clues are obscure; later clues are easier. That means the state improves over time.",
                  max_size=24, min_size=14, fill=TEXT_SOFT)
    draw_bullets(d, (86, 248, 426, 360), [
        "obscure clue only experts get",
        "stronger clue narrows the field",
        "giveaway clue makes the answer obvious",
    ])
    rounded(d, (86, 390, 430, 462), WHITE, outline=BLUE, radius=16, width=2)
    draw_text_fit(d, (96, 398, 420, 454),
                  "RL intuition\nThe answer space is finite; the hard part is deciding when to stop listening.",
                  max_size=20, min_size=12, bold=True, fill=NAVY)

    card(d, (506, 96, 916, 486), "Multiple-choice twist", title_color=PURPLE)
    draw_text_fit(d, (536, 142, 886, 174),
                  "Options are fixed from the start:", max_size=22, min_size=14, fill=TEXT_SOFT)
    draw_options_grid(d, (536, 184, 884, 360))
    rounded(d, (536, 428, 886, 486), "#EAE6FB", outline=BLUE, radius=16, width=2)
    draw_text_fit(d, (548, 438, 874, 478),
                  "RL question\nAt each prefix, should the policy WAIT or BUZZ with one of the answers?",
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
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "As clues arrive, posterior mass concentrates on one option")

    rounded(d, (44, 86, 916, 482), WHITE, outline=BORDER, radius=16, width=2)
    draw_progress_dots(d, (72, 110, 664, 136), stage, 5)
    draw_text_fit(d, (72, 142, 664, 174), "prefix t, only the clue seen so far is visible", max_size=15, min_size=10, fill=TEXT_SOFT)

    rounded(d, (72, 176, 664, 232), WHITE, outline=BORDER, radius=12, width=2)
    prefix_texts = [
        "only a very obscure clue is visible",
        "a second clue rules out one distractor",
        "a third clue makes option A increasingly plausible",
        "a later clue makes A strongly favored",
        "one more clue pushes the argmax over the commit threshold",
    ]
    draw_text_fit(d, (88, 188, 648, 220), prefix_texts[stage], max_size=17, min_size=11, fill=TEXT)

    draw_posterior_bars(d, (72, 254, 664, 388), [
        [0.16, 0.23, 0.29, 0.32],
        [0.29, 0.23, 0.24, 0.24],
        [0.52, 0.20, 0.16, 0.12],
        [0.74, 0.12, 0.08, 0.06],
        [0.88, 0.06, 0.04, 0.02],
    ][stage])

    rounded(d, (694, 182, 758, 254), WHITE if stage < 4 else GREEN_SOFT,
            outline=BORDER if stage < 4 else GREEN, radius=14, width=2)
    draw_text_fit(d, (700, 197, 752, 240), "WAIT" if stage < 4 else "BUZZ", max_size=18, min_size=12, bold=True, align="center", valign="center", fill=ORANGE if stage < 4 else GREEN)

    if stage < 4:
        rounded(d, (774, 170, 902, 286), ORANGE_SOFT, outline=ORANGE, radius=14, width=2)
        draw_text_fit(d, (786, 182, 890, 274),
                      "Not enough value yet:\nwaiting still has positive future benefit because one more clue can move the posterior.",
                      max_size=17, min_size=10, fill=ORANGE)
    else:
        rounded(d, (774, 170, 902, 286), GREEN_SOFT, outline=GREEN, radius=14, width=2)
        draw_text_fit(d, (786, 182, 890, 274),
                      "Now the best argmax is good enough: expected commit value exceeds the value of waiting.",
                      max_size=17, min_size=10, fill=GREEN)

    rounded(d, (128, 420, 832, 462), "#F7F8FA", outline=BORDER, radius=10, width=1)
    draw_text_fit(d, (142, 430, 818, 454),
                  "In a finite answer space, the policy does not discover new options. It decides whether the current posterior is sharp enough to commit.",
                  max_size=14, min_size=9, fill=TEXT_SOFT, align="center", valign="center")

    footer(d, spec["footer"])
    return img

def render_value_chart(spec: Dict) -> Image.Image:
    stage = spec["stage"]
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Buzz when committing beats the continuation value of waiting")

    rounded(d, (50, 92, 910, 470), WHITE, outline=BORDER, radius=16, width=2)
    draw_line_chart(d, (90, 130, 850, 382), stage)

    messages = [
        "Early on, future clues are worth a lot, so WAIT dominates.",
        "The curves get closer as evidence improves.",
        "Near the crossing, timing is the whole problem.",
        "Once the current curve crosses the continuation curve, BUZZ becomes optimal.",
        "Later, waiting only delays an answer that is already good enough.",
    ]
    rounded(d, (160, 408, 800, 448), "#F7F8FA", outline=BORDER, radius=10, width=1)
    draw_text_fit(d, (178, 416, 782, 440), messages[stage], max_size=15, min_size=10, fill=TEXT_SOFT, align="center", valign="center")

    footer(d, spec["footer"])
    return img

def render_abstain(spec: Dict) -> Image.Image:
    stage = spec["stage"]
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Important semantic point: ABSTAIN is not 'none of the above'")

    rounded(d, (46, 96, 914, 472), WHITE, outline=BORDER, radius=16, width=2)
    card(d, (72, 130, 396, 388), "The gold answer is already in the option set")
    draw_options_grid(d, (92, 176, 376, 312))
    draw_text_fit(d, (92, 326, 376, 374),
                  "Abstaining means the full value of one more clue is higher than the value of committing now.",
                  max_size=16, min_size=10, fill=TEXT_SOFT, align="center", valign="center")

    card(d, (426, 130, 888, 388), "What changes over time")
    draw_posterior_bars(d, (452, 186, 790, 300), [
        [0.30, 0.24, 0.24, 0.22],
        [0.52, 0.20, 0.16, 0.12],
        [0.78, 0.11, 0.07, 0.04],
    ][stage])

    draw_text_fit(d, (500, 316, 706, 348), "ABSTAIN", max_size=26, min_size=16, bold=True, fill=ORANGE, align="center")
    draw_text_fit(d, (492, 344, 716, 382), "means continuation because another clue is worth it", max_size=16, min_size=10, fill=TEXT_SOFT, align="center")

    d.line((760, 334, 856, 370), fill=RED, width=3)
    d.line((760, 370, 856, 334), fill=RED, width=3)
    draw_text_fit(d, (760, 310, 858, 334), "NOTA", max_size=18, min_size=11, bold=True, fill=RED, align="center")

    bottom_msgs = [
        "Only architecture or loss design can make abstain behave like a reject option sink. That is an optimization artifact, not the Bayes-optimal semantics.",
        "The gold answer can already be present while waiting is still rational because another clue has positive value of information.",
        "So abstain is about continuation value, not exhaustivity of the answer set.",
    ]
    rounded(d, (160, 416, 800, 454), "#F7F8FA", outline=BORDER, radius=10, width=1)
    draw_text_fit(d, (176, 424, 784, 446), bottom_msgs[stage], max_size=13, min_size=9, fill=TEXT_SOFT, align="center", valign="center")

    footer(d, spec["footer"])
    return img

def render_factorization(spec: Dict) -> Image.Image:
    stage = spec["stage"]
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Implementation intuition: separate answer quality from stop timing")

    rounded(d, (46, 96, 914, 472), WHITE, outline=BORDER, radius=16, width=2)

    rounded(d, (84, 162, 214, 214), WHITE, outline=BORDER, radius=12, width=2)
    draw_text_fit(d, (96, 170, 202, 206), "prefix h_t\n+ option set C", max_size=18, min_size=11, align="center", valign="center")

    rounded(d, (300, 122, 500, 238), BLUE_SOFT if stage >= 0 else WHITE, outline=BLUE, radius=14, width=2)
    draw_text_fit(d, (316, 140, 484, 176), "Answer module", max_size=22, min_size=13, bold=True, align="center")
    draw_text_fit(d, (320, 178, 480, 220), "outputs p_ans(i)", max_size=20, min_size=12, align="center", valign="center")

    rounded(d, (300, 286, 500, 402), ORANGE_SOFT if stage >= 1 else WHITE, outline=ORANGE, radius=14, width=2)
    draw_text_fit(d, (316, 304, 484, 340), "Stop module", max_size=22, min_size=13, bold=True, align="center")
    draw_text_fit(d, (320, 344, 480, 384), "outputs p_buzz(h_t)", max_size=20, min_size=12, align="center", valign="center")

    d.line((214, 188, 300, 178), fill=BLUE, width=4)
    d.line((214, 188, 300, 344), fill=ORANGE, width=4)

    rounded(d, (608, 180, 866, 344), PURPLE_SOFT if stage >= 1 else WHITE, outline=PURPLE, radius=14, width=2)
    draw_text_fit(d, (624, 198, 850, 234), "Optional flat interface", max_size=22, min_size=13, bold=True, align="center")
    draw_text_fit(d, (626, 248, 848, 320),
                  "P(WAIT) = 1 - p_buzz\nP(BUZZ i) = p_buzz * p_ans(i)",
                  max_size=18, min_size=10, align="center", valign="center")

    rounded(d, (158, 424, 804, 462), "#F7F8FA", outline=BORDER, radius=10, width=1)
    draw_text_fit(d, (176, 432, 786, 454),
                  "Same decision semantics, cleaner implementation: factor stop and answer internally, then expose a flat API only if the RL stack needs it.",
                  max_size=13, min_size=9, fill=TEXT_SOFT, align="center", valign="center")

    footer(d, spec["footer"])
    return img

def render_recipe(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "A practical training recipe")

    rounded(d, (48, 104, 912, 472), WHITE, outline=BORDER, radius=16, width=2)

    steps = [
        ("1. answer model first", "learn p_ans(i | h_t)", BLUE_SOFT, BLUE),
        ("2. hazard-loss warm start", "learn when to commit", ORANGE_SOFT, ORANGE),
        ("3. PPO fine-tuning", "optimize task reward", GREEN_SOFT, GREEN),
    ]
    xs = [84, 328, 572]
    for x, (a, b, fill, outline) in zip(xs, steps):
        rounded(d, (x, 202, x + 200, 314), fill, outline=outline, radius=14, width=2)
        draw_text_fit(d, (x + 12, 218, x + 188, 246), a, max_size=20, min_size=12, bold=True, fill=NAVY, align="center")
        draw_text_fit(d, (x + 14, 254, x + 186, 296), b, max_size=18, min_size=11, fill=TEXT_SOFT, align="center", valign="center")

    rounded(d, (154, 394, 806, 438), "#F7F8FA", outline=BORDER, radius=10, width=1)
    draw_text_fit(d, (170, 404, 790, 430),
                  "RL then learns mostly the stopping problem, not which answer plus whether to stop all at once.",
                  max_size=15, min_size=10, fill=TEXT_SOFT, align="center", valign="center")
    footer(d, spec["footer"])
    return img

def render_summary(spec: Dict) -> Image.Image:
    stage = spec["stage"]
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Quiz bowl MC stopping, in one picture")

    rounded(d, (48, 98, 912, 474), WHITE, outline=BORDER, radius=16, width=2)
    rounded(d, (74, 144, 650, 418), "#FAFBFD", outline=BORDER, radius=12, width=2)

    bullets = [
        ("1. clues arrive from hard to easy", BLUE),
        ("2. posterior over fixed answer choices sharpens", ORANGE),
        ("3. the RL decision is mostly when to stop listening", PURPLE),
        ("4. abstain means another clue is worth it, not 'none of these are right'", GREEN),
    ]
    y = 174
    visible = bullets[: stage + 2]
    for i, (txt, color) in enumerate(visible):
        draw_text_fit(d, (96, y, 624, y + 24), txt, max_size=18, min_size=11, bold=(i == 0), fill=color)
        y += 48

    rounded(d, (94, 350, 620, 398), WHITE, outline=BORDER, radius=10, width=1)
    draw_text_fit(d, (108, 360, 606, 388),
                  "Good mental model: selective prediction + optimal stopping + fixed answer set.",
                  max_size=15, min_size=10, fill=TEXT_SOFT, align="center", valign="center")

    rounded(d, (688, 146, 892, 256), "#EEF2F8", outline=BORDER, radius=12, width=2)
    draw_text_fit(d, (702, 160, 878, 188), "For RL people", max_size=18, min_size=12, bold=True, fill=NAVY, align="center")
    draw_text_fit(d, (702, 196, 878, 242),
                  "At each prefix, compare the value of committing now to the value of one more clue.",
                  max_size=15, min_size=10, fill=TEXT_SOFT, align="center", valign="center")

    if stage >= 1:
        rounded(d, (688, 282, 892, 388), GREEN_SOFT, outline=GREEN, radius=12, width=2)
        draw_text_fit(d, (702, 296, 878, 374),
                      "If you keep a flat action interface, factor it internally:\nP(WAIT)=1-p_buzz\nP(BUZZ i)=p_buzz * p_ans(i)",
                      max_size=15, min_size=9, fill=GREEN, align="center", valign="center")

    if stage >= 2:
        rounded(d, (86, 434, 860, 464), BLUE_SOFT, outline=BLUE, radius=10, width=2)
        draw_text_fit(d, (100, 440, 846, 458),
                      "Key takeaway: pyramidal quiz bowl turns MCQA into a stopping problem because information improves before action.",
                      max_size=13, min_size=9, fill=NAVY_DARK, align="center", valign="center")

    footer(d, spec["footer"])
    return img

# ============================================================
# Dispatcher
# ============================================================
RENDERERS = {
    "intro": render_intro,
    "section": render_section,
    "posterior": render_posterior,
    "value_chart": render_value_chart,
    "abstain": render_abstain,
    "factorization": render_factorization,
    "recipe": render_recipe,
    "summary": render_summary,
}

# ============================================================
# Build frames
# ============================================================
frames: List[Image.Image] = []
for spec in SCENES:
    frames.append(RENDERERS[spec["kind"]](spec))

assert len(frames) == 25

# ============================================================
# Durations
# ============================================================
durations = []
section_frames = {2, 8, 14, 18, 21}  # 1-indexed
for i in range(1, len(frames) + 1):
    if i == 1:
        durations.append(1000)
    elif i in section_frames:
        durations.append(900)
    elif i in {7, 13, 20, 25}:
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