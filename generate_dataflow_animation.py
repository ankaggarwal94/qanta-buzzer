#!/usr/bin/env python3
"""Generate a CoVe-validated animated overview of qanta-buzzer data flows.

This script intentionally derives its scenes from verified source paths rather
than a hand-wavy architecture sketch. It emits:

- an animated GIF
- per-frame PNGs
- a contact sheet
- a markdown CoVe validation log with claim-by-claim evidence

The scene grouping is an inference over verified entrypoints, but every node,
arrow, and note in the animation is backed by code references captured below.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "generated" / "qanta_buzzer_e2e_data_flows"
OUT_DIR = Path(os.getenv("QB_DATAFLOW_OUT_DIR", str(DEFAULT_OUT_DIR))).expanduser()
FRAMES_DIR = OUT_DIR / "frames"
GIF_OUT = OUT_DIR / "qanta_buzzer_e2e_data_flows.gif"
CONTACT_OUT = OUT_DIR / "qanta_buzzer_e2e_data_flows_contact.png"
VALIDATION_OUT = OUT_DIR / "cove_validation.md"


# ---------------------------------------------------------------------------
# Canvas + palette
# ---------------------------------------------------------------------------
W, H = 1440, 900

BG = "#F5F2EB"
PAPER = "#FCFBF8"
WHITE = "#FFFFFF"
TEXT = "#1F2933"
TEXT_SOFT = "#64707D"
BORDER = "#D7CEC0"
GRID = "#E9E2D6"
SHADOW = "#D9D0C2"

NAVY = "#244C7C"
BLUE = "#396BCF"
BLUE_SOFT = "#E3EDFF"
TEAL = "#1B8A89"
TEAL_SOFT = "#E3F7F7"
GREEN = "#2E8B57"
GREEN_SOFT = "#E6F5EA"
ORANGE = "#C97A16"
ORANGE_SOFT = "#FFF0DB"
RED = "#B7564D"
RED_SOFT = "#FBE4E1"
PURPLE = "#7450C7"
PURPLE_SOFT = "#EEE6FF"
GOLD = "#9A6A05"
GOLD_SOFT = "#FFF3D7"
INK = "#0F1720"


# ---------------------------------------------------------------------------
# Fonts
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


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = "bold" if bold else "regular"
    for path in _FONT_PATHS[key]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    for ttc_path, reg_idx, bold_idx in _TTC_FALLBACKS:
        if Path(ttc_path).exists():
            try:
                return ImageFont.truetype(
                    ttc_path,
                    size=size,
                    index=bold_idx if bold else reg_idx,
                )
            except Exception:
                continue
    return ImageFont.load_default()


FONTS = {
    "title": get_font(44, bold=True),
    "subtitle": get_font(22),
    "scene": get_font(28, bold=True),
    "caption": get_font(18),
    "body": get_font(19),
    "body_bold": get_font(19, bold=True),
    "small": get_font(16),
    "small_bold": get_font(16, bold=True),
    "tiny": get_font(13),
    "tiny_bold": get_font(13, bold=True),
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Claim:
    claim_id: str
    claim: str
    evidence: tuple[str, ...]
    verdict: str = "Verified"


@dataclass(frozen=True)
class Node:
    node_id: str
    title: str
    detail: str
    box: tuple[int, int, int, int]
    accent: str
    soft: str


@dataclass(frozen=True)
class Edge:
    edge_id: str
    start: str
    end: str
    label: str = ""


@dataclass(frozen=True)
class Step:
    caption: str
    active_nodes: tuple[str, ...]
    active_edges: tuple[str, ...]


@dataclass(frozen=True)
class Scene:
    scene_id: str
    title: str
    subtitle: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    steps: tuple[Step, ...]
    claim_ids: tuple[str, ...]
    note: str = ""


CLAIMS: tuple[Claim, ...] = (
    Claim(
        "C1",
        "The build_mc_dataset entrypoint loads questions from CSV when present, "
        "falls back to HuggingFace only if configured, builds answer profiles, "
        "constructs MC questions, creates stratified splits, and saves dataset JSONs.",
        (
            "scripts/build_mc_dataset.py:267-355",
            "qb_data/answer_profiles.py:11-142",
            "qb_data/dataset_splits.py:create_stratified_splits",
        ),
    ),
    Claim(
        "C2",
        "MCBuilder supports category_random, tfidf_profile, sbert_profile, and "
        "openai_profile distractor strategies and applies four anti-artifact guards.",
        (
            "qb_data/mc_builder.py:63-97",
            "qb_data/mc_builder.py:205-249",
            "qb_data/mc_builder.py:251-367",
        ),
    ),
    Claim(
        "C3",
        "Question history is represented as run_indices plus cumulative_prefixes, "
        "where each prefix is the text revealed up to a clue boundary.",
        (
            "qb_data/data_loader.py:65-74",
            "qb_data/data_loader.py:156-168",
        ),
    ),
    Claim(
        "C4",
        "The belief-feature path builds a configurable likelihood model over MC "
        "questions; supported backends are tfidf, sbert, openai, and t5 variants.",
        (
            "scripts/_common.py:44-53",
            "models/likelihoods.py:374-731",
        ),
    ),
    Claim(
        "C5",
        "TossupMCEnv supports from_scratch belief recomputation from cumulative "
        "prefixes and sequential_bayes updates from newly revealed fragments, then "
        "exposes belief features [belief..., top_p, margin, entropy, stability, progress, clue_idx_norm].",
        (
            "qb_env/tossup_env.py:93-114",
            "qb_env/tossup_env.py:151-158",
            "qb_env/tossup_env.py:351-388",
            "models/features.py:50-108",
        ),
    ),
    Claim(
        "C6",
        "run_baselines executes ThresholdBuzzer, SoftmaxProfileBuzzer, "
        "SequentialBayesBuzzer, and AlwaysBuzzFinal, then saves per-agent runs "
        "and a baseline_summary artifact.",
        (
            "scripts/run_baselines.py:5-12",
            "scripts/run_baselines.py:141-225",
        ),
    ),
    Claim(
        "C7",
        "train_ppo loads MC questions, builds a likelihood model, precomputes "
        "belief trajectories, creates TossupMCEnv from config, trains PPOBuzzer "
        "with SB3 MlpPolicy, and writes ppo_model, ppo_runs, and ppo_summary.",
        (
            "scripts/train_ppo.py:82-176",
            "agents/ppo_buzzer.py:68-120",
            "qb_env/tossup_env.py:598-660",
        ),
    ),
    Claim(
        "C8",
        "evaluate_all loads MC questions and baseline outputs, picks the best "
        "softmax threshold from baseline_summary, runs full evaluation plus "
        "choices-only, shuffle, and alias controls, then writes evaluation_report "
        "and plotting artifacts.",
        (
            "scripts/evaluate_all.py:87-115",
            "scripts/evaluate_all.py:152-255",
            "scripts/evaluate_all.py:258-328",
        ),
    ),
    Claim(
        "C9",
        "In the current end-to-end pipeline, alias substitution is effectively "
        "fed by an empty lookup when alias_lookup.json is absent, and "
        "build_mc_dataset.py does not write that file.",
        (
            "scripts/evaluate_all.py:158-163",
            "scripts/build_mc_dataset.py:338-355",
        ),
    ),
    Claim(
        "C10",
        "train_t5_policy loads an MC dataset, splits questions into train/val/test, "
        "runs supervised warm-start on complete-question text, then PPO fine-tunes "
        "a T5PolicyModel on incremental text observations.",
        (
            "scripts/train_t5_policy.py:1-236",
            "training/train_supervised_t5.py:52-72",
            "training/train_ppo_t5.py:299-380",
        ),
    ),
    Claim(
        "C11",
        "TextObservationWrapper exposes visible history as "
        "'CLUES: <prefix> | CHOICES: ...', and T5PolicyModel consumes that "
        "text to produce wait logits, answer logits, and values.",
        (
            "qb_env/text_wrapper.py:70-120",
            "models/t5_policy.py:132-216",
            "models/t5_policy.py:321-396",
        ),
    ),
    Claim(
        "C12",
        "T5 PPO rollouts still instantiate TossupMCEnv with a TF-IDF likelihood "
        "for environment scoring/reward computation, while the T5 policy itself "
        "reads text observations through TextObservationWrapper.",
        (
            "training/train_ppo_t5.py:317-345",
            "training/train_ppo_t5.py:351-380",
        ),
    ),
    Claim(
        "C13",
        "compare_policies evaluates an MLP PPO policy on belief features and a "
        "T5 policy on text observations, then saves a comparison JSON.",
        (
            "scripts/compare_policies.py:58-142",
            "scripts/compare_policies.py:145-260",
            "scripts/compare_policies.py:393-464",
        ),
    ),
)

CLAIM_MAP = {claim.claim_id: claim for claim in CLAIMS}


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def measure(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    max_width: int,
) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            if measure(draw, trial, font)[0] <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font,
    fill: str,
    line_gap: int = 4,
) -> None:
    x0, y0, x1, y1 = box
    lines = wrap_text(draw, text, font, max_width=max(1, x1 - x0))
    _, line_h = measure(draw, "Ag", font)
    y = y0
    for line in lines:
        if y + line_h > y1:
            break
        draw.text((x0, y), line, font=font, fill=fill)
        y += line_h + line_gap


def lerp_color(a: str, b: str, t: float) -> str:
    def to_rgb(value: str) -> tuple[int, int, int]:
        value = value.lstrip("#")
        return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)

    ar, ag, ab = to_rgb(a)
    br, bg, bb = to_rgb(b)
    rr = round(ar + (br - ar) * t)
    rg = round(ag + (bg - ag) * t)
    rb = round(ab + (bb - ab) * t)
    return f"#{rr:02X}{rg:02X}{rb:02X}"


def draw_background(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 0, W, H), fill=BG)
    for y in range(0, H, 48):
        alpha = 0.18 if (y // 48) % 2 == 0 else 0.08
        fill = lerp_color(BG, GRID, alpha)
        draw.rectangle((0, y, W, min(H, y + 48)), fill=fill)

    draw.rounded_rectangle(
        (36, 30, W - 36, H - 30),
        radius=28,
        fill=PAPER,
        outline=BORDER,
        width=2,
    )


def draw_header(
    draw: ImageDraw.ImageDraw,
    scene: Scene,
    scene_index: int,
    total_scenes: int,
    step_index: int,
    total_steps: int,
) -> None:
    draw.text((70, 56), "qanta-buzzer - verified E2E data flows", font=FONTS["title"], fill=INK)
    draw.text((72, 112), scene.subtitle, font=FONTS["subtitle"], fill=TEXT_SOFT)

    badge_box = (W - 280, 56, W - 72, 102)
    draw.rounded_rectangle(badge_box, radius=18, fill=GREEN_SOFT, outline=GREEN, width=2)
    draw.text((badge_box[0] + 16, badge_box[1] + 11), "CoVe validated scene", font=FONTS["small_bold"], fill=GREEN)

    scene_label = f"Scene {scene_index + 1}/{total_scenes}: {scene.title}"
    draw.text((70, 154), scene_label, font=FONTS["scene"], fill=NAVY)

    bar_x0 = 70
    bar_x1 = W - 72
    bar_y0 = 194
    bar_y1 = 208
    draw.rounded_rectangle((bar_x0, bar_y0, bar_x1, bar_y1), radius=8, fill="#E8E2D8", outline=None)
    overall = (scene_index + (step_index + 1) / max(1, total_steps)) / max(1, total_scenes)
    fill_x = bar_x0 + int((bar_x1 - bar_x0) * overall)
    draw.rounded_rectangle((bar_x0, bar_y0, fill_x, bar_y1), radius=8, fill=BLUE, outline=None)


def draw_node(
    draw: ImageDraw.ImageDraw,
    node: Node,
    active: bool,
    pulse: float,
) -> None:
    x0, y0, x1, y1 = node.box
    shadow_offset = 6
    draw.rounded_rectangle(
        (x0 + shadow_offset, y0 + shadow_offset, x1 + shadow_offset, y1 + shadow_offset),
        radius=22,
        fill=SHADOW,
        outline=None,
    )

    fill = node.soft if active else WHITE
    outline = node.accent if active else BORDER
    width = 4 if active else 2
    if active:
        fill = lerp_color(fill, WHITE, 0.10 * (1.0 - pulse))
    draw.rounded_rectangle(node.box, radius=22, fill=fill, outline=outline, width=width)

    pill = (x0 + 18, y0 + 16, x0 + 166, y0 + 44)
    draw.rounded_rectangle(pill, radius=14, fill=node.accent, outline=None)
    draw.text((pill[0] + 12, pill[1] + 6), node.title, font=FONTS["small_bold"], fill=WHITE)

    draw_wrapped_text(
        draw,
        (x0 + 18, y0 + 56, x1 - 18, y1 - 16),
        node.detail,
        FONTS["body"],
        TEXT if active else TEXT_SOFT,
        line_gap=4,
    )


def node_center(node: Node) -> tuple[int, int]:
    x0, y0, x1, y1 = node.box
    return (x0 + x1) // 2, (y0 + y1) // 2


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str,
    width: int,
) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head_len = 14
    left = (
        end[0] - head_len * math.cos(angle - math.pi / 6),
        end[1] - head_len * math.sin(angle - math.pi / 6),
    )
    right = (
        end[0] - head_len * math.cos(angle + math.pi / 6),
        end[1] - head_len * math.sin(angle + math.pi / 6),
    )
    draw.polygon([end, left, right], fill=color)


def edge_points(nodes: dict[str, Node], edge: Edge) -> tuple[tuple[int, int], tuple[int, int]]:
    start_node = nodes[edge.start]
    end_node = nodes[edge.end]
    sx, sy = node_center(start_node)
    ex, ey = node_center(end_node)
    if ex >= sx:
        start = (start_node.box[2], sy)
        end = (end_node.box[0], ey)
    else:
        start = (start_node.box[0], sy)
        end = (end_node.box[2], ey)
    if abs(ex - sx) < abs(ey - sy):
        if ey >= sy:
            start = (sx, start_node.box[3])
            end = (ex, end_node.box[1])
        else:
            start = (sx, start_node.box[1])
            end = (ex, end_node.box[3])
    return start, end


def draw_edge(
    draw: ImageDraw.ImageDraw,
    nodes: dict[str, Node],
    edge: Edge,
    active: bool,
    pulse: float,
) -> None:
    start, end = edge_points(nodes, edge)
    color = BLUE if active else "#C7CFDB"
    if active:
        color = lerp_color(BLUE, NAVY, 0.35 + 0.35 * pulse)
    width = 5 if active else 3
    draw_arrow(draw, start, end, color, width)
    if edge.label:
        mx = (start[0] + end[0]) // 2
        my = (start[1] + end[1]) // 2
        tw, th = measure(draw, edge.label, FONTS["tiny_bold"])
        pad = 8
        label_box = (mx - tw // 2 - pad, my - th // 2 - pad, mx + tw // 2 + pad, my + th // 2 + pad)
        draw.rounded_rectangle(label_box, radius=12, fill=PAPER, outline=color if active else BORDER, width=2)
        draw.text((label_box[0] + pad, label_box[1] + pad - 1), edge.label, font=FONTS["tiny_bold"], fill=color if active else TEXT_SOFT)


def draw_footer(
    draw: ImageDraw.ImageDraw,
    scene: Scene,
    step: Step,
) -> None:
    foot_box = (58, H - 210, W - 58, H - 58)
    draw.rounded_rectangle(foot_box, radius=22, fill=WHITE, outline=BORDER, width=2)

    caption_title = "Verified flow step"
    draw.text((foot_box[0] + 18, foot_box[1] + 16), caption_title, font=FONTS["small_bold"], fill=ORANGE)
    draw_wrapped_text(
        draw,
        (foot_box[0] + 18, foot_box[1] + 44, foot_box[0] + 520, foot_box[3] - 16),
        step.caption,
        FONTS["body_bold"],
        TEXT,
        line_gap=5,
    )

    claims_text = "Claims: " + ", ".join(scene.claim_ids)
    draw.text((foot_box[0] + 540, foot_box[1] + 18), claims_text, font=FONTS["small_bold"], fill=PURPLE)

    evidence_lines: list[str] = []
    for claim_id in scene.claim_ids:
        claim = CLAIM_MAP[claim_id]
        evidence_lines.append(f"{claim_id}: {claim.evidence[0]}")
        if len(claim.evidence) > 1:
            evidence_lines.append(f"    + {claim.evidence[1]}")

    evidence_text = "\n".join(evidence_lines[:6])
    draw_wrapped_text(
        draw,
        (foot_box[0] + 540, foot_box[1] + 48, foot_box[3] - 18, foot_box[3] - 18),
        evidence_text,
        FONTS["small"],
        TEXT_SOFT,
        line_gap=3,
    )

    if scene.note:
        note_box = (foot_box[0] + 18, foot_box[1] + 104, foot_box[0] + 500, foot_box[3] - 18)
        draw_wrapped_text(
            draw,
            note_box,
            f"Note: {scene.note}",
            FONTS["small"],
            TEXT_SOFT,
            line_gap=3,
        )


def box(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    return x, y, x + w, y + h


# ---------------------------------------------------------------------------
# Scene definitions
# ---------------------------------------------------------------------------
SCENES: tuple[Scene, ...] = (
    Scene(
        scene_id="build",
        title="Build MC dataset",
        subtitle="Validated source path: raw tossups -> answer profiles -> guarded MC artifacts",
        nodes=(
            Node("csv", "questions.csv", "Primary local QANTA CSV input.", box(80, 250, 220, 120), BLUE, BLUE_SOFT),
            Node("hf", "HuggingFace", "Optional fallback only if CSV is unavailable and configured.", box(80, 400, 220, 120), BLUE, BLUE_SOFT),
            Node("loader", "Loader", "QANTADatasetLoader or HuggingFace loader creates TossupQuestion rows.", box(360, 320, 300, 120), TEAL, TEAL_SOFT),
            Node("profiles", "Profiles", "AnswerProfileBuilder groups question text by answer and supports leave-one-out gold profiles.", box(730, 240, 300, 120), ORANGE, ORANGE_SOFT),
            Node("mc", "MCBuilder", "Builds MCQuestion objects with strategy branches and anti-artifact guards.", box(730, 400, 300, 120), ORANGE, ORANGE_SOFT),
            Node("splits", "Splits + JSON", "create_stratified_splits then save mc_dataset, train, val, test, and answer_profiles debug JSON.", box(1090, 320, 270, 120), GREEN, GREEN_SOFT),
        ),
        edges=(
            Edge("e1", "csv", "loader", "preferred"),
            Edge("e2", "hf", "loader", "fallback"),
            Edge("e3", "loader", "profiles"),
            Edge("e4", "loader", "mc"),
            Edge("e5", "profiles", "mc"),
            Edge("e6", "mc", "splits"),
        ),
        steps=(
            Step("Load tossups from the local CSV when it exists; otherwise the script can fall back to HuggingFace.", ("csv", "hf", "loader"), ("e1", "e2")),
            Step("Build reusable answer profiles from grouped historical question text.", ("loader", "profiles"), ("e3",)),
            Step("Construct guarded MC questions using category_random, tfidf_profile, sbert_profile, or openai_profile ranking.", ("profiles", "mc"), ("e4", "e5")),
            Step("Create stratified train/val/test splits and persist the dataset artifacts consumed downstream.", ("mc", "splits"), ("e6",)),
        ),
        claim_ids=("C1", "C2", "C3"),
    ),
    Scene(
        scene_id="belief",
        title="Belief construction branch",
        subtitle="Validated source path: MC dataset -> likelihood scores -> belief modes -> feature observation",
        nodes=(
            Node("mcdata", "MC dataset", "MCQuestion records carry options, option profiles, run_indices, and cumulative_prefixes.", box(90, 320, 230, 120), BLUE, BLUE_SOFT),
            Node("lm", "Likelihood", "build_likelihood_model delegates to tfidf, sbert, openai, or t5 backends.", box(380, 320, 250, 120), ORANGE, ORANGE_SOFT),
            Node("scratch", "from_scratch", "Re-score the entire revealed cumulative prefix each step.", box(720, 230, 260, 110), PURPLE, PURPLE_SOFT),
            Node("seq", "sequential_bayes", "Score only the newly revealed fragment and multiply into the prior.", box(720, 390, 260, 110), PURPLE, PURPLE_SOFT),
            Node("env", "TossupMCEnv", "Environment updates belief and exposes a Discrete(K+1) action space.", box(1050, 320, 260, 120), TEAL, TEAL_SOFT),
            Node("features", "Belief features", "Observation is [belief..., top_p, margin, entropy, stability, progress, clue_idx_norm].", box(1050, 480, 260, 120), GREEN, GREEN_SOFT),
        ),
        edges=(
            Edge("e1", "mcdata", "lm"),
            Edge("e2", "lm", "scratch", "prefix"),
            Edge("e3", "lm", "seq", "fragment"),
            Edge("e4", "scratch", "env"),
            Edge("e5", "seq", "env"),
            Edge("e6", "env", "features"),
        ),
        steps=(
            Step("MC questions provide the visible prefix history plus option profiles for scoring.", ("mcdata", "lm"), ("e1",)),
            Step("One branch recomputes belief from the full revealed prefix at each step.", ("lm", "scratch", "env"), ("e2", "e4")),
            Step("The other branch performs sequential Bayes updates from newly revealed fragments.", ("lm", "seq", "env"), ("e3", "e5")),
            Step("Both branches feed the same environment observation features for downstream agents.", ("env", "features"), ("e6",)),
        ),
        claim_ids=("C3", "C4", "C5"),
    ),
    Scene(
        scene_id="baselines",
        title="Baseline sweep",
        subtitle="Validated source path: shared precomputed beliefs -> four non-RL buzzer families -> baseline summary",
        nodes=(
            Node("mcdata", "MC dataset", "Loaded from artifacts/<split>/mc_dataset.json or fallback data/processed.", box(80, 330, 220, 120), BLUE, BLUE_SOFT),
            Node("lm", "Likelihood", "One model instance scores prefixes or fragments for the whole sweep.", box(340, 330, 220, 120), ORANGE, ORANGE_SOFT),
            Node("cache", "Precompute", "precompute_beliefs and precompute_sequential_beliefs collapse repeated scoring.", box(600, 330, 260, 120), TEAL, TEAL_SOFT),
            Node("rules", "Rule agents", "ThresholdBuzzer, SoftmaxProfileBuzzer, SequentialBayesBuzzer, AlwaysBuzzFinal.", box(920, 250, 300, 120), PURPLE, PURPLE_SOFT),
            Node("summary", "Artifacts", "baseline_*_runs.json plus baseline_summary.json.", box(920, 410, 300, 120), GREEN, GREEN_SOFT),
        ),
        edges=(
            Edge("e1", "mcdata", "lm"),
            Edge("e2", "lm", "cache"),
            Edge("e3", "cache", "rules"),
            Edge("e4", "rules", "summary"),
        ),
        steps=(
            Step("Load MC questions and build one shared likelihood model for the baseline run.", ("mcdata", "lm"), ("e1",)),
            Step("Precompute beliefs once so the threshold sweeps do not rescore text repeatedly.", ("lm", "cache"), ("e2",)),
            Step("Run the four non-RL buzzer families over the cached trajectories.", ("cache", "rules"), ("e3",)),
            Step("Persist per-agent run traces and the aggregate baseline summary used downstream.", ("rules", "summary"), ("e4",)),
        ),
        claim_ids=("C4", "C5", "C6"),
    ),
    Scene(
        scene_id="ppo_mlp",
        title="Belief-feature PPO",
        subtitle="Validated source path: precomputed beliefs -> TossupMCEnv -> SB3 PPO MlpPolicy -> saved traces",
        nodes=(
            Node("mcdata", "MC dataset", "Same MCQuestion artifacts as the baseline path.", box(80, 320, 220, 120), BLUE, BLUE_SOFT),
            Node("lm", "Likelihood", "Configured backend from the likelihood factory.", box(340, 320, 220, 120), ORANGE, ORANGE_SOFT),
            Node("beliefs", "precompute_beliefs", "Caches (question_idx, step_idx) -> belief vectors for O(1) env lookups.", box(600, 320, 270, 120), TEAL, TEAL_SOFT),
            Node("env", "TossupMCEnv", "make_env_from_config wires reward mode, penalties, K, beta, and belief_mode.", box(930, 240, 260, 120), PURPLE, PURPLE_SOFT),
            Node("ppo", "PPOBuzzer", "Stable-Baselines3 PPO with MlpPolicy over belief features.", box(930, 400, 260, 120), RED, RED_SOFT),
            Node("artifacts", "Saved outputs", "ppo_model.zip, ppo_runs.json, ppo_summary.json.", box(1230, 320, 150, 120), GREEN, GREEN_SOFT),
        ),
        edges=(
            Edge("e1", "mcdata", "lm"),
            Edge("e2", "lm", "beliefs"),
            Edge("e3", "beliefs", "env"),
            Edge("e4", "env", "ppo"),
            Edge("e5", "ppo", "artifacts"),
        ),
        steps=(
            Step("Load MC questions, build the configured likelihood model, and compute cached beliefs.", ("mcdata", "lm", "beliefs"), ("e1", "e2")),
            Step("Instantiate TossupMCEnv from config using the precomputed belief cache.", ("beliefs", "env"), ("e3",)),
            Step("Train the learned belief-feature buzzer as an SB3 PPO MlpPolicy.", ("env", "ppo"), ("e4",)),
            Step("Evaluate the trained policy and save both the checkpoint and post-training summaries.", ("ppo", "artifacts"), ("e5",)),
        ),
        claim_ids=("C4", "C5", "C7"),
    ),
    Scene(
        scene_id="evaluate",
        title="Comprehensive evaluation",
        subtitle="Validated source path: best-softmax eval + controls + plots + evaluation report",
        nodes=(
            Node("baseline", "baseline_summary", "Used to pick the best softmax threshold by mean S_q.", box(70, 240, 250, 120), BLUE, BLUE_SOFT),
            Node("ppo", "ppo_summary", "Optional PPO summary loaded into the final report and comparison table.", box(70, 410, 250, 120), BLUE, BLUE_SOFT),
            Node("mcdata", "MC dataset", "Questions and option profiles are reloaded for evaluation.", box(360, 325, 220, 120), ORANGE, ORANGE_SOFT),
            Node("full", "Full eval", "SoftmaxProfileBuzzer runs on precomputed beliefs at the selected threshold.", box(660, 210, 260, 120), TEAL, TEAL_SOFT),
            Node("controls", "Controls", "choices-only, shuffle, and alias substitution controls.", box(660, 390, 260, 120), PURPLE, PURPLE_SOFT),
            Node("report", "Report + plots", "evaluation_report.json, calibration.png, entropy_vs_clue.png, comparison.csv.", box(1000, 325, 320, 120), GREEN, GREEN_SOFT),
        ),
        edges=(
            Edge("e1", "baseline", "full", "pick best"),
            Edge("e2", "mcdata", "full"),
            Edge("e3", "mcdata", "controls"),
            Edge("e4", "full", "report"),
            Edge("e5", "controls", "report"),
            Edge("e6", "ppo", "report"),
        ),
        steps=(
            Step("Load the baseline summary so evaluation can select the best softmax threshold.", ("baseline", "full"), ("e1",)),
            Step("Run the main evaluation on reloaded MC questions using precomputed beliefs.", ("mcdata", "full"), ("e2",)),
            Step("Run control experiments: choices-only, shuffle, and alias substitution.", ("mcdata", "controls"), ("e3",)),
            Step("Assemble the report and plotting artifacts, merging in baseline and PPO summaries.", ("full", "controls", "ppo", "report"), ("e4", "e5", "e6")),
        ),
        claim_ids=("C8", "C9"),
        note="Alias substitution currently falls back to an empty lookup when alias_lookup.json is absent in this path.",
    ),
    Scene(
        scene_id="t5_supervised",
        title="T5 supervised warm-start",
        subtitle="Validated source path: MC dataset -> train/val/test split -> full-question text -> answer head training",
        nodes=(
            Node("mcdata", "MC dataset", "train_t5_policy loads the MC dataset from artifacts/main, artifacts/smoke, or data/processed.", box(80, 330, 240, 120), BLUE, BLUE_SOFT),
            Node("split", "split_questions", "Random split into train/val/test inside the T5 training script.", box(380, 330, 240, 120), TEAL, TEAL_SOFT),
            Node("text", "Full question text", "Supervised warm-start formats ALL clues as 'CLUES: ... | CHOICES: ...'.", box(680, 250, 280, 120), ORANGE, ORANGE_SOFT),
            Node("t5", "T5PolicyModel", "T5 encoder plus wait, answer, and value heads.", box(680, 410, 280, 120), PURPLE, PURPLE_SOFT),
            Node("ckpt", "Supervised ckpt", "Best model saved under checkpoint_dir/supervised/best_model.", box(1030, 330, 280, 120), GREEN, GREEN_SOFT),
        ),
        edges=(
            Edge("e1", "mcdata", "split"),
            Edge("e2", "split", "text"),
            Edge("e3", "text", "t5"),
            Edge("e4", "t5", "ckpt"),
        ),
        steps=(
            Step("Load MC questions and split them into train, validation, and test partitions.", ("mcdata", "split"), ("e1",)),
            Step("For supervised warm-start, format complete questions with all clue text visible.", ("split", "text"), ("e2",)),
            Step("Feed the formatted question text into T5PolicyModel and train the answer head with cross-entropy.", ("text", "t5"), ("e3",)),
            Step("Save the best supervised checkpoint for later PPO fine-tuning or evaluation.", ("t5", "ckpt"), ("e4",)),
        ),
        claim_ids=("C10", "C11"),
    ),
    Scene(
        scene_id="t5_ppo",
        title="T5 PPO fine-tuning",
        subtitle="Validated source path: incremental text observations -> T5 policy -> custom PPO updates",
        nodes=(
            Node("ckpt", "Start model", "Either the supervised checkpoint or a provided pretrained T5 checkpoint.", box(80, 330, 240, 120), BLUE, BLUE_SOFT),
            Node("env", "TossupMCEnv", "Rollout env still uses a TF-IDF likelihood model for belief/reward computation.", box(380, 240, 260, 120), TEAL, TEAL_SOFT),
            Node("wrap", "Text wrapper", "TextObservationWrapper exposes visible history as 'CLUES: prefix | CHOICES: ...'.", box(380, 420, 260, 120), ORANGE, ORANGE_SOFT),
            Node("t5", "T5PolicyModel", "Reads text observations and outputs wait logits, answer logits, and values.", box(720, 330, 280, 120), PURPLE, PURPLE_SOFT),
            Node("ppo", "Custom PPO", "RolloutBuffer + GAE + clipped PPO updates in training/train_ppo_t5.py.", box(1060, 330, 280, 120), RED, RED_SOFT),
        ),
        edges=(
            Edge("e1", "ckpt", "t5"),
            Edge("e2", "env", "wrap"),
            Edge("e3", "wrap", "t5"),
            Edge("e4", "t5", "ppo"),
        ),
        steps=(
            Step("Begin from the supervised checkpoint or another provided pretrained T5 policy model.", ("ckpt", "t5"), ("e1",)),
            Step("Collect rollouts from TossupMCEnv, where environment scoring still uses a lightweight TF-IDF likelihood.", ("env", "wrap"), ("e2",)),
            Step("Convert each incremental state into text and let T5PolicyModel act on the visible prefix.", ("wrap", "t5"), ("e3",)),
            Step("Store rollout steps, compute returns/advantages, and run custom PPO updates on the T5 policy.", ("t5", "ppo"), ("e4",)),
        ),
        claim_ids=("C10", "C11", "C12"),
        note="The T5 policy is text-native, but the rollout environment still computes internal beliefs with TF-IDF in this trainer.",
    ),
    Scene(
        scene_id="compare",
        title="Comparison experiment",
        subtitle="Validated source path: MLP belief-feature policy vs end-to-end T5 policy",
        nodes=(
            Node("mcdata", "MC dataset", "compare_policies reloads the MC dataset, shuffles with seed 42, and takes the last 15 percent as test.", box(80, 330, 250, 120), BLUE, BLUE_SOFT),
            Node("mlp", "MLP eval", "TfIdfLikelihood + TossupMCEnv + PPOBuzzer.load(checkpoint) for belief-feature evaluation.", box(420, 240, 320, 120), ORANGE, ORANGE_SOFT),
            Node("t5", "T5 eval", "T5PolicyModel.load_pretrained + TextObservationWrapper + TF-IDF rollout env.", box(420, 420, 320, 120), PURPLE, PURPLE_SOFT),
            Node("metrics", "Common metrics", "accuracy, mean_sq, ece, brier, avg_buzz_pos, mean_reward.", box(840, 330, 240, 120), TEAL, TEAL_SOFT),
            Node("json", "comparison JSON", "save_json(output_path, comparison).", box(1160, 330, 180, 120), GREEN, GREEN_SOFT),
        ),
        edges=(
            Edge("e1", "mcdata", "mlp"),
            Edge("e2", "mcdata", "t5"),
            Edge("e3", "mlp", "metrics"),
            Edge("e4", "t5", "metrics"),
            Edge("e5", "metrics", "json"),
        ),
        steps=(
            Step("Reload the MC dataset, derive a test split, and feed the same test questions to both families.", ("mcdata", "mlp", "t5"), ("e1", "e2")),
            Step("Evaluate the learned belief-feature PPO policy on belief observations.", ("mlp", "metrics"), ("e3",)),
            Step("Evaluate the end-to-end T5 policy on text observations with the same metric family.", ("t5", "metrics"), ("e4",)),
            Step("Write the final comparison JSON artifact.", ("metrics", "json"), ("e5",)),
        ),
        claim_ids=("C12", "C13"),
    ),
)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_scene(scene: Scene, scene_index: int, pulse: float, step_index: int) -> Image.Image:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw_background(draw)
    draw_header(draw, scene, scene_index, len(SCENES), step_index, len(scene.steps))

    nodes = {node.node_id: node for node in scene.nodes}
    step = scene.steps[step_index]
    active_node_ids = set(step.active_nodes)
    active_edge_ids = set(step.active_edges)

    for edge in scene.edges:
        draw_edge(draw, nodes, edge, active=edge.edge_id in active_edge_ids, pulse=pulse)
    for node in scene.nodes:
        draw_node(draw, node, active=node.node_id in active_node_ids, pulse=pulse)

    draw_footer(draw, scene, step)
    return image


def build_contact_sheet(frames: Sequence[Image.Image]) -> Image.Image:
    cols = 4
    rows = math.ceil(len(frames) / cols)
    thumb_w = 320
    thumb_h = int(thumb_w * H / W)
    margin = 24
    gap = 16
    sheet_w = margin * 2 + cols * thumb_w + (cols - 1) * gap
    sheet_h = margin * 2 + 70 + rows * thumb_h + (rows - 1) * gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), BG)
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, margin), "qanta-buzzer E2E data flows - contact sheet", font=FONTS["scene"], fill=INK)
    draw.text((margin, margin + 36), "Every frame is driven by a CoVe-validated scene definition.", font=FONTS["small"], fill=TEXT_SOFT)

    y0 = margin + 70
    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        x = margin + col * (thumb_w + gap)
        y = y0 + row * (thumb_h + gap)
        thumb = frame.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x, y))
        draw.rounded_rectangle((x, y, x + thumb_w, y + thumb_h), radius=14, outline=BORDER, width=2)
        label = f"{idx + 1:02d}"
        tw, th = measure(draw, label, FONTS["tiny_bold"])
        pill = (x + 10, y + 10, x + 10 + tw + 16, y + 10 + th + 10)
        draw.rounded_rectangle(pill, radius=11, fill=WHITE, outline=BORDER, width=2)
        draw.text((pill[0] + 8, pill[1] + 4), label, font=FONTS["tiny_bold"], fill=TEXT)
    return sheet


def write_validation_markdown() -> None:
    lines: list[str] = [
        "# CoVe Validation for qanta-buzzer E2E data-flow animation",
        "",
        "This bundle was generated from verified source paths in the local repo.",
        "The scene grouping is an inference over entrypoints, but each animated",
        "claim below is backed by source evidence.",
        "",
        "## Verdict",
        "",
        "- All constituent claims used in the animation were validated from source.",
        "- Scene grouping into eight flows is an inference from the verified scripts and trainers.",
        "- Known caveat carried into the animation: alias substitution currently falls back to an empty lookup in this end-to-end path.",
        "",
        "## Verified claims",
        "",
    ]

    for claim in CLAIMS:
        lines.append(f"- `{claim.claim_id}` {claim.claim}")
        lines.append(f"  Verdict: `{claim.verdict}`")
        lines.append("  Evidence:")
        for ref in claim.evidence:
            lines.append(f"  - `{ref}`")
        lines.append("")

    lines.extend(
        [
            "## Scene mapping",
            "",
        ]
    )
    for scene in SCENES:
        lines.append(f"- `{scene.title}` -> {', '.join(scene.claim_ids)}")

    VALIDATION_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    frames: list[Image.Image] = []
    frame_index = 1
    pulse_values = (0.15, 0.60)

    for scene_index, scene in enumerate(SCENES):
        for step_index, _step in enumerate(scene.steps):
            for pulse in pulse_values:
                frame = render_scene(scene, scene_index, pulse, step_index)
                frame_path = FRAMES_DIR / f"frame_{frame_index:02d}.png"
                frame.save(frame_path)
                frames.append(frame)
                frame_index += 1

    durations = [180] * len(frames)
    if durations:
        durations[-1] = 1200
    frames[0].save(
        GIF_OUT,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )

    contact = build_contact_sheet(frames)
    contact.save(CONTACT_OUT)
    write_validation_markdown()

    print(f"Wrote {len(frames)} frames to {FRAMES_DIR}")
    print(f"Wrote GIF to {GIF_OUT}")
    print(f"Wrote contact sheet to {CONTACT_OUT}")
    print(f"Wrote CoVe validation log to {VALIDATION_OUT}")


if __name__ == "__main__":
    main()
