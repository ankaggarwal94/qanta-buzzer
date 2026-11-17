"""Small synthetic jane-paired-v1 input for engineering smokes only."""
from __future__ import annotations


OBJECTS = (
    ("Mercury", "This planet has a large iron core.", "Its year lasts about 88 Earth days."),
    ("Venus", "This planet rotates retrograde.", "Its atmosphere is rich in carbon dioxide."),
    ("Earth", "This planet has one natural satellite.", "It is our home planet."),
    ("Mars", "This planet has two small moons.", "Olympus Mons rises on its surface."),
    ("Jupiter", "This planet has a long-lived anticyclonic storm.", "The storm is the Great Red Spot."),
    ("Saturn", "This planet has conspicuous rings.", "Titan is its largest moon."),
    ("Uranus", "This planet has a highly tilted rotation axis.", "William Herschel discovered it in 1781."),
    ("Neptune", "This planet has a moon in a retrograde orbit.", "That moon is Triton."),
    ("Ceres", "This dwarf planet lies in the asteroid belt.", "The Dawn spacecraft visited it."),
    ("Pluto", "This dwarf planet has a large moon.", "That moon is Charon."),
)


def inputs():
    questions = []
    for i, (label, first, clue) in enumerate(OBJECTS):
        full = first + " " + clue
        words = full.split()
        prefixes = [{"text": " ".join(words[:len(words)*j//5])} for j in range(1, 6)]
        alternatives = [OBJECTS[(i + j) % 8][0] for j in range(1, 4)]
        alternatives.insert(i % 4, label)
        questions.append({
            "qid": f"fixture-{i}", "group_id": f"body-{i}",
            "split": "train" if i < 8 else ("val" if i == 8 else "test"),
            "source": {"id": f"fixture-{i}", "canonical_answer": label,
                       "category": "Inner Solar System" if i < 4 or i == 8 else "Outer Solar System"},
            "question": full, "prefixes": prefixes,
            "answer": {"accepted": [label], "rejected": [], "prompt": []},
            "menus": [{"condition": "independent_pool", "gold_option_id": str(i % 4 + 1),
                       "options": [{"id": str(j), "text": a} for j, a in enumerate(alternatives, 1)]}],
        })
    return {"schema_version": "jane-paired-v1", "evidence_scope": "synthetic_fixture",
            "questions": questions}
