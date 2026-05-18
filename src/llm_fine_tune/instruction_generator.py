from __future__ import annotations

import random

LANGUAGE_DISPLAY_NAMES: dict[str, str] = {
    "cpp": "C++",
    "java": "Java",
    "python": "Python",
}

INSTRUCTION_TEMPLATES: list[str] = [
    "Rewrite the following {source} code in {target}.",
    "Convert this {source} code to {target}.",
    "Translate the following {source} solution into {target}.",
    "Can you convert this {source} code to {target}?",
    "Port the following {source} code to {target}.",
    "Reimplement this {source} solution in {target}.",
    "Given this {source} code, produce an equivalent {target} implementation.",
    "Transform the following {source} code into {target}.",
    "I have a {source} solution. Convert it to {target}.",
    "Please rewrite this {source} code as {target}.",
]


def generate_instruction(source_lang: str, target_lang: str, rng: random.Random) -> str:
    """Pick a random template and format it with the given source and target language names."""
    source = LANGUAGE_DISPLAY_NAMES[source_lang]
    target = LANGUAGE_DISPLAY_NAMES[target_lang]
    return rng.choice(INSTRUCTION_TEMPLATES).format(source=source, target=target)
