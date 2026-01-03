from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class DieSpec:
    name: str
    # Face-indexed probabilities, faces 1..6 stored at indices 0..5
    probabilities: tuple[float, float, float, float, float, float]
    max_quantity: int | None
    wildcard_faces: frozenset[int]


def load_dice_specs(path: str | Path) -> list[DieSpec]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    dice_specs: list[DieSpec] = []

    for die in data["dice"]:
        name = str(die["name"])
        max_quantity = die.get("maxQuantity", None)
        if max_quantity is not None:
            max_quantity = int(max_quantity)

        wildcard_faces = frozenset(int(x) for x in die.get("wildcardFaces", []))

        probs_raw = die["probabilities"]
        probs_dict: dict[int, float] = {}
        for k, v in probs_raw.items():
            face = int(k)
            probs_dict[face] = float(v)

        # Light validation
        if set(probs_dict.keys()) != {1, 2, 3, 4, 5, 6}:
            raise ValueError(f"Die '{name}' must define probabilities for faces 1..6")
        total = sum(probs_dict.values())
        # The JSON uses rounded decimals (e.g., 0.286), so totals may be slightly off 1.0.
        # Auto-normalize within a reasonable tolerance.
        if not (0.98 <= total <= 1.02):
            raise ValueError(f"Die '{name}' probabilities sum to {total}, expected ~1.0")
        if total != 1.0:
            for face in probs_dict:
                probs_dict[face] = probs_dict[face] / total

        probs = (
            probs_dict[1],
            probs_dict[2],
            probs_dict[3],
            probs_dict[4],
            probs_dict[5],
            probs_dict[6],
        )

        dice_specs.append(
            DieSpec(
                name=name,
                probabilities=probs,
                max_quantity=max_quantity,
                wildcard_faces=wildcard_faces,
            )
        )

    return dice_specs


def get_die_by_name(dice: Sequence[DieSpec], name: str) -> DieSpec:
    for d in dice:
        if d.name == name:
            return d
    raise KeyError(f"Unknown die name: {name!r}")
