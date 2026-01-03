from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class RolledDie:
    face: int
    is_wild: bool
    die_name: str


@dataclass(frozen=True, slots=True)
class ScoreResult:
    points: int


def _base_three_of_kind_points(face: int) -> int:
    if face == 1:
        return 1000
    return face * 100


def _of_a_kind_points(face: int, count: int) -> int:
    # count >= 3
    base = _base_three_of_kind_points(face)
    return base * (2 ** (count - 3))


def best_score_for_subset(dice: Sequence[RolledDie]) -> ScoreResult:
    """Compute best achievable score for a chosen kept subset.

    Wild dice (Devil's Head wildcard faces) can substitute to complete combos,
    but wild dice alone cannot score (no combo of only wilds counts), and a wild
    does not count as a single '1'.

    Implemented scoring (KCD2):
    - Single 1 = 100 (non-wild only)
    - Single 5 = 50
    - Three-of-kind: (face*100), except 1s = 1000
    - Each additional die beyond 3 doubles that set's value (4,5,6 of a kind)
    - Straights: 1-6 = 1500, 1-5 = 500, 2-6 = 750
    """

    faces = tuple(d.face for d in dice)
    wild_mask = tuple(1 if d.is_wild else 0 for d in dice)

    # Convert to counts + wilds for a compact DP state
    counts = [0] * 7
    wilds = 0
    for f, w in zip(faces, wild_mask, strict=True):
        if w:
            wilds += 1
        else:
            counts[f] += 1

    return ScoreResult(points=_best_score_from_counts(tuple(counts[1:]), wilds))


@lru_cache(maxsize=None)
def _best_score_from_counts(counts_1_to_6: tuple[int, int, int, int, int, int], wilds: int) -> int:
    c = list(counts_1_to_6)

    # If there are only wilds left, they cannot score.
    if wilds > 0 and sum(c) == 0:
        return 0

    best = 0

    # Singles
    if c[0] > 0:  # face 1 (non-wild)
        c[0] -= 1
        best = max(best, 100 + _best_score_from_counts(tuple(c), wilds))
        c[0] += 1

    if c[4] > 0:  # face 5
        c[4] -= 1
        best = max(best, 50 + _best_score_from_counts(tuple(c), wilds))
        c[4] += 1

    # Straights (require at least one non-wild die participating)
    # 1-6
    needed = [1, 1, 1, 1, 1, 1]
    missing = sum(max(0, needed[i] - c[i]) for i in range(6))
    if missing <= wilds and sum(min(needed[i], c[i]) for i in range(6)) >= 1:
        c2 = c[:]
        w2 = wilds
        for i in range(6):
            use_real = min(needed[i], c2[i])
            c2[i] -= use_real
            w2 -= (needed[i] - use_real)
        best = max(best, 1500 + _best_score_from_counts(tuple(c2), w2))

    # 1-5
    needed = [1, 1, 1, 1, 1, 0]
    missing = sum(max(0, needed[i] - c[i]) for i in range(6))
    if missing <= wilds and sum(min(needed[i], c[i]) for i in range(6)) >= 1:
        c2 = c[:]
        w2 = wilds
        for i in range(6):
            use_real = min(needed[i], c2[i])
            c2[i] -= use_real
            w2 -= (needed[i] - use_real)
        best = max(best, 500 + _best_score_from_counts(tuple(c2), w2))

    # 2-6
    needed = [0, 1, 1, 1, 1, 1]
    missing = sum(max(0, needed[i] - c[i]) for i in range(6))
    if missing <= wilds and sum(min(needed[i], c[i]) for i in range(6)) >= 1:
        c2 = c[:]
        w2 = wilds
        for i in range(6):
            use_real = min(needed[i], c2[i])
            c2[i] -= use_real
            w2 -= (needed[i] - use_real)
        best = max(best, 750 + _best_score_from_counts(tuple(c2), w2))

    # N-of-a-kind (3..6) using wilds as needed; must include >=1 real die of that face
    for face_index in range(6):
        real = c[face_index]
        if real <= 0:
            continue

        for k in range(3, 7):
            need_wild = max(0, k - real)
            if need_wild > wilds:
                continue
            take_real = min(real, k)
            c2 = c[:]
            c2[face_index] -= take_real
            best = max(
                best,
                _of_a_kind_points(face_index + 1, k) + _best_score_from_counts(tuple(c2), wilds - need_wild),
            )

    return best


def enumerate_scoring_subsets(roll: Sequence[RolledDie]) -> list[tuple[int, tuple[int, ...]]]:
    """Return list of (points, kept_indices) for all non-zero scoring subsets."""

    n = len(roll)
    out: list[tuple[int, tuple[int, ...]]] = []

    for mask in range(1, 1 << n):
        kept_idx: list[int] = []
        subset: list[RolledDie] = []
        for i in range(n):
            if mask & (1 << i):
                kept_idx.append(i)
                subset.append(roll[i])

        points = best_score_for_subset(subset).points
        if points > 0:
            out.append((points, tuple(kept_idx)))

    # De-duplicate by kept indices (score engine is deterministic), sort high-to-low
    out.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
    return out


def _run_self_test() -> None:
    # Singles
    assert best_score_for_subset([RolledDie(1, False, "x")]).points == 100
    assert best_score_for_subset([RolledDie(5, False, "x")]).points == 50

    # Of-a-kind + doubling
    assert best_score_for_subset([RolledDie(2, False, "x")] * 3).points == 200
    assert best_score_for_subset([RolledDie(2, False, "x")] * 4).points == 400
    assert best_score_for_subset([RolledDie(5, False, "x")] * 6).points == 4000

    # Straights
    assert best_score_for_subset([RolledDie(i, False, "x") for i in (1, 2, 3, 4, 5, 6)]).points == 1500
    assert best_score_for_subset([RolledDie(i, False, "x") for i in (1, 2, 3, 4, 5)]).points == 500
    assert best_score_for_subset([RolledDie(i, False, "x") for i in (2, 3, 4, 5, 6)]).points == 750

    # Wildcard behavior
    wild = RolledDie(1, True, "Devil's head die")
    assert best_score_for_subset([wild]).points == 0  # wild is not a single 1
    assert best_score_for_subset([wild, wild]).points == 0
    assert best_score_for_subset([wild, wild, wild]).points == 0  # wilds alone never score

    # Wild completes combos
    assert best_score_for_subset([RolledDie(2, False, "x"), RolledDie(2, False, "x"), wild]).points == 200
    assert best_score_for_subset([RolledDie(2, False, "x"), RolledDie(2, False, "x"), RolledDie(2, False, "x"), wild]).points == 400

    # Wild completes straights
    assert (
        best_score_for_subset(
            [RolledDie(1, False, "x"), RolledDie(2, False, "x"), RolledDie(3, False, "x"), RolledDie(4, False, "x"), RolledDie(5, False, "x"), wild]
        ).points
        == 1500
    )
    assert (
        best_score_for_subset(
            [RolledDie(2, False, "x"), RolledDie(3, False, "x"), RolledDie(4, False, "x"), RolledDie(5, False, "x"), RolledDie(6, False, "x"), wild]
        ).points
        == 1500
    )


if __name__ == "__main__":
    _run_self_test()
    print("OK: scoring self-test passed")
