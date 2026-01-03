from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .dice_specs import DieSpec, get_die_by_name, load_dice_specs
from .scoring import RolledDie, enumerate_scoring_subsets


@dataclass(frozen=True, slots=True)
class DieInstance:
    spec: DieSpec
    cdf: tuple[float, float, float, float, float, float]

    @staticmethod
    def from_spec(spec: DieSpec) -> "DieInstance":
        p = spec.probabilities
        c1 = p[0]
        c2 = c1 + p[1]
        c3 = c2 + p[2]
        c4 = c3 + p[3]
        c5 = c4 + p[4]
        c6 = c5 + p[5]
        return DieInstance(spec=spec, cdf=(c1, c2, c3, c4, c5, c6))


def _sample_face_from_cdf(rng: random.Random, cdf: tuple[float, float, float, float, float, float]) -> int:
    r = rng.random()
    if r <= cdf[0]:
        return 1
    if r <= cdf[1]:
        return 2
    if r <= cdf[2]:
        return 3
    if r <= cdf[3]:
        return 4
    if r <= cdf[4]:
        return 5
    return 6


def roll_dice(rng: random.Random, dice: Sequence[DieInstance]) -> list[RolledDie]:
    out: list[RolledDie] = []
    for d in dice:
        face = _sample_face_from_cdf(rng, d.cdf)
        is_wild = face in d.spec.wildcard_faces
        out.append(RolledDie(face=face, is_wild=is_wild, die_name=d.spec.name))
    return out


@dataclass(frozen=True, slots=True)
class TurnMetrics:
    busted: bool
    rolls: int
    hot_dice: int


class Strategy:
    name: str = "strategy"

    def choose_keep(
        self,
        rng: random.Random,
        *,
        roll: Sequence[RolledDie],
        rolled_from: Sequence[DieInstance],
        hand: Sequence[DieInstance],
        turn_points: int,
        cap: int,
        goal: int,
    ) -> tuple[int, tuple[int, ...]]:
        raise NotImplementedError

    def should_bank(
        self,
        rng: random.Random,
        *,
        turn_points: int,
        dice_remaining: int,
        cap: int,
        goal: int,
    ) -> bool:
        raise NotImplementedError


class GreedyThresholdStrategy(Strategy):
    def __init__(self, bank_threshold: int):
        self.name = f"greedy_bank_{bank_threshold}"
        self.bank_threshold = bank_threshold

    def choose_keep(
        self,
        rng: random.Random,
        *,
        roll: Sequence[RolledDie],
        rolled_from: Sequence[DieInstance],
        hand: Sequence[DieInstance],
        turn_points: int,
        cap: int,
        goal: int,
    ) -> tuple[int, tuple[int, ...]]:
        options = enumerate_scoring_subsets(roll)
        # Prefer max points; tie-break by using fewer dice (keeps more for reroll)
        options.sort(key=lambda x: (x[0], -len(x[1])), reverse=True)
        return options[0]

    def should_bank(
        self,
        rng: random.Random,
        *,
        turn_points: int,
        dice_remaining: int,
        cap: int,
        goal: int,
    ) -> bool:
        return turn_points >= self.bank_threshold


class HotDiceFirstStrategy(Strategy):
    def __init__(self, bank_threshold: int):
        self.name = f"hotdice_bank_{bank_threshold}"
        self.bank_threshold = bank_threshold

    def choose_keep(
        self,
        rng: random.Random,
        *,
        roll: Sequence[RolledDie],
        rolled_from: Sequence[DieInstance],
        hand: Sequence[DieInstance],
        turn_points: int,
        cap: int,
        goal: int,
    ) -> tuple[int, tuple[int, ...]]:
        options = enumerate_scoring_subsets(roll)
        # Prefer scoring all dice (hot dice), then points
        options.sort(key=lambda x: (len(x[1]), x[0]), reverse=True)
        return options[0]

    def should_bank(
        self,
        rng: random.Random,
        *,
        turn_points: int,
        dice_remaining: int,
        cap: int,
        goal: int,
    ) -> bool:
        return turn_points >= self.bank_threshold


class RolloutStrategy(Strategy):
    """Goal-aware strategy using Monte Carlo rollouts to pick keep + bank.

    This is substantially stronger than fixed thresholds for p_goal-style objectives.
    """

    def __init__(
        self,
        *,
        rollouts_per_decision: int,
        metric: str,
        bank_min_points: int = 0,
        max_steps: int = 80,
    ):
        self.rollouts_per_decision = int(rollouts_per_decision)
        self.metric = metric
        self.bank_min_points = int(bank_min_points)
        self.max_steps = int(max_steps)
        self.name = f"rollout_{metric}_{self.rollouts_per_decision}"

    def _objective(self, final_points: int, *, goal: int, metric: str) -> float:
        if metric == "p_goal":
            return 1.0 if final_points >= goal else 0.0
        if metric == "mean":
            return float(final_points)
        # For mean_minus_std we handle aggregation outside per-sample.
        return float(final_points)

    def _finish_turn_hotdice(
        self,
        rng: random.Random,
        *,
        hand: Sequence[DieInstance],
        remaining: list[DieInstance],
        points: int,
        cap: int,
        goal: int,
        max_steps: int,
    ) -> int:
        # Lightweight default continuation policy: keep hot dice if possible, else max points.
        for _ in range(max_steps):
            if points >= cap:
                return cap

            if not remaining:
                remaining = list(hand)

            roll = roll_dice(rng, remaining)
            options = enumerate_scoring_subsets(roll)
            if not options:
                return 0

            # Prefer using all dice (hot dice), then points.
            options.sort(key=lambda x: (len(x[1]), x[0]), reverse=True)
            keep_points, kept_idx = options[0]
            points += keep_points
            if points >= cap:
                return cap

            kept_set = set(kept_idx)
            remaining = [d for i, d in enumerate(remaining) if i not in kept_set]

            # Simple goal-aware bank heuristic to keep rollouts bounded.
            if points >= self.bank_min_points:
                # Bank if we're already at goal, or if we have only a couple dice left.
                if points >= goal or len(remaining) <= 2:
                    return points

        return min(points, cap)

    def _estimate_from_state(
        self,
        rng: random.Random,
        *,
        hand: Sequence[DieInstance],
        remaining: list[DieInstance],
        points: int,
        cap: int,
        goal: int,
    ) -> float:
        n = max(1, self.rollouts_per_decision)
        if self.metric == "mean_minus_std":
            # Small-sample estimate for (mean - std).
            vals: list[float] = []
            for _ in range(n):
                final_points = self._finish_turn_hotdice(
                    rng,
                    hand=hand,
                    remaining=list(remaining),
                    points=points,
                    cap=cap,
                    goal=goal,
                    max_steps=self.max_steps,
                )
                vals.append(float(final_points))
            mean = sum(vals) / len(vals)
            var = 0.0
            if len(vals) > 1:
                var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
            return mean - (var**0.5)

        total = 0.0
        for _ in range(n):
            final_points = self._finish_turn_hotdice(
                rng,
                hand=hand,
                remaining=list(remaining),
                points=points,
                cap=cap,
                goal=goal,
                max_steps=self.max_steps,
            )
            total += self._objective(final_points, goal=goal, metric=self.metric)
        return total / n

    def choose_keep(
        self,
        rng: random.Random,
        *,
        roll: Sequence[RolledDie],
        rolled_from: Sequence[DieInstance],
        hand: Sequence[DieInstance],
        turn_points: int,
        cap: int,
        goal: int,
    ) -> tuple[int, tuple[int, ...]]:
        options = enumerate_scoring_subsets(roll)
        # Consider a manageable number of candidates: keep best-by-points and best-by-hotdice.
        options.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        top_by_points = options[:12]
        options.sort(key=lambda x: (len(x[1]), x[0]), reverse=True)
        top_by_hot = options[:12]
        candidates = {tuple(k): (p, k) for p, k in (top_by_points + top_by_hot)}

        best: tuple[int, tuple[int, ...]] | None = None
        best_score = float("-inf")

        for points, kept_idx in candidates.values():
            new_points = turn_points + points
            if new_points >= cap:
                return points, kept_idx

            kept_set = set(kept_idx)
            remaining = [d for i, d in enumerate(rolled_from) if i not in kept_set]
            # Hot dice: if we used everything, remaining becomes full hand.
            if not remaining:
                remaining = list(hand)

            score = self._estimate_from_state(
                rng,
                hand=hand,
                remaining=remaining,
                points=new_points,
                cap=cap,
                goal=goal,
            )
            if score > best_score:
                best_score = score
                best = (points, kept_idx)

        assert best is not None
        return best

    def should_bank(
        self,
        rng: random.Random,
        *,
        turn_points: int,
        dice_remaining: int,
        cap: int,
        goal: int,
    ) -> bool:
        # Always bank once goal is reached.
        if turn_points >= goal:
            return True
        if turn_points < self.bank_min_points:
            return False
        # If low dice remain, rolling again is high-variance; bank more often.
        if dice_remaining <= 2 and turn_points >= 500:
            return True
        return False


def simulate_turn(
    rng: random.Random,
    hand: Sequence[DieInstance],
    strategy: Strategy,
    *,
    cap: int = 4000,
    goal: int = 4000,
) -> tuple[int, TurnMetrics]:
    remaining: list[DieInstance] = list(hand)
    points = 0
    rolls = 0
    hot_dice = 0

    for _step in range(200):
        if not remaining:
            remaining = list(hand)
            hot_dice += 1

        roll = roll_dice(rng, remaining)
        rolls += 1

        options = enumerate_scoring_subsets(roll)
        if not options:
            return 0, TurnMetrics(busted=True, rolls=rolls, hot_dice=hot_dice)

        keep_points, kept_idx = strategy.choose_keep(
            rng,
            roll=roll,
            rolled_from=remaining,
            hand=hand,
            turn_points=points,
            cap=cap,
            goal=goal,
        )
        points += keep_points

        # Once cap is reached, additional rolling is never rewarded.
        if points >= cap:
            return cap, TurnMetrics(busted=False, rolls=rolls, hot_dice=hot_dice)

        kept_set = set(kept_idx)
        remaining = [d for i, d in enumerate(remaining) if i not in kept_set]

        if strategy.should_bank(rng, turn_points=points, dice_remaining=len(remaining), cap=cap, goal=goal):
            return points, TurnMetrics(busted=False, rolls=rolls, hot_dice=hot_dice)

    # Safety cap (shouldn't normally hit)
    return points, TurnMetrics(busted=False, rolls=rolls, hot_dice=hot_dice)


@dataclass(frozen=True, slots=True)
class SummaryStats:
    trials: int
    mean: float
    stddev: float
    bust_rate: float
    p_ge_goal: float
    avg_rolls: float
    avg_hot_dice: float


def summarize_turns(
    rng: random.Random,
    hand: Sequence[DieInstance],
    strategy: Strategy,
    *,
    trials: int,
    goal: int | None = 4000,
    cap: int = 4000,
) -> SummaryStats:
    # Welford's algorithm for mean/stddev
    mean = 0.0
    m2 = 0.0
    busts = 0
    hits = 0
    total_rolls = 0
    total_hot = 0

    for i in range(1, trials + 1):
        pts, m = simulate_turn(rng, hand, strategy, cap=cap, goal=(goal if goal is not None else cap))
        reward = pts if pts <= cap else cap
        x = float(reward)
        delta = x - mean
        mean += delta / i
        m2 += delta * (x - mean)

        busts += 1 if m.busted else 0
        if goal is not None and pts >= goal:
            hits += 1
        total_rolls += m.rolls
        total_hot += m.hot_dice

    var = (m2 / (trials - 1)) if trials > 1 else 0.0
    std = var**0.5
    return SummaryStats(
        trials=trials,
        mean=mean,
        stddev=std,
        bust_rate=busts / trials,
        p_ge_goal=(hits / trials) if goal is not None else 0.0,
        avg_rolls=total_rolls / trials,
        avg_hot_dice=total_hot / trials,
    )


def parse_hand(dice_specs: Sequence[DieSpec], hand_arg: str) -> list[DieInstance]:
    # Comma-separated list of 6 names.
    names = [x.strip() for x in hand_arg.split(",") if x.strip()]
    if len(names) != 6:
        raise ValueError("--hand must contain exactly 6 comma-separated die names")

    specs = [get_die_by_name(dice_specs, n) for n in names]

    # Validate maxQuantity constraints
    counts: dict[str, int] = {}
    spec_by_name: dict[str, DieSpec] = {}
    for s in specs:
        counts[s.name] = counts.get(s.name, 0) + 1
        spec_by_name.setdefault(s.name, s)

    for name, count in counts.items():
        s = spec_by_name[name]
        if s.max_quantity is not None and count > s.max_quantity:
            raise ValueError(f"Hand exceeds maxQuantity for {name!r}: {count} > {s.max_quantity}")

    return [DieInstance.from_spec(spec=s) for s in specs]


def build_strategy_preset(name: str, *, goal: int, cap: int, metric: str) -> Strategy:
    if name.startswith("greedy_bank_"):
        return GreedyThresholdStrategy(int(name.removeprefix("greedy_bank_")))
    if name.startswith("hotdice_bank_"):
        return HotDiceFirstStrategy(int(name.removeprefix("hotdice_bank_")))
    if name.startswith("rollout_"):
        # rollout_<metric>_<N> or rollout_<N> (defaults to global metric)
        rest = name.removeprefix("rollout_")
        parts = rest.split("_")
        if len(parts) == 1:
            return RolloutStrategy(rollouts_per_decision=int(parts[0]), metric=metric)
        if len(parts) == 2:
            m, n = parts
            return RolloutStrategy(rollouts_per_decision=int(n), metric=m)
        raise ValueError(f"Unknown rollout preset: {name}")
    raise ValueError(f"Unknown strategy preset: {name}")


def _split_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _seed_for_eval(seed: int, combo_name: str, strategy_name: str) -> int:
    h = hashlib.blake2b(f"{seed}|{combo_name}|{strategy_name}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, byteorder="big", signed=False)


def load_hand_combos_json(path: str) -> list[tuple[str, str]]:
    """Load fixed 6-die hands from a JSON file.

    Returns list of (combo_name, hand_arg) where hand_arg matches --hand format.
    """

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    combos = data.get("combos")
    if not isinstance(combos, list):
        raise ValueError("Combos JSON must contain a list field 'combos'")

    out: list[tuple[str, str]] = []
    for idx, c in enumerate(combos, start=1):
        if not isinstance(c, dict):
            raise ValueError(f"combos[{idx}] must be an object")
        name = c.get("name")
        hand = c.get("hand")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"combos[{idx}].name must be a non-empty string")
        if not isinstance(hand, list) or any(not isinstance(x, str) for x in hand):
            raise ValueError(f"combos[{idx}].hand must be a list of strings")
        if len(hand) != 6:
            raise ValueError(f"combos[{idx}].hand must contain exactly 6 die names")
        hand_arg = ",".join(hand)
        out.append((name, hand_arg))

    return out


def load_hand_combo_map(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, hand_arg in load_hand_combos_json(path):
        if name in out:
            raise ValueError(f"Duplicate combo name in {path!r}: {name}")
        out[name] = hand_arg
    return out


def random_hand(rng: random.Random, dice_specs: Sequence[DieSpec], pool: Sequence[DieSpec] | None = None) -> list[DieInstance]:
    specs = list(pool) if pool is not None else list(dice_specs)
    if not specs:
        raise ValueError("Empty dice pool")

    chosen: list[DieSpec] = []
    counts: dict[str, int] = {}

    for _ in range(6):
        # rejection-sample until maxQuantity constraints satisfied
        for _attempt in range(1000):
            s = rng.choice(specs)
            have = counts.get(s.name, 0)
            if s.max_quantity is not None and have >= s.max_quantity:
                continue
            chosen.append(s)
            counts[s.name] = have + 1
            break
        else:
            raise RuntimeError("Failed to sample a valid hand from pool (constraints too tight?)")

    return [DieInstance.from_spec(spec=s) for s in chosen]


def _hand_names(hand: Sequence[DieInstance]) -> tuple[str, ...]:
    return tuple(d.spec.name for d in hand)


def _counts_by_name(names: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for n in names:
        out[n] = out.get(n, 0) + 1
    return out


def mutate_hand(
    rng: random.Random,
    *,
    hand: Sequence[DieInstance],
    dice_specs: Sequence[DieSpec],
    pool: Sequence[DieSpec] | None,
) -> list[DieInstance]:
    """Randomly replace 1 die while respecting maxQuantity."""

    specs = list(pool) if pool is not None else list(dice_specs)
    if not specs:
        raise ValueError("Empty dice pool")

    names = list(_hand_names(hand))
    counts = _counts_by_name(names)
    idx = rng.randrange(6)
    old = names[idx]
    counts[old] -= 1

    # Rejection-sample a replacement that satisfies constraints
    for _attempt in range(2000):
        s = rng.choice(specs)
        have = counts.get(s.name, 0)
        if s.max_quantity is not None and have >= s.max_quantity:
            continue
        names[idx] = s.name
        break
    else:
        # fallback: return original
        names[idx] = old

    return parse_hand(dice_specs, ",".join(names))


def hand_to_arg(hand: Sequence[DieInstance]) -> str:
    # Canonical (order-independent) representation: sort names.
    # Duplicates are preserved (multiset), but permutations compare equal.
    return ",".join(sorted(d.spec.name for d in hand))


@dataclass(frozen=True, slots=True)
class SearchResult:
    strategy: str
    hand: str
    stats: SummaryStats


@dataclass(frozen=True, slots=True)
class BestOfStrategies:
    best: SearchResult
    all_results: list[SearchResult]


def _fitness_value(stats: SummaryStats, *, metric: str) -> float:
    if metric == "mean":
        return stats.mean
    if metric == "p_goal":
        return stats.p_ge_goal
    if metric == "mean_minus_std":
        return stats.mean - stats.stddev
    raise ValueError("Unknown metric")


def evaluate_hand_best_strategy(
    *,
    rng: random.Random,
    hand: Sequence[DieInstance],
    hand_arg: str,
    strategies: Sequence[str],
    trials: int,
    goal: int,
    cap: int,
    metric: str,
    cache: dict[tuple[str, str, int, int, int], SummaryStats],
) -> BestOfStrategies:
    results: list[SearchResult] = []
    best: SearchResult | None = None
    best_fit = float("-inf")

    for strat_name in strategies:
        key = (hand_arg, strat_name, trials, goal, cap)
        stats = cache.get(key)
        if stats is None:
            strat = build_strategy_preset(strat_name, goal=goal, cap=cap, metric=metric)
            stats = summarize_turns(rng, hand, strat, trials=trials, goal=goal, cap=cap)
            cache[key] = stats

        r = SearchResult(strategy=strat_name, hand=hand_arg, stats=stats)
        results.append(r)
        fit = _fitness_value(stats, metric=metric)
        if fit > best_fit:
            best_fit = fit
            best = r

    assert best is not None
    return BestOfStrategies(best=best, all_results=results)


def run_search(
    *,
    rng: random.Random,
    dice_specs: Sequence[DieSpec],
    pool: Sequence[DieSpec] | None,
    hands: int,
    trials_per_hand: int,
    strategies: Sequence[str],
    top_k: int,
    goal: int,
    metric: str,
    cap: int,
    progress_every: int,
    progress_seconds: float,
    initial_hands: Sequence[Sequence[DieInstance]] | None = None,
) -> list[SearchResult]:
    results: list[SearchResult] = []

    seen_hands: set[str] = set()

    total_evals = hands * max(1, len(strategies))
    completed = 0

    start_t = time.monotonic()
    last_report_t = start_t
    best_so_far: SearchResult | None = None
    best_fit = float("-inf")

    def maybe_report(*, force: bool = False) -> None:
        nonlocal last_report_t
        if progress_seconds <= 0 and not force:
            return
        now = time.monotonic()
        if not force and (now - last_report_t) < progress_seconds:
            return
        last_report_t = now

        elapsed = now - start_t
        rate = (completed / elapsed) if elapsed > 0 else 0.0
        eta = ((total_evals - completed) / rate) if rate > 0 else float("inf")
        pct = 100.0 * completed / max(1, total_evals)

        best_part = "best=(none)"
        if best_so_far is not None:
            s = best_so_far.stats
            exp_rounds = (1.0 / s.p_ge_goal) if s.p_ge_goal > 0 else float("inf")
            best_part = (
                f"best(metric={metric})={best_fit:.4f} strat={best_so_far.strategy} "
                f"p>={goal}={s.p_ge_goal:.4f} E[rounds]~={exp_rounds:.1f} mean={s.mean:.1f} std={s.stddev:.1f} bust={s.bust_rate:.3f}"
            )

        print(
            f"[search] {completed}/{total_evals} ({pct:.1f}%) elapsed={elapsed:.0f}s eta~={eta:.0f}s {best_part}",
            file=__import__("sys").stderr,
        )

    seeds = list(initial_hands) if initial_hands is not None else []
    for _ in range(min(hands, len(seeds))):
        hand = list(seeds.pop(0))
        hand_arg = hand_to_arg(hand)
        if hand_arg in seen_hands:
            continue
        seen_hands.add(hand_arg)

        for strat_name in strategies:
            strat = build_strategy_preset(strat_name, goal=goal, cap=cap, metric=metric)
            stats = summarize_turns(rng, hand, strat, trials=trials_per_hand, goal=goal, cap=cap)
            sr = SearchResult(strategy=strat_name, hand=hand_arg, stats=stats)
            results.append(sr)

            fit = _fitness_value(stats, metric=metric)
            if fit > best_fit:
                best_fit = fit
                best_so_far = sr

            completed += 1
            if progress_every > 0 and (completed % progress_every) == 0:
                # Lightweight progress indicator (stderr) so stdout stays parseable.
                pct = 100.0 * completed / max(1, total_evals)
                print(f"[search] {completed}/{total_evals} ({pct:.1f}%)", file=__import__("sys").stderr)
            maybe_report()

    maybe_report(force=True)

    target_random = hands - min(hands, len(initial_hands) if initial_hands is not None else 0)
    added_random = 0
    while added_random < target_random:
        hand = random_hand(rng, dice_specs, pool=pool)
        hand_arg = hand_to_arg(hand)
        if hand_arg in seen_hands:
            continue
        seen_hands.add(hand_arg)

        for strat_name in strategies:
            strat = build_strategy_preset(strat_name, goal=goal, cap=cap, metric=metric)
            stats = summarize_turns(rng, hand, strat, trials=trials_per_hand, goal=goal, cap=cap)
            results.append(SearchResult(strategy=strat_name, hand=hand_arg, stats=stats))

            completed += 1
            if progress_every > 0 and (completed % progress_every) == 0:
                # Lightweight progress indicator (stderr) so stdout stays parseable.
                pct = 100.0 * completed / max(1, total_evals)
                print(f"[search] {completed}/{total_evals} ({pct:.1f}%)", file=__import__("sys").stderr)

        added_random += 1

    if metric == "mean":
        results.sort(key=lambda r: (r.stats.mean, -r.stats.stddev, -r.stats.bust_rate), reverse=True)
    elif metric == "p_goal":
        results.sort(key=lambda r: (r.stats.p_ge_goal, r.stats.mean, -r.stats.bust_rate), reverse=True)
    elif metric == "mean_minus_std":
        results.sort(key=lambda r: (r.stats.mean - r.stats.stddev, r.stats.mean), reverse=True)
    else:
        raise ValueError("Unknown metric")
    return results[:top_k]


def run_evolutionary_search(
    *,
    rng: random.Random,
    dice_specs: Sequence[DieSpec],
    pool: Sequence[DieSpec] | None,
    strategies: Sequence[str],
    hands: int,
    trials_per_hand: int,
    final_trials: int,
    final_candidates: int,
    elite_frac: float,
    generations: int,
    goal: int,
    cap: int,
    metric: str,
    progress_every: int,
    progress_seconds: float,
    top_k: int,
    initial_population: Sequence[Sequence[DieInstance]] | None = None,
) -> list[SearchResult]:
    """Smarter search: keep elites, mutate, cache evaluations.

    Evaluates each hand by choosing the best strategy according to `metric`.
    """

    if hands < 2:
        raise ValueError("--hands must be >= 2")
    elite_n = max(1, int(hands * elite_frac))
    cache: dict[tuple[str, str, int, int, int], SummaryStats] = {}

    population: list[list[DieInstance]] = []
    if initial_population is not None:
        # de-dup seeds by hand arg while preserving order
        seen: set[str] = set()
        for h in initial_population:
            arg = hand_to_arg(h)
            if arg in seen:
                continue
            seen.add(arg)
            population.append(list(h))
            if len(population) >= hands:
                break

    while len(population) < hands:
        population.append(random_hand(rng, dice_specs, pool=pool))
    best_overall: SearchResult | None = None
    best_logged_fit = float("-inf")
    evals = 0

    seen_hands: set[str] = set()

    start_t = time.monotonic()
    last_report_t = start_t

    total_evals = generations * hands * max(1, len(strategies))

    def maybe_report(*, gen: int, gen_total: int, force: bool = False) -> None:
        nonlocal last_report_t
        if progress_seconds <= 0 and not force:
            return
        now = time.monotonic()
        if not force and (now - last_report_t) < progress_seconds:
            return
        last_report_t = now

        elapsed = now - start_t
        rate = (evals / elapsed) if elapsed > 0 else 0.0
        eta = ((total_evals - evals) / rate) if rate > 0 else float("inf")
        pct = 100.0 * evals / max(1, total_evals)

        best_part = "best=(none)"
        if best_overall is not None:
            s = best_overall.stats
            exp_rounds = (1.0 / s.p_ge_goal) if s.p_ge_goal > 0 else float("inf")
            best_part = (
                f"best(metric={metric})={_fitness_value(s, metric=metric):.4f} strat={best_overall.strategy} hand={best_overall.hand} "
                f"p>={goal}={s.p_ge_goal:.4f} E[rounds]~={exp_rounds:.1f} mean={s.mean:.1f} std={s.stddev:.1f} bust={s.bust_rate:.3f}"
            )

        print(
            f"[evolve] gen={gen}/{gen_total} ({pct:.1f}%) evals~={evals}/{total_evals} elapsed={elapsed:.0f}s eta~={eta:.0f}s uniq={len(seen_hands)} {best_part}",
            file=__import__("sys").stderr,
        )

    for gen in range(1, generations + 1):
        scored: list[tuple[float, SearchResult, list[DieInstance]]] = []

        for hand in population:
            hand_arg = hand_to_arg(hand)
            seen_hands.add(hand_arg)
            bos = evaluate_hand_best_strategy(
                rng=rng,
                hand=hand,
                hand_arg=hand_arg,
                strategies=strategies,
                trials=trials_per_hand,
                goal=goal,
                cap=cap,
                metric=metric,
                cache=cache,
            )
            evals += len(strategies)
            fit = _fitness_value(bos.best.stats, metric=metric)
            scored.append((fit, bos.best, hand))

            # Update best-so-far immediately so time-based progress can show it.
            if best_overall is None or fit > _fitness_value(best_overall.stats, metric=metric):
                best_overall = bos.best

            maybe_report(gen=gen, gen_total=generations)

        scored.sort(key=lambda x: x[0], reverse=True)
        elites = scored[:elite_n]

        # Track best (log when improved)
        if elites and elites[0][0] > best_logged_fit:
            best_logged_fit = elites[0][0]
            best_overall = elites[0][1]
            s = best_overall.stats
            exp_rounds = (1.0 / s.p_ge_goal) if s.p_ge_goal > 0 else float("inf")
            print(
                f"[best] gen={gen} metric={metric}={_fitness_value(s, metric=metric):.4f} "
                f"p>={goal}={s.p_ge_goal:.4f} E[rounds]~={exp_rounds:.1f} mean={s.mean:.1f} std={s.stddev:.1f} bust={s.bust_rate:.3f} "
                f"strat={best_overall.strategy} hand={best_overall.hand}",
                file=__import__("sys").stderr,
            )

        if progress_every > 0 and (gen == 1 or (gen % progress_every) == 0 or gen == generations):
            b = elites[0][1]
            s = b.stats
            exp_rounds = (1.0 / s.p_ge_goal) if s.p_ge_goal > 0 else float("inf")
            print(
                f"[evolve] gen={gen}/{generations} evals~={evals} "
                f"best(metric={metric})={_fitness_value(s, metric=metric):.4f} "
                f"p>={goal}={s.p_ge_goal:.4f} E[rounds]~={exp_rounds:.1f} mean={s.mean:.1f} std={s.stddev:.1f} bust={s.bust_rate:.3f} "
                f"strat={b.strategy} hand={b.hand}",
                file=__import__("sys").stderr,
            )

        # Next generation: keep elites, fill rest with mutated elites
        next_pop: list[list[DieInstance]] = [h for (_fit, _best, h) in elites]
        next_seen: set[str] = {hand_to_arg(h) for h in next_pop}
        while len(next_pop) < hands:
            parent = rng.choice(next_pop)
            # Try to avoid re-adding an already-seen multiset hand.
            for _attempt in range(50):
                child = mutate_hand(rng, hand=parent, dice_specs=dice_specs, pool=pool)
                key = hand_to_arg(child)
                if key in next_seen:
                    continue
                next_seen.add(key)
                next_pop.append(child)
                break
            else:
                # Give up after a few attempts to avoid stalling; allow duplicates.
                child = mutate_hand(rng, hand=parent, dice_specs=dice_specs, pool=pool)
                next_pop.append(child)
        population = next_pop

    maybe_report(gen=generations, gen_total=generations, force=True)

    if best_overall is None:
        raise RuntimeError("No results")

    # Optional high-precision final evaluation (can be expensive)
    if final_trials <= 0:
        # Return a ranked top_k from the final population using the same
        # (lower precision) trials_per_hand evaluation.
        uniq: dict[str, list[DieInstance]] = {}
        for h in population:
            uniq.setdefault(hand_to_arg(h), h)

        finals: list[SearchResult] = []
        for hand_arg, hand in uniq.items():
            bos = evaluate_hand_best_strategy(
                rng=rng,
                hand=hand,
                hand_arg=hand_arg,
                strategies=strategies,
                trials=trials_per_hand,
                goal=goal,
                cap=cap,
                metric=metric,
                cache=cache,
            )
            finals.append(bos.best)

        finals.sort(key=lambda r: _fitness_value(r.stats, metric=metric), reverse=True)
        return finals[:top_k]

    # Re-evaluate top unique hands (by preliminary fitness) at higher precision
    uniq: dict[str, list[DieInstance]] = {}
    for h in population:
        uniq.setdefault(hand_to_arg(h), h)

    prelim_cache: dict[tuple[str, str, int, int, int], SummaryStats] = {}
    prelim_scored: list[tuple[float, str, list[DieInstance], SearchResult]] = []
    for hand_arg, hand in uniq.items():
        bos = evaluate_hand_best_strategy(
            rng=rng,
            hand=hand,
            hand_arg=hand_arg,
            strategies=strategies,
            trials=trials_per_hand,
            goal=goal,
            cap=cap,
            metric=metric,
            cache=prelim_cache,
        )
        fit = _fitness_value(bos.best.stats, metric=metric)
        prelim_scored.append((fit, hand_arg, hand, bos.best))

    prelim_scored.sort(key=lambda x: x[0], reverse=True)
    candidates = prelim_scored[: max(top_k, final_candidates)]

    final_cache: dict[tuple[str, str, int, int, int], SummaryStats] = {}
    finals: list[SearchResult] = []
    best_final: SearchResult | None = None

    final_start_t = time.monotonic()
    final_last_report_t = final_start_t

    def maybe_report_final(*, idx: int, total: int, force: bool = False) -> None:
        nonlocal final_last_report_t
        if progress_seconds <= 0 and not force:
            return
        now = time.monotonic()
        if not force and (now - final_last_report_t) < progress_seconds:
            return
        final_last_report_t = now

        elapsed = now - final_start_t
        rate = (idx / elapsed) if elapsed > 0 else 0.0
        eta = ((total - idx) / rate) if rate > 0 else float("inf")
        pct = 100.0 * idx / max(1, total)

        bf = best_final
        best_part = "best=(none)"
        if bf is not None:
            s = bf.stats
            exp_rounds = (1.0 / s.p_ge_goal) if s.p_ge_goal > 0 else float("inf")
            best_part = (
                f"best(metric={metric})={_fitness_value(s, metric=metric):.4f} p>={goal}={s.p_ge_goal:.4f} E[rounds]~={exp_rounds:.1f} "
                f"mean={s.mean:.1f} std={s.stddev:.1f} strat={bf.strategy} hand={bf.hand}"
            )

        print(
            f"[final] {idx}/{total} ({pct:.1f}%) elapsed={elapsed:.0f}s eta~={eta:.0f}s {best_part}",
            file=__import__("sys").stderr,
        )
    for idx, (_fit, hand_arg, hand, _best_prelim) in enumerate(candidates, start=1):
        bos = evaluate_hand_best_strategy(
            rng=rng,
            hand=hand,
            hand_arg=hand_arg,
            strategies=strategies,
            trials=final_trials,
            goal=goal,
            cap=cap,
            metric=metric,
            cache=final_cache,
        )
        finals.append(bos.best)

        if best_final is None or _fitness_value(bos.best.stats, metric=metric) > _fitness_value(best_final.stats, metric=metric):
            best_final = bos.best

        if progress_every > 0:
            bf = best_final if best_final is not None else bos.best
            s = bf.stats
            exp_rounds = (1.0 / s.p_ge_goal) if s.p_ge_goal > 0 else float("inf")
            print(
                f"[final] {idx}/{len(candidates)} best(metric={metric})={_fitness_value(s, metric=metric):.4f} "
                f"p>={goal}={s.p_ge_goal:.4f} E[rounds]~={exp_rounds:.1f} mean={s.mean:.1f} std={s.stddev:.1f} "
                f"strat={bf.strategy} hand={bf.hand}",
                file=__import__("sys").stderr,
            )

        maybe_report_final(idx=idx, total=len(candidates))

    maybe_report_final(idx=len(candidates), total=len(candidates), force=True)

    finals.sort(key=lambda r: _fitness_value(r.stats, metric=metric), reverse=True)
    return finals[:top_k]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="KCD2 Farkle (dice) Monte Carlo simulator")
    ap.add_argument(
        "--dice-json",
        default=str(Path(__file__).resolve().parent / "dice_probabilities.json"),
        help="Path to dice_probabilities.json",
    )

    sp = ap.add_subparsers(dest="cmd", required=True)

    sp_sim = sp.add_parser("simulate", help="Simulate one hand + strategy")
    sim_hand_group = sp_sim.add_mutually_exclusive_group(required=True)
    sim_hand_group.add_argument(
        "--hand",
        help="6 dice names, comma-separated (must match JSON 'name' exactly)",
    )
    sim_hand_group.add_argument(
        "--combo",
        help="Name of a combo in combos JSON (see tools/kcd2_farkle/dice_combos.json)",
    )
    sp_sim.add_argument(
        "--combos-json",
        default=str(Path(__file__).resolve().parent / "dice_combos.json"),
        help="Path to combos JSON (used by --combo)",
    )
    sp_sim.add_argument("--seed", type=int, default=1)
    sp_sim.add_argument(
        "--strategy",
        default="greedy_bank_500",
        help="Strategy preset (e.g. greedy_bank_500, hotdice_bank_500)",
    )
    sp_sim.add_argument("--trials", type=int, default=10000)
    sp_sim.add_argument("--goal", type=int, default=4000, help="Goal score for p(score>=goal)")
    sp_sim.add_argument("--cap", type=int, default=4000, help="Cap rewards at this score")

    sp_search = sp.add_parser("search", help="Random-search best hands/strategies")
    sp_search.add_argument("--seed", type=int, default=1)
    sp_search.add_argument("--hands", type=int, default=200, help="Number of random hands to evaluate")
    sp_search.add_argument("--trials", type=int, default=5000, help="Trials per hand per strategy")
    sp_search.add_argument("--final-trials", type=int, default=50000, help="High-precision trials for top candidates")
    sp_search.add_argument("--final-candidates", type=int, default=30, help="How many candidate hands to re-evaluate at final-trials")
    sp_search.add_argument("--goal", type=int, default=4000, help="Goal score for p(score>=goal)")
    sp_search.add_argument("--cap", type=int, default=4000, help="Cap rewards at this score")
    sp_search.add_argument("--metric", choices=["mean", "p_goal", "mean_minus_std"], default="p_goal")
    sp_search.add_argument("--progress-every", type=int, default=10, help="Print progress every N evals (0 disables)")
    sp_search.add_argument(
        "--progress-seconds",
        type=float,
        default=20.0,
        help="Emit a progress line about every N seconds (0 disables)",
    )
    sp_search.add_argument("--algo", choices=["random", "evolve"], default="evolve")
    sp_search.add_argument("--generations", type=int, default=30)
    sp_search.add_argument("--elite-frac", type=float, default=0.2)
    sp_search.add_argument(
        "--strategies",
        default="greedy_bank_250,greedy_bank_500,greedy_bank_750,greedy_bank_1000,hotdice_bank_500,hotdice_bank_750",
        help=(
            "Comma-separated strategy presets. Available: greedy_bank_<N>, hotdice_bank_<N>, "
            "rollout_<N> (uses --metric), rollout_<metric>_<N> (e.g. rollout_p_goal_8)."
        ),
    )
    sp_search.add_argument("--top", type=int, default=10, help="Show top N results")
    sp_search.add_argument(
        "--pool",
        default="",
        help="Optional comma-separated dice names to restrict sampling pool",
    )
    sp_search.add_argument(
        "--combos-json",
        default=str(Path(__file__).resolve().parent / "dice_combos.json"),
        help="Path to combos JSON (used by --seed-combos)",
    )
    sp_search.add_argument(
        "--seed-combos",
        default="*",
        help=(
            "Comma-separated combo names to seed the initial hand population (evolve) or to include in evaluation (random). "
            "Default '*' means seed from all combos in --combos-json."
        ),
    )
    sp_search.add_argument(
        "--no-seed-combos",
        action="store_true",
        help="Disable seeding from combos (ignore --seed-combos)",
    )

    sp_compare = sp.add_parser("compare", help="Evaluate fixed hands from a combos JSON (no evolution)")
    sp_compare.add_argument(
        "--combos-json",
        default=str(Path(__file__).resolve().parent / "dice_combos.json"),
        help="Path to combos JSON (see tools/kcd2_farkle/dice_combos.json)",
    )
    sp_compare.add_argument("--seed", type=int, default=1)
    sp_compare.add_argument("--trials", type=int, default=20000)
    sp_compare.add_argument("--goal", type=int, default=4000)
    sp_compare.add_argument("--cap", type=int, default=4000)
    sp_compare.add_argument("--metric", choices=["mean", "p_goal", "mean_minus_std"], default="p_goal")
    sp_compare.add_argument(
        "--strategies",
        default="greedy_bank_250,greedy_bank_500,greedy_bank_750,greedy_bank_1000,hotdice_bank_500,hotdice_bank_750",
        help=(
            "Comma-separated strategy presets. Available: greedy_bank_<N>, hotdice_bank_<N>, "
            "rollout_<N> (uses --metric), rollout_<metric>_<N> (e.g. rollout_p_goal_8)."
        ),
    )
    sp_compare.add_argument("--top", type=int, default=5, help="Show top N results")
    sp_compare.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Also print progress every N evals (0 disables; time-based progress still applies)",
    )
    sp_compare.add_argument(
        "--progress-seconds",
        type=float,
        default=20.0,
        help="Emit a progress line about every N seconds (0 disables)",
    )

    args = ap.parse_args(argv)

    dice_specs = load_dice_specs(args.dice_json)

    if args.cmd == "simulate":
        rng = random.Random(args.seed)
        hand_arg = args.hand
        if hand_arg is None:
            combo_map = load_hand_combo_map(args.combos_json)
            if args.combo not in combo_map:
                available = ", ".join(sorted(combo_map.keys()))
                raise ValueError(f"Unknown combo {args.combo!r}. Available: {available}")
            hand_arg = combo_map[args.combo]

        hand = parse_hand(dice_specs, hand_arg)
        strat = build_strategy_preset(args.strategy, goal=args.goal, cap=args.cap, metric="p_goal")
        stats = summarize_turns(rng, hand, strat, trials=args.trials, goal=args.goal, cap=args.cap)
        print(f"strategy={strat.name}")
        print(f"trials={stats.trials}")
        print(f"cap={args.cap}")
        print(f"mean_capped_turn_points={stats.mean:.2f}")
        print(f"stddev_capped_turn_points={stats.stddev:.2f}")
        print(f"bust_rate={stats.bust_rate:.4f}")
        print(f"p_turn_points_ge_{args.goal}={stats.p_ge_goal:.4f}")
        if stats.p_ge_goal > 0:
            print(f"expected_rounds_to_ge_{args.goal}~={1.0 / stats.p_ge_goal:.2f}")
        else:
            print(f"expected_rounds_to_ge_{args.goal}~=inf")
        print(f"avg_rolls_per_turn={stats.avg_rolls:.2f}")
        print(f"avg_hot_dice_per_turn={stats.avg_hot_dice:.2f}")
        return 0

    if args.cmd == "search":
        rng = random.Random(args.seed)
        strategy_names = _split_csv(args.strategies)
        pool_specs: list[DieSpec] | None = None
        if args.pool.strip():
            pool_names = _split_csv(args.pool)
            pool_specs = [get_die_by_name(dice_specs, n) for n in pool_names]

        seed_hands: list[list[DieInstance]] = []
        if not args.no_seed_combos:
            combo_map = load_hand_combo_map(args.combos_json)
            seed_spec = args.seed_combos.strip()
            if seed_spec in ("", "*", "all"):
                # Default behavior: seed from all known combos.
                for combo_name in combo_map.keys():
                    seed_hands.append(parse_hand(dice_specs, combo_map[combo_name]))
            else:
                for combo_name in _split_csv(seed_spec):
                    if combo_name not in combo_map:
                        available = ", ".join(sorted(combo_map.keys()))
                        raise ValueError(f"Unknown seed combo {combo_name!r}. Available: {available}")
                    seed_hands.append(parse_hand(dice_specs, combo_map[combo_name]))

        if args.algo == "random":
            top = run_search(
                rng=rng,
                dice_specs=dice_specs,
                pool=pool_specs,
                hands=args.hands,
                trials_per_hand=args.trials,
                strategies=strategy_names,
                top_k=args.top,
                goal=args.goal,
                metric=args.metric,
                cap=args.cap,
                progress_every=args.progress_every,
                progress_seconds=args.progress_seconds,
                initial_hands=seed_hands if seed_hands else None,
            )
        else:
            top = run_evolutionary_search(
                rng=rng,
                dice_specs=dice_specs,
                pool=pool_specs,
                strategies=strategy_names,
                hands=args.hands,
                trials_per_hand=args.trials,
                final_trials=args.final_trials,
                final_candidates=args.final_candidates,
                elite_frac=args.elite_frac,
                generations=args.generations,
                goal=args.goal,
                cap=args.cap,
                metric=args.metric,
                progress_every=args.progress_every,
                progress_seconds=args.progress_seconds,
                top_k=args.top,
                initial_population=seed_hands if seed_hands else None,
            )

        for i, r in enumerate(top, start=1):
            s = r.stats
            exp_rounds = (1.0 / s.p_ge_goal) if s.p_ge_goal > 0 else float("inf")
            print(
                f"#{i} p>={args.goal}={s.p_ge_goal:.4f} E[rounds]~={exp_rounds:.1f} mean_capped={s.mean:.1f} std_capped={s.stddev:.1f} bust={s.bust_rate:.3f} "
                f"rolls={s.avg_rolls:.2f} hot={s.avg_hot_dice:.2f} strat={r.strategy} hand={r.hand}"
            )
        return 0

    if args.cmd == "compare":
        strategy_names = _split_csv(args.strategies)
        combos = load_hand_combos_json(args.combos_json)

        results: list[tuple[str, SearchResult]] = []
        total_evals = len(combos) * max(1, len(strategy_names))
        completed = 0

        start_t = time.monotonic()
        last_report_t = start_t
        best_so_far: tuple[float, str, SearchResult] | None = None

        def maybe_report(*, force: bool = False) -> None:
            nonlocal last_report_t
            if args.progress_seconds <= 0 and not force:
                return
            now = time.monotonic()
            if not force and (now - last_report_t) < args.progress_seconds:
                return
            last_report_t = now

            elapsed = now - start_t
            rate = (completed / elapsed) if elapsed > 0 else 0.0
            eta = ((total_evals - completed) / rate) if rate > 0 else float("inf")
            pct = 100.0 * completed / max(1, total_evals)

            best_part = "best=(none)"
            if best_so_far is not None:
                fit, combo_name, r = best_so_far
                s = r.stats
                exp_rounds = (1.0 / s.p_ge_goal) if s.p_ge_goal > 0 else float("inf")
                best_part = (
                    f"best(metric={args.metric})={fit:.4f} combo={combo_name} strat={r.strategy} "
                    f"p>={args.goal}={s.p_ge_goal:.4f} E[rounds]~={exp_rounds:.1f} mean={s.mean:.1f} std={s.stddev:.1f} bust={s.bust_rate:.3f}"
                )

            print(
                f"[compare] {completed}/{total_evals} ({pct:.1f}%) elapsed={elapsed:.0f}s eta~={eta:.0f}s {best_part}",
                file=__import__("sys").stderr,
            )

        for combo_name, hand_arg in combos:
            hand = parse_hand(dice_specs, hand_arg)
            for strat_name in strategy_names:
                eval_seed = _seed_for_eval(args.seed, combo_name, strat_name)
                rng = random.Random(eval_seed)
                strat = build_strategy_preset(strat_name, goal=args.goal, cap=args.cap, metric=args.metric)
                stats = summarize_turns(rng, hand, strat, trials=args.trials, goal=args.goal, cap=args.cap)
                sr = SearchResult(strategy=strat_name, hand=hand_arg, stats=stats)
                results.append((combo_name, sr))

                fit = _fitness_value(stats, metric=args.metric)
                if best_so_far is None or fit > best_so_far[0]:
                    best_so_far = (fit, combo_name, sr)

                completed += 1
                if args.progress_every > 0 and (completed % args.progress_every) == 0:
                    pct = 100.0 * completed / max(1, total_evals)
                    print(f"[compare] {completed}/{total_evals} ({pct:.1f}%)", file=__import__("sys").stderr)
                maybe_report()

        maybe_report(force=True)

        results.sort(key=lambda x: _fitness_value(x[1].stats, metric=args.metric), reverse=True)
        top = results[: args.top]

        for i, (combo_name, r) in enumerate(top, start=1):
            s = r.stats
            exp_rounds = (1.0 / s.p_ge_goal) if s.p_ge_goal > 0 else float("inf")
            print(
                f"#{i} combo={combo_name} p>={args.goal}={s.p_ge_goal:.4f} E[rounds]~={exp_rounds:.1f} mean_capped={s.mean:.1f} std_capped={s.stddev:.1f} bust={s.bust_rate:.3f} "
                f"rolls={s.avg_rolls:.2f} hot={s.avg_hot_dice:.2f} strat={r.strategy} hand={r.hand}"
            )

        return 0

    raise ValueError("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
