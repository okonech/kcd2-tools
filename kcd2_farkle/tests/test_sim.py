import random
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from tools.kcd2_farkle.dice_specs import DieSpec
from tools.kcd2_farkle.sim import (
    DieInstance,
    GreedyThresholdStrategy,
    HotDiceFirstStrategy,
    RolloutStrategy,
    parse_hand,
    hand_to_arg,
    main,
    run_evolutionary_search,
    simulate_turn,
)


def _deterministic_die(name: str, face: int) -> DieSpec:
    probs = [0.0] * 6
    probs[face - 1] = 1.0
    return DieSpec(name=name, probabilities=tuple(probs), max_quantity=None, wildcard_faces=frozenset())


class TestSim(unittest.TestCase):
    def test_parse_hand_requires_six(self) -> None:
        dice_specs = [_deterministic_die("A", 1), _deterministic_die("B", 5)]
        with self.assertRaises(ValueError):
            parse_hand(dice_specs, "A,B")

    def test_simulate_turn_caps_and_stops(self) -> None:
        # Always roll 1s -> always can score.
        spec = _deterministic_die("One", 1)
        hand = [DieInstance.from_spec(spec) for _ in range(6)]

        rng = random.Random(0)
        strat = GreedyThresholdStrategy(bank_threshold=10**9)  # never bank early
        pts, metrics = simulate_turn(rng, hand, strat, cap=1000)

        self.assertEqual(pts, 1000)
        # With all ones, the best subset is 6-of-a-kind (high value), so the cap is hit immediately.
        self.assertEqual(metrics.rolls, 1)
        self.assertFalse(metrics.busted)

    def test_evolve_returns_top_k_when_final_trials_zero(self) -> None:
        # Regression test for: final_trials<=0 returning only 1 result.
        dice_specs = [
            _deterministic_die("One", 1),
            _deterministic_die("Five", 5),
            _deterministic_die("Two", 2),
            _deterministic_die("Three", 3),
            _deterministic_die("Four", 4),
            _deterministic_die("Six", 6),
        ]

        rng = random.Random(1)
        top = run_evolutionary_search(
            rng=rng,
            dice_specs=dice_specs,
            pool=None,
            strategies=["greedy_bank_500", "hotdice_bank_500"],
            hands=12,
            trials_per_hand=10,
            final_trials=0,
            final_candidates=10,
            elite_frac=0.25,
            generations=3,
            goal=4000,
            cap=4000,
            metric="p_goal",
            progress_every=0,
            progress_seconds=0,
            top_k=3,
            initial_population=None,
        )

        self.assertEqual(len(top), 3)
        # Ensure sorted by fitness descending.
        self.assertGreaterEqual(top[0].stats.p_ge_goal, top[-1].stats.p_ge_goal)

    def test_hotdice_strategy_prefers_using_all_dice(self) -> None:
        # Sanity check strategy name and non-throwing behavior.
        strat = HotDiceFirstStrategy(bank_threshold=500)
        self.assertIn("hotdice_bank_", strat.name)

    def test_hand_identity_is_order_independent(self) -> None:
        dice_specs = [
            _deterministic_die("One", 1),
            _deterministic_die("Five", 5),
            _deterministic_die("Two", 2),
            _deterministic_die("Three", 3),
            _deterministic_die("Four", 4),
            _deterministic_die("Six", 6),
        ]

        h1 = parse_hand(dice_specs, "One,Two,Three,Four,Five,Six")
        h2 = parse_hand(dice_specs, "Six,Five,Four,Three,Two,One")
        self.assertEqual(hand_to_arg(h1), hand_to_arg(h2))

    def test_rollout_strategy_smoke(self) -> None:
        spec = _deterministic_die("One", 1)
        hand = [DieInstance.from_spec(spec) for _ in range(6)]

        rng = random.Random(0)
        strat = RolloutStrategy(rollouts_per_decision=2, metric="p_goal")
        pts, _metrics = simulate_turn(rng, hand, strat, cap=1000, goal=1000)
        self.assertLessEqual(pts, 1000)

    def test_search_default_seeds_from_combos_json(self) -> None:
        # No --seed-combos provided => should seed from all combos by default.
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            dice_json = td_path / "dice.json"
            dice_list = []
            # Provide 6 deterministic dice A..F.
            for name, face in [("A", 1), ("B", 2), ("C", 3), ("D", 4), ("E", 5), ("F", 6)]:
                probs = {str(i): (1.0 if i == face else 0.0) for i in range(1, 7)}
                dice_list.append(
                    {
                        "name": name,
                        "probabilities": probs,
                        "maxQuantity": None,
                        "wildcardFaces": [],
                    }
                )
            dice_json.write_text(json.dumps({"dice": dice_list}), encoding="utf-8")

            combos_json = td_path / "combos.json"
            combos_json.write_text(
                json.dumps({"combos": [{"name": "combo1", "hand": ["A", "B", "C", "D", "E", "F"]}]}),
                encoding="utf-8",
            )

            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                rc = main(
                    [
                        "--dice-json",
                        str(dice_json),
                        "search",
                        "--algo",
                        "random",
                        "--hands",
                        "1",
                        "--trials",
                        "5",
                        "--top",
                        "1",
                        "--goal",
                        "4000",
                        "--cap",
                        "4000",
                        "--metric",
                        "p_goal",
                        "--combos-json",
                        str(combos_json),
                        "--progress-seconds",
                        "0",
                        "--progress-every",
                        "0",
                        "--strategies",
                        "greedy_bank_500",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertIn("hand=A,B,C,D,E,F", out.getvalue())

    def test_search_no_seed_combos_disables_seeding(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            dice_json = td_path / "dice.json"
            dice_list = []
            # Six dice A..F and one extra die X.
            for name, face in [("A", 1), ("B", 2), ("C", 3), ("D", 4), ("E", 5), ("F", 6), ("X", 1)]:
                probs = {str(i): (1.0 if i == face else 0.0) for i in range(1, 7)}
                dice_list.append(
                    {
                        "name": name,
                        "probabilities": probs,
                        "maxQuantity": None,
                        "wildcardFaces": [],
                    }
                )
            dice_json.write_text(json.dumps({"dice": dice_list}), encoding="utf-8")

            combos_json = td_path / "combos.json"
            combos_json.write_text(
                json.dumps({"combos": [{"name": "combo1", "hand": ["A", "B", "C", "D", "E", "F"]}]}),
                encoding="utf-8",
            )

            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                rc = main(
                    [
                        "--dice-json",
                        str(dice_json),
                        "search",
                        "--algo",
                        "random",
                        "--hands",
                        "1",
                        "--trials",
                        "3",
                        "--top",
                        "1",
                        "--goal",
                        "4000",
                        "--cap",
                        "4000",
                        "--metric",
                        "p_goal",
                        "--combos-json",
                        str(combos_json),
                        "--no-seed-combos",
                        "--pool",
                        "X",
                        "--seed",
                        "1",
                        "--progress-seconds",
                        "0",
                        "--progress-every",
                        "0",
                        "--strategies",
                        "greedy_bank_500",
                    ]
                )
            self.assertEqual(rc, 0)
            # With pool restricted to X, the only possible hand is X repeated.
            self.assertIn("hand=X,X,X,X,X,X", out.getvalue())

    def test_search_seed_combos_subset(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            dice_json = td_path / "dice.json"
            dice_list = []
            for name, face in [("A", 1), ("B", 2), ("C", 3), ("D", 4), ("E", 5), ("F", 6), ("X", 1)]:
                probs = {str(i): (1.0 if i == face else 0.0) for i in range(1, 7)}
                dice_list.append(
                    {
                        "name": name,
                        "probabilities": probs,
                        "maxQuantity": None,
                        "wildcardFaces": [],
                    }
                )
            dice_json.write_text(json.dumps({"dice": dice_list}), encoding="utf-8")

            combos_json = td_path / "combos.json"
            combos_json.write_text(
                json.dumps(
                    {
                        "combos": [
                            {"name": "combo1", "hand": ["A", "B", "C", "D", "E", "F"]},
                            {"name": "combo2", "hand": ["X", "X", "X", "X", "X", "X"]},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                rc = main(
                    [
                        "--dice-json",
                        str(dice_json),
                        "search",
                        "--algo",
                        "random",
                        "--hands",
                        "1",
                        "--trials",
                        "3",
                        "--top",
                        "1",
                        "--goal",
                        "4000",
                        "--cap",
                        "4000",
                        "--metric",
                        "p_goal",
                        "--combos-json",
                        str(combos_json),
                        "--seed-combos",
                        "combo1",
                        "--progress-seconds",
                        "0",
                        "--progress-every",
                        "0",
                        "--strategies",
                        "greedy_bank_500",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertIn("hand=A,B,C,D,E,F", out.getvalue())


if __name__ == "__main__":
    unittest.main()
