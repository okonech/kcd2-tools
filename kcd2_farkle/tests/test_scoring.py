import unittest

from tools.kcd2_farkle.scoring import RolledDie, best_score_for_subset


class TestScoring(unittest.TestCase):
    def test_singles(self) -> None:
        self.assertEqual(best_score_for_subset([RolledDie(1, False, "x")]).points, 100)
        self.assertEqual(best_score_for_subset([RolledDie(5, False, "x")]).points, 50)

    def test_of_a_kind_and_doubling(self) -> None:
        self.assertEqual(best_score_for_subset([RolledDie(2, False, "x")] * 3).points, 200)
        self.assertEqual(best_score_for_subset([RolledDie(2, False, "x")] * 4).points, 400)
        self.assertEqual(best_score_for_subset([RolledDie(5, False, "x")] * 6).points, 4000)

    def test_straights(self) -> None:
        self.assertEqual(
            best_score_for_subset([RolledDie(i, False, "x") for i in (1, 2, 3, 4, 5, 6)]).points,
            1500,
        )
        self.assertEqual(
            best_score_for_subset([RolledDie(i, False, "x") for i in (1, 2, 3, 4, 5)]).points,
            500,
        )
        self.assertEqual(
            best_score_for_subset([RolledDie(i, False, "x") for i in (2, 3, 4, 5, 6)]).points,
            750,
        )

    def test_wildcards_do_not_score_alone_or_as_single_ones(self) -> None:
        wild = RolledDie(1, True, "Devil's head die")
        self.assertEqual(best_score_for_subset([wild]).points, 0)
        self.assertEqual(best_score_for_subset([wild, wild]).points, 0)
        self.assertEqual(best_score_for_subset([wild, wild, wild]).points, 0)

    def test_wildcards_complete_of_a_kind(self) -> None:
        wild = RolledDie(1, True, "Devil's head die")
        self.assertEqual(
            best_score_for_subset([RolledDie(2, False, "x"), RolledDie(2, False, "x"), wild]).points,
            200,
        )
        self.assertEqual(
            best_score_for_subset([RolledDie(2, False, "x"), RolledDie(2, False, "x"), RolledDie(2, False, "x"), wild]).points,
            400,
        )

    def test_wildcards_complete_straights(self) -> None:
        wild = RolledDie(1, True, "Devil's head die")
        self.assertEqual(
            best_score_for_subset(
                [
                    RolledDie(1, False, "x"),
                    RolledDie(2, False, "x"),
                    RolledDie(3, False, "x"),
                    RolledDie(4, False, "x"),
                    RolledDie(5, False, "x"),
                    wild,
                ]
            ).points,
            1500,
        )
        self.assertEqual(
            best_score_for_subset(
                [
                    RolledDie(2, False, "x"),
                    RolledDie(3, False, "x"),
                    RolledDie(4, False, "x"),
                    RolledDie(5, False, "x"),
                    RolledDie(6, False, "x"),
                    wild,
                ]
            ).points,
            1500,
        )


if __name__ == "__main__":
    unittest.main()
