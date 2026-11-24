import time
import unittest
from solution import count_reduced_fractions


class TestHiddenTotientSum(unittest.TestCase):
    def test_edge_cases(self):
        self.assertEqual(count_reduced_fractions(0), 0)
        self.assertEqual(count_reduced_fractions(2), 1)

    def test_medium_and_large_performance(self):
        self.assertEqual(count_reduced_fractions(50_000), 759_924_263)
        t0 = time.perf_counter()
        ans = count_reduced_fractions(1_000_000)
        elapsed = time.perf_counter() - t0
        self.assertEqual(ans, 303_963_552_391)
        self.assertLess(elapsed, 5.0, f"Too slow for max_d=1_000_000: {elapsed:.2f}s")


if __name__ == "__main__":
    unittest.main()
