import time
import unittest
from solution import partition_mod


class TestHiddenPartitions(unittest.TestCase):
    def test_medium(self):
        self.assertEqual(partition_mod(1000, 1_000_000), 727991)

    def test_large_performance(self):
        t0 = time.perf_counter()
        ans = partition_mod(50_000, 10**9 + 7)
        elapsed = time.perf_counter() - t0
        self.assertEqual(ans, 114_250_830)
        self.assertLess(elapsed, 6.0, f"Too slow for n=50_000: {elapsed:.2f}s")


if __name__ == "__main__":
    unittest.main()
