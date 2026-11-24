import time
import unittest
from solution import sum_primes_upto


class TestHiddenPrimeSum(unittest.TestCase):
    def test_edge(self):
        self.assertEqual(sum_primes_upto(0), 0)
        self.assertEqual(sum_primes_upto(2), 2)

    def test_sublinear_billion_scale(self):
        t0 = time.perf_counter()
        self.assertEqual(sum_primes_upto(10**9), 24_739_512_092_254_535)
        self.assertEqual(sum_primes_upto(2_345_678_901), 130_609_358_898_584_058)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 6.0, f"Too slow at billion scale: {elapsed:.2f}s")


if __name__ == "__main__":
    unittest.main()
