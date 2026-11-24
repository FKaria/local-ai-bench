import time
import unittest
from solution import max_x_pell_d, solve_pell


class TestHiddenPell(unittest.TestCase):
    def test_invalid_square(self):
        with self.assertRaises(ValueError):
            solve_pell(9)

    def test_hard_d_values(self):
        self.assertEqual(solve_pell(61), (1_766_319_049, 226_153_980))
        self.assertEqual(solve_pell(109), (158_070_671_986_249, 15_140_424_455_100))

    def test_limits_and_speed(self):
        t0 = time.perf_counter()
        self.assertEqual(max_x_pell_d(1000), 661)
        self.assertEqual(max_x_pell_d(1500), 1381)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 3.0, f"Too slow for limit=1500: {elapsed:.2f}s")


if __name__ == "__main__":
    unittest.main()
