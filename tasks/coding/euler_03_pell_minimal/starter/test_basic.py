import unittest
from solution import max_x_pell_d, solve_pell


class TestBasicPell(unittest.TestCase):
    def test_small(self):
        self.assertEqual(solve_pell(2), (3, 2))
        self.assertEqual(solve_pell(5), (9, 4))
        self.assertEqual(solve_pell(7), (8, 3))
        self.assertEqual(solve_pell(13), (649, 180))
        self.assertEqual(max_x_pell_d(7), 5)


if __name__ == "__main__":
    unittest.main()
