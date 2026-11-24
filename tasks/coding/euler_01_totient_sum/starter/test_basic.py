import unittest
from solution import count_reduced_fractions


class TestBasicTotientSum(unittest.TestCase):
    def test_small(self):
        self.assertEqual(count_reduced_fractions(1), 0)
        self.assertEqual(count_reduced_fractions(8), 21)
        self.assertEqual(count_reduced_fractions(100), 3043)


if __name__ == "__main__":
    unittest.main()
