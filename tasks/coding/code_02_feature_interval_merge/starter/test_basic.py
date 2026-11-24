import unittest
from intervals import merge_intervals, total_covered_length


class TestBasicIntervals(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(merge_intervals([(1, 4), (2, 6)]), [(1, 6)])
        self.assertEqual(total_covered_length([(1, 4), (2, 6)]), 5)

    def test_empty_and_touching(self):
        self.assertEqual(merge_intervals([]), [])
        self.assertEqual(merge_intervals([(1, 3), (3, 5)]), [(1, 5)])
        self.assertEqual(total_covered_length([(1, 3), (3, 5)]), 4)

    def test_unsorted(self):
        self.assertEqual(merge_intervals([(5, 7), (1, 2)]), [(1, 2), (5, 7)])


if __name__ == "__main__":
    unittest.main()
