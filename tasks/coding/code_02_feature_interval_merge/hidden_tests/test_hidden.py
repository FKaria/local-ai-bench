import unittest
from intervals import merge_intervals, total_covered_length


class TestHiddenIntervals(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(merge_intervals([]), [])
        self.assertEqual(total_covered_length([]), 0)

    def test_unsorted_and_touching(self):
        data = [(8, 10), (1, 3), (3, 5), (15, 18), (2, 4)]
        self.assertEqual(merge_intervals(data), [(1, 5), (8, 10), (15, 18)])
        self.assertEqual(total_covered_length(data), 9)

    def test_nested_and_duplicates(self):
        data = [(1, 10), (2, 3), (4, 9), (1, 10)]
        self.assertEqual(merge_intervals(data), [(1, 10)])
        self.assertEqual(total_covered_length(data), 9)

    def test_negative_and_zero_length(self):
        data = [(-10, -5), (-6, -2), (0, 0), (5, 7)]
        self.assertEqual(merge_intervals(data), [(-10, -2), (0, 0), (5, 7)])
        self.assertEqual(total_covered_length(data), 10)


if __name__ == "__main__":
    unittest.main()
