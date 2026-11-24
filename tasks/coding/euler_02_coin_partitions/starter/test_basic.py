import unittest
from solution import partition_mod


class TestBasicPartitions(unittest.TestCase):
    def test_small(self):
        self.assertEqual(partition_mod(0, 1000), 1)
        self.assertEqual(partition_mod(5, 1000), 7)
        self.assertEqual(partition_mod(10, 1000), 42)
        self.assertEqual(partition_mod(50, 10**9 + 7), 204226)


if __name__ == "__main__":
    unittest.main()
