import unittest
from rle import encode, decode


class TestBasicRLE(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(encode("aaabbc"), "3a2b1c")
        self.assertEqual(decode("3a2b1c"), "aaabbc")

    def test_empty(self):
        self.assertEqual(encode(""), "")
        self.assertEqual(decode(""), "")

    def test_multidigit(self):
        self.assertEqual(encode("a" * 12 + "b" * 3), "12a3b")
        self.assertEqual(decode("12a3b"), "a" * 12 + "b" * 3)


if __name__ == "__main__":
    unittest.main()
