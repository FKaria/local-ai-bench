import unittest
from rle import encode, decode


class TestHiddenRLE(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(encode(""), "")
        self.assertEqual(decode(""), "")

    def test_single_char(self):
        self.assertEqual(encode("z"), "1z")
        self.assertEqual(decode("1z"), "z")

    def test_multidigit_counts(self):
        raw = "a" * 12 + "b" * 3 + "c" * 105
        encoded = "12a3b105c"
        self.assertEqual(encode(raw), encoded)
        self.assertEqual(decode(encoded), raw)

    def test_roundtrip_alternating(self):
        raw = "abababababab"
        self.assertEqual(decode(encode(raw)), raw)


if __name__ == "__main__":
    unittest.main()
