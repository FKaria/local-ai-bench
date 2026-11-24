import unittest
from solution import sum_primes_upto


class TestBasicPrimeSum(unittest.TestCase):
    def test_small(self):
        self.assertEqual(sum_primes_upto(1), 0)
        self.assertEqual(sum_primes_upto(10), 17)
        self.assertEqual(sum_primes_upto(100), 1060)
        self.assertEqual(sum_primes_upto(10_000), 5_736_396)


if __name__ == "__main__":
    unittest.main()
