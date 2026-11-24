import unittest
from stats import summarize_prices


class TestBasicStats(unittest.TestCase):
    def test_odd_simple(self):
        csv_text = "item,price\napple,1.5\nbanana,2.5\ncherry,5.0\n"
        res = summarize_prices(csv_text)
        self.assertEqual(res["count"], 3)
        self.assertAlmostEqual(res["mean"], 3.0)
        self.assertAlmostEqual(res["median"], 2.5)

    def test_quoted_comma_and_even_median(self):
        csv_text = 'item,price\n"apple, red",1.0\nbanana,3.0\n'
        res = summarize_prices(csv_text)
        self.assertEqual(res["count"], 2)
        self.assertAlmostEqual(res["mean"], 2.0)
        self.assertAlmostEqual(res["median"], 2.0)

    def test_empty(self):
        self.assertEqual(summarize_prices(""), {"count": 0, "mean": 0.0, "median": 0.0})


if __name__ == "__main__":
    unittest.main()
