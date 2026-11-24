import unittest
from stats import summarize_prices


class TestHiddenStats(unittest.TestCase):
    def test_quoted_commas_and_even_median(self):
        csv_text = (
            'item,price\n'
            '"Widget, Large", 10.0\n'
            '"Gadget, Mini, Pro", 20.0\n'
            '\n'
            'Plain, 30.0\n'
            '"Cable, USB-C", 40.0\n'
        )
        res = summarize_prices(csv_text)
        self.assertEqual(res["count"], 4)
        self.assertAlmostEqual(res["mean"], 25.0)
        self.assertAlmostEqual(res["median"], 25.0)

    def test_empty_and_header_only(self):
        self.assertEqual(summarize_prices(""), {"count": 0, "mean": 0.0, "median": 0.0})
        self.assertEqual(summarize_prices("item,price\n\n"), {"count": 0, "mean": 0.0, "median": 0.0})

    def test_negative_and_unsorted(self):
        csv_text = 'item,price\n"Refund, A",-5.0\nB,15.0\nC,5.0\nD,25.0\n'
        res = summarize_prices(csv_text)
        self.assertEqual(res["count"], 4)
        self.assertAlmostEqual(res["mean"], 10.0)
        self.assertAlmostEqual(res["median"], 10.0)


if __name__ == "__main__":
    unittest.main()
