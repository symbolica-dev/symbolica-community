"""Unit checks for acceptance arithmetic; these are not Pyodide physics evidence."""
import unittest

from gg_hg_pyodide_compare import compare, number


def encoded(numerator, denominator=1, bits=200):
    return [str(numerator), str(denominator), bits]


class ExactComparisonTests(unittest.TestCase):
    def test_encoding_preserves_sub_binary64_difference(self):
        self.assertNotEqual(number(encoded(2**100 + 1, 2**100)), number(encoded(1)))
        self.assertEqual(number(encoded(2**100 + 1, 2**100)), number([f'{2**100 + 1}/{2**100}', 101]))

    def test_reject_non_dyadic_or_insufficient_precision(self):
        for record in (encoded(1, 3), encoded(2**100 + 1, 2**100, 96)):
            with self.subTest(record=record), self.assertRaises(AssertionError):
                number(record)

    def test_astro_requested_precision_kept_separate_from_word_storage(self):
        value = encoded(2**223 + 1, 2**223, 216)
        self.assertEqual(number(value), number([f'{2**223 + 1}/{2**223}', 224]))
        with self.assertRaises(AssertionError):
            number(encoded(2**224 + 1, 2**224, 216))
        with self.assertRaises(AssertionError):
            number([f'{2**223 + 1}/{2**223}', 216])

    def test_combined_errors_accept_actual_small_complex_difference(self):
        result = compare([encoded(2**100 + 1, 2**100), encoded(1, 2**100)], encoded(1, 2**100),
                         [encoded(1), encoded(0)], encoded(1, 2**100), relative=True)
        self.assertNotEqual(result['difference_squared'], '0')

    def test_reject_difference_outside_combined_errors(self):
        with self.assertRaisesRegex(AssertionError, 'combined admitted errors'):
            compare([encoded(2**100 + 1, 2**100), encoded(0)], encoded(0),
                    [encoded(1), encoded(0)], encoded(0))

    def test_reject_uncertainty_even_when_values_equal(self):
        with self.assertRaisesRegex(AssertionError, 'individual requested20 error'):
            compare([encoded(1), encoded(0)], encoded(1, 2**20),
                    [encoded(1), encoded(0)], encoded(1, 2**20))

    def test_relative_zero_rejected_but_mixed_zero_allowed(self):
        zero = [encoded(0), encoded(0)]
        compare(zero, encoded(0), zero, encoded(0))
        with self.assertRaises(AssertionError):
            compare(zero, encoded(0), zero, encoded(0), relative=True)


if __name__ == '__main__':
    unittest.main()
