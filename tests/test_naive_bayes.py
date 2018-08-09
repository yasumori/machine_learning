#!/usr/bin/env python3

from collections import Counter, defaultdict
from math import exp
import sys
import unittest
sys.path.append('code')

from naive_bayes import NaiveBayes

class TestNaiveBayes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._naive_bayes = NaiveBayes()

    def test_get_likelihood(self):
        word_counter = defaultdict(Counter)
        word_counter[1] = Counter(["foo", "foo", "bar"])
        word_totals = Counter(["foo", "foo", "foo", "foo", "bar"])

        p_x_given_c = \
            self._naive_bayes._get_likelihood(word_counter, word_totals)

        exp_p_foo = 0.5
        exp_p_bar = 1.0
        self.assertEqual(exp(p_x_given_c[1]["foo"]), exp_p_foo)
        self.assertEqual(exp(p_x_given_c[1]["bar"]), exp_p_bar)

    def test_get_prior(self):
        label_counter = Counter([1, 2, 2, 2, 2])

        p_c = self._naive_bayes._get_prior(label_counter)

        exp_p_1 = 0.2
        exp_p_2 = 0.8
        self.assertEqual(exp(p_c[1]), exp_p_1)
        self.assertEqual(exp(p_c[2]), exp_p_2)

if __name__ == "__main__":
    unittest.main()
