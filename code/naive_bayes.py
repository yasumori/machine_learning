#!/usr/bin/env python3

from collections import Counter, defaultdict
from numpy import argmax, log

class NaiveBayes:
    def __init__(self):
        self.p_c = None
        self.p_x_given_c = None
        self.labels = None

    def train(self, data):
        """function that trains a naive bayes model
           data: (counts, labels)
               - counts: a list of Counter objects
               - labels: a list of integers"""
        counts, labels = data
        # obtain frequencies
        label_counter = Counter(labels)
        word_counter = defaultdict(Counter)
        word_totals = Counter()
        for counts, label in zip(counts, labels):
            word_counter[label] += counts
            word_totals += counts

        # update probabilities
        self.p_c = self._get_prior(label_counter)
        self.p_x_given_c = self._get_likelihood(word_counter, word_totals)
        self.labels = sorted(self.p_c.keys())

    @staticmethod
    def _get_likelihood(word_counter, word_totals):
        """compute p(x|c), probability of a data point given a class"""
        p_x_given_c = defaultdict(lambda: defaultdict(float))
        for label in word_counter:
            for word in word_counter[label]:
                p_x_given_c[label][word] = \
                    log(word_counter[label][word] / word_totals[word])
        return p_x_given_c

    @staticmethod
    def _get_prior(label_counter):
        """compute p(c), probability of a class"""
        p_c = defaultdict(float)
        total = sum(label_counter.values())
        for label in label_counter:
            p_c[label] = log(label_counter[label] / total)
        return p_c

    def predict(self, data):
        """compute p(c|x), probability of a class given a data point
           for each data point, and return predictions for each data point
           data: (counts, labels)
               - counts: a list of Counter objects
               - labels: a list of integers"""
        counts, labels = data
        predictions = list()
        for count in counts:
            probs = list()
            for label in self.labels:
                p_c_given_x = self.p_c[label]
                for word in count:
                    p_c_given_x += \
                        (self.p_x_given_c[label][word] * count[word])
                probs.append(p_c_given_x)
            prediction = argmax(probs)
            predictions.append(prediction)
        return predictions
