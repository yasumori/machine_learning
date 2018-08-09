#!/usr/bin/env python3

from collections import Counter
from math import exp
import sys
sys.path.append('code')

from naive_bayes import NaiveBayes

# 1 seal related document, 0 non-seal related document
train_data_raw = [("seals swim in the sea", 1),
              ("I saw seals swim in the pool", 1),
              ("dogs don't like a pool", 0),
              ("I like to swim in the pool", 0),
              ("seals like swim in the sea", 1),
              ("seals like dogs", 1),
              ("I like dogs", 0)]

test_data_raw = [("dogs like the pool", 0),
                 ("seals like the sea", 1)]

def create_data(data):
    counts = list()
    labels = list()
    for text, label in data:
        counts.append(Counter(text.split()))
        labels.append(label)
    return (counts, labels)

if __name__ == "__main__":
    train_data = create_data(train_data_raw)
    print("word_counts: {0}\nlabels: {1}\n"
            .format(train_data[0], train_data[1]))
    NB = NaiveBayes()
    NB.train(train_data)
    print("p_label0: {0}\np_label1: {1}\n"
            .format(exp(NB.p_c[0]), exp(NB.p_c[1])))
    label0_p_x_given_c = [(word, exp(prob)) for word, prob in
            NB.p_x_given_c[0].items()]
    label1_p_x_given_c = [(word, exp(prob)) for word, prob in
            NB.p_x_given_c[1].items()]
    print("p_x_given_c_label0: {0}\np_x_given_c_label1: {1}\n"
        .format(label0_p_x_given_c, label1_p_x_given_c))
    test_data = create_data(test_data_raw)
    predictions = NB.predict(test_data)
    data, labels = test_data
    NB.report(predictions, labels)
