#!/usr/bin/env python3


import math
from random import randint, random
from collections import defaultdict

vec = [randint(0, 10) for _ in range(5)]
vectors = [[randint(0, 10) for _ in range(5)] for _ in range(15)]
results = [randint(1000, 1500) for _ in range(15)]


def get_distance(vec1: list, vec2: list) —> float:
    """Return Euclidean distance of 2 vectors"""
    return math.sqrt(sum([pow(i - j, 2) for i, j in zip(vec1, vec2)]))


def knn(vec, vectors, k=1):
    """Return k-nearest neighbors of vec compared to each vector in vectors"""
    distances = [(idx, get_distance(vec, vecx))
                    for idx, vecx in enumerate(vectors)]
    return sorted(distances, key=lambda x: x[1])[:k]


def regr_predict(vec, vectors, results, k, weighted=True, weight_func=gauss):
    """Regression prediction"""
    neighbors = knn(vec, vectors, k)
    weights, total = 0, 0
    for idx, distance in neighbors:
        if weighted:
            weight = weight_func(distance)
            total += results[idx] * weight
            weights += weight
        else:
            total += results[idx]
            weights += 1
    # return avg
    return total / weights

def cls_predict(vec, vectors, results, k, weighted=True, weight_func=gauss):
    """Class prediction"""
    neighbors = knn(vec, vectors, k)
    predictions = defaultdict(int)
    for idx, distance in neighbors:
        if weighted:
            weight = weight_func(distance)
            predictions[results[idx]] += weight
        else:
            predictions[results[idx]] += 1
    return predictions


# TODO: test_cls_predict()
# TODO: test_regr_predict()
# TODO: select best fit models (regression and classification)


def get_train_test_sets(data, results, train=0.75):
    x_train, x_test, y_train, y_test = [], [], [], []
    for idx, sample in enumerate(data):
        if random() < train:
            x_train.append(sample)
            y_train.append(results[idx])
        else:
            x_test.append(sample)
            y_test.append(results[idx])
    return x_train, x_test, y_train, y_test


# Weight functions
def gauss(dist, sigma=10.0):
    return math.e ** (-dist ** 2 / (2 * sigma ** 2))


def inverse(dist):
    return 1 / (dist + 1)


