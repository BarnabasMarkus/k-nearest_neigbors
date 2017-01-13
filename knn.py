#!/usr/bin/env python3
# K N N   A L G O R I T H M

# Project   KNN Algorithm Implementation
# Author    Barnabas Markus
# Email     barnabasmarkus@gmail.com
# Date      13.01.2017
# Python    3.5.1
# License   MIT


import math
from random import random
from collections import defaultdict


def get_train_test_sets(data, results, train=0.75):
    """Split data and results into train and test sets"""
    x_train, x_test, y_train, y_test = [], [], [], []
    for idx, sample in enumerate(data):
        if random() < train:
            x_train.append(sample)
            y_train.append(results[idx])
        else:
            x_test.append(sample)
            y_test.append(results[idx])
    return x_train, x_test, y_train, y_test


def gauss(dist, sigma=10.0):
    """Gauss weight function"""
    return math.e ** (-dist ** 2 / (2 * sigma ** 2))


def inverse(dist):
    """Inverse weight function"""
    return 1 / (dist + 1)


def get_distance(vec1: list, vec2: list) -> float:
    """Return Euclidean distance of 2 vectors"""
    return math.sqrt(sum([pow(i - j, 2) for i, j in zip(vec1, vec2)]))


def knn(vec, vectors, k):
    """Return k-nearest neighbors of vec compared to each vector in vectors"""
    distances = [(idx, get_distance(vec, vecx))
                    for idx, vecx in enumerate(vectors)]
    return sorted(distances, key=lambda x: x[1])[:k]


def regr_predict(vec, vectors, results, k, weighted=True, weight_func=inverse):
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


def cls_predict(vec, vectors, results, k, weighted=True, weight_func=inverse):
    """Class prediction"""
    neighbors = knn(vec, vectors, k)
    predictions = defaultdict(int)
    for idx, distance in neighbors:
        if weighted:
            weight = weight_func(distance)
            predictions[results[idx]] += weight
        else:
            predictions[results[idx]] += 1
    return max(predictions)


def regr_error_rate(x_train, y_train, x_test, y_test, k):
    """Return regression prediction error rate on given data sets
    with specified k"""
    error = 0.0
    for x_test_i, y_test_i in zip(x_test, y_test):
        pred = regr_predict(x_test_i, x_train, y_train, k)
        error += abs(pred - y_test_i) / y_test_i
    error_rate = error / len(y_test)
    return error_rate


def cls_error_rate(x_train, y_train, x_test, y_test, k):
    """Return classification prediction error rate on given data sets
    with specified k"""
    error = 0.0
    for x_test_i, y_test_i in zip(x_test, y_test):
        pred = cls_predict(x_test_i, x_train, y_train, k)
        # Compare predicted and real results
        if pred != y_test_i:
            error += 1
    error_rate = error / len(y_test)
    return error_rate


def get_best_fit_model(x_train, y_train, x_test, y_test):
    """Return the best fit number of k (lower is prefered)
    for prediction on given data sets"""
    k_max = int(len(y_train) / 3)
    best_model = (None, 1.0)

    # Classification or regression?
    if isinstance(y_train[0], str) or isinstance(y_train[0], bool):
        func = cls_error_rate
    else:
        func = regr_error_rate

    # Test all value for k
    for k in range(1, k_max):
        error_rate = func(x_train, y_train, x_test, y_test, k)
        if error_rate < best_model[1]:
            best_model = (k, error_rate)

    # Return lowest best fit number of k
    return best_model[0]
