import sklearn.neighbors

import solution

base_settings = {"normalize": True}


def get_classifier() -> object:
    return sklearn.neighbors.NearestCentroid()


def get_knn_classifier_score(settings) -> (float, float):
    knn_classifier = get_classifier()
    score_train, score_test = solution.get_classifier_score(knn_classifier, settings | base_settings)

    return score_train, score_test


def find_best_knn_classifier_score() -> (float, dict):
    knn_classifier = get_classifier()
    best_test_score, best_settings = solution.find_best_classifier_score(knn_classifier, base_settings)

    return best_test_score, best_settings
