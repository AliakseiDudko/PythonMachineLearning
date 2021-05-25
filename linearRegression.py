import sklearn.linear_model

import solution


def get_classifier() -> object:
    return sklearn.linear_model.LinearRegression(n_jobs=-1)


def get_linear_classifier_score(settings) -> (float, float):
    linear_classifier = get_classifier()
    score_train, score_test = solution.get_classifier_score(linear_classifier, settings)

    return score_train, score_test


def find_best_linear_classifier_score() -> (float, dict):
    linear_classifier = get_classifier()
    best_test_score, best_settings = solution.find_best_classifier_score(linear_classifier)

    return best_test_score, best_settings
