import sklearn.linear_model

import solution


def get_classifier() -> object:
    return sklearn.linear_model.LogisticRegression(penalty="l2",
                                                   solver="newton-cg",
                                                   class_weight="balanced")


def get_logistic_classifier_score(settings) -> (float, float):
    logistic_classifier = get_classifier()
    score_train, score_test = solution.get_classifier_score(logistic_classifier, settings)

    return score_train, score_test


def find_best_logistic_classifier_score() -> (float, dict):
    logistic_classifier = get_classifier()
    best_test_score, best_settings = solution.find_best_classifier_score(logistic_classifier)

    return best_test_score, best_settings
