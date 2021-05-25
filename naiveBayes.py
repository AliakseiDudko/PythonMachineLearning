import sklearn.naive_bayes

import solution


def get_classifier() -> object:
    return sklearn.naive_bayes.GaussianNB()


def get_bayes_classifier_score(settings) -> (float, float):
    bayes_classifier = get_classifier()
    score_train, score_test = solution.get_classifier_score(bayes_classifier, settings)

    return score_train, score_test


def find_best_bayes_classifier_score() -> (float, dict):
    bayes_classifier = get_classifier()
    best_test_score, best_settings = solution.find_best_classifier_score(bayes_classifier)

    return best_test_score, best_settings
