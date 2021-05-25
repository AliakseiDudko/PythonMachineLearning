import sklearn.svm

import solution

base_settings = {"Normalize": True}


def get_classifier() -> object:
    return sklearn.svm.SVC(kernel="poly")


def get_svm_classifier_score(settings) -> (float, float):
    svm_classifier = get_classifier()
    score_train, score_test = solution.get_classifier_score(svm_classifier, settings | base_settings)

    return score_train, score_test


def find_best_svm_classifier_score() -> (float, dict):
    svm_classifier = get_classifier()
    best_test_score, best_settings = solution.find_best_classifier_score(svm_classifier, base_settings)

    return best_test_score, best_settings
