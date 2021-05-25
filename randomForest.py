import sklearn.ensemble

import solution


def get_classifier() -> object:
    return sklearn.ensemble.RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1)


def get_random_forest_classifier_score(settings) -> (float, float):
    random_forest_classifier = get_classifier()
    score_train, score_test = solution.get_classifier_score(random_forest_classifier, settings)

    return score_train, score_test


def find_best_random_forest_classifier_score() -> (float, dict):
    random_forest_classifier = get_classifier()
    best_test_score, best_settings = solution.find_best_classifier_score(random_forest_classifier)

    return best_test_score, best_settings
