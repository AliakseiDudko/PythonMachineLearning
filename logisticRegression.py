import sklearn.linear_model

import featureEngineering
import progress


def get_data_frame_logistic_classifier_score(tbl) -> (float, float):
    train_data, test_data, train_survived_data, test_survived_data = featureEngineering.split_data_frame(tbl)

    logistic_classifier = sklearn.linear_model.LogisticRegression(penalty="l2",
                                                                  solver="newton-cg",
                                                                  class_weight="balanced")
    logistic_classifier.fit(train_data, train_survived_data.values.ravel())
    score_train = logistic_classifier.score(train_data, train_survived_data)
    score_test = logistic_classifier.score(test_data, test_survived_data)

    return score_train, score_test


def get_logistic_classifier_score(settings) -> (float, float):
    # Prepare DataFrame
    tbl = featureEngineering.get_featured_data_frame("data.csv", settings)

    return get_data_frame_logistic_classifier_score(tbl)


def find_best_logistic_classifier_score() -> (float, dict):
    best_test_score = 0.0
    best_settings = None
    variations_count = featureEngineering.get_settings_variations_count()
    attempt_count = 5

    progress_log = progress.Progress(variations_count * attempt_count)

    for settings_seed in range(0, variations_count):
        # Get variation of features
        settings = featureEngineering.get_settings_variation(settings_seed)

        # Prepare DataFrame
        tbl = featureEngineering.get_featured_data_frame("data.csv", settings)

        test_score_sum = 0
        for attempt in range(0, attempt_count):
            # Get results of classification
            score_test, score_test = get_data_frame_logistic_classifier_score(tbl)
            test_score_sum += score_test

            progress_log.log()

        score_test_avg = test_score_sum / attempt_count
        if best_test_score < score_test_avg:
            best_test_score = score_test_avg
            best_settings = settings

    return best_test_score, best_settings
