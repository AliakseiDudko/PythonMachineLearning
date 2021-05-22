import sklearn.naive_bayes

import featureEngineering


def get_data_frame_bayes_classifier_score(tbl) -> (float, float):
    train_data, test_data, train_survived_data, test_survived_data = featureEngineering.split_data_frame(tbl)

    bayes_classifier = sklearn.naive_bayes.GaussianNB()
    bayes_classifier.fit(train_data, train_survived_data.values.ravel())
    score_train = bayes_classifier.score(train_data, train_survived_data)
    score_test = bayes_classifier.score(test_data, test_survived_data)

    return score_train, score_test


def get_bayes_classifier_score(settings) -> (float, float):
    # Prepare DataFrame
    tbl = featureEngineering.get_featured_data_frame("data.csv", settings)

    return get_data_frame_bayes_classifier_score(tbl)


def find_best_bayes_classifier_score() -> (float, dict):
    best_test_score = 0.0
    best_settings = None
    attempt_count = 5

    for settings_seed in range(0, featureEngineering.get_settings_variations_count()):
        # Get variation of features
        settings = featureEngineering.get_settings_variation(settings_seed)

        # Prepare DataFrame
        tbl = featureEngineering.get_featured_data_frame("data.csv", settings)

        test_score_sum = 0
        for attempt in range(0, attempt_count):
            # Get results of classification
            score_train, score_test = get_data_frame_bayes_classifier_score(tbl)
            test_score_sum += score_test

        score_test_avg = test_score_sum / attempt_count
        if best_test_score < score_test_avg:
            best_test_score = score_test_avg
            best_settings = settings
            print(f"Naive Bayes best configuration: {best_test_score}, settings: {best_settings}")

    return best_test_score, best_settings
