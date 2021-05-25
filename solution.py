import featureEngineering
import progress


def get_classifier_score(classifier, settings=None) -> (float, float):
    tbl = featureEngineering.get_featured_data_frame(settings)
    train_data, test_data, train_survived_data, test_survived_data = featureEngineering.split_data_frame(tbl)

    classifier.fit(train_data, train_survived_data.values.ravel())
    score_train = classifier.score(train_data, train_survived_data)
    score_test = classifier.score(test_data, test_survived_data)

    return score_train, score_test


def find_best_classifier_score(classifier, base_settings=None) -> (float, dict):
    best_test_score = 0.0
    best_settings = None
    variations_count = featureEngineering.get_settings_variations_count()
    attempt_count = 3

    progress_log = progress.Progress(variations_count * attempt_count)

    for settings_seed in range(0, variations_count):
        settings = featureEngineering.get_settings_variation(settings_seed)
        if base_settings is not None:
            settings = base_settings | settings

        tbl = featureEngineering.get_featured_data_frame(settings)

        test_score_sum = 0
        for attempt in range(0, attempt_count):
            train_data, test_data, train_survived_data, test_survived_data = featureEngineering.split_data_frame(tbl)
            classifier.fit(train_data, train_survived_data.values.ravel())

            score_test = classifier.score(test_data, test_survived_data)
            test_score_sum += score_test

            progress_log.log()

        score_test_avg = test_score_sum / attempt_count
        if best_test_score < score_test_avg:
            best_test_score = score_test_avg
            best_settings = settings

    return best_test_score, best_settings
