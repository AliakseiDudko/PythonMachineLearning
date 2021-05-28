import featureEngineering
import progress


def get_classifier_score(classifier, settings={}) -> (float, float):
    tbl = featureEngineering.get_featured_data_frame(settings)
    train_data, test_data, train_survived_data, test_survived_data = featureEngineering.split_data_frame(tbl)

    classifier.fit(train_data, train_survived_data.values.ravel())
    score_train = classifier.score(train_data, train_survived_data)
    score_test = classifier.score(test_data, test_survived_data)

    return score_train, score_test


def find_best_classifier_score(classifier, base_settings={}) -> (float, dict):
    variations_count = featureEngineering.get_settings_variations_count()
    attempt_count = 10
    progress_log = progress.Progress(variations_count * attempt_count)

    results = list()
    for settings_seed in range(0, variations_count):
        settings = base_settings | featureEngineering.get_settings_variation(settings_seed)
        avg_test_score = get_avg_test_score(classifier, settings, attempt_count, progress_log)
        results.append((avg_test_score, settings))
    best_settings = get_best_settings(results)

    return results[0][0], best_settings


def get_avg_test_score(classifier, settings, attempt_count, progress_log) -> float:
    tbl = featureEngineering.get_featured_data_frame(settings)
    sum_test_score = 0
    for attempt in range(0, attempt_count):
        train_data, test_data, train_survived_data, test_survived_data = featureEngineering.split_data_frame(tbl)
        classifier.fit(train_data, train_survived_data.values.ravel())

        score_test = classifier.score(test_data, test_survived_data)
        sum_test_score += score_test

        progress_log.log()

    avg_test_score = sum_test_score / attempt_count

    return avg_test_score


def get_best_settings(results) -> object:
    results.sort(key=lambda x: x[0], reverse=True)
    low_test_score = results[0][0] * 0.95
    results = list(filter(lambda x: x[0] >= low_test_score, results))

    settings = featureEngineering.get_settings_variation(0)
    for setting_key in settings:
        true_count = 0
        false_count = 0
        for result in results:
            if result[1][setting_key]:
                true_count += 1
            else:
                false_count += 1

        settings[setting_key] = true_count >= false_count

    return settings
