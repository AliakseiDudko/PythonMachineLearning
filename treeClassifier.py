import sklearn.tree
import featureEngineering


def get_data_frame_tree_classifier_score(max_depth, tbl) -> (float, float):
    train_data, test_data, train_survived_data, test_survived_data = featureEngineering.split_data_frame(tbl)

    tree_classifier = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth)
    tree_classifier.fit(train_data, train_survived_data.values.ravel())
    score_train = tree_classifier.score(train_data, train_survived_data)
    score_test = tree_classifier.score(test_data, test_survived_data)

    return score_train, score_test


def get_tree_classifier_score(settings) -> (float, float):
    # Prepare DataFrame
    tbl = featureEngineering.get_featured_data_frame("data.csv", settings)

    return get_data_frame_tree_classifier_score(settings["max_depth"], tbl)


def find_best_tree_classifier_score(depth_min, depth_max) -> (float, dict):
    best_test_score = 0.0
    best_settings = None
    attempt_count = 100

    for depth in range(depth_min, depth_max + 1):
        print(f"Depth:{depth}")
        for settings_seed in range(0, featureEngineering.get_settings_variations_count()):
            # Get variation of features
            settings = featureEngineering.get_settings_variation(settings_seed)
            settings["max_depth"] = depth

            # Prepare DataFrame
            tbl = featureEngineering.get_featured_data_frame("data.csv", settings)

            train_score_sum = 0
            for attempt in range(0, attempt_count):
                # Get results of classification
                score_train, score_test = get_data_frame_tree_classifier_score(depth, tbl)
                train_score_sum += score_train

            score_train_avg = train_score_sum / attempt_count
            if best_test_score < score_train_avg:
                best_test_score = score_train_avg
                best_settings = settings
                print(f"Decision tree best configuration: {best_test_score}, settings: {best_settings}")

    return best_test_score, best_settings
