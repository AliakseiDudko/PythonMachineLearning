import sklearn.tree
import featureEngineering


def get_data_frame_tree_classifier_score(max_depth, tbl) -> (float, float):
    train_data, test_data, train_survived_data, test_survived_data = featureEngineering.split_data_frame(tbl)

    tree_classifier = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth)
    tree_classifier.fit(train_data, train_survived_data.values.ravel())
    score_train = tree_classifier.score(train_data, train_survived_data)
    score_test = tree_classifier.score(test_data, test_survived_data)

    return score_train, score_test


def get_tree_classifier_score(max_depth,
                              fill_embarked=False,
                              fill_age=False,
                              fill_fare=False,
                              title_feature=False,
                              ticket_group_feature=False,
                              family_size_feature=False,
                              deck_feature=False) -> (float, float):
    # Prepare DataFrame
    tbl = featureEngineering.get_featured_data_frame("data.csv",
                                                     False,
                                                     fill_embarked,
                                                     fill_age,
                                                     fill_fare,
                                                     title_feature,
                                                     ticket_group_feature,
                                                     family_size_feature,
                                                     deck_feature)

    return get_data_frame_tree_classifier_score(max_depth, tbl)


def find_best_tree_classifier_score(depth_min, depth_max) -> float:
    best_train_score = 0.0
    attempt_count = 100

    for depth in range(depth_min, depth_max + 1):
        print(f"Depth:{depth}")
        for settings in range(0, 128):
            # Get variation of features
            fill_embarked, fill_age, fill_fare, title_feature, ticket_group_feature, family_size_feature, deck_feature = \
                featureEngineering.get_features_vector(settings)

            # Prepare DataFrame
            tbl = featureEngineering.get_featured_data_frame("data.csv",
                                                             False,
                                                             fill_embarked,
                                                             fill_age,
                                                             fill_fare,
                                                             title_feature,
                                                             ticket_group_feature,
                                                             family_size_feature,
                                                             deck_feature)

            train_score_sum = 0
            for attempt in range(0, attempt_count):
                # Get results of classification
                score_train, score_test = get_data_frame_tree_classifier_score(depth, tbl)
                train_score_sum += score_train

            score_train_avg = train_score_sum / attempt_count
            if best_train_score < score_train_avg:
                best_train_score = score_train_avg
                print(f"Decision tree best configuration: {best_train_score}, depth:{depth}, settings: {featureEngineering.get_features_vector(settings)}")

    return best_train_score
