import sklearn.tree
import featureEngineering


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
                                                     fill_embarked,
                                                     fill_age,
                                                     fill_fare,
                                                     title_feature,
                                                     ticket_group_feature,
                                                     family_size_feature,
                                                     deck_feature)

    data = tbl.drop(["Survived"], axis=1)
    survived_data = tbl[["Survived"]]
    train_data, test_data, train_survived_data, test_survived_data = \
        sklearn.model_selection.train_test_split(data, survived_data)

    tree_classifier = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth)
    tree_classifier.fit(train_data, train_survived_data.values.ravel())
    score_train = tree_classifier.score(train_data, train_survived_data)
    score_test = tree_classifier.score(test_data, test_survived_data)

    return score_train, score_test


def get_best_tree_classifier_score() -> float:
    best_max_depth = 0
    best_test_score = 0.0
    best_settings = 1

    for depth in range(1, 20):
        print(f"Depth:{depth}")
        for i in range(0, 128):
            # Get variation of features
            fill_embarked, fill_age, fill_fare, title_feature, ticket_group_feature, family_size_feature, deck_feature = \
                featureEngineering.get_features_vector(i)

            # Get results of classification
            score_train, score_test = get_tree_classifier_score(depth,
                                                                fill_embarked,
                                                                fill_age,
                                                                fill_fare,
                                                                title_feature,
                                                                ticket_group_feature,
                                                                family_size_feature,
                                                                deck_feature)
            if best_test_score < score_test:
                best_test_score = score_test
                best_settings = i
                best_max_depth = depth
                print(f"Decision tree best configuration: {best_test_score}, depth:{best_max_depth}, settings: {featureEngineering.get_features_vector(best_settings)}")

    return best_test_score
