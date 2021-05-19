import sklearn.naive_bayes
import featureEngineering


def get_bayes_classifier_score(fill_embarked=False,
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

    bayes_classifier = sklearn.naive_bayes.GaussianNB()
    bayes_classifier.fit(train_data, train_survived_data.values.ravel())
    score_train = bayes_classifier.score(train_data, train_survived_data)
    score_test = bayes_classifier.score(test_data, test_survived_data)

    return score_train, score_test


def get_best_bayes_classifier_score() -> float:
    best_test_score = 0.0
    best_settings = 0

    for i in range(0, 128):
        # Get variation of features
        fill_embarked, fill_age, fill_fare, title_feature, ticket_group_feature, family_size_feature, deck_feature = \
            featureEngineering.get_features_vector(i)

        # Get results of classification
        score_train, score_test = get_bayes_classifier_score(fill_embarked,
                                                             fill_age,
                                                             fill_fare,
                                                             title_feature,
                                                             ticket_group_feature,
                                                             family_size_feature,
                                                             deck_feature)
        if best_test_score < score_test:
            best_test_score = score_test
            best_settings = i
            print(
                f"Naive Bayes best configuration: {best_test_score}, settings: {featureEngineering.get_features_vector(best_settings)}")

    return best_test_score
