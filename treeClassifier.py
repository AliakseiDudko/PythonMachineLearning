import dtreeviz
import graphviz
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
    attempt_count = 5

    for depth in range(depth_min, depth_max + 1):
        print(f"Depth:{depth}")
        for settings_seed in range(0, featureEngineering.get_settings_variations_count()):
            # Get variation of features
            settings = featureEngineering.get_settings_variation(settings_seed)
            settings["max_depth"] = depth

            # Prepare DataFrame
            tbl = featureEngineering.get_featured_data_frame("data.csv", settings)

            test_score_sum = 0
            for attempt in range(0, attempt_count):
                # Get results of classification
                score_train, score_test = get_data_frame_tree_classifier_score(depth, tbl)
                test_score_sum += score_test

            score_test_avg = test_score_sum / attempt_count
            if best_test_score < score_test_avg:
                best_test_score = score_test_avg
                best_settings = settings
                print(f"Decision tree best configuration: {best_test_score}, settings: {best_settings}")

    return best_test_score, best_settings


def plot_tree_graphviz(settings):
    tbl = featureEngineering.get_featured_data_frame("data.csv", settings)
    train_data, test_data, train_survived_data, test_survived_data = featureEngineering.split_data_frame(tbl)

    tree_classifier = sklearn.tree.DecisionTreeClassifier(max_depth=settings["max_depth"])
    tree_classifier.fit(train_data, train_survived_data.values.ravel())

    # Draw using graphviz
    feature_names = ["Age", "Siblings Spouse", "Parent Children", "Fare", "Women", "1st Class",
                     "2nd Class", "3rd Class", "Cherbourg", "Queenstown", "Southhampton",
                     "Master", "Miss", "Mister", "Missis", "Other Title",
                     "Deck A", "Deck B", "Deck C", "Deck D", "Deck E", "Deck F", "Deck G",
                     "Deck N/A", "Deck T"]
    dot_data = sklearn.tree.export_graphviz(tree_classifier,
                                            feature_names=feature_names,
                                            class_names=["Drowned", "Survived"],
                                            out_file=None,
                                            filled=True)
    graph = graphviz.Source(dot_data, format="svg")
    graph.render(filename="Titanic_Graphviz")


def plot_tree_dteeviz(settings):
    tbl = featureEngineering.get_featured_data_frame("data.csv", settings)
    train_data, test_data, train_survived_data, test_survived_data = featureEngineering.split_data_frame(tbl)

    tree_classifier = sklearn.tree.DecisionTreeClassifier(max_depth=settings["max_depth"])
    tree_classifier.fit(train_data, train_survived_data.values.ravel())

    # Draw using dtreeviz
    feature_names = ["Age", "Siblings Spouse", "Parent Children", "Fare", "Women", "1st Class",
                     "2nd Class", "3rd Class", "Cherbourg", "Queenstown", "Southhampton",
                     "Master", "Miss", "Mister", "Missis", "Other Title",
                     "Deck A", "Deck B", "Deck C", "Deck D", "Deck E", "Deck F", "Deck G",
                     "Deck N/A", "Deck T"]
    viz = dtreeviz.trees.dtreeviz(tree_classifier,
                                  tbl.drop(["Survived"], axis=1),
                                  tbl["Survived"],
                                  feature_names=feature_names,
                                  class_names=["Drowned", "Survived"])
    viz.save("Titanic_Dtreeviz.svg")
