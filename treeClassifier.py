import dtreeviz
import graphviz
import sklearn.tree

import featureEngineering
import solution


def get_classifier(max_depth) -> object:
    return sklearn.tree.DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced")


def get_tree_classifier_score(settings) -> (float, float):
    tree_classifier = get_classifier(settings["max_depth"])
    score_train, score_test = solution.get_classifier_score(tree_classifier, settings)

    return score_train, score_test


def find_best_tree_classifier_score(depth_min, depth_max) -> (float, dict):
    best_test_score = 0.0
    best_settings = None

    depths = list(range(depth_min, depth_max + 1))
    depths.append(None)
    for depth in depths:
        print(f"Depth={depth} ", end="")

        tree_classifier = get_classifier(depth)
        current_best_test_score, current_best_settings = solution.find_best_classifier_score(tree_classifier)

        if best_test_score < current_best_test_score:
            best_test_score = current_best_test_score
            best_settings = current_best_settings
            best_settings["max_depth"] = depth

    return best_test_score, best_settings


def plot_tree_graphviz(settings):
    tbl = featureEngineering.get_featured_data_frame(settings)
    train_data, test_data, train_survived_data, test_survived_data = featureEngineering.split_data_frame(tbl)

    tree_classifier = sklearn.tree.DecisionTreeClassifier(max_depth=settings["max_depth"])
    tree_classifier.fit(train_data, train_survived_data.values.ravel())

    # Draw using graphviz
    feature_names = ["Age", "Siblings Spouse", "Parent Children", "Fare", "Women", "1st Class", "2nd Class",
                     "3rd Class", "Cherbourg", "Queenstown", "Southhampton", "Master", "Miss", "Mister", "Missis",
                     "Other Title", "Alone", "Big Family", "Middle Family", "Deck A", "Deck B", "Deck C", "Deck D",
                     "Deck E", "Deck F", "Deck G", "Deck N/A", "Deck T"]
    dot_data = sklearn.tree.export_graphviz(tree_classifier,
                                            feature_names=feature_names,
                                            class_names=["Drowned", "Survived"],
                                            out_file=None,
                                            filled=True)
    graph = graphviz.Source(dot_data, format="svg")
    graph.render(filename="Titanic_Graphviz")


def plot_tree_dteeviz(settings):
    tbl = featureEngineering.get_featured_data_frame(settings)
    train_data, test_data, train_survived_data, test_survived_data = featureEngineering.split_data_frame(tbl)

    tree_classifier = sklearn.tree.DecisionTreeClassifier(max_depth=settings["max_depth"])
    tree_classifier.fit(train_data, train_survived_data.values.ravel())

    # Draw using dtreeviz
    feature_names = ["Age", "Siblings Spouse", "Parent Children", "Fare", "Women", "1st Class", "2nd Class",
                     "3rd Class", "Cherbourg", "Queenstown", "Southhampton", "Master", "Miss", "Mister", "Missis",
                     "Other Title", "Alone", "Big Family", "Middle Family", "Deck A", "Deck B", "Deck C", "Deck D",
                     "Deck E", "Deck F", "Deck G", "Deck N/A", "Deck T"]
    viz = dtreeviz.trees.dtreeviz(tree_classifier,
                                  tbl.drop(["Survived"], axis=1),
                                  tbl["Survived"],
                                  feature_names=feature_names,
                                  class_names=["Drowned", "Survived"])
    viz.save("Titanic_Dtreeviz.svg")
