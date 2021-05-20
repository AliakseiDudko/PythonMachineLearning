import graphviz
import matplotlib.pyplot as plt
import numpy as np
import knnClassifier
import treeClassifier
import naiveBayes

# Set seed if it is required to get stable results for debugging
# np.random.seed(5)

knnClassifier.find_best_knn_classifier_score()
# KNN best configuration: 0.7979341317365269, settings: (True, True, False, True, False, True, False)
knn_score_train, knn_score_test = knnClassifier.get_knn_classifier_score(True, True, False, True, False, True, False)
print(f"KNN           train score: {knn_score_train}, test score: {knn_score_test}")

treeClassifier.find_best_tree_classifier_score(7, 7)
# Decision tree best configuration: 0.9040718562874253, depth:7, settings: (False, True, True, False, False, True, True)
tree_score_train, tree_score_test = treeClassifier.get_tree_classifier_score(7, False, True, True, False, False, True, True)
print(f"Decision tree train score: {tree_score_train}, test score: {tree_score_test}")

naiveBayes.find_best_bayes_classifier_score()
# Naive Bayes best configuration: 0.8230988023952094, settings: (False, False, True, True, False, True, False)
best_bayes_score_train, best_bayes_score_test = naiveBayes.get_bayes_classifier_score(False, False, True, True, False, True, False)
print(f"Naive Bayes   train score: {best_bayes_score_train}, test score: {best_bayes_score_test}")

# Show Age histogram
# tbl["Age"].hist()
# plt.show()

# Show Survived by Title histogram
# tbl["Survived"].groupby(tbl["Title"]).mean().plot(kind='bar')
# plt.show()

# print(tbl.to_string())
# print(tbl.corr().to_string())

# Draw using graphviz
# feature_names = ["Age", "SibSp", "Parch", "Fare", "IsAlone", "Sex_female", "Pclass_1",
#                  "Pclass_2", "Pclass_3", "Embarked_C", "Embarked_Q", "Embarked_S",
#                  "Title_Master", "Title_Miss", "Title_Mr", "Title_Ms", "Title_Senior"]
# dot_data = sklearn.tree.export_graphviz(treeClassifier,
#                                         feature_names=feature_names,
#                                         class_names=["Drowned", "Survived"],
#                                         out_file=None,
#                                         filled=True)
# graph = graphviz.Source(dot_data, format="svg")
# graph.render(filename="Titanic_Graphviz")

# Draw using dtreeviz
# viz = dtree.dtreeviz(treeClassifier,
#                      tbl[feature_names],
#                      tbl["Survived"],
#                      feature_names=feature_names,
#                      class_names=["Drowned", "Survived"])
# viz.save("Titanic_Dtreeviz.svg")