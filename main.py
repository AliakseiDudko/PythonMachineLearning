import graphviz
import matplotlib.pyplot as plt
import knnClassifier
import treeClassifier
import naiveBayes

# Set seed if it is required to get stable results for debugging
# np.random.seed(5)

# best_knn_score, best_knn_settings = knnClassifier.find_best_knn_classifier_score()
# print(f"Best KNN           test score: {best_knn_score}, settings: {best_knn_settings}")
# # Best KNN           test score: 0.7976497005988024, settings: {'fill_embarked': False, 'fill_age': True, 'fill_fare': True, 'title_feature': True, 'ticket_group_feature': False, 'family_size_feature': True, 'deck_feature': False}
#
# best_tree_score, best_tree_settings = treeClassifier.find_best_tree_classifier_score(7, 7)
# print(f"Best decision tree test score: {best_tree_score}, settings: {best_tree_settings}")
# # Best decision tree test score: 0.902949101796407, depth:7, settings: {'fill_embarked': False, 'fill_age': False, 'fill_fare': True, 'title_feature': True, 'ticket_group_feature': False, 'family_size_feature': False, 'deck_feature': True}
#
# best_bayes_score, best_bayes_settings = naiveBayes.find_best_bayes_classifier_score()
# print(f"Best naive Bayes   test score: {best_bayes_score}, settings: {best_bayes_settings}")
# # Best naive Bayes   test score: 0.8221856287425147, settings: {'fill_embarked': False, 'fill_age': False, 'fill_fare': True, 'title_feature': True, 'ticket_group_feature': True, 'family_size_feature': True, 'deck_feature': False}

knn_settings = {'fill_embarked': False, 'fill_age': True, 'fill_fare': True, 'title_feature': True, 'ticket_group_feature': False, 'family_size_feature': True, 'deck_feature': False}
knn_score_train, knn_score_test = knnClassifier.get_knn_classifier_score(knn_settings)
print(f"KNN           train score: {knn_score_train}, test score: {knn_score_test}")

tree_settings = {'max_depth': 7, 'fill_embarked': False, 'fill_age': False, 'fill_fare': True, 'title_feature': True, 'ticket_group_feature': False, 'family_size_feature': False, 'deck_feature': True}
tree_score_train, tree_score_test = treeClassifier.get_tree_classifier_score(tree_settings)
print(f"Decision tree train score: {tree_score_train}, test score: {tree_score_test}")

bayes_settings = {'fill_embarked': False, 'fill_age': False, 'fill_fare': True, 'title_feature': True, 'ticket_group_feature': True, 'family_size_feature': True, 'deck_feature': False}
best_bayes_score_train, best_bayes_score_test = naiveBayes.get_bayes_classifier_score(bayes_settings)
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