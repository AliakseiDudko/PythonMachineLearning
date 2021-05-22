import knnClassifier
import logisticRegression
import naiveBayes
import svmClassifier
import treeClassifier
import treeVisualization

# Set seed if it is required to get stable results for debugging
# np.random.seed(5)

# best_knn_score, best_knn_settings = knnClassifier.find_best_knn_classifier_score()
# print(f"Best KNN           test score: {best_knn_score}, settings: {best_knn_settings}")
# # Best KNN           test score: 0.8121076233183857, settings: {'fill_embarked': False, 'fill_age': True, 'fill_fare': True, 'title_feature': True, 'ticket_group_feature': False, 'family_size_feature': True, 'deck_feature': True}
#
# best_tree_score, best_tree_settings = treeClassifier.find_best_tree_classifier_score(1, 5)
# print(f"Best decision tree test score: {best_tree_score}, settings: {best_tree_settings}")
# # Best decision tree test score: 0.8493273542600898, settings: {'fill_embarked': True, 'fill_age': True, 'fill_fare': True, 'title_feature': True, 'ticket_group_feature': False, 'family_size_feature': False, 'deck_feature': True, 'max_depth': 3}
#
# best_bayes_score, best_bayes_settings = naiveBayes.find_best_bayes_classifier_score()
# print(f"Best naive Bayes   test score: {best_bayes_score}, settings: {best_bayes_settings}")
# # Best naive Bayes   test score: 0.8219730941704038, settings: {'fill_embarked': False, 'fill_age': False, 'fill_fare': False, 'title_feature': True, 'ticket_group_feature': True, 'family_size_feature': False, 'deck_feature': False}
#
# best_logistic_score, best_logistic_settings = logisticRegression.find_best_logistic_classifier_score()
# print(f"Best logistic reg. test score: {best_logistic_score}, settings: {best_logistic_settings}")
# # Best logistic reg. test score: 0.8304932735426009, settings: {'fill_embarked': True, 'fill_age': False, 'fill_fare': False, 'title_feature': True, 'ticket_group_feature': True, 'family_size_feature': True, 'deck_feature': False}
#
# best_svm_score, best_svm_settings = svmClassifier.find_best_svm_classifier_score()
# print(f"Best SVM           test score: {best_svm_score}, settings: {best_svm_settings}")
# # Best SVM           test score: 0.841255605381166, settings: {'fill_embarked': True, 'fill_age': False, 'fill_fare': False, 'title_feature': True, 'ticket_group_feature': True, 'family_size_feature': False, 'deck_feature': False}


knn_settings = {'fill_embarked': False, 'fill_age': True, 'fill_fare': True, 'title_feature': True,
                'ticket_group_feature': False, 'family_size_feature': True, 'deck_feature': True}
knn_score_train, knn_score_test = knnClassifier.get_knn_classifier_score(knn_settings)
print(f"KNN           train score: {knn_score_train}, test score: {knn_score_test}")

tree_settings = {'fill_embarked': True, 'fill_age': True, 'fill_fare': True, 'title_feature': True,
                 'ticket_group_feature': False, 'family_size_feature': False, 'deck_feature': True, 'max_depth': 3}
tree_score_train, tree_score_test = treeClassifier.get_tree_classifier_score(tree_settings)
print(f"Decision tree train score: {tree_score_train}, test score: {tree_score_test}")

bayes_settings = {'fill_embarked': False, 'fill_age': False, 'fill_fare': False, 'title_feature': True,
                  'ticket_group_feature': True, 'family_size_feature': False, 'deck_feature': False}
best_bayes_score_train, best_bayes_score_test = naiveBayes.get_bayes_classifier_score(bayes_settings)
print(f"Naive Bayes   train score: {best_bayes_score_train}, test score: {best_bayes_score_test}")

logistic_settings = {'fill_embarked': True, 'fill_age': False, 'fill_fare': False, 'title_feature': True,
                     'ticket_group_feature': True, 'family_size_feature': True, 'deck_feature': False}
best_logistic_score_train, best_logistic_score_test = logisticRegression.get_logistic_classifier_score(
    logistic_settings)
print(f"Logistic Reg. train score: {best_logistic_score_train}, test score: {best_logistic_score_test}")

svm_settings = {'fill_embarked': True, 'fill_age': False, 'fill_fare': False, 'title_feature': True,
                'ticket_group_feature': True, 'family_size_feature': False, 'deck_feature': False}
best_svm_score_train, best_svm_score_test = svmClassifier.get_svm_classifier_score(svm_settings)
print(f"SVM           train score: {best_svm_score_train}, test score: {best_svm_score_test}")

# Show Age histogram
# tbl["Age"].hist()
# plt.show()

# Show Survived by Title histogram
# tbl["Survived"].groupby(tbl["Title"]).mean().plot(kind='bar')
# plt.show()

# print(tbl.to_string())
# print(tbl.corr().to_string())
