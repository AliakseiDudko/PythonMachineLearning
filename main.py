import knnClassifier
import linearRegression
import logisticRegression
import naiveBayes
import randomForest
import svmClassifier
import treeClassifier
import votingClassifier

# Set seed if it is required to get stable results for debugging
# np.random.seed(5)

# best_knn_score, best_knn_settings = knnClassifier.find_best_knn_classifier_score()
# print(f"Best KNN test score: {best_knn_score}, settings: {best_knn_settings}")
knn_settings = {'fill_embarked': False, 'fill_age': False, 'fill_fare': True, 'title_feature': True,
                'ticket_group_feature': False, 'family_size_feature': False, 'deck_feature': False, 'drop_male': False,
                'drop_age': True, 'drop_sibsp_parch': True}
knn_score_train, knn_score_test = knnClassifier.get_knn_classifier_score(knn_settings)
print(f"KNN           train score: {knn_score_train}, test score: {knn_score_test}")

# best_tree_score, best_tree_settings = treeClassifier.find_best_tree_classifier_score(3, 5)
# print(f"Best decision tree test score: {best_tree_score}, settings: {best_tree_settings}")
tree_settings = {'fill_embarked': True, 'fill_age': False, 'fill_fare': False, 'title_feature': True,
                 'ticket_group_feature': False, 'family_size_feature': True, 'deck_feature': True, 'drop_male': False,
                 'drop_age': False, 'drop_sibsp_parch': True, 'max_depth': 5}
tree_score_train, tree_score_test = treeClassifier.get_tree_classifier_score(tree_settings)
print(f"Decision tree train score: {tree_score_train}, test score: {tree_score_test}")
# # Visualize decision tree
# treeClassifier.plot_tree_graphviz(tree_settings)
# treeClassifier.plot_tree_dteeviz(tree_settings)

# best_bayes_score, best_bayes_settings = naiveBayes.find_best_bayes_classifier_score()
# print(f"Best naive Bayes test score: {best_bayes_score}, settings: {best_bayes_settings}")
bayes_settings = {'fill_embarked': True, 'fill_age': True, 'fill_fare': True, 'title_feature': True,
                  'ticket_group_feature': True, 'family_size_feature': True, 'deck_feature': False, 'drop_male': False,
                  'drop_age': True, 'drop_sibsp_parch': False}
bayes_score_train, bayes_score_test = naiveBayes.get_bayes_classifier_score(bayes_settings)
print(f"Naive Bayes   train score: {bayes_score_train}, test score: {bayes_score_test}")

# best_logistic_score, best_logistic_settings = logisticRegression.find_best_logistic_classifier_score()
# print(f"Best logistic reg. test score: {best_logistic_score}, settings: {best_logistic_settings}")
logistic_settings = {'fill_embarked': False, 'fill_age': True, 'fill_fare': True, 'title_feature': True,
                     'ticket_group_feature': True, 'family_size_feature': True, 'deck_feature': True,
                     'drop_male': False, 'drop_age': False, 'drop_sibsp_parch': False}
logistic_score_train, logistic_score_test = logisticRegression.get_logistic_classifier_score(logistic_settings)
print(f"Logistic Reg. train score: {logistic_score_train}, test score: {logistic_score_test}")

# best_svm_score, best_svm_settings = svmClassifier.find_best_svm_classifier_score()
# print(f"Best SVM test score: {best_svm_score}, settings: {best_svm_settings}")
svm_settings = {'fill_embarked': False, 'fill_age': True, 'fill_fare': True, 'title_feature': True,
                'ticket_group_feature': True, 'family_size_feature': True, 'deck_feature': False, 'drop_male': True,
                'drop_age': False, 'drop_sibsp_parch': False}
svm_score_train, svm_score_test = svmClassifier.get_svm_classifier_score(svm_settings)
print(f"SVM           train score: {svm_score_train}, test score: {svm_score_test}")

# best_linear_score, best_linear_settings = linearRegression.find_best_linear_classifier_score()
# print(f"Best linear reg. test score: {best_linear_score}, settings: {best_linear_settings}")
linear_settings = {'fill_embarked': True, 'fill_age': True, 'fill_fare': False, 'title_feature': True,
                   'ticket_group_feature': False, 'family_size_feature': True, 'deck_feature': False, 'drop_male': True,
                   'drop_age': True, 'drop_sibsp_parch': False}
linear_score_train, linear_score_test = linearRegression.get_linear_classifier_score(linear_settings)
print(f"Linear Reg.   train score: {linear_score_train}, test score: {linear_score_test}")

# best_random_forest_score, best_random_forest_settings = randomForest.find_best_random_forest_classifier_score()
# print(f"Best random forest : {best_random_forest_score}, settings: {best_random_forest_settings}")
random_forest_settings = {'fill_embarked': True, 'fill_age': True, 'fill_fare': True, 'title_feature': True,
                          'ticket_group_feature': False, 'family_size_feature': False, 'deck_feature': False,
                          'drop_male': False, 'drop_age': False, 'drop_sibsp_parch': True}
random_forest_score_train, random_forest_score_test = randomForest.get_random_forest_classifier_score(
    random_forest_settings)
print(f"Random forest train score: {random_forest_score_train}, test score: {random_forest_score_test}")

voting_score_train, voting_score_test = votingClassifier.get_voting_classifier_score()
print(f"Voting        train score: {voting_score_train}, test score: {voting_score_test}")

# Show Age histogram
# tbl["Age"].hist()
# plt.show()

# Show Survived by Title histogram
# tbl["Survived"].groupby(tbl["Title"]).mean().plot(kind='bar')
# plt.show()

# print(tbl.to_string())
# print(tbl.corr().to_string())
