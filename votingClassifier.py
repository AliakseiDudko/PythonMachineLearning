import sklearn

import knnClassifier
import logisticRegression
import naiveBayes
import randomForest
import solution
import svmClassifier
import treeClassifier


def get_classifier() -> object:
    estimators = [('knn_classifier', knnClassifier.get_classifier()),
                  ('logistic_classifier', logisticRegression.get_classifier()),
                  ('naive_bayes_classifier', naiveBayes.get_classifier()),
                  ('random_forest_classifier', randomForest.get_classifier()),
                  ('svm_classifier', svmClassifier.get_classifier()),
                  ('tree_classifier', treeClassifier.get_classifier(5))]
    voting_classifier = sklearn.ensemble.VotingClassifier(estimators=estimators, voting="hard")

    return voting_classifier


def get_voting_classifier_score() -> (float, float):
    voting_classifier = get_classifier()
    score_train, score_test = solution.get_classifier_score(voting_classifier)

    return score_train, score_test
