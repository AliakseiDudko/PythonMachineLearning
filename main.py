import graphviz
import numpy
import pandas
import sklearn.neighbors
import sklearn.tree
import sklearn.naive_bayes
import dtreeviz.trees as dtree

# Set random seed to get stable results for debugging
numpy.random.seed(5)

# Fill missing Embarked with most frequent value
tbl = pandas.read_csv("data.csv")
tbl["Embarked"] = tbl["Embarked"].fillna(tbl["Embarked"].value_counts().index[0])

# Fill missing Fare based on mean ticket price for each Pclass
pclassFareMean = tbl[tbl["Fare"] > 0].groupby(["Pclass", "Embarked"])["Fare"].mean()
for item, fare in pclassFareMean.iteritems():
    tbl.loc[(tbl["Pclass"] == item[0]) & (tbl["Embarked"] == item[1]) & (tbl["Fare"].isnull()), "Fare"] = fare

# Create and reduce Title column
titleDictionary = {"Master": "Master", "Miss": "Miss", "Mlle": "Miss", "Mme": "Ms", "Ms": "Ms", "Mr": "Mr",
                   "Countess": "Ms", "Mrs": "Ms", "Jonkheer": "Mr", "Don": "Mr", "Dr": "Mr", "Rev": "Mr", "Lady": "Ms",
                   "Major": "Senior", "Sir": "Senior", "Col": "Senior", "Capt": "Senior"}
tbl["Title"] = tbl["Name"].str.extract("([A-Za-z]+)\.", expand=True)
tbl = tbl.replace({"Title": titleDictionary})

# Fill missing Age based on mean age for each title
titleAgeMean = tbl[tbl["Age"] > 0].groupby("Title")["Age"].mean()
for title, age in titleAgeMean.iteritems():
    tbl.loc[(tbl["Title"] == title) & (tbl["Age"].isnull()), "Age"] = age

# Add new features IsAlone
tbl["IsAlone"] = (tbl["SibSp"] + tbl["Parch"] == 0) * 1

# Build dummy columns
tbl = pandas.get_dummies(tbl, columns=["Sex", "Pclass", "Embarked", "Title"], drop_first=False)

# Drop extra columns
tbl = tbl.drop(["PassengerId", "Name", "Ticket", "Cabin", "Sex_male"], axis=1)

# Normalize columns
tbl = ((tbl - tbl.min()) / (tbl.max() - tbl.min()))

# print(tbl.to_string())
# print(tbl.corr().to_string())

X = tbl.drop(["Survived"], axis=1)
Y = tbl[["Survived"]]

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y)
print("-------------------------------------------------------")

# KNN algorithm

knnClassifier = sklearn.neighbors.NearestCentroid()
knnClassifier.fit(X_train, Y_train.values.ravel())
score_train = knnClassifier.score(X_train, Y_train)
score_test = knnClassifier.score(X_test, Y_test)
print(f"KNN Train: {score_train},  Test: {score_test}")
print("-------------------------------------------------------")

# Decision Tree algorithm
# for depth in range(1, 20):
#     treeClassifier = sklearn.tree.DecisionTreeClassifier(max_depth=depth)
#     treeClassifier.fit(X_train, Y_train.values.ravel())
#
#     score_train = treeClassifier.score(X_train, Y_train)
#     score_test = treeClassifier.score(X_test, Y_test)
#     print(f"Decision Tree Train (depth={depth}): {score_train}, Test: {score_test}")
# print("-------------------------------------------------------")

treeClassifier = sklearn.tree.DecisionTreeClassifier(max_depth=15)
treeClassifier.fit(X_train, Y_train.values.ravel())
score_train = treeClassifier.score(X_train, Y_train)
score_test = treeClassifier.score(X_test, Y_test)
print(f"Decision Tree Train (depth={15}): {score_train}, Test: {score_test}")
print("-------------------------------------------------------")

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

# print(tbl.corr().to_string())
X = tbl.drop(["Age", "SibSp", "Parch", "Embarked_Q", "Title_Master", "Title_Senior", "Survived"], axis=1)
Y = tbl[["Survived"]]
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y)

bayesClassifier = sklearn.naive_bayes.GaussianNB()
bayesClassifier.fit(X_train, Y_train.values.ravel())
score_train = bayesClassifier.score(X_train, Y_train)
score_test = bayesClassifier.score(X_test, Y_test)
print(f"Bayes Naive Train: {score_train}, Test: {score_test}")
print("-------------------------------------------------------")
