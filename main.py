import numpy
import pandas
import sklearn.neighbors
import sklearn.tree
import graphviz

# Set random seed to get stable results for debugging
numpy.random.seed(5)

# Fill missing Embarked with most frequent value
tbl = pandas.read_csv("data.csv")
tbl["Embarked"] = tbl["Embarked"].fillna(tbl["Embarked"].value_counts().index[0])

# Fill missing Fare based on mean ticket price for each Pclass
pclassFareMean = tbl[tbl["Fare"] > 0].groupby(["Pclass"])["Fare"].mean()
for pclass, fare in pclassFareMean.iteritems():
    tbl.loc[(tbl["Pclass"] == pclass) & (tbl["Fare"].isnull()), "Fare"] = fare

# Add Title column
tbl["Title"] = tbl["Name"].str.extract("([A-Za-z]+)\.", expand=True)

# Fill missing Age based on mean age for each title
titleAgeMean = tbl[tbl["Age"] > 0].groupby("Title")["Age"].mean()
for title, age in titleAgeMean.iteritems():
    tbl.loc[(tbl["Title"] == title) & (tbl["Age"].isnull()), "Age"] = age

# Add new feature IsAlone, HasCabin
tbl["IsAlone"] = tbl["SibSp"] + tbl["Parch"] == 0
tbl["HasCabin"] = tbl["Cabin"].isnull() != True

# print(tbl.to_string())

# Build dummy columns
tbl = pandas.get_dummies(tbl, columns=["Sex", "Pclass", "Embarked", "Title"], drop_first=False)

# Drop extra columns
tbl = tbl.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Normalize columns
for column in ["Age", "SibSp", "Parch", "Fare"]:
    tbl[column] = ((tbl[column] - tbl[column].min()) / (tbl[column].max() - tbl[column].min())) * 1

# print(tbl.to_string())
# print(tbl.corr().to_string())

# KNN algorithm
X = tbl.drop(["Survived"], axis=1)
Y = tbl[["Survived"]]

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y)

alg = sklearn.neighbors.NearestCentroid()
alg.fit(X_train, Y_train.values.ravel())

score_train = alg.score(X_train, Y_train)
score_test = alg.score(X_test, Y_test)
print("-------------------------------------------------------")
print(f"KNN Train: {score_train},  Test: {score_test}")

# Decision Tree algorithm
print("-------------------------------------------------------")
for depth in range(1, 20):
    classifier = sklearn.tree.DecisionTreeClassifier(max_depth=depth)
    classifier.fit(X_train, Y_train.values.ravel())

    score_train = classifier.score(X_train, Y_train)
    score_test = classifier.score(X_test, Y_test)
    print(f"Decision Tree Train (depth={depth}): {score_train},  Test: {score_test}")

classifier = sklearn.tree.DecisionTreeClassifier(max_depth=15)
classifier.fit(X_train, Y_train.values.ravel())
print("-------------------------------------------------------")

# Get dot-data list
feature_names = ["Age", "SibSp", "Parch", "Fare", "IsAlone", "HasCabin", "Sex_female", "Sex_male", "Pclass_1",
                 "Pclass_2", "Pclass_3", "Embarked_C", "Embarked_Q", "Embarked_S", "Title_Capt", "Title_Col",
                 "Title_Countess", "Title_Don", "Title_Dr", "Title_Jonkheer", "Title_Lady", "Title_Major",
                 "Title_Master", "Title_Miss", "Title_Mlle", "Title_Mme", "Title_Mr", "Title_Mrs", "Title_Ms",
                 "Title_Rev", "Title_Sir"]
dot_data = sklearn.tree.export_graphviz(classifier, feature_names=feature_names, out_file=None, filled=True)

graph = graphviz.Source(dot_data, format="svg")
# graph.render(filename="Titanic")
