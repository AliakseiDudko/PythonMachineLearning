import numpy
import pandas
import sklearn.neighbors
import sklearn.tree
import sklearn.linear_model

# Set random seed to get stable results for debugging
numpy.random.seed(5)

# Read data to DataFrame
tbl = pandas.read_csv("data.csv")

# Fill missing Embarked with most frequent value
tbl["Embarked"] = tbl["Embarked"].fillna(tbl["Embarked"].value_counts().index[0])

# Fill missing Fare based on mean ticket price for each Pclass
pclassFareMean = tbl[tbl["Fare"] > 0].groupby(["Pclass", "Embarked"])["Fare"].mean()
for item, fare in pclassFareMean.iteritems():
    tbl.loc[(tbl["Pclass"] == item[0]) & (tbl["Embarked"] == item[1]) & (tbl["Fare"].isnull()), "Fare"] = fare

# Create and reduce Title column
titleDictionary = {"Master": "Master", "Mr": "Mr", "Jonkheer": "Mr", "Don": "Mr",
                   "Miss": "Miss", "Mlle": "Miss",  "Mme": "Ms", "Ms": "Ms", "Mrs": "Ms",
                   "Countess": "Other", "Dr": "Other", "Rev": "Other", "Lady": "Other",
                   "Major": "Other", "Sir": "Other", "Col": "Other", "Capt": "Other"}
tbl["Title"] = tbl["Name"].str.extract("([A-Za-z]+)\.", expand=True)
tbl = tbl.replace({"Title": titleDictionary})

# Fill missing Age based on mean age for each title
titleAgeMean = tbl[tbl["Age"] > 0].groupby("Title")["Age"].mean()
for title, age in titleAgeMean.iteritems():
    tbl.loc[(tbl["Title"] == title) & (tbl["Age"].isnull()), "Age"] = age

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

# Logistic regression algorithm
classifier = sklearn.linear_model.LogisticRegression(penalty="l2", solver="newton-cg", class_weight="balanced")
classifier.fit(X_train, Y_train.values.ravel())
score_train = classifier.score(X_train, Y_train)
score_test = classifier.score(X_test, Y_test)
print(f"Logistic regression Train: {score_train},  Test: {score_test}")
print("-------------------------------------------------------")