import pandas as pd
import sklearn
from pandas import DataFrame


def get_featured_data_frame(file_name,
                            settings,
                            normalize=False):
    # Read data to DataFrame
    tbl = pd.read_csv(file_name)

    # Fill missing Embarked with the most frequent value
    if settings["fill_embarked"]:
        tbl["Embarked"] = tbl["Embarked"].fillna(tbl["Embarked"].value_counts().index[0])

    # Fill missing Fare based on mean ticket price for each Pclass and Embarked
    if settings["fill_fare"]:
        pclass_fare_mean = tbl[tbl["Fare"] > 0].groupby(["Pclass", "Embarked"])["Fare"].mean()
        for item, fare in pclass_fare_mean.iteritems():
            tbl.loc[(tbl["Pclass"] == item[0]) & (tbl["Embarked"] == item[1]) & (tbl["Fare"].isnull()), "Fare"] = fare

    # Create and reduce Title column
    if settings["title_feature"]:
        title_dictionary = {"Master": "Master", "Mr": "Mr", "Jonkheer": "Mr", "Don": "Mr",
                            "Miss": "Miss", "Mlle": "Miss", "Mme": "Ms", "Ms": "Ms", "Mrs": "Ms",
                            "Countess": "Other", "Dr": "Other", "Rev": "Other", "Lady": "Other",
                            "Major": "Other", "Sir": "Other", "Col": "Other", "Capt": "Other"}
        tbl["Title"] = tbl["Name"].str.extract("([A-Za-z]+)\.", expand=True)
        tbl = tbl.replace({"Title": title_dictionary})

    # Fill missing Age based on mean age for each title
    if settings["fill_age"] & settings["title_feature"]:
        title_age_mean = tbl[tbl["Age"] > 0].groupby("Title")["Age"].mean()
        for title, age in title_age_mean.iteritems():
            tbl.loc[(tbl["Title"] == title) & (tbl["Age"].isnull()), "Age"] = age

    # Add TicketGroupSize
    if settings["ticket_group_feature"]:
        tickets_count = tbl.groupby(tbl["Ticket"])["Ticket"].count()
        for ticket, count in tickets_count.iteritems():
            tbl.loc[tbl["Ticket"] == ticket, "TicketGroupSize"] = count

    # Add FamilySize. Alone: no family members, Middle: 1-3 members, Big: 4 and more members
    if settings["family_size_feature"]:
        tbl.loc[tbl["Parch"] + tbl["SibSp"] == 0, "FamilySize"] = "Alone"
        tbl.loc[(tbl["Parch"] + tbl["SibSp"] >= 1) & (tbl["Parch"] + tbl["SibSp"] <= 3), "FamilySize"] = "Middle"
        tbl.loc[tbl["Parch"] + tbl["SibSp"] > 3, "FamilySize"] = "Big"

    # Add deck feature
    if settings["deck_feature"]:
        tbl["Deck"] = tbl["Cabin"].str.extract("([A-Z])", expand=True)
        tbl["Deck"] = tbl["Deck"].fillna("N/A")

    # Fill empty values with 0
    tbl = tbl.fillna(0)

    # Build dummy columns
    tbl = pd.get_dummies(tbl, columns=["Sex", "Pclass", "Embarked"], drop_first=False)
    if settings["title_feature"]:
        tbl = pd.get_dummies(tbl, columns=["Title"], drop_first=False)
    if settings["family_size_feature"]:
        tbl = pd.get_dummies(tbl, columns=["FamilySize"], drop_first=False)
    if settings["deck_feature"]:
        tbl = pd.get_dummies(tbl, columns=["Deck"], drop_first=False)

    # Drop extra columns
    tbl = tbl.drop(["PassengerId", "Name", "Ticket", "Cabin", "Sex_male"], axis=1)

    # Normalize columns
    if normalize:
        tbl = ((tbl - tbl.min()) / (tbl.max() - tbl.min()))

    return tbl


def split_data_frame(tbl) -> (DataFrame, DataFrame, DataFrame, DataFrame):
    data = tbl.drop(["Survived"], axis=1)
    survived_data = tbl[["Survived"]]
    train_data, test_data, train_survived_data, test_survived_data = \
        sklearn.model_selection.train_test_split(data, survived_data)

    return train_data, test_data, train_survived_data, test_survived_data


def get_settings_variation(seed) -> dict:
    keys = ["fill_embarked", "fill_age", "fill_fare", "title_feature",
            "ticket_group_feature", "family_size_feature", "deck_feature"]
    settings = dict.fromkeys(keys)
    for index in range(0, len(settings)):
        key = keys[index]
        settings[key] = seed & 2 ** index > 0

    return settings


def get_settings_variations_count() -> int:
    return 2 ** len(get_settings_variation(0))
