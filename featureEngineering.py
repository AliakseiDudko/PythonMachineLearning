import pandas as pd
import sklearn
from pandas import DataFrame


def get_featured_data_frame(settings):
    # Get settings
    fill_embarked = settings.get("fill_embarked", False)
    fill_age = settings.get("fill_age", False)
    fill_fare = settings.get("fill_fare", False)
    title_feature = settings.get("title_feature", False)
    ticket_group_feature = settings.get("ticket_group_feature", False)
    family_size_feature = settings.get("family_size_feature", False)
    deck_feature = settings.get("deck_feature", False)
    drop_male = settings.get("drop_male", False)
    drop_age = settings.get("drop_age", False)
    drop_sibsp_parch = settings.get("drop_sibsp_parch", False)
    normalize = settings.get("normalize", False)

    # Read data to DataFrame
    tbl = pd.read_csv("data.csv")

    # Fill missing Embarked with the most frequent value
    if fill_embarked:
        tbl["Embarked"] = tbl["Embarked"].fillna(tbl["Embarked"].value_counts().index[0])

    # Fill missing Fare based on mean ticket price for each Pclass and Embarked
    if fill_fare:
        pclass_fare_mean = tbl[tbl["Fare"] > 0].groupby(["Pclass", "Embarked"])["Fare"].mean()
        for item, fare in pclass_fare_mean.iteritems():
            tbl.loc[(tbl["Pclass"] == item[0]) & (tbl["Embarked"] == item[1]) & (tbl["Fare"].isnull()), "Fare"] = fare

    # Create and reduce Title column
    if title_feature:
        title_dictionary = {"Master": "Master", "Mr": "Mr", "Jonkheer": "Mr", "Don": "Mr",
                            "Miss": "Miss", "Mlle": "Miss", "Mme": "Ms", "Ms": "Ms", "Mrs": "Ms",
                            "Countess": "Other", "Dr": "Other", "Rev": "Other", "Lady": "Other",
                            "Major": "Other", "Sir": "Other", "Col": "Other", "Capt": "Other"}
        tbl["Title"] = tbl["Name"].str.extract("([A-Za-z]+)\.", expand=True)
        tbl = tbl.replace({"Title": title_dictionary})

    # Fill missing Age based on mean age for each title
    if fill_age & title_feature:
        title_age_mean = tbl[tbl["Age"] > 0].groupby("Title")["Age"].mean()
        for title, age in title_age_mean.iteritems():
            tbl.loc[(tbl["Title"] == title) & (tbl["Age"].isnull()), "Age"] = age

    # Add TicketGroupSize
    if ticket_group_feature:
        tickets_count = tbl.groupby(tbl["Ticket"])["Ticket"].count()
        for ticket, count in tickets_count.iteritems():
            tbl.loc[tbl["Ticket"] == ticket, "TicketGroupSize"] = count

    # Add FamilySize. Alone: no family members, Middle: 1-3 members, Big: 4 and more members
    if family_size_feature:
        tbl.loc[tbl["Parch"] + tbl["SibSp"] == 0, "FamilySize"] = "Alone"
        tbl.loc[(tbl["Parch"] + tbl["SibSp"] >= 1) & (tbl["Parch"] + tbl["SibSp"] <= 3), "FamilySize"] = "Middle"
        tbl.loc[tbl["Parch"] + tbl["SibSp"] > 3, "FamilySize"] = "Big"

    # Add deck feature
    if deck_feature:
        tbl["Deck"] = tbl["Cabin"].str.extract("([A-Z])", expand=True)
        tbl["Deck"] = tbl["Deck"].fillna("N/A")

    # Fill empty values with 0
    tbl = tbl.fillna(0)

    # Build dummy columns
    tbl = pd.get_dummies(tbl, columns=["Sex", "Pclass", "Embarked"], drop_first=False)
    if title_feature:
        tbl = pd.get_dummies(tbl, columns=["Title"], drop_first=False)
    if family_size_feature:
        tbl = pd.get_dummies(tbl, columns=["FamilySize"], drop_first=False)
    if deck_feature:
        tbl = pd.get_dummies(tbl, columns=["Deck"], drop_first=False)

    # Drop extra columns
    tbl = tbl.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    if drop_male:
        tbl = tbl.drop(["Sex_male"], axis=1)
    if drop_age:
        tbl = tbl.drop(["Age"], axis=1)
    if drop_sibsp_parch:
        tbl = tbl.drop(["SibSp", "Parch"], axis=1)

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
    keys = ["fill_embarked", "fill_age", "fill_fare", "title_feature", "ticket_group_feature", "family_size_feature",
            "deck_feature", "drop_male", "drop_age", "drop_sibsp_parch"]
    settings = dict.fromkeys(keys)
    for index in range(0, len(settings)):
        key = keys[index]
        settings[key] = seed & 2 ** index > 0

    return settings


def get_settings_variations_count() -> int:
    return 2 ** len(get_settings_variation(0))
