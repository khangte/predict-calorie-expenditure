
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_features(df):
    df = df.copy()
    # df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    df["Duration_Heart"] = df["Duration"] * df["Heart_Rate"]
    df["Duration_Temp"] = df["Duration"] * df["Body_Temp"]
    df["log_Duration"] = np.log1p(df["Duration"])
    df["log_Heart_Rate"] = np.log1p(df["Heart_Rate"])
    df = pd.concat([df, pd.get_dummies(df["Sex"], prefix="Sex")], axis=1)
    df.drop(columns=["Sex"], inplace=True, errors="ignore")
    return df

def load_and_preprocess(train_path="data/train.csv", test_path="data/test.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    y_train = np.log1p(train["Calories"])  # log1p 적용
    train.drop(columns=["Calories"], inplace=True)

    test_ids = test["id"]

    train = create_features(train)
    test = create_features(test)

    train.drop(columns=["id"], inplace=True, errors="ignore")
    test.drop(columns=["id"], inplace=True, errors="ignore")

    numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
                    'Duration_Heart', 'Duration_Temp', 'log_Duration', 'log_Heart_Rate']

    scaler = StandardScaler()
    train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
    test[numeric_cols] = scaler.transform(test[numeric_cols])

    return train, y_train, test, test_ids
