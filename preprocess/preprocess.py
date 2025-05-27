# preprocess.py - 공통 전처리 함수 정의

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess():
    # 데이터 로딩
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    # One-hot encoding for 'Sex'
    train = pd.concat([train.drop("Sex", axis=1), pd.get_dummies(train["Sex"], prefix="Sex")], axis=1)
    test = pd.concat([test.drop("Sex", axis=1), pd.get_dummies(test["Sex"], prefix="Sex")], axis=1)
    for col in ['Sex_female', 'Sex_male']:
        if col not in test.columns:
            test[col] = 0

    # 수치형 특성
    numeric_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train[numeric_features])
    X = pd.concat([
        pd.DataFrame(X_scaled, columns=numeric_features),
        train[['Sex_female', 'Sex_male']]
    ], axis=1)
    y = np.log1p(train['Calories'])

    # 테스트셋 처리
    X_test = pd.concat([
        pd.DataFrame(scaler.transform(test[numeric_features]), columns=numeric_features),
        test[['Sex_female', 'Sex_male']]
    ], axis=1)

    return X, y, X_test, test['id']
