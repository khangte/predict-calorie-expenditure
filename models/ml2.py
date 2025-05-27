# ml2.py - 저장된 best_params.json을 사용해 학습 및 제출 파일 생성

import json
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# 데이터 로딩
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
train = pd.concat([train.drop('Sex', axis=1), pd.get_dummies(train['Sex'], prefix='Sex')], axis=1)
test = pd.concat([test.drop('Sex', axis=1), pd.get_dummies(test['Sex'], prefix='Sex')], axis=1)
for col in ['Sex_female', 'Sex_male']:
    if col not in test.columns:
        test[col] = 0

# 스케일링 및 특성 준비
# train 데이터는 fit_transform 스케일러 학습 후 표준화 함
numeric_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train[numeric_features])
X = pd.concat([
    pd.DataFrame(X_scaled, columns=numeric_features),
    train[['Sex_female', 'Sex_male']]
], axis=1)
y = np.log1p(train['Calories'])

# test 데이터는 transform 표준화만 함
X_test = pd.concat([
    pd.DataFrame(scaler.transform(test[numeric_features]), columns=numeric_features),
    test[['Sex_female', 'Sex_male']]
], axis=1)

# 저장된 최적 파라미터 불러오기
with open("data/best_params_catboost.json", "r") as f:
    best_params = json.load(f)
best_params["random_seed"] = 42
best_params["logging_level"] = "Silent"

# 모델 학습 및 예측
model = CatBoostRegressor(**best_params)
model.fit(X, y)
test_pred_log = model.predict(X_test)
test_pred = np.expm1(test_pred_log)

# 제출 파일 저장
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': test_pred
})
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"submissions/submission_catboost_optuna_{current_time}.csv"
submission.to_csv(filename, index=False)
print(f"\U0001F4C1 제출 파일 저장 완료: {filename}")
