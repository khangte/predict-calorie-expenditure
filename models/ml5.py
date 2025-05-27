# ml5.py - CatBoost 기반 블렌딩 예측 및 제출 (파일명 및 Optuna 기준 명확화)

import json
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

# 데이터 로딩
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# One-hot encoding
train = pd.concat([train.drop('Sex', axis=1), pd.get_dummies(train['Sex'], prefix='Sex')], axis=1)
test = pd.concat([test.drop('Sex', axis=1), pd.get_dummies(test['Sex'], prefix='Sex')], axis=1)
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

# 테스트셋 준비
X_test = pd.concat([
    pd.DataFrame(scaler.transform(test[numeric_features]), columns=numeric_features),
    test[['Sex_female', 'Sex_male']]
], axis=1)

# 최적 파라미터 불러오기
with open("data/best_params_catboost.json", "r") as f:
    best_params_cat = json.load(f)
best_params_cat["random_seed"] = 42
best_params_cat["logging_level"] = "Silent"

with open("data/best_params_lgb.json", "r") as f:
    best_params_lgb = json.load(f)
best_params_lgb["random_state"] = 42

with open("data/best_params_xgb.json", "r") as f:
    best_params_xgb = json.load(f)
best_params_xgb["random_state"] = 42

# 모델 학습
cat_model = CatBoostRegressor(**best_params_cat)
cat_model.fit(X, y)
cat_pred_log = cat_model.predict(X_test)

lgb_model = LGBMRegressor(**best_params_lgb)
lgb_model.fit(X, y)
lgb_pred_log = lgb_model.predict(X_test)

xgb_model = XGBRegressor(**best_params_xgb)
xgb_model.fit(X, y)
xgb_pred_log = xgb_model.predict(X_test)

# 예측값 블렌딩 (평균)
blended_pred_log = (cat_pred_log + lgb_pred_log + xgb_pred_log) / 3
blended_pred = np.expm1(blended_pred_log)

# 제출 파일 저장
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': blended_pred
})
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"submissions/submission_catboost_blended_{current_time}.csv"
submission.to_csv(filename, index=False)
print(f"\U0001F4C1 제출 파일 저장 완료: {filename}")
