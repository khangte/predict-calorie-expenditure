# ml3.py - BMI 파생변수 추가 후 CatBoost, LightGBM 비교 학습 및 제출

import json
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error
from datetime import datetime

# 데이터 로딩
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# BMI 파생 변수 추가
train["BMI"] = train["Weight"] / ((train["Height"] / 100) ** 2)
test["BMI"] = test["Weight"] / ((test["Height"] / 100) ** 2)

# One-hot encoding
train = pd.concat([train.drop('Sex', axis=1), pd.get_dummies(train['Sex'], prefix='Sex')], axis=1)
test = pd.concat([test.drop('Sex', axis=1), pd.get_dummies(test['Sex'], prefix='Sex')], axis=1)
for col in ['Sex_female', 'Sex_male']:
    if col not in test.columns:
        test[col] = 0

# 수치형 특성 + BMI
numeric_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']

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

# 저장된 최적 CatBoost 파라미터 불러오기
with open("data/best_params_catboost.json", "r") as f:
    best_params = json.load(f)
best_params["random_seed"] = 42
best_params["logging_level"] = "Silent"

# CatBoost 학습 및 예측
cat_model = CatBoostRegressor(**best_params)
cat_model.fit(X, y)
cat_pred_log = cat_model.predict(X_test)
cat_pred = np.expm1(cat_pred_log)

# LightGBM 기본 설정 학습 및 예측
lgb_model = LGBMRegressor(random_state=42)
lgb_model.fit(X, y)
lgb_pred_log = lgb_model.predict(X_test)
lgb_pred = np.expm1(lgb_pred_log)

# 검증 데이터로 RMSLE 비교
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

cat_model.fit(X_train, y_train)
y_val_actual = np.expm1(y_val)
y_pred_cat = np.expm1(cat_model.predict(X_val))
rmsle_cat = np.sqrt(mean_squared_log_error(y_val_actual, y_pred_cat))

lgb_model.fit(X_train, y_train)
y_pred_lgb = np.expm1(lgb_model.predict(X_val))
rmsle_lgb = np.sqrt(mean_squared_log_error(y_val_actual, y_pred_lgb))

print(f"CatBoost RMSLE: {rmsle_cat:.5f}")
print(f"LightGBM RMSLE: {rmsle_lgb:.5f}")

# 더 좋은 모델 선택
final_pred = cat_pred if rmsle_cat < rmsle_lgb else lgb_pred
best_model_name = "catboost" if rmsle_cat < rmsle_lgb else "lightgbm"

# 제출 파일 저장
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': final_pred
})
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"submissions/submission_bmi_{best_model_name}_{current_time}.csv"
submission.to_csv(filename, index=False)
print(f"\U0001F4C1 제출 파일 저장 완료: {filename}")
