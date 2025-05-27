# ml4.py - 최적화된 CatBoost, LightGBM, XGBoost 모델 비교 및 제출

import json
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from datetime import datetime

# 데이터 로딩
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

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

# 검증용 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
y_val_actual = np.expm1(y_val)

# CatBoost 최적 파라미터 불러오기
with open("best_params_catboost.json", "r") as f:
    best_params_cat = json.load(f)
best_params_cat["random_seed"] = 42
best_params_cat["logging_level"] = "Silent"

cat_model = CatBoostRegressor(**best_params_cat)
cat_model.fit(X_train, y_train)
y_pred_cat = np.expm1(cat_model.predict(X_val))
rmsle_cat = np.sqrt(mean_squared_log_error(y_val_actual, y_pred_cat))

# LightGBM 최적 파라미터 불러오기
with open("best_params_lgb.json", "r") as f:
    best_params_lgb = json.load(f)
best_params_lgb["random_state"] = 42

lgb_model = LGBMRegressor(**best_params_lgb)
lgb_model.fit(X_train, y_train)
y_pred_lgb = np.expm1(lgb_model.predict(X_val))
rmsle_lgb = np.sqrt(mean_squared_log_error(y_val_actual, y_pred_lgb))

# XGBoost 최적 파라미터 불러오기
with open("best_params_xgb.json", "r") as f:
    best_params_xgb = json.load(f)
best_params_xgb["random_state"] = 42

xgb_model = XGBRegressor(**best_params_xgb)
xgb_model.fit(X_train, y_train)
y_pred_xgb = np.expm1(xgb_model.predict(X_val))
rmsle_xgb = np.sqrt(mean_squared_log_error(y_val_actual, y_pred_xgb))

# 결과 출력
print(f"CatBoost RMSLE:  {rmsle_cat:.5f}")
print(f"LightGBM RMSLE: {rmsle_lgb:.5f}")
print(f"XGBoost RMSLE:  {rmsle_xgb:.5f}")

# 가장 좋은 모델 선택
best_model_name, best_model_class, best_params = min([
    ("catboost", CatBoostRegressor, best_params_cat),
    ("lightgbm", LGBMRegressor, best_params_lgb),
    ("xgboost", XGBRegressor, best_params_xgb)
], key=lambda x: np.sqrt(mean_squared_log_error(y_val_actual, np.expm1(x[1](**x[2]).fit(X_train, y_train).predict(X_val)))))

# 전체 데이터로 재학습 및 예측
model = best_model_class(**best_params)
model.fit(X, y)
test_pred_log = model.predict(X_test)
test_pred = np.expm1(test_pred_log)

# 제출 파일 저장
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': test_pred
})
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"submission_bmi_{best_model_name}_{current_time}.csv"
submission.to_csv(filename, index=False)
print(f"\U0001F4C1 제출 파일 저장 완료: {filename}")
