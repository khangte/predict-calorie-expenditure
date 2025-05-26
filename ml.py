from datetime import datetime
import time

import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# 한글출력
plt.rcParams['font.family'] = 'Malgun Gothic' #  Windows 'Malgun Gothic' 
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로딩
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
assert train.isnull().sum().sum() == 0
assert test.isnull().sum().sum() == 0

# One-hot encoding
train = pd.concat([train.drop('Sex', axis=1), pd.get_dummies(train['Sex'], prefix='Sex')], axis=1)
test = pd.concat([test.drop('Sex', axis=1), pd.get_dummies(test['Sex'], prefix='Sex')], axis=1)
for col in ['Sex_female', 'Sex_male']:
    if col not in test.columns:
        test[col] = 0

numeric_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
scaler = StandardScaler()
X = pd.concat([
    pd.DataFrame(scaler.fit_transform(train[numeric_features]), columns=numeric_features),
    train[['Sex_female', 'Sex_male']]
], axis=1)
y_log = np.log1p(train['Calories'])

X_train, X_val, y_train, y_val = train_test_split(X, y_log, test_size=0.2, random_state=42)

# 평가 함수
def evaluate(name, model, X_val, y_val_log):
    y_pred_log = model.predict(X_val)
    y_true = np.expm1(y_val_log)
    y_pred = np.expm1(y_pred_log)
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{name}] RMSLE: {rmsle:.4f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
    return name, rmsle, model

# 모델 리스트
models = []

# 1. 다중 회귀
lr = LinearRegression()
lr.fit(X_train, y_train)
models.append(evaluate("Linear Regression", lr, X_val, y_val))

# 2. 결정 트리
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
models.append(evaluate("Decision Tree", dt, X_val, y_val))

# 3. 경사하강법
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)
models.append(evaluate("Gradient Boosting", gbr, X_val, y_val))

# 4. LightGBM
lgb_model = lgb.LGBMRegressor(random_state=42)
lgb_model.fit(X_train, y_train)
models.append(evaluate("LightGBM", lgb_model, X_val, y_val))

# 5. XGBoost
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
models.append(evaluate("XGBoost", xgb_model, X_val, y_val))

# 6. CatBoost
cat_model = cb.CatBoostRegressor(verbose=0, random_state=42)
cat_model.fit(X_train, y_train)
models.append(evaluate("CatBoost", cat_model, X_val, y_val))

# 7. 딥러닝
model_dl = Sequential([
    Input(shape=(X_train.shape[1],)),   # ← 여기 중요!
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

model_dl.compile(optimizer=Adam(0.01), loss='mse')
# 학습시간 측정 시작
start_time = time.time()
# 딥러닝 학습
model_dl.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
# 학습시간 측정 끝
end_time = time.time()
elapsed_time = end_time - start_time
print(f"⏱ 딥러닝 모델 학습 시간: {elapsed_time:.2f}초")

y_pred_dl_log = model_dl.predict(X_val).flatten()
y_val_actual = np.expm1(y_val)
y_pred_dl = np.expm1(y_pred_dl_log)
rmsle_dl = np.sqrt(mean_squared_log_error(y_val_actual, y_pred_dl))
rmse_dl = np.sqrt(mean_squared_error(y_val_actual, y_pred_dl))
r2_dl = r2_score(y_val_actual, y_pred_dl)
print(f"[Deep Learning] RMSLE: {rmsle_dl:.4f}, RMSE: {rmse_dl:.2f}, R2: {r2_dl:.2f}")
models.append(("Deep Learning", rmsle_dl, model_dl))

# ✅ 가장 RMSLE 낮은 모델 선택
best_model_name, _, best_model = sorted(models, key=lambda x: x[1])[0]
print(f"\n🎯 가장 좋은 모델: {best_model_name}")

# 테스트셋 예측
X_test = pd.concat([
    pd.DataFrame(scaler.transform(test[numeric_features]), columns=numeric_features),
    test[['Sex_female', 'Sex_male']]
], axis=1)

if best_model_name == "Deep Learning":
    test_pred_log = best_model.predict(X_test).flatten()
else:
    test_pred_log = best_model.predict(X_test)

test_pred = np.expm1(test_pred_log)

# 제출 파일 저장
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': test_pred
})
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
submission.to_csv(f"submission_{best_model_name}_{current_time}.csv", index=False)
print(f"📁 제출 완료: submission_{best_model_name}_{current_time}.csv")
