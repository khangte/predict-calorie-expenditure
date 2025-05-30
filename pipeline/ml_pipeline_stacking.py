import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error
import json

# 데이터 불러오기
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
train['Calories'] = np.log1p(train['Calories'])

# # 조합 피처
# train['HR_Duration'] = train['Heart_Rate'] * train['Duration']
# train['BT_Duration'] = train['Body_Temp'] * train['Duration']
# test['HR_Duration'] = test['Heart_Rate'] * test['Duration']
# test['BT_Duration'] = test['Body_Temp'] * test['Duration']

# 피처 분리
drop_cols = ['id', 'Calories']
numeric_feats = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_feats = train.select_dtypes(include=['object']).columns.tolist()
numeric_feats = [col for col in numeric_feats if col not in drop_cols]
categorical_feats = [col for col in categorical_feats if col not in drop_cols]

# for feat in ['HR_Duration', 'BT_Duration']:
#     if feat not in numeric_feats:
#         numeric_feats.append(feat)

main_features = numeric_feats + categorical_feats

X = train[main_features]
y = train['Calories']
X_test = test[main_features]

# 전처리기 정의
numeric_transformer = Pipeline([('scaler', StandardScaler())])
categorical_transformer = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_feats),
    ('cat', categorical_transformer, categorical_feats)
])

# 최적 파라미터 로딩
with open("data/best_params_catboost.json") as f:
    best_params_cat = json.load(f)
best_params_cat["random_seed"] = 42
best_params_cat["logging_level"] = "Silent"

with open("data/best_params_lgb.json") as f:
    best_params_lgb = json.load(f)
best_params_lgb["random_state"] = 42

with open("data/best_params_xgb.json") as f:
    best_params_xgb = json.load(f)
best_params_xgb["random_state"] = 42

# 개별 모델 정의
estimators = [
    ('cat', CatBoostRegressor(**best_params_cat)),
    ('lgb', LGBMRegressor(**best_params_lgb)),
    ('xgb', XGBRegressor(**best_params_xgb))
]

# 메타모델 정의
final_estimator = RidgeCV(alphas=[0.1, 1.0, 10.0])

# 스태킹 모델 구성
stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator,
    passthrough=False,
    cv=5,
    n_jobs=-1
)

# 전체 파이프라인 구성
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('stack', stacking_model)
])

# 학습 및 예측
model_pipeline.fit(X, y)

y_pred_log = model_pipeline.predict(X)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y)
rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
print(f"\n📊 Stacking RMSLE (with StackingRegressor): {rmsle:.4f}")

# 테스트셋 예측 및 제출
y_test_pred = np.expm1(model_pipeline.predict(X_test))
submission = pd.DataFrame({'id': test['id'], 'Calories': y_test_pred})
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
submission.to_csv(f"submissions/submission_stacking_regressor_{current_time}.csv", index=False)
print("✅ 제출 완료")
