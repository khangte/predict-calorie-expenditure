
# ml_pipeline_add_feature_rmsle_safe_fixed.py - 조합 피처 + RMSLE 평가 + 중복 제거
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb
import warnings
import joblib

warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# 데이터 불러오기
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# log1p 적용 (RMSLE 기준)
train["Calories"] = np.log1p(train["Calories"])

# 조합 피처 생성
train['HR_Duration'] = train['Heart_Rate'] * train['Duration']
train['BT_Duration'] = train['Body_Temp'] * train['Duration']
test['HR_Duration'] = test['Heart_Rate'] * test['Duration']
test['BT_Duration'] = test['Body_Temp'] * test['Duration']

# 피처 구성
drop_cols = ['id', 'Calories']
numeric_feats = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_feats = train.select_dtypes(include=['object']).columns.tolist()

# 중복 방지: drop_cols 제거
numeric_feats = [col for col in numeric_feats if col not in drop_cols]
categorical_feats = [col for col in categorical_feats if col not in drop_cols]

# 조합 피처 중복 없이 추가
for feat in ['HR_Duration', 'BT_Duration']:
    if feat not in numeric_feats:
        numeric_feats.append(feat)

main_features = numeric_feats + categorical_feats

# 훈련/검증 데이터 분할
X = train[main_features]
y = train["Calories"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 전처리 파이프라인 구성
numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_feats),
    ('cat', categorical_transformer, categorical_feats)
])

# 모델 후보군
models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'XGBoost': xgb.XGBRegressor(tree_method='hist', random_state=42),
    'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
    'CatBoost': CatBoostRegressor(verbose=0, random_state=42)
}

# 학습 및 평가
results = {}
best_model = None
best_rmsle = float('inf')

for name, model in models.items():
    print(f'\n==== {name} ====')
    try:
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('reg', model)
        ])
        pipe.fit(X_train, y_train)
        y_val_pred_log = pipe.predict(X_val)

        # 복원 후 RMSLE 계산
        y_val_pred = np.expm1(y_val_pred_log)
        y_val_actual = np.expm1(y_val)
        rmsle = np.sqrt(mean_squared_log_error(y_val_actual, y_val_pred))
        print(f'Validation RMSLE: {rmsle:.4f}')

        results[name] = rmsle
        if rmsle < best_rmsle:
            best_rmsle = rmsle
            best_model = pipe
    except Exception as e:
        print(f"Error in {name}: {str(e)}")

# 예외 처리: 학습 실패한 경우
if best_model is None:
    raise ValueError("❌ No model was successfully trained. Check logs above.")

# 결과 정리 출력
print("\n==== Final Results ====")
results_df = pd.DataFrame(results.items(), columns=["Model", "RMSLE"])
results_df = results_df.sort_values("RMSLE")
print(results_df)

# 테스트 예측
print("\n==== Test Data Prediction ====")
X_test = test[main_features]
test_pred_log = best_model.predict(X_test)
test_pred = np.expm1(test_pred_log)

# 제출 파일 생성
submission = pd.DataFrame({
    "id": test["id"],
    "Calories": test_pred
})
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission_filename = f"submissions/submission_{timestamp}.csv"
submission.to_csv(submission_filename, index=False)
print(f"\n📁 제출 파일 저장 완료: {submission_filename}")

# 모델 저장
model_filename = f"models/best_model_{timestamp}.joblib"
joblib.dump(best_model, model_filename)
print(f"💾 최고 성능 모델 저장 완료: {model_filename}")
