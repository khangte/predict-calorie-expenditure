
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# 데이터 불러오기
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# log1p 변환
train['Calories'] = np.log1p(train['Calories'])

# 피처 구성
drop_cols = ['id', 'Calories']
numeric_feats = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_feats = train.select_dtypes(include=['object']).columns.tolist()
numeric_feats = [col for col in numeric_feats if col not in drop_cols]
categorical_feats = [col for col in categorical_feats if col not in drop_cols]
main_features = numeric_feats + categorical_feats

# # 조합 피처
# train['HR_Duration'] = train['Heart_Rate'] * train['Duration']
# train['BT_Duration'] = train['Body_Temp'] * train['Duration']
# test['HR_Duration'] = test['Heart_Rate'] * test['Duration']
# test['BT_Duration'] = test['Body_Temp'] * test['Duration']
# main_features += ['HR_Duration', 'BT_Duration']
# numeric_feats += ['HR_Duration', 'BT_Duration']

# 데이터 분리
X = train[main_features]
y = train['Calories']
X_test = test[main_features]

# 전처리
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

# KFold 설정
fold_cnt = 5
kf = KFold(n_splits=fold_cnt, shuffle=True, random_state=42)

# OOF 및 테스트 예측 저장
oof_cat, oof_lgb, oof_xgb = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
test_pred_cat, test_pred_lgb, test_pred_xgb = [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}/{fold_cnt}")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val = X.iloc[val_idx]

    cat_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', CatBoostRegressor(**best_params_cat))
    ])
    lgb_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LGBMRegressor(**best_params_lgb))
    ])
    xgb_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(**best_params_xgb))
    ])

    cat_pipe.fit(X_train, y_train)
    lgb_pipe.fit(X_train, y_train)
    xgb_pipe.fit(X_train, y_train)

    oof_cat[val_idx] = cat_pipe.predict(X_val)
    oof_lgb[val_idx] = lgb_pipe.predict(X_val)
    oof_xgb[val_idx] = xgb_pipe.predict(X_val)

    test_pred_cat.append(cat_pipe.predict(X_test))
    test_pred_lgb.append(lgb_pipe.predict(X_test))
    test_pred_xgb.append(xgb_pipe.predict(X_test))

# 스태킹 메타모델 학습
stack_X = np.vstack([oof_cat, oof_lgb, oof_xgb]).T
stack_X_test = np.vstack([
    np.mean(test_pred_cat, axis=0),
    np.mean(test_pred_lgb, axis=0),
    np.mean(test_pred_xgb, axis=0)
]).T

# 메타모델 선정
meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
# meta_model = LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)

meta_model.fit(stack_X, y)

# RMSLE 평가
stack_oof_pred = meta_model.predict(stack_X)
stack_oof_pred_actual = np.expm1(stack_oof_pred)

y_actual = np.expm1(y)
rmsle = np.sqrt(mean_squared_log_error(y_actual, stack_oof_pred_actual))
print(f"\nStacking RMSLE: {rmsle:.4f}")

# 테스트 예측 및 저장
stacked_pred_log = meta_model.predict(stack_X_test)
stacked_pred = np.expm1(stacked_pred_log)

submission = pd.DataFrame({
    "id": test['id'],
    "Calories": stacked_pred
})
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"submissions/submission_stacking_kf{fold_cnt}_{current_time}.csv"
submission.to_csv(filename, index=False)
print(f"\n제출 파일 저장 완료: {filename}")
