# Predict Calorie Expenditure
[Playground Series - Season 5, Episode 5](https://www.kaggle.com/competitions/playground-series-s5e5/)

---

- 첫 번째 실험
    + ```ml.py```
        - [다중회귀, 결정트리, 경사하강법, LightGBM, XGBoost, CatBoost, 딥러닝]
        - 위 7가지 방법 중 RMSLE가 가장 낮은 모델 선택
    + model: CatBoost
    + RMSLE: 0.0595
    + score: **0.05755**
    + 제출파일: submission_CatBoost_20250526_163800.csv

- 두번째 실험
    + ```ml2.py```+```optuna_tune_catboost.py```
    + ```Optuna``` 사용
        - '첫번째 제출'에서 선정된 모델(catboost)의 하이퍼파라미터를 자동으로 조합
        - 30번 모델 학습
        - 평가지표가 가장 낮은 최적의 하이퍼파라미터 조합을 찾음
    + RMSLE: 0.0592
    + score: **0.05739** (성능 향상)
    + 제출파일: `submission_catboost_optuna_20250526_173823.csv`

- 세번째 실험
    + ```ml3.py```
    + BMI 파생 변수 추가, Optuna 튜닝된 CatBoost + LightGBM 비교
    + 더 낮은 RMSLE 모델 선택 (CatBoost)
    + 출력결과:
        - CatBoost RMSLE: 0.05919
        - LightGBM RMSLE: 0.06211
    + score: **0.05746** (성능 미세 하락)
    + 제출파일: `submission_bmi_catboost_20250526_222929.csv`

- 네 번째 실험
    + `ml4.py`
    + BMI = Weight / (Height/100)^2 파생 변수 추가
    + Optuna에서 저장된 최적 하이퍼파라미터(`best_params_catboost.json`, `best_params_lgb.json`, `best_params_xgb.json`)를 이용해
    + CatBoost, LightGBM, XGBoost 모델을 모두 학습
    + 검증 데이터에서 RMSLE가 가장 낮은 모델을 자동으로 선정
    + 출력 결과:
        - CatBoost RMSLE:  0.05919 <= 이전 결과 동일
        - LightGBM RMSLE: 0.05995
        - XGBoost RMSLE:  0.06022
    + 선택된 모델: **CatBoost**
    + 최종 Score: **0.05746** (이전과 동일)
    + 제출 파일: `submission_bmi_catboost_20250527_120910.csv`
    + 미제출

- 다섯 번째 실험
    + `ml5.py`
    + Optuna를 통해 튜닝한 `best_params_catboost.json`을 기반으로 CatBoost, LightGBM, XGBoost 세 모델을 학습
    + 각각의 예측 결과를 log 스케일에서 평균내어 블렌딩 수행
    + 출력 결과 (개별 모델 RMSLE는 측정하지 않음, 블렌딩 기반 제출): - blending 방식: (CatBoost + LightGBM + XGBoost) / 3
    + score: **0.05713** (성능 미세 향상)
    + 제출 파일: `submission_catboost_blended_20250526_1759.csv`

- 여섯 번째 실험
    + `ml6.py`
    + Optuna로 튜닝된 `best_params_catboost.json`, `best_params_lgb.json`, `best_params_xgb.json`을 각각 불러와 모델 학습
    + CatBoost, LightGBM, XGBoost 모델의 예측 결과를 **정규화된 RMSLE 기준으로 가중 평균 블렌딩**
        - CatBoost RMSLE: 0.05919 → 가중치: 0.3364
        - LightGBM RMSLE: 0.05995 → 가중치: 0.3326
        - XGBoost RMSLE: 0.06022 → 가중치: 0.3310
    + blending 방식: `blended_pred_log = 0.3364 * cat + 0.3326 * lgb + 0.3310 * xgb`
    + 최종 예측값을 `np.expm1()`로 복원 후 제출
    + score: **0.05713** (이전과 동일)
    + 제출 파일: `submission_weighted_blend_20250527_122812.csv`

- 일곱 번째 실험
    + `ml7_kfold_blend.py`
    + KFold(n_splits=5) 기반으로 CatBoost, LightGBM, XGBoost 모델을 각각 fold별 학습
    + 각 모델의 Fold 예측 결과를 평균한 후,
    + 정규화된 RMSLE 기반 가중치로 블렌딩
        - CatBoost RMSLE: 0.05919 → 가중치: 0.3364
        - LightGBM RMSLE: 0.05995 → 가중치: 0.3326
        - XGBoost RMSLE: 0.06022 → 가중치: 0.3310
    + blending 방식: `blended_pred_log = 0.3364 * cat + 0.3326 * lgb + 0.3310 * xgb`
    + 최종 예측값을 `np.expm1()`로 복원 후 제출
    + 최고 score: **0.05703** (성능 향상)
    + 제출 파일: `submission_kfold_blend_20250527_124155.csv`

- 여덟 번째 실험
    + `ml8_stacking.py`
    + CatBoost, LightGBM, XGBoost 모델을 대상으로 KFold(n=5)로 OOF(Out-of-Fold) 예측 생성
    + 각 모델의 OOF 예측값을 수집하여 메타 모델 입력 피처(`stack_X`) 구성
    + 테스트셋 예측 결과도 각 모델 평균 후 메타 모델 입력 생성
    + 메타 모델로 `RidgeCV` 사용하여 최종 예측 수행 (Stacking Ensemble)
    + 최종 예측값을 `np.expm1()`로 복원 후 제출
    + ✅ score: **0.05698** (성능 미세 향상, 최고 성능)
    + 제출 파일: `submission_stacking_20250527_124906.csv`

- 아홉 번째 실험 (미제출)
    + `ml9_gender_split.py`
    + 성별 분할 (male/female) 별로 모델 학습 후 병합
    + 남성의 RMSLE 높음 (0.06851) → 전체 성능 낮음 가능성

- 열 번째 실험
    + `ml10_sexbmi_split.py`
    + BMI = Weight / (Height/100)^2 계산 후,
      성별(Sex) + BMI 구간을 조합하여 **8개 그룹**으로 분할
        - 예: `male_under`, `female_normal`, ...
    + 각 그룹별로 CatBoost, LightGBM, XGBoost를 학습하여
      정규화된 가중 평균으로 예측 수행
        - 가중치: Cat: 0.3364, LGB: 0.3326, XGB: 0.3310
    + 그룹별 검증용 RMSLE를 출력하며,
      전체 test 예측 결과를 통합하여 제출
    + 최종 score: **0.05754** (성능 하락)
    + 제출 파일: `submission_sexbmi_split_20250527_142906.csv`
    + 성능하락 원인 : group 수 과도

 - 열한 번째 실험 (미제출)
    + `ml11_sexbmi_2group.py`
    + 성별 + BMI 2구간 분할 (총 4개 그룹)
    + 전체 평균 RMSLE: **0.05948**
    + 이전 stacking 방식보다 낮은 성능
    + group 수를 줄였는데도 성능 하락

- 열두 번째 실험: Stacking 앙상블 fold 5->10 증가
    - **파일명**: `ml8_stacking.py`
    - **모델**: CatBoost, LightGBM, XGBoost + RidgeCV (메타모델)
    - **전처리**: `preprocess.py`에서 공통 전처리 함수 사용
    - **튜닝**: 각 모델별로 Optuna를 활용한 best_params 적용
    - **앙상블 방식**:
    - `KFold(n_splits=10)`로 훈련 데이터를 분할하여 OOF 예측 수행
        - 각 모델의 OOF 결과를 메타모델(RidgeCV) 학습에 사용
        - 테스트 데이터는 각 모델의 예측값을 평균내어 메타모델에 입력
    - **성능 평가**: `utils/evaluations.py`의 `evaluate()` 함수로 전체 RMSLE 출력
    - **결과**:
        - 최종 RMSLE: **0.0592** (성능 하락)
        - 제출 파일: `submission_stacking_20250528_XXXXXX.csv` (날짜 및 시간 자동 생성)
    - **비고**:
        - 기존 5-Fold → 10-Fold로 증가시켜 일반화 성능 개선 유도
        - 3개 모델의 예측을 결합해 강건한 앙상블 효과 기대
    - 미제출

### 🔬 열세 번째 실험
- **파일명**: `ml13_stacking_with_rf.py`
- **주요 특징**:
  - 기존 stacking 모델(`CatBoost`, `LightGBM`, `XGBoost`)에 `RandomForestRegressor` 추가
  - 총 4개 모델의 OOF(Out-of-Fold) 예측값을 메타 특성으로 사용
  - 메타모델로 `RidgeCV` 사용
  - `KFold(n_splits=7)`로 교차검증 수행

- **변경 사항**:
  - 기존 3개 모델에 더해 `RandomForest`의 예측값을 stacking 입력에 추가
  - `utils.evaluations.evaluate` 함수로 RMSLE 출력 추가

- **성능 결과**:
  - RMSLE: **0.05917**
  - 기존 실험(`ml8_stacking.py`)과 동일한 결과

- **제출 파일**:
  - `submission_stacking_rf_YYYYMMDD_HHMMSS.csv`

- **해석**:
  - `RandomForest`를 stacking에 추가했지만 성능 향상은 없었음
  - 기존 모델들이 비슷한 예측 경향을 가져 메타모델에 기여하지 못한 것으로 보임
  - 다양한 구조/성격의 모델을 더하거나, 다른 메타모델을 고려할 필요 있음

### 열네 번째 실험
- 