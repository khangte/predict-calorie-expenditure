# Predict Calorie Expenditure
[Playground Series - Season 5, Episode 5](https://www.kaggle.com/competitions/playground-series-s5e5/)

---

- 첫 번째 제출
    + ```ml.py```
        - [다중회귀, 결정트리, 경사하강법, LightGBM, XGBoost, CatBoost, 딥러닝]
        - 위 7가지 방법 중 RMSLE가 가장 낮은 모델 선택
    + model: CatBoost
    + RMSLE: 0.0595
    + score: **0.05755**
    + 제출파일: submission_CatBoost_20250526_163800.csv

- 두번째 제출
    + ```ml2.py```+```optuna_tune_catboost.py```
    + ```Optuna``` 사용
        - '첫번째 제출'에서 선정된 모델(catboost)의 하이퍼파라미터를 자동으로 조합
        - 30번 모델 학습
        - 평가지표가 가장 낮은 최적의 하이퍼파라미터 조합을 찾음
    + RMSLE: 0.0592
    + score: **0.05739** (성능 향상)
    + 제출파일: `submission_catboost_optuna_20250526_173823.csv`

- 세번째 제출
    + ```ml3.py```
    + BMI 파생 변수 추가, Optuna 튜닝된 CatBoost + LightGBM 비교
    + 더 낮은 RMSLE 모델 선택 (CatBoost)
    + 출력결과:
        - CatBoost RMSLE: 0.05919
        - LightGBM RMSLE: 0.06211
    + score: **0.05746** (성능 미세 하락)
    + 제출파일: `submission_bmi_catboost_20250526_222929.csv`

- 네 번째 제출
    + `ml4.py`
    + BMI = Weight / (Height/100)^2 파생 변수 추가
    + Optuna에서 저장된 최적 하이퍼파라미터(`best_params_catboost.json`, `best_params_lgb.json`, `best_params_xgb.json`)를 이용해
    + CatBoost, LightGBM, XGBoost 모델을 모두 학습
    + 검증 데이터에서 RMSLE가 가장 낮은 모델을 자동으로 선정
    + 출력 결과:
        - CatBoost RMSLE:  0.05919 <= 이전 결과랑 같아서 제출안함
        - LightGBM RMSLE: 0.05995
        - XGBoost RMSLE:  0.06022
    + 선택된 모델: **CatBoost**
    + 최종 Score: **0.05746** (이전과 동일)
    + 제출 파일: `submission_bmi_catboost_20250527_120910.csv`

- 다섯 번째 제출
    + `ml5.py`
    + Optuna를 통해 튜닝한 `best_params_catboost.json`을 기반으로 CatBoost, LightGBM, XGBoost 세 모델을 학습
    + 각각의 예측 결과를 log 스케일에서 평균내어 블렌딩 수행
    + 출력 결과 (개별 모델 RMSLE는 측정하지 않음, 블렌딩 기반 제출): - blending 방식: (CatBoost + LightGBM + XGBoost) / 3
    + score: **0.05713** (성능 미세 향상)
    + 제출 파일: `submission_catboost_blended_20250526_1759.csv`

- 공통 전처리 함수 정의
    + `preprocess.py`
    + `load_and_preprocess()` 함수 하나로 전처리 자동화
        - train/test 로딩
        - 성별 One-Hot 인코딩
        - 수치형 피처 스케일링
        - `np.log1p()`로 타깃 변환
        - 반환값: `X`, `y`, `X_test`, `test_ids`
    + 모든 `ml*.py`에서 코드 중복 제거 및 재사용성 극대화

- 여섯 번째 제출
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

- 일곱 번째 제출
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

- 여덟 번째 제출
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

- 열 번째 제출
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
