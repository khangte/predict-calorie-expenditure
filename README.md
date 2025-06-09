# 📊 Predict Calorie Expenditure  
[Playground Series - Season 5, Episode 5](https://www.kaggle.com/competitions/playground-series-s5e5/)

---

## 최종 결과
| 항목         | 내용              |
|--------------|-------------------|
| Rank         | **365** / 4316        |
| Score        | 0.05865           |
| 선정 실험    | 8번째 실험        |


### ✅ 실험 요약

- **베이스 모델**: CatBoost, LightGBM, XGBoost  
- **메타 모델**: RidgeCV  
- **Stacking 방식**: K-Fold 기반 OOF stacking (5-Fold)  
- **하이퍼파라미터 최적화**: Optuna를 이용해 각 베이스 모델의 최적 파라미터 사전 탐색  
- **파라미터 적용 방식**: JSON 파일로 저장된 Optuna 결과 불러와 각 모델에 적용  
- **평가 지표**: 로그 스케일 RMSLE 사용  
- **예측 후처리**: `np.expm1()`으로 로그 예측값을 원래 스케일로 복원  
- **성능 결과**: 가장 낮은 RMSLE을 기록한 최고의 성능 모델  

---

| 실험 번호 | 파일명                              | 주요 내용                                      | RMSLE   | Score   | 제출 파일명                                        |
|-----------|--------------------------------------|-----------------------------------------------|---------|---------|---------------------------------------------------|
| 1         | `ml.py`                              | 7가지 모델 비교 → CatBoost 선택               | 0.0595  | 0.05755 | `submission_CatBoost_20250526_163800.csv`         |
| 2         | `ml2.py + optuna_tune_catboost.py`   | Optuna를 사용한 CatBoost 튜닝                | 0.0592  | 0.05739 | `submission_catboost_optuna_20250526_173823.csv`  |
| 3         | `ml3.py`                             | BMI 파생 변수 추가 후 CatBoost 선택          | 0.05919 | 0.05746 | `submission_bmi_catboost_20250526_222929.csv`     |
| 4         | `ml4.py`                             | BMI 유지 + Optuna 하이퍼파라미터 비교         | 0.05919 | 0.05746 | `submission_bmi_catboost_20250527_120910.csv`     |
| 5         | `ml5.py`                             | 3개 모델 예측 평균 블렌딩                    |         | 0.05713 | `submission_catboost_blended_20250526_1759.csv`   |
| 6         | `ml6.py`                             | RMSLE 기반 가중 평균 블렌딩                  |         | 0.05713 | `submission_weighted_blend_20250527_122812.csv`   |
| 7         | `ml7_kfold_blend.py`                 | KFold + RMSLE 정규화 가중 평균               |         | 0.05703 | `submission_kfold_blend_20250527_124155.csv`      |
| **8**     | **`ml8_stacking.py`**                | **KFold OOF → RidgeCV 메타모델 stacking**    |         | **0.05698** | **`submission_stacking_20250527_124906.csv`**     |
| 9         | `ml9_gender_split.py`                | 성별 기반 분할 학습                          | 0.06851 |         | 미제출                                               |
| 10        | `ml10_sexbmi_split.py`               | 성별 + BMI 구간 조합, 가중 평균              |         | 0.05754 | `submission_sexbmi_split_20250527_142906.csv`     |
| 11        | `ml11_sexbmi_2group.py`              | 성별 + BMI 2구간 → 4그룹                    | 0.05948 |         | 미제출                                               |
| 12        | `ml8_stacking.py`                    | KFold 10분할 RidgeCV 스태킹                  | 0.0592  | 0.05713 | `submission_stacking_kf5_20250528_133852.csv`         |
| 13        | `ml13_stacking_with_rf.py`           | RandomForest 추가 + RidgeCV                  | 0.05917 | 0.05945 | `submission_stacking_kf5_20250528_161452.csv`      |
| 14        | `ml_pipeline.py`            | 메타모델 LGBM, 조합 특성 추가               | 0.0597  | 0.05914 | `submission_20250529_122336.csv`     |
| 15        | `ml_pipeline_stacking_improved.py`   | 예측값 차이 피처 추가                        | 0.0593  | 0.05708 | `submission_stacking_improved_20250529_162009.csv`|
| 16        | `ml_pipeline_stacking_improved.py`   | 조합 피처: BMI, Temp_per_Duration 추가       | 0.0595  | 0.05708 | 미제출                                               |
| 17        | `ml_pipeline_stacking.py`            | StackingRegressor 사용, RidgeCV 메타모델     | 0.0556  | 0.05711 | `submission_stacking_regressor_20250530_112240.csv`         |

