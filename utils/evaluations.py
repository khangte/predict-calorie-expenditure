import numpy as np
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score

def evaluate(name, model, X_val, y_val_log):
    y_pred_log = model.predict(X_val)
    y_true = np.expm1(y_val_log)
    y_pred = np.expm1(y_pred_log)

    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"[{name}] RMSLE: {rmsle:.5f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
    return name, rmsle, model
