import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    y_true_safe = np.where(np.array(y_true) == 0, 1e-9, y_true)
    mape = float(np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / y_true_safe)) * 100)
    return {"MAE": float(mae), "RMSE": rmse, "MAPE": mape}
