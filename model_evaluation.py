from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def predict_eval(model, data, features, name):
    y_preds = model.predict(data[features])
    rmse = mean_squared_error(data.log_trip_duration, y_preds, squared=False)
    r2 = r2_score(data.log_trip_duration, y_preds)
    print(f"{name} RMSE = {rmse:.4f} - {name} R2 = {r2:.4f}")

    return rmse, r2
