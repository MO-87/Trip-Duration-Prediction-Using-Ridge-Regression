from approach_1_Baseline_Model import *
from approach_2_feature_engineering_applied import *
from approach_3_polynomial_applied import *
from saving_loading_model import *
from model_evaluation import *


def model_approach_1(rmse, r2, train, val, test):
    approach_1_model, train_1, val_1, features = approach1(train, val)

    saving_model(approach_1_model, "models/approach_1_model.pkl")

    rmse_1, r2_1 = predict_eval(approach_1_model, train_1, features, "train")
    rmse.append(rmse_1), r2.append(r2_1)

    rmse_2, r2_2 = predict_eval(approach_1_model, val_1, features, "val")
    rmse.append(rmse_2), r2.append(r2_2)

    approach_1_loaded_model = loading_model("models/approach_1_model.pkl")

    rmse_3, r2_3 = predict_eval(approach_1_loaded_model, test, features, "test")
    rmse.append(rmse_3), r2.append(r2_3)

    return rmse, r2


def model_approach_2(rmse, r2, train, val, test):
    approach_2_model, train_2, val_2, test_2, features = approach2(train, val, test)

    saving_model(approach_2_model, "models/approach_2_model.pkl")

    rmse_4, r2_4 = predict_eval(approach_2_model, train_2, features, "train")
    rmse.append(rmse_4), r2.append(r2_4)

    rmse_5, r2_5 = predict_eval(approach_2_model, val_2, features, "val")
    rmse.append(rmse_5), r2.append(r2_5)

    approach_2_loaded_model = loading_model("models/approach_2_model.pkl")

    rmse_6, r2_6 = predict_eval(approach_2_loaded_model, test_2, features, "test")
    rmse.append(rmse_6), r2.append(r2_6)

    return rmse, r2


def model_approach_3(rmse, r2, train, val, test):
    approach_3_model, train_3, val_3, test_3, features = approach3(train, val, test)

    saving_model(approach_3_model, "models/approach_3_model.pkl")

    rmse_7, r2_7 = predict_eval(approach_3_model, train_3, features, "train")
    rmse.append(rmse_7), r2.append(r2_7)

    rmse_8, r2_8 = predict_eval(approach_3_model, val_3, features, "val")
    rmse.append(rmse_8), r2.append(r2_8)

    approach_3_loaded_model = loading_model("models/approach_3_model.pkl")

    rmse_9, r2_9 = predict_eval(approach_3_loaded_model, test_3, features, "test")
    rmse.append(rmse_9), r2.append(r2_9)

    return rmse, r2
