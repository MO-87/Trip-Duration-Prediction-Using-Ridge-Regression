from data_preprocessing import *
from sklearn.linear_model import Ridge


def approach1(train, val):
    numeric_features = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    categorical_features = ['dayofweek', 'month', 'hour', 'dayofyear', 'passenger_count']
    features = categorical_features + numeric_features

    column_transformer = preprocessing_data(numeric_features, categorical_features)

    pipeline = pipelining_model([
        ('ohe', column_transformer),
        ('regression', Ridge())
    ])

    model = pipeline.fit(train[features], train.log_trip_duration)

    return model, train, val, features
