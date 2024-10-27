from feature_engineering import *
from data_preprocessing import *
from sklearn.linear_model import Ridge


def approach2(train, val, test):
    numeric_features = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',
                        'haversine_distance', 'bearing_angle', 'bearing_sin', 'bearing_cos',
                        'bearing_hour_interaction', 'bearing_dayofweek_interaction', 'is_rush_hour',
                        'is_daytime', 'is_weekend', 'rolling_mean_duration_by_hour', 'distance_hour_interaction',
                        'passenger_rush_interaction', 'log_haversine_distance'
                        ]

    categorical_features = ['dayofweek', 'month', 'hour', 'dayofyear', 'passenger_count',
                            'distance_bucket']

    train = feature_engineering_data_for_approach2(train)
    val = feature_engineering_data_for_approach2(val)
    test = feature_engineering_data_for_approach2(test)
    train = removing_furthest_outliers(train, numeric_features)
    val = removing_furthest_outliers(val, numeric_features)
    test = removing_furthest_outliers(test, numeric_features)

    features = categorical_features + numeric_features

    column_transformer = preprocessing_data(numeric_features, categorical_features)

    pipeline = pipelining_model([
        ('ohe', column_transformer),
        ('regression', Ridge(alpha=1))
    ])

    model = pipeline.fit(train[features], train.log_trip_duration)

    return model, train, val, test, features
