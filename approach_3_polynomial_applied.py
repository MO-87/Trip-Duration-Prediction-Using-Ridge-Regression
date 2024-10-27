from feature_engineering import *
from data_preprocessing import *
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


def approach3(train, val, test):
    # same as approach#2 but with polynomial features

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

    poly = PolynomialFeatures(degree=2, include_bias=True)
    poly.fit(train[numeric_features])

    train_poly = poly.transform(train[numeric_features])
    val_poly = poly.transform(val[numeric_features])
    test_poly = poly.transform(test[numeric_features])

    poly_feature_names = poly.get_feature_names_out(numeric_features)

    train_poly = pd.DataFrame(train_poly, columns=poly_feature_names, index=train.index)
    val_poly = pd.DataFrame(val_poly, columns=poly_feature_names, index=val.index)
    test_poly = pd.DataFrame(test_poly, columns=poly_feature_names, index=test.index)

    train = pd.concat([train.drop(columns=numeric_features), train_poly], axis=1)
    val = pd.concat([val.drop(columns=numeric_features), val_poly], axis=1)
    test = pd.concat([test.drop(columns=numeric_features), test_poly], axis=1)

    column_transformer = preprocessing_data(list(poly_feature_names), categorical_features)

    pipeline = pipelining_model([
        ('ohe', column_transformer),
        ('regression', Ridge(alpha=1))
    ])

    features = categorical_features + list(poly_feature_names)

    model = pipeline.fit(train[features], train.log_trip_duration)

    return model, train, val, test, features
