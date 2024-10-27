import numpy as np
import pandas as pd


def feature_engineering_data_for_approach2(data):
    data['haversine_distance'], data['bearing_angle'] = haversine_distance_and_bearing_angle(data['pickup_latitude'],
                                                                                             data['pickup_longitude'],
                                                                                             data['dropoff_latitude'],
                                                                                             data['dropoff_longitude'])

    data['log_haversine_distance'] = np.log1p(data['haversine_distance'])


    data['bearing_sin'] = np.sin(np.radians(data['bearing_angle']))
    data['bearing_cos'] = np.cos(np.radians(data['bearing_angle']))

    data['bearing_hour_interaction'] = data['bearing_angle'] * data['hour']
    data['bearing_hour_interaction'] = np.log1p(data['bearing_hour_interaction'])
    data['bearing_dayofweek_interaction'] = data['bearing_angle'] * data['dayofweek']
    data['bearing_dayofweek_interaction'] = np.log1p(data['bearing_dayofweek_interaction'])

    data['is_rush_hour'] = data['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0)
    data['is_daytime'] = data['hour'].apply(lambda x: 1 if 6 <= x <= 20 else 0)

    data['is_weekend'] = data['dayofweek'].apply(lambda x: 1 if x in [5, 6] else 0)

    data['distance_hour_interaction'] = data['haversine_distance'] * data['hour']
    data['passenger_rush_interaction'] = data['passenger_count'] * data['is_rush_hour']

    data['distance_rush_interaction'] = data['haversine_distance'] * data['is_rush_hour']
    data['hour_day_interaction'] = data['hour'] * data['dayofweek']

    data['distance_bucket'] = pd.cut(data['haversine_distance'], bins=[0, 1, 3, 5, 10, np.inf],
                                     labels=[1, 2, 3, 4, 5])


    data['rolling_mean_duration_by_hour'] = data.groupby('hour')['trip_duration'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean())


    return data


def removing_furthest_outliers(data, numeric_features):
    z_scores = np.abs((data[numeric_features] - data[numeric_features].mean()) / data[numeric_features].std())
    data = data[(z_scores < 5).all(axis=1)]

    return data


def haversine_distance_and_bearing_angle(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    initial_bearing = np.arctan2(x, y)
    bearing = (np.degrees(initial_bearing) + 360) % 360

    # Radius of Earth in kilometers (mean value)
    r = 6371
    return c * r, bearing


def manhattan_distance(lat1, lon1, lat2, lon2):
    return np.abs(lat1 - lat2) + np.abs(lon1 - lon2)


def euclidean_distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)
