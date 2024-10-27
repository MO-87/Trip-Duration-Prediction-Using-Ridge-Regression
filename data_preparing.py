import numpy as np
import pandas as pd


def prepare_data(data):
    data.drop(columns=['id'], inplace=True)

    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    data['dayofweek'] = data.pickup_datetime.dt.dayofweek
    data['month'] = data.pickup_datetime.dt.month
    data['hour'] = data.pickup_datetime.dt.hour
    data['dayofyear'] = data.pickup_datetime.dt.dayofyear

    data['log_trip_duration'] = np.log1p(data.trip_duration)
