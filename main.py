from data_preparing import *
from models_projection import *
from visualizing_models_performance import *

import pandas as pd
import warnings

warnings.filterwarnings("ignore")
pd.options.display.max_columns = None
pd.options.display.max_rows = None


if __name__ == '__main__':
    train = pd.read_csv('Sample_Data/train.csv')
    val = pd.read_csv('Sample_Data/val.csv')
    test = pd.read_csv('Sample_Data/test.csv')

    prepare_data(train)
    prepare_data(val)
    prepare_data(test)


    rmse, r2 = [], []

    print("Approach #1 stats:")
    rmse, r2 = model_approach_1(rmse, r2, train, val, test)

    print("\nApproach #2 stats:")
    rmse, r2 = model_approach_2(rmse, r2, train, val, test)

    print("\nApproach #3 stats:")
    rmse, r2 = model_approach_3(rmse, r2, train, val, test)


    visualize(rmse, r2)

