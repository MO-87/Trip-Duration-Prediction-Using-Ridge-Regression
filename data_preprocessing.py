from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocessing_data(numeric_features, categorical_features):
    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ('scaling', StandardScaler(), numeric_features)
    ]
        , remainder='passthrough'
    )
    return column_transformer


def pipelining_model(steps):
    pipeline = Pipeline(steps)

    return pipeline
