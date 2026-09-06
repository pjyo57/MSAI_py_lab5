from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

def split_data(X, y, test_size=0.2, random_state=0):
    """
    Split features and target into training and test sets.
    """
    return train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=y)


def build_preprocessor(num_cols, cat_cols):
    """
    Build preprocessing for numeric and categorical features.

    Numeric:
        median imputation -> standard scaling

    Categorical:
        most-frequent imputation -> one-hot encoding
    """

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, num_cols),
        ("categorical", categorical_pipeline, cat_cols)
    ])

    return preprocessor


def build_pipeline(num_cols, cat_cols, model):
    """
    Combine preprocessing and a model into one pipeline.
    """
    preprocessor = build_preprocessor(num_cols,cat_cols)
    return Pipeline([("preprocessor", preprocessor),("model", model)])

def engineer_features(df):
    df = df.copy()

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    return df

def get_column_groups(X):
    numeric = [
        c for c in X.select_dtypes(include=np.number).columns
        if c != "Pclass"
    ]
    categorical = [
        c for c in X.columns
        if c not in numeric
    ]
    return numeric, categorical