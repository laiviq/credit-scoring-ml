import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler



def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, header=1)
    return df


def prepare_features_and_target(df: pd.DataFrame):
    df = df.copy()

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    target_column = "default payment next month"

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def split_data(X, y, random_state: int = 42):

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=random_state,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.1765,
        random_state=random_state,
        stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor():
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

    numerical_features = [
        "LIMIT_BAL", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )


    return preprocessor
