from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def build_logreg_model(preprocessor):

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=1000,
            random_state=42
        ))
    ])

    return model