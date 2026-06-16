from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier


def build_catboost_model(preprocessor):

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=0,
            random_seed=42
        ))
    ])

    return model