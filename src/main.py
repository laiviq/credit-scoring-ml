import os
import numpy as np
import tensorflow as tf

from data_preprocessing import (
    load_data,
    prepare_features_and_target,
    split_data,
    build_preprocessor
)
from train_logreg import build_logreg_model
from train_catboost import build_catboost_model
from train_mlp import build_mlp_model, get_early_stopping
from evaluate import (
    calculate_metrics,
    plot_conf_matrix,
    get_roc_curve_data,
    plot_all_roc_curves,
    plot_mlp_history,
    save_metrics_table
)


def ensure_directories(outputs_dir, models_dir):
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "data", "default_of_credit_card_clients.xls")
    outputs_dir = os.path.join(base_dir, "outputs")
    models_dir = os.path.join(base_dir, "models")

    ensure_directories(outputs_dir, models_dir)

    df = load_data(file_path)

    X, y = prepare_features_and_target(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    preprocessor = build_preprocessor()

    metrics_list = []
    roc_data = {}

    logreg_model = build_logreg_model(preprocessor)
    logreg_model.fit(X_train, y_train)

    y_pred_logreg = logreg_model.predict(X_test)
    y_proba_logreg = logreg_model.predict_proba(X_test)[:, 1]

    metrics_logreg = calculate_metrics(y_test, y_pred_logreg, y_proba_logreg, "LogReg")
    metrics_list.append(metrics_logreg)

    plot_conf_matrix(y_test, y_pred_logreg, "LogReg", outputs_dir)
    roc_data["LogReg"] = get_roc_curve_data(y_test, y_proba_logreg)

    catboost_model = build_catboost_model(preprocessor)
    catboost_model.fit(X_train, y_train)

    y_pred_cat = catboost_model.predict(X_test)
    y_pred_cat = np.array(y_pred_cat).astype(int)

    y_proba_cat = catboost_model.predict_proba(X_test)[:, 1]

    metrics_cat = calculate_metrics(y_test, y_pred_cat, y_proba_cat, "CatBoost")
    metrics_list.append(metrics_cat)

    plot_conf_matrix(y_test, y_pred_cat, "CatBoost", outputs_dir)
    roc_data["CatBoost"] = get_roc_curve_data(y_test, y_proba_cat)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
        X_val_processed = X_val_processed.toarray()
        X_test_processed = X_test_processed.toarray()

    mlp_model = build_mlp_model(input_dim=X_train_processed.shape[1])

    history = mlp_model.fit(
        X_train_processed,
        y_train,
        validation_data=(X_val_processed, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[get_early_stopping()],
        verbose=1
    )

    y_proba_mlp = mlp_model.predict(X_test_processed).ravel()
    y_pred_mlp = (y_proba_mlp >= 0.5).astype(int)

    metrics_mlp = calculate_metrics(y_test, y_pred_mlp, y_proba_mlp, "MLP")
    metrics_list.append(metrics_mlp)

    plot_conf_matrix(y_test, y_pred_mlp, "MLP", outputs_dir)
    roc_data["MLP"] = get_roc_curve_data(y_test, y_proba_mlp)
    plot_mlp_history(history, outputs_dir)

    mlp_model.save(os.path.join(models_dir, "best_mlp.keras"))

    plot_all_roc_curves(roc_data, outputs_dir)
    df_metrics = save_metrics_table(metrics_list, outputs_dir)

    print("\nИтоговые метрики:")
    print(df_metrics)
    print("\nСодержимое outputs:", os.listdir(outputs_dir))
    print("Содержимое models:", os.listdir(models_dir))


if __name__ == "__main__":
    main()