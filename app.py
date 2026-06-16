import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostClassifier


st.set_page_config(page_title="Оценка кредитоспособности", page_icon="💳", layout="wide")


@st.cache_resource
def load_and_train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "default_of_credit_card_clients.xls")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Файл датасета не найден: {data_path}. "
            f"Положи default_of_credit_card_clients.xls в папку data рядом с этим файлом."
        )

    df = pd.read_excel(data_path, header=1)

    target_column = "default payment next month"
    df = df.drop(columns=["ID"])

    X = df.drop(columns=[target_column])
    y = df[target_column]

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
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=0,
        random_seed=42,
    )
    model.fit(X_processed, y)

    return model, preprocessor


def make_prediction(model, preprocessor, client_data: dict):
    input_df = pd.DataFrame([client_data])

    expected_columns = list(preprocessor.feature_names_in_)
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    input_processed = preprocessor.transform(input_df)

    if hasattr(input_processed, "toarray"):
        input_processed = input_processed.toarray()

    probability_default = float(model.predict_proba(input_processed)[0][1])
    prediction = int(probability_default >= 0.5)

    return prediction, probability_default


st.title("Система оценки кредитоспособности клиента")
st.write(
    "Интерфейс использует модель CatBoost для оценки вероятности дефолта клиента. "
    "По результатам расчёта система показывает вероятность дефолта и итоговый вывод о кредитоспособности."
)

st.info(
    "Инструкция по использованию:\n"
    "1. Введите основные данные клиента.\n"
    "2. Укажите историю просрочек за последние месяцы.\n"
    "3. Заполните суммы задолженности по счетам и суммы фактических платежей.\n"
    "4. Нажмите кнопку «Оценить кредитоспособность».\n"
    "5. Система рассчитает вероятность дефолта и выдаст итоговое заключение."
)

st.markdown(
    """
    **Пояснение:**
    - вероятность дефолта ниже 0.5 означает, что клиент считается кредитоспособным;
    - вероятность дефолта 0.5 и выше означает, что клиент относится к группе риска.
    """
)

with st.spinner("Загрузка данных и обучение модели..."):
    model, preprocessor = load_and_train_model()

st.success("Модель готова к прогнозированию.")

left, right = st.columns(2)

with left:
    st.subheader("Общие данные клиента")
    limit_bal = st.number_input("Размер кредитного лимита (LIMIT_BAL)", min_value=0, value=50000, step=1000)
    sex = st.selectbox("Пол (SEX)", options=[1, 2], format_func=lambda x: "Мужской" if x == 1 else "Женский")
    education = st.selectbox(
        "Образование (EDUCATION)",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Магистратура / аспирантура",
            2: "Университет",
            3: "Школа",
            4: "Другое",
        }[x],
    )
    marriage = st.selectbox(
        "Семейное положение (MARRIAGE)",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Женат / замужем", 2: "Холост / не замужем", 3: "Другое"}[x],
    )
    age = st.number_input("Возраст клиента (AGE)", min_value=18, max_value=100, value=30, step=1)

    st.subheader("История просрочек")
    pay_0 = st.number_input("Статус погашения в последнем месяце (PAY_0)", min_value=-1, max_value=9, value=0, step=1)
    pay_2 = st.number_input("Статус погашения два месяца назад (PAY_2)", min_value=-1, max_value=9, value=0, step=1)
    pay_3 = st.number_input("Статус погашения три месяца назад (PAY_3)", min_value=-1, max_value=9, value=0, step=1)
    pay_4 = st.number_input("Статус погашения четыре месяца назад (PAY_4)", min_value=-1, max_value=9, value=0, step=1)
    pay_5 = st.number_input("Статус погашения пять месяцев назад (PAY_5)", min_value=-1, max_value=9, value=0, step=1)
    pay_6 = st.number_input("Статус погашения шесть месяцев назад (PAY_6)", min_value=-1, max_value=9, value=0, step=1)

with right:
    st.subheader("Суммы по счетам")
    bill_amt1 = st.number_input("Сумма задолженности за последний месяц (BILL_AMT1)", value=20000, step=1000)
    bill_amt2 = st.number_input("Сумма задолженности два месяца назад (BILL_AMT2)", value=18000, step=1000)
    bill_amt3 = st.number_input("Сумма задолженности три месяца назад (BILL_AMT3)", value=15000, step=1000)
    bill_amt4 = st.number_input("Сумма задолженности четыре месяца назад (BILL_AMT4)", value=12000, step=1000)
    bill_amt5 = st.number_input("Сумма задолженности пять месяцев назад (BILL_AMT5)", value=10000, step=1000)
    bill_amt6 = st.number_input("Сумма задолженности шесть месяцев назад (BILL_AMT6)", value=8000, step=1000)

    st.subheader("Суммы платежей")
    pay_amt1 = st.number_input("Сумма фактического платежа за последний месяц (PAY_AMT1)", value=5000, step=500)
    pay_amt2 = st.number_input("Сумма фактического платежа два месяца назад (PAY_AMT2)", value=4000, step=500)
    pay_amt3 = st.number_input("Сумма фактического платежа три месяца назад (PAY_AMT3)", value=3000, step=500)
    pay_amt4 = st.number_input("Сумма фактического платежа четыре месяца назад (PAY_AMT4)", value=3000, step=500)
    pay_amt5 = st.number_input("Сумма фактического платежа пять месяцев назад (PAY_AMT5)", value=2000, step=500)
    pay_amt6 = st.number_input("Сумма фактического платежа шесть месяцев назад (PAY_AMT6)", value=2000, step=500)

client_data = {
    "LIMIT_BAL": limit_bal,
    "SEX": sex,
    "EDUCATION": education,
    "MARRIAGE": marriage,
    "AGE": age,
    "PAY_0": pay_0,
    "PAY_2": pay_2,
    "PAY_3": pay_3,
    "PAY_4": pay_4,
    "PAY_5": pay_5,
    "PAY_6": pay_6,
    "BILL_AMT1": bill_amt1,
    "BILL_AMT2": bill_amt2,
    "BILL_AMT3": bill_amt3,
    "BILL_AMT4": bill_amt4,
    "BILL_AMT5": bill_amt5,
    "BILL_AMT6": bill_amt6,
    "PAY_AMT1": pay_amt1,
    "PAY_AMT2": pay_amt2,
    "PAY_AMT3": pay_amt3,
    "PAY_AMT4": pay_amt4,
    "PAY_AMT5": pay_amt5,
    "PAY_AMT6": pay_amt6,
}

if st.button("Оценить кредитоспособность", use_container_width=True):
    prediction, probability_default = make_prediction(model, preprocessor, client_data)
    creditworthy = prediction == 0

    st.subheader("Результат оценки")
    st.metric("Вероятность дефолта", f"{probability_default * 100:.2f}%")

    if creditworthy:
        st.success("Клиент кредитоспособен. Риск дефолта низкий.")
    else:
        st.error("Клиент относится к группе риска. Вероятность дефолта высокая.")

    st.write("### Интерпретация")
    st.write(
        "Решение формируется на основе порога 0.5: если вероятность дефолта меньше 0.5, "
        "клиент считается кредитоспособным; если 0.5 и выше — клиент относится к рискованным."
    )

with st.expander("Пояснение по шкале просрочек (PAY_0 ... PAY_6)"):
    st.write(
        "-1 — платёж внесён вовремя; 0 — отсутствие просрочки; "
        "1–9 — задержка платежа на соответствующее число месяцев."
    )

st.caption("Для практической реализации выбрана модель CatBoost как показавшая лучший результат по итогам сравнения моделей.")