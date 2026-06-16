import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("default_of_credit_card_clients.xls", header=1)

df.columns = df.columns.astype(str).str.strip()

print(df.columns.tolist())
print(df.head())

target_col = "default payment next month"

class_counts = df[target_col].value_counts().sort_index()

print("Распределение классов:")
print(class_counts)

plt.figure(figsize=(7, 5))

bars = plt.bar(
    ["0 — дефолт отсутствует", "1 — дефолт произошёл"],
    class_counts.values
)

plt.title("Распределение клиентов по целевой переменной")
plt.xlabel("Класс целевой переменной")
plt.ylabel("Количество клиентов")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(height),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig("Рисунок_3_3_Распределение_классов.png", dpi=300)
plt.show()