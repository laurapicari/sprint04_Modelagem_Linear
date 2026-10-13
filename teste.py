import os
import glob

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

path = kagglehub.dataset_download("itsrohithere/renewable-energy-production-dataset-2010-2020")
csv_file = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)[0]

df = pd.read_csv(csv_file)
print("Formato:", df.shape)
print(df.head())
df.info()
print(df.isna().sum())


df.to_csv("renewable_energy_dataset.csv", index=False)


TARGET = "Energy_Class"
print(df[TARGET].value_counts())

le = LabelEncoder()
df["Energy_Class_enc"] = le.fit_transform(df[TARGET])
print("Mapeamento:", dict(zip(le.classes_, le.transform(le.classes_))))


df_num = df.drop(columns=[TARGET]).select_dtypes(include="number")
corr = df_num.corr()

plt.figure(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Matriz de correlação")
plt.tight_layout()
plt.savefig("matriz_correlacao.png", dpi=150)
plt.show()

print("\nCorrelação de cada feature com o target:")
print(corr["Energy_Class_enc"].drop("Energy_Class_enc")
      .sort_values(key=abs, ascending=False))


FEATURES = None

if FEATURES is None:
    FEATURES = [c for c in df_num.columns if c != "Energy_Class_enc"]
print("\nFeatures utilizadas:", FEATURES)

X = df[FEATURES]
y = df["Energy_Class_enc"]


def rodar_cenario(test_size):
    pct = int(test_size * 100)
    print("\n" + "=" * 60)
    print(f"Cenário: {pct}% para teste")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

    modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusão:\n", cm)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"Acurácia          : {acc:.4f}")
    print(f"Precisão (macro)  : {prec:.4f}")
    print(f"Recall (macro)    : {rec:.4f}")
    print("\nRelatório por classe:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_],
                                zero_division=0))

    disp = ConfusionMatrixDisplay(cm, display_labels=[str(c) for c in le.classes_])
    disp.plot(cmap="Blues")
    plt.title(f"Matriz de confusão - teste {pct}%")
    plt.savefig(f"matriz_confusão_{pct}.png", dpi=150)
    plt.show()

    return {"teste": f"{pct}%", "acurácia": acc, "precisão macro": prec, "recall macro": rec}


resultados = [rodar_cenario(0.40), rodar_cenario(0.15)]

print("\nComparação entre cenários:")
print(pd.DataFrame(resultados).round(4).to_string(index=False))