import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)

# 1. Geração da massa de dados sintética
X, y = make_classification(
    n_samples=500,        # Total de exemplos
    n_features=4,         # Número de atributos (variáveis continuas)
    n_informative=3,      # Atributos com relevância para a predição
    n_redundant=1,        # Atributos redundantes
    n_classes=2,          # Classificação binária (0 ou 1)
    random_state=42
)

# 2. Divisão entre treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Inicialização e treinamento do modelo Naïve Bayes Gaussiano
model = GaussianNB()
model.fit(X_train, y_train)

# 4. Avaliação do modelo
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # probabilidade da classe positiva (1)

print("=== Nível Básico: Gaussian Naïve Bayes ===")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}\n")
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 5. Matriz de Confusão
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="Blues")
plt.title("Matriz de Confusão - Naive Bayes")
plt.savefig("matriz_confusao.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------
# 6. Curva ROC
# ---------------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"Curva ROC (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Classificador aleatório")
plt.xlabel("Taxa de Falsos Positivos (FPR)")
plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
plt.title("Curva ROC - Naive Bayes")
plt.legend(loc="lower right")
plt.savefig("curva_roc.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"AUC (Área sob a curva ROC): {auc:.2f}")

# ---------------------------------------------------------
# 7. Gravar o modelo treinado em arquivo .pkl
# ---------------------------------------------------------
with open("modelo_naive_bayes.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModelo salvo em 'modelo_naive_bayes.pkl'")

# Exemplo de como carregar o modelo salvo novamente:
# with open("modelo_naive_bayes.pkl", "rb") as f:
#     modelo_carregado = pickle.load(f)
# y_pred_novo = modelo_carregado.predict(X_test)
