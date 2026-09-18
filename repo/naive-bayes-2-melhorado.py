import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)

# 1. Geração de massa de dados categóricos estruturados (Pandas Dataframe)
np.random.seed(42)
n_samples = 1000

dados = pd.DataFrame({
    'escolaridade': np.random.choice(['Medio', 'Superior', 'Pos-Graduacao'], size=n_samples, p=[0.4, 0.4, 0.2]),
    'renda_faixa': np.random.choice(['Baixa', 'Media', 'Alta'], size=n_samples, p=[0.3, 0.5, 0.2]),
    'historico_credito': np.random.choice(['Ruim', 'Bom', 'Excelente'], size=n_samples, p=[0.2, 0.6, 0.2]),
    'perfil_risco': np.random.choice(['Baixo', 'Alto'], size=n_samples, p=[0.7, 0.3])  # Target
})

# Separação de features (X) e rótulo (y)
X = dados[['escolaridade', 'renda_faixa', 'historico_credito']]
y = dados['perfil_risco']

categorical_cols = X.columns.tolist()

# Codifica o alvo para uso na curva ROC (Alto = 1, Baixo = 0)
le = LabelEncoder()
y_bin = le.fit_transform(y)  # classes_ = ['Alto', 'Baixo'] em ordem alfabética
classe_positiva = 'Baixo'  # ajuste se quiser tratar 'Alto' como positivo

# 2. Pré-processamento e construção da Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', CategoricalNB())
])

# 3. Validação Cruzada (5-folds) para verificar estabilidade do modelo
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

# 4. Divisão Treino/Teste e Ajuste Final
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
pipeline.fit(X_train, y_train)

# 5. Avaliação do Modelo
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)

print("=== Nível Intermediário: Categorical Naïve Bayes ===")
print(f"Média Acurácia (Cross-Validation 5-Fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")
print("Matriz de Confusão:")
cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
print(cm)
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# Matriz de Confusão (gráfico)
# ---------------------------------------------------------
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
disp.plot(cmap="Blues")
plt.title("Matriz de Confusão - Categorical Naive Bayes")
plt.savefig("matriz_confusao_categorico.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------
# Curva ROC
# ---------------------------------------------------------
idx_classe_pos = list(pipeline.classes_).index(classe_positiva)
y_test_bin = (y_test == classe_positiva).astype(int)
y_score = y_proba[:, idx_classe_pos]

fpr, tpr, _ = roc_curve(y_test_bin, y_score)
auc = roc_auc_score(y_test_bin, y_score)

plt.figure()
plt.plot(fpr, tpr, label=f"Curva ROC (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Classificador aleatório")
plt.xlabel("Taxa de Falsos Positivos (FPR)")
plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
plt.title(f"Curva ROC - Categorical Naive Bayes (classe positiva: {classe_positiva})")
plt.legend(loc="lower right")
plt.savefig("curva_roc_categorico.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"AUC (Área sob a curva ROC): {auc:.2f}")

# ---------------------------------------------------------
# Gravar o modelo treinado em arquivo .pkl
# ---------------------------------------------------------
with open("modelo_categorico.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\nModelo salvo em 'modelo_categorico.pkl'")

# Para carregar depois:
# with open("modelo_categorico.pkl", "rb") as f:
#     modelo_carregado = pickle.load(f)
