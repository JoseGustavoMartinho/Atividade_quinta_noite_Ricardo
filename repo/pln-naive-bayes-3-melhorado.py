import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)

# 1. Geração da massa de dados de texto (Exemplo: Avaliações de Produtos)
textos = [
    "Excelente produto, entrega rápida e ótimo atendimento",
    "Gostei muito, superou minhas expectativas e chegou antes do prazo",
    "Muito bom, recomendo a todos a compra",
    "Qualidade excelente, comprarei novamente com certeza",
    "Produto maravilhoso, cumpre o que promete",
    "Péssima qualidade, veio com defeito e quebrado",
    "Pior compra que já fiz, não recomendo a ninguém",
    "Horrível, o produto demorou semanas e veio errado",
    "Muito insatisfeito, atendimento péssimo e sem suporte",
    "Não funciona direito, dinheiro jogado fora"
] * 20  # Multiplicando para gerar 200 amostras sintéticas

# Rótulos: 1 = Positivo, 0 = Negativo
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 20

df = pd.DataFrame({'texto': textos, 'sentimento': labels})

# 2. Divisão dos dados em treino (75%) e teste (25%)
X_train, X_test, y_train, y_test = train_test_split(
    df['texto'],
    df['sentimento'],
    test_size=0.25,
    random_state=42,
    stratify=df['sentimento']
)

# 3. Construção da Pipeline NLP (TF-IDF + MultinomialNB)
pipeline_nlp = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        strip_accents='unicode'
    )),
    ('classifier', MultinomialNB(alpha=1.0))
])

# 4. Treinamento do Modelo
pipeline_nlp.fit(X_train, y_train)

# 5. Avaliação
y_pred = pipeline_nlp.predict(X_test)
y_proba = pipeline_nlp.predict_proba(X_test)[:, 1]  # probabilidade da classe "Positivo" (1)

print("=== NLP: Multinomial Naïve Bayes ===")
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(cm)
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Negativo', 'Positivo']))

# ---------------------------------------------------------
# Matriz de Confusão (gráfico)
# ---------------------------------------------------------
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negativo', 'Positivo'])
disp.plot(cmap="Blues")
plt.title("Matriz de Confusão - NLP Naive Bayes")
plt.savefig("matriz_confusao_nlp.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------
# Curva ROC
# ---------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"Curva ROC (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Classificador aleatório")
plt.xlabel("Taxa de Falsos Positivos (FPR)")
plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
plt.title("Curva ROC - NLP Naive Bayes")
plt.legend(loc="lower right")
plt.savefig("curva_roc_nlp.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"AUC (Área sob a curva ROC): {auc:.2f}")

# ---------------------------------------------------------
# Gravar o modelo (pipeline completa: TF-IDF + classificador) em .pkl
# ---------------------------------------------------------
with open("modelo_nlp.pkl", "wb") as f:
    pickle.dump(pipeline_nlp, f)

print("\nModelo salvo em 'modelo_nlp.pkl'")

# 6. Testando com novos textos inéditos
novos_exemplos = [
    "O suporte respondeu rápido e resolveu meu problema, gostei",
    "Veio quebrado e o atendimento foi horrível",
    "Chegou rápido, mas a qualidade é bem fraca"
]

predicoes = pipeline_nlp.predict(novos_exemplos)
probas = pipeline_nlp.predict_proba(novos_exemplos)

print("\n=== Predição em Novos Textos ===")
for texto, pred, prob in zip(novos_exemplos, predicoes, probas):
    classe = "Positivo" if pred == 1 else "Negativo"
    confianca = np.max(prob) * 100
    print(f"Texto: '{texto}'")
    print(f"-> Classe: {classe} ({confianca:.1f}% de confiança)\n")

# Para carregar o modelo depois:
# with open("modelo_nlp.pkl", "rb") as f:
#     modelo_carregado = pickle.load(f)
