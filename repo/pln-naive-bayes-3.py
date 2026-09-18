import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

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
# CountVectorizer / TfidfVectorizer converte os textos em matrizes numéricas
pipeline_nlp = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True,         # Converte texto para minúsculas
        ngram_range=(1, 2),     # Considera unigramas e bigramas (ex: "muito bom")
        strip_accents='unicode' # Remove acentos
    )),
    ('classifier', MultinomialNB(alpha=1.0)) # Alpha = parâmetro de suavização (Laplace Smoothing)
])

# 4. Treinamento do Modelo
pipeline_nlp.fit(X_train, y_train)

# 5. Avaliação
y_pred = pipeline_nlp.predict(X_test)

print("=== NLP: Multinomial Naïve Bayes ===")
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Negativo', 'Positivo']))

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