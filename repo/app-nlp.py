import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

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

MODEL_PATH = "modelo_nlp.pkl"

st.set_page_config(page_title="NLP Naive Bayes - App", layout="wide")
st.title("Análise de Sentimento - Multinomial Naive Bayes")
st.caption("Classificação de avaliações de produtos em Positivo / Negativo")

@st.cache_data
def gerar_dados():
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
    ] * 20
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 20
    return pd.DataFrame({'texto': textos, 'sentimento': labels})

df = gerar_dados()
X_train, X_test, y_train, y_test = train_test_split(
    df['texto'], df['sentimento'], test_size=0.25, random_state=42, stratify=df['sentimento']
)

if "model" not in st.session_state:
    st.session_state.model = None

st.sidebar.header("Ações")

if st.sidebar.button("Treinar modelo"):
    pipeline_nlp = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, ngram_range=(1, 2), strip_accents='unicode')),
        ('classifier', MultinomialNB(alpha=1.0))
    ])
    pipeline_nlp.fit(X_train, y_train)
    st.session_state.model = pipeline_nlp
    st.sidebar.success("Modelo treinado!")

if st.sidebar.button("Gravar modelo (.pkl)"):
    if st.session_state.model is None:
        st.sidebar.error("Treine o modelo antes de salvar.")
    else:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(st.session_state.model, f)
        st.sidebar.success(f"Modelo salvo em '{MODEL_PATH}'")

if st.sidebar.button("Carregar modelo (.pkl)"):
    try:
        with open(MODEL_PATH, "rb") as f:
            st.session_state.model = pickle.load(f)
        st.sidebar.success("Modelo carregado do arquivo .pkl!")
    except FileNotFoundError:
        st.sidebar.error("Nenhum arquivo .pkl encontrado. Treine e grave o modelo primeiro.")

pipeline_nlp = st.session_state.model

if pipeline_nlp is None:
    st.info("Clique em **Treinar modelo** na barra lateral para começar.")
else:
    y_pred = pipeline_nlp.predict(X_test)
    y_proba = pipeline_nlp.predict_proba(X_test)[:, 1]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Métricas do modelo")
        report = classification_report(y_test, y_pred, target_names=['Negativo', 'Positivo'], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    with col2:
        st.subheader("Matriz de Confusão")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negativo', 'Positivo']).plot(cmap="Blues", ax=ax_cm)
        st.pyplot(fig_cm)

    st.subheader("Curva ROC")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"Curva ROC (AUC = {auc:.2f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Classificador aleatório")
    ax_roc.set_xlabel("Taxa de Falsos Positivos (FPR)")
    ax_roc.set_ylabel("Taxa de Verdadeiros Positivos (TPR)")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    st.subheader("Testar um novo texto")
    texto_novo = st.text_area("Digite uma avaliação de produto:", "Chegou rápido, mas a qualidade é bem fraca")

    if st.button("Analisar sentimento"):
        pred = pipeline_nlp.predict([texto_novo])[0]
        proba = pipeline_nlp.predict_proba([texto_novo])[0]
        classe = "Positivo" if pred == 1 else "Negativo"
        st.success(f"Sentimento previsto: {classe}")
        st.write(f"P(Negativo) = {proba[0]:.2%} | P(Positivo) = {proba[1]:.2%}")
