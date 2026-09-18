import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

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

MODEL_PATH = "modelo_naive_bayes.pkl"

st.set_page_config(page_title="Naive Bayes - App", layout="wide")
st.title("Classificação com Naive Bayes")
st.caption("Modelos de Classificação - Naive Bayes")

# ---------------------------------------------------------
# Dados e sessão
# ---------------------------------------------------------
@st.cache_data
def gerar_dados():
    X, y = make_classification(
        n_samples=500,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    return X, y

X, y = gerar_dados()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

if "model" not in st.session_state:
    st.session_state.model = None

# ---------------------------------------------------------
# Barra lateral - Ações
# ---------------------------------------------------------
st.sidebar.header("Ações")

if st.sidebar.button("Treinar modelo"):
    model = GaussianNB()
    model.fit(X_train, y_train)
    st.session_state.model = model
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

model = st.session_state.model

# ---------------------------------------------------------
# Corpo principal
# ---------------------------------------------------------
if model is None:
    st.info("Clique em **Treinar modelo** na barra lateral para começar.")
else:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    col1, col2 = st.columns(2)

    # ---- Métricas ----
    with col1:
        st.subheader("Métricas do modelo")
        acc = accuracy_score(y_test, y_pred)
        st.metric("Acurácia", f"{acc:.2%}")

        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    # ---- Matriz de confusão ----
    with col2:
        st.subheader("Matriz de Confusão")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_).plot(
            cmap="Blues", ax=ax_cm
        )
        st.pyplot(fig_cm)

    # ---- Curva ROC ----
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

    # ---- Predição manual ----
    st.subheader("Testar uma nova amostra")
    st.write("Informe os valores dos 4 atributos para obter a predição do modelo:")

    cols = st.columns(4)
    valores = []
    for i, c in enumerate(cols):
        val = c.number_input(f"Atributo {i+1}", value=float(X[:, i].mean()))
        valores.append(val)

    if st.button("Prever"):
        entrada = np.array(valores).reshape(1, -1)
        pred = model.predict(entrada)[0]
        proba = model.predict_proba(entrada)[0]
        st.success(f"Classe prevista: {pred}")
        st.write(f"Probabilidades: Classe 0 = {proba[0]:.2%} | Classe 1 = {proba[1]:.2%}")
