import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
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

MODEL_PATH = "modelo_categorico.pkl"

st.set_page_config(page_title="Naive Bayes Categórico - App", layout="wide")
st.title("Perfil de Risco - Categorical Naive Bayes")
st.caption("Escolaridade, renda e histórico de crédito -> perfil de risco")

OPCOES = {
    'escolaridade': ['Medio', 'Superior', 'Pos-Graduacao'],
    'renda_faixa': ['Baixa', 'Media', 'Alta'],
    'historico_credito': ['Ruim', 'Bom', 'Excelente'],
}
CLASSE_POSITIVA = 'Baixo'

@st.cache_data
def gerar_dados():
    np.random.seed(42)
    n_samples = 1000
    dados = pd.DataFrame({
        'escolaridade': np.random.choice(OPCOES['escolaridade'], size=n_samples, p=[0.4, 0.4, 0.2]),
        'renda_faixa': np.random.choice(OPCOES['renda_faixa'], size=n_samples, p=[0.3, 0.5, 0.2]),
        'historico_credito': np.random.choice(OPCOES['historico_credito'], size=n_samples, p=[0.2, 0.6, 0.2]),
        'perfil_risco': np.random.choice(['Baixo', 'Alto'], size=n_samples, p=[0.7, 0.3]),
    })
    return dados

dados = gerar_dados()
X = dados[['escolaridade', 'renda_faixa', 'historico_credito']]
y = dados['perfil_risco']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

if "model" not in st.session_state:
    st.session_state.model = None

st.sidebar.header("Ações")

if st.sidebar.button("Treinar modelo"):
    preprocessor = ColumnTransformer(
        transformers=[('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), X.columns.tolist())]
    )
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', CategoricalNB())])
    pipeline.fit(X_train, y_train)
    st.session_state.model = pipeline
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

pipeline = st.session_state.model

if pipeline is None:
    st.info("Clique em **Treinar modelo** na barra lateral para começar.")
else:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Métricas do modelo")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    with col2:
        st.subheader("Matriz de Confusão")
        cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_).plot(cmap="Blues", ax=ax_cm)
        st.pyplot(fig_cm)

    st.subheader("Curva ROC")
    idx_pos = list(pipeline.classes_).index(CLASSE_POSITIVA)
    y_test_bin = (y_test == CLASSE_POSITIVA).astype(int)
    y_score = y_proba[:, idx_pos]
    fpr, tpr, _ = roc_curve(y_test_bin, y_score)
    auc = roc_auc_score(y_test_bin, y_score)

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"Curva ROC (AUC = {auc:.2f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Classificador aleatório")
    ax_roc.set_xlabel("Taxa de Falsos Positivos (FPR)")
    ax_roc.set_ylabel("Taxa de Verdadeiros Positivos (TPR)")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    st.subheader("Testar um novo perfil")
    c1, c2, c3 = st.columns(3)
    escolaridade = c1.selectbox("Escolaridade", OPCOES['escolaridade'])
    renda = c2.selectbox("Faixa de renda", OPCOES['renda_faixa'])
    credito = c3.selectbox("Histórico de crédito", OPCOES['historico_credito'])

    if st.button("Prever"):
        entrada = pd.DataFrame([{
            'escolaridade': escolaridade,
            'renda_faixa': renda,
            'historico_credito': credito,
        }])
        pred = pipeline.predict(entrada)[0]
        proba = pipeline.predict_proba(entrada)[0]
        st.success(f"Perfil de risco previsto: {pred}")
        for classe, p in zip(pipeline.classes_, proba):
            st.write(f"P({classe}) = {p:.2%}")
