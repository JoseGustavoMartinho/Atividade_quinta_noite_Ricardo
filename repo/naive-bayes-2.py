import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix, classification_report

# 1. Geração de massa de dados categóricos estruturados (Pandas Dataframe)
np.random.seed(42)
n_samples = 1000

dados = pd.DataFrame({
    'escolaridade': np.random.choice(['Medio', 'Superior', 'Pos-Graduacao'], size=n_samples, p=[0.4, 0.4, 0.2]),
    'renda_faixa': np.random.choice(['Baixa', 'Media', 'Alta'], size=n_samples, p=[0.3, 0.5, 0.2]),
    'historico_credito': np.random.choice(['Ruim', 'Bom', 'Excelente'], size=n_samples, p=[0.2, 0.6, 0.2]),
    'perfil_risco': np.random.choice(['Baixo', 'Alto'], size=n_samples, p=[0.7, 0.3]) # Target
})

# Separação de features (X) e rótulo (y)
X = dados[['escolaridade', 'renda_faixa', 'historico_credito']]
y = dados['perfil_risco']

categorical_cols = X.columns.tolist()

# 2. Pré-processamento e construção da Pipeline
# O CategoricalNB exige dados codificados numericamente para categorias
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

print("=== Nível Intermediário: Categorical Naïve Bayes ===")
print(f"Média Acurácia (Cross-Validation 5-Fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))