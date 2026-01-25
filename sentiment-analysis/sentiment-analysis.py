import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc

# 1. CARGA DE DATOS
CURRENT_DIR = os.path.dirname(__file__)

def file_path(filename):
    return os.path.join(CURRENT_DIR, filename)

dataset_path = file_path('IMDB Dataset.csv')
df = pd.read_csv(dataset_path)

# Preprocesamiento rápido: convertir etiquetas a números
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# 2. DIVISIÓN DE DATOS
# Usamos una muestra de 10k si tu PC es lenta, o el total (50k) para el reporte final
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
)

# 3. VECTORIZACIÓN (TF-IDF)
# Esto convierte el texto en una matriz numérica basada en la importancia de las palabras
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 4. DEFINICIÓN DE MODELOS
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# 5. ENTRENAMIENTO Y MATRICES DE CONFUSIÓN
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

print("Entrenando modelos de NLP (esto puede tardar unos minutos)...")

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    # Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[i], cbar=False)
    axes[i].set_title(f'Matriz: {name}')
    axes[i].set_xlabel('Predicción')
    axes[i].set_ylabel('Real')
    axes[i].xaxis.set_ticklabels(['Negativo', 'Positivo'])
    axes[i].yaxis.set_ticklabels(['Negativo', 'Positivo'])
    
    print(f"\nReporte para {name}:")
    print(classification_report(y_test, y_pred))

plt.tight_layout()
plt.savefig(file_path('matrices_sentiment.png'), dpi=300)
plt.show()

# 6. CURVAS PRECISION-RECALL
plt.figure(figsize=(10, 7))
for name, model in models.items():
    y_probs = model.predict_proba(X_test_tfidf)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    area_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{name} (AUPRC = {area_auc:.3f})')

plt.title('Comparativa Precision-Recall: IMDB Sentiment Analysis')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(file_path('curvas_pr_sentiment.png'), dpi=300)
plt.show()