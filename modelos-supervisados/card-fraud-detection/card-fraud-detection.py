import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import os
from xgboost import XGBClassifier

# 1. Load the dataset
CURRENT_DIR = os.path.dirname(__file__)

def file_path(filename):
    return os.path.join(CURRENT_DIR, filename)

dataset_path = file_path('creditcard.csv')
data = pd.read_csv(dataset_path)
print("Dataset cargado exitosamente.")

# 2. Escalado de Variable Amount
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data[['Amount']])
data = data.drop(['Amount', "Time"], axis=1)
print("Variable 'Amount' escalada y columnas innecesarias eliminadas.")

# 3. Split the dataset into features and target variable
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Dataset separado en training and testing.")
print(f"Casos de fraude en TEST real: {y_test.sum()}")
# 4. Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("Clases desbalanceadas manejadas usando SMOTE.")

# 5. Create Models
models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

# 5. Evaluation and Visualization of Matrixes
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
resumen_metricas = []

# 6. Train and Evaluate Models
print("Entrenamiento y evaluación de modelos...")
for i, (name, model) in enumerate(models.items()):
    # Training
    model.fit(X_train_resampled, y_train_resampled)
    
    # Prediction
    y_pred = model.predict(X_test)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
    axes[i].set_title(f'Matriz: {name}')
    axes[i].set_xlabel('Predicción')
    axes[i].set_ylabel('Real')
    axes[i].xaxis.set_ticklabels(['Normal', 'Fraude'])
    axes[i].yaxis.set_ticklabels(['Normal', 'Fraude'])

    # Print to console
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred))

    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    resumen_metricas.append({
        'Modelo': name,
        'Precisión': precision,
        'Recall': recall,
        'F1-Score': f1
    })

plt.tight_layout()
plt.savefig(file_path('matrices_confusion.png'), dpi=300) # Guarda la imagen
plt.show()

# 7. Precision-Recall Curve
plt.figure(figsize=(10, 7))
for name, model in models.items():
    y_scores = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{name} (AUC={pr_auc:.2f})')

plt.title('Comparativa de Curvas Precision-Recall')
plt.xlabel('Recall (Capacidad de detectar fraudes)')
plt.ylabel('Precision (Confiabilidad de la alerta)')
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.savefig(file_path('curvas_pr.png'), dpi=300)
plt.show()

