import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.decomposition import PCA

# 1. CARGA DE DATOS

CURRENT_DIR = os.path.dirname(__file__)
def file_path(filename):
    return os.path.join(CURRENT_DIR, filename)

df = pd.read_csv(file_path('creditcard.csv'))

#df = df.sample(frac=0.2, random_state=42) 

print(f"Procesando {len(df)} transacciones...")

# 2. PREPROCESAMIENTO
y_true = df['Class']
X = df.drop(['Class', 'Time'], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Proporción real de fraude para calibrar modelos
outlier_fraction = y_true.sum() / len(y_true)
print(f"Fracción de anomalías detectada: {outlier_fraction:.4f}")

# ---------------------------------------------------------
# 3. ENTRENAMIENTO DE MODELOS
# ---------------------------------------------------------

# A. Isolation Forest
print("Entrenando Isolation Forest...")
iforest = IsolationForest(contamination=outlier_fraction, random_state=42)
y_pred_if = iforest.fit_predict(X_scaled)
y_pred_if = [1 if i == -1 else 0 for i in y_pred_if]

# B. K-Means (Detección por distancia al percentil crítico)
print("Entrenando K-Means...")
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
kmeans.fit(X_scaled)
distances = np.min(kmeans.transform(X_scaled), axis=1)
threshold = np.percentile(distances, 100 * (1 - outlier_fraction))
y_pred_km = [1 if d > threshold else 0 for d in distances]

# C. DBSCAN (Detección por densidad)
print("Entrenando DBSCAN...")
dbscan = DBSCAN(eps=3.0, min_samples=10)
clusters_db = dbscan.fit_predict(X_scaled)
y_pred_db = [1 if c == -1 else 0 for c in clusters_db]

# ---------------------------------------------------------
# 4. VISUALIZACIÓN DE RESULTADOS (PCA 2D)
# ---------------------------------------------------------
print("Generando visualizaciones...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plot_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
plot_df['Realidad'] = y_true.values
plot_df['IF'] = y_pred_if
plot_df['KM'] = y_pred_km
plot_df['DB'] = y_pred_db

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
titulos = ['Distribución Real (Fraude)', 'Isolation Forest', 'K-Means (Distancia)', 'DBSCAN (Densidad)']
columnas = ['Realidad', 'IF', 'KM', 'DB']
paletas = ['viridis', 'rocket', 'mako', 'flare']

for i, ax in enumerate(axes.flat):
    sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue=columnas[i], ax=ax, palette=paletas[i], alpha=0.6)
    ax.set_title(titulos[i])

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 5. MATRICES DE CONFUSIÓN Y MÉTRICAS
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
modelos = [("Isolation Forest", y_pred_if), ("K-Means", y_pred_km), ("DBSCAN", y_pred_db)]

for i, (nombre, y_p) in enumerate(modelos):
    cm = confusion_matrix(y_true, y_p)
    ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Fraude']).plot(ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(f'{nombre}\nROC AUC: {roc_auc_score(y_true, y_p):.3f}')
    
    print(f"\nReporte {nombre}:")
    print(classification_report(y_true, y_p))

plt.show()