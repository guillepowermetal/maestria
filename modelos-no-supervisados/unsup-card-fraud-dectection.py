import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# 1. CARGA DE DATOS

CURRENT_DIR = os.path.dirname(__file__)
def file_path(filename):
    return os.path.join(CURRENT_DIR, filename)

RESULTS_DIR = os.path.join(CURRENT_DIR, 'results')

df = pd.read_csv(file_path('creditcard.csv'))

#df = df.sample(frac=0.2, random_state=42) 

print(f"Procesando {len(df)} transacciones...")

# 2. PREPROCESAMIENTO
import warnings
warnings.filterwarnings('ignore')

y_true = df['Class']
X = df.drop(['Class'], axis=1)

# --- Feature Engineering ---
# 1. Amount per V1 cluster (KMeans on V1)
from sklearn.preprocessing import KBinsDiscretizer
kbins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
X['V1_bin'] = kbins.fit_transform(X[['V1']])
amount_per_v1 = X.groupby('V1_bin')['Amount'].transform('mean')
X['Amount_per_V1bin'] = amount_per_v1

# 2. Time-based features
X['Hour'] = (df['Time'] // 3600) % 24
X['DayPart'] = pd.cut(X['Hour'], bins=[-1,6,12,18,24], labels=['Night','Morning','Afternoon','Evening'])
X = pd.get_dummies(X, columns=['DayPart'], drop_first=True)

# 3. Remove original Time column for modeling
if 'Time' in X.columns:
    X = X.drop('Time', axis=1)

# 4. Dimensionality reduction for visualization (PCA and UMAP)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

try:
    import umap
    umap_model = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_model.fit_transform(X_scaled)
except ImportError:
    X_umap = None

# Proporción real de fraude para calibrar modelos
outlier_fraction = y_true.sum() / len(y_true)
print(f"Fracción de anomalías detectada: {outlier_fraction:.4f}")

# ---------------------------------------------------------
# 3. ENTRENAMIENTO DE MODELOS
# ---------------------------------------------------------



# --- Hyperparameter Tuning (basic grid) ---
print("Entrenando Isolation Forest (tuning)...")
best_iforest = None
best_y_pred_if = None
best_score_if = -np.inf
for contamination in [outlier_fraction, outlier_fraction*2, 0.01, 0.05]:
    iforest = IsolationForest(contamination=contamination, random_state=42)
    y_pred_if = iforest.fit_predict(X_scaled)
    # For anomaly detection, use average anomaly score as a metric
    if hasattr(iforest, 'decision_function'):
        scores = -iforest.decision_function(X_scaled)
        avg_score = np.mean(scores)
    else:
        avg_score = np.mean(y_pred_if)
    if avg_score > best_score_if:
        best_score_if = avg_score
        best_iforest = iforest
        best_y_pred_if = y_pred_if
print(f"Mejor Isolation Forest average anomaly score: {best_score_if:.4f}")

print("Entrenando K-Means (tuning)...")
best_kmeans = None
best_distances = None
best_y_pred_km = None
best_km_sil = -1
for n_clusters in [4, 8, 12]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    sil = silhouette_score(X_scaled, labels)
    if sil > best_km_sil:
        best_km_sil = sil
        best_kmeans = kmeans
        best_distances = np.min(kmeans.transform(X_scaled), axis=1)
        best_labels_km = labels
print(f"Mejor K-Means Silhouette: {best_km_sil:.4f}")

print("Entrenando DBSCAN (tuning)...")
best_dbscan = None
best_clusters_db = None
best_db_sil = -1
for eps in [1.5, 2.0, 3.0]:
    dbscan = DBSCAN(eps=eps, min_samples=10)
    clusters_db = dbscan.fit_predict(X_scaled)
    # Only compute silhouette if more than 1 cluster and less than all noise
    if len(set(clusters_db)) > 1 and np.sum(clusters_db == -1) < len(clusters_db):
        sil = silhouette_score(X_scaled, clusters_db)
        if sil > best_db_sil:
            best_db_sil = sil
            best_dbscan = dbscan
            best_clusters_db = clusters_db
print(f"Mejor DBSCAN Silhouette: {best_db_sil:.4f}")

# Use best models for all downstream plots
iforest = best_iforest
y_pred_if = best_y_pred_if
kmeans = best_kmeans
distances = best_distances
labels_km = best_labels_km
dbscan = best_dbscan
clusters_db = best_clusters_db

# ---------------------------------------------------------
# 4. VISUALIZACIÓN DE RESULTADOS (PCA 2D)
# --- Feature Importance Analysis ---
print("Calculando importancia de variables...")
feature_names = list(X.columns)

# Isolation Forest feature importances (mean decrease in impurity)
if hasattr(iforest, 'feature_importances_'):
    importances = iforest.feature_importances_
else:
    # Use mean absolute value of tree splits
    importances = np.mean([np.abs(tree.feature_importances_) for tree in iforest.estimators_], axis=0)

plt.figure(figsize=(12,6))
indices = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.title('Isolation Forest Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'feature-importance-iforest.png'))
plt.close()

# --- PCA 2D Scatter Plots ---
print("Generando visualizaciones...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# PCA explained variance plot
plt.figure(figsize=(8,5))
plt.plot(np.arange(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, marker='o')
plt.title('PCA Explained Variance by Component')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'pca-explained-variance.png'))
plt.close()
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- PCA 2D Scatter Plots ---

plot_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
plot_df['Realidad'] = y_true.values
plot_df['IF'] = y_pred_if
plot_df['KM'] = labels_km
plot_df['DB'] = clusters_db
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
titulos = ['Distribución Real (Fraude)', 'Isolation Forest', 'K-Means (Cluster)', 'DBSCAN (Cluster)']
columnas = ['Realidad', 'IF', 'KM', 'DB']
paletas = ['viridis', 'rocket', 'mako', 'flare']
for i, ax in enumerate(axes.flat):
    sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue=columnas[i], ax=ax, palette=paletas[i], alpha=0.6)
    ax.set_title(titulos[i])
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'pca-scatter-all.png'))
plt.close()

# --- UMAP 2D Scatter Plots (if available) ---
if X_umap is not None:
    plot_df_umap = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])
    plot_df_umap['Realidad'] = y_true.values
    plot_df_umap['IF'] = y_pred_if
    plot_df_umap['KM'] = labels_km
    plot_df_umap['DB'] = clusters_db
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for i, ax in enumerate(axes.flat):
        sns.scatterplot(data=plot_df_umap, x='UMAP1', y='UMAP2', hue=columnas[i], ax=ax, palette=paletas[i], alpha=0.6)
        ax.set_title(titulos[i] + ' (UMAP)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'umap-scatter-all.png'))
    plt.close()


# --- Unsupervised Metrics and Plots ---
print("Calculando métricas no supervisadas...")
metrics_unsup = {}

# KMeans metrics
metrics_unsup['KMeans'] = {
    'Silhouette': silhouette_score(X_scaled, labels_km),
    'Davies-Bouldin': davies_bouldin_score(X_scaled, labels_km),
    'Calinski-Harabasz': calinski_harabasz_score(X_scaled, labels_km)
}

# DBSCAN metrics (if valid)
if clusters_db is not None and len(set(clusters_db)) > 1 and np.sum(clusters_db == -1) < len(clusters_db):
    metrics_unsup['DBSCAN'] = {
        'Silhouette': silhouette_score(X_scaled, clusters_db),
        'Davies-Bouldin': davies_bouldin_score(X_scaled, clusters_db),
        'Calinski-Harabasz': calinski_harabasz_score(X_scaled, clusters_db)
    }
else:
    metrics_unsup['DBSCAN'] = {'Silhouette': np.nan, 'Davies-Bouldin': np.nan, 'Calinski-Harabasz': np.nan}

# Isolation Forest anomaly score histogram
if hasattr(iforest, 'decision_function'):
    scores_if = -iforest.decision_function(X_scaled)
    plt.figure(figsize=(10,6))
    plt.hist(scores_if, bins=30, alpha=0.7, color='purple')
    plt.title('Isolation Forest Anomaly Score Distribution')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'iforest-anomaly-score-hist.png'))
    plt.close()
    metrics_unsup['IsolationForest'] = {'Average Anomaly Score': np.mean(scores_if)}
else:
    metrics_unsup['IsolationForest'] = {'Average Anomaly Score': np.nan}

# Save metrics as CSV and print
metrics_unsup_df = pd.DataFrame(metrics_unsup).T
metrics_unsup_df.to_csv(os.path.join(RESULTS_DIR, 'unsupervised-metrics-summary.csv'))
print("\nResumen de métricas no supervisadas:")
print(metrics_unsup_df)


# --- End of script: only unsupervised metrics and plots are generated ---