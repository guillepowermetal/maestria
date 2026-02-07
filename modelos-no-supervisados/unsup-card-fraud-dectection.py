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
best_if_auc = 0
for contamination in [outlier_fraction, outlier_fraction*2, 0.01, 0.05]:
    iforest = IsolationForest(contamination=contamination, random_state=42)
    y_pred_if = iforest.fit_predict(X_scaled)
    y_pred_if = [1 if i == -1 else 0 for i in y_pred_if]
    auc = roc_auc_score(y_true, y_pred_if)
    if auc > best_if_auc:
        best_if_auc = auc
        best_iforest = iforest
        best_y_pred_if = y_pred_if
print(f"Mejor Isolation Forest AUC: {best_if_auc:.4f}")

print("Entrenando K-Means (tuning)...")
best_km_auc = 0
for n_clusters in [4, 8, 12]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    distances = np.min(kmeans.transform(X_scaled), axis=1)
    threshold = np.percentile(distances, 100 * (1 - outlier_fraction))
    y_pred_km = [1 if d > threshold else 0 for d in distances]
    auc = roc_auc_score(y_true, y_pred_km)
    if auc > best_km_auc:
        best_km_auc = auc
        best_kmeans = kmeans
        best_distances = distances
        best_y_pred_km = y_pred_km
print(f"Mejor K-Means AUC: {best_km_auc:.4f}")

print("Entrenando DBSCAN (tuning)...")
best_db_auc = 0
for eps in [1.5, 2.0, 3.0]:
    dbscan = DBSCAN(eps=eps, min_samples=10)
    clusters_db = dbscan.fit_predict(X_scaled)
    y_pred_db = [1 if c == -1 else 0 for c in clusters_db]
    auc = roc_auc_score(y_true, y_pred_db)
    if auc > best_db_auc:
        best_db_auc = auc
        best_dbscan = dbscan
        best_clusters_db = clusters_db
        best_y_pred_db = y_pred_db
print(f"Mejor DBSCAN AUC: {best_db_auc:.4f}")

# Use best models for all downstream plots
iforest = best_iforest
y_pred_if = best_y_pred_if
kmeans = best_kmeans
distances = best_distances
y_pred_km = best_y_pred_km
dbscan = best_dbscan
clusters_db = best_clusters_db
y_pred_db = best_y_pred_db

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
plt.savefig(os.path.join(RESULTS_DIR, 'pca-scatter-all.png'))
plt.close()

# --- UMAP 2D Scatter Plots (if available) ---
if X_umap is not None:
    plot_df_umap = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])
    plot_df_umap['Realidad'] = y_true.values
    plot_df_umap['IF'] = y_pred_if
    plot_df_umap['KM'] = y_pred_km
    plot_df_umap['DB'] = y_pred_db
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for i, ax in enumerate(axes.flat):
        sns.scatterplot(data=plot_df_umap, x='UMAP1', y='UMAP2', hue=columnas[i], ax=ax, palette=paletas[i], alpha=0.6)
        ax.set_title(titulos[i] + ' (UMAP)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'umap-scatter-all.png'))
    plt.close()

# --- Confusion Matrices ---
from sklearn.metrics import ConfusionMatrixDisplay
for name, y_pred in zip(['Isolation Forest', 'K-Means', 'DBSCAN'], [y_pred_if, y_pred_km, y_pred_db]):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Normal','Fraude'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.savefig(os.path.join(RESULTS_DIR, f'confusion-{name.replace(" ", "_").lower()}.png'))
    plt.close()

# --- ROC Curves ---
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
plt.figure(figsize=(8,6))
for name, y_pred in zip(['Isolation Forest', 'K-Means', 'DBSCAN'], [y_pred_if, y_pred_km, y_pred_db]):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve Comparison')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'roc-compare.png'))
plt.close()

# --- Precision-Recall Curves ---
plt.figure(figsize=(8,6))
for name, y_pred in zip(['Isolation Forest', 'K-Means', 'DBSCAN'], [y_pred_if, y_pred_km, y_pred_db]):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    plt.plot(recall, precision, label=f'{name} (AP={ap:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'pr-compare.png'))
plt.close()

# --- Error Distribution Histograms ---
plt.figure(figsize=(10,6))
plt.hist(distances[y_true==0], bins=30, alpha=0.5, label='K-Means Normal')
plt.hist(distances[y_true==1], bins=30, alpha=0.5, label='K-Means Fraude')
plt.title('K-Means Distance Error Distribution')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'error-hist-kmeans.png'))
plt.close()

if hasattr(iforest, 'decision_function'):
    scores_if = -iforest.decision_function(X_scaled)
    plt.figure(figsize=(10,6))
    plt.hist(scores_if[y_true==0], bins=30, alpha=0.5, label='IF Normal')
    plt.hist(scores_if[y_true==1], bins=30, alpha=0.5, label='IF Fraude')
    plt.title('Isolation Forest Score Distribution')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'error-hist-iforest.png'))
    plt.close()

plt.figure(figsize=(10,6))
plt.hist(clusters_db[y_true==0], bins=30, alpha=0.5, label='DBSCAN Normal')
plt.hist(clusters_db[y_true==1], bins=30, alpha=0.5, label='DBSCAN Fraude')
plt.title('DBSCAN Cluster Distribution')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'error-hist-dbscan.png'))
plt.close()

# --- Summary Bar Chart ---
from sklearn.metrics import precision_score, recall_score, f1_score
metrics = []
for name, y_pred in zip(['Isolation Forest', 'K-Means', 'DBSCAN'], [y_pred_if, y_pred_km, y_pred_db]):
    auc = roc_auc_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    metrics.append({'model':name, 'auc':auc, 'precision':prec, 'recall':rec, 'f1':f1})
metrics_df = pd.DataFrame(metrics)
plt.figure(figsize=(10,6))
bar_width = 0.2
index = np.arange(len(metrics_df))
plt.bar(index-bar_width*1.5, metrics_df['auc'], bar_width, label='AUC')
plt.bar(index-bar_width*0.5, metrics_df['precision'], bar_width, label='Precision')
plt.bar(index+bar_width*0.5, metrics_df['recall'], bar_width, label='Recall')
plt.bar(index+bar_width*1.5, metrics_df['f1'], bar_width, label='F1')
plt.xticks(index, metrics_df['model'])
plt.ylim([0,1.1])
plt.title('Model Metrics Comparison')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'metrics-bar.png'))
plt.close()

# --- Metrics Heatmap ---
plt.figure(figsize=(8,6))
sns.heatmap(metrics_df.set_index('model'), annot=True, cmap='RdYlGn', vmin=0, vmax=1)
plt.title('Model Metrics Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'metrics-heatmap.png'))
plt.close()

# --- Metrics Summary Table (CSV and PNG) ---
metrics_df_rounded = metrics_df.copy()
metrics_df_rounded[['auc','precision','recall','f1']] = metrics_df_rounded[['auc','precision','recall','f1']].round(4)
metrics_df_rounded.to_csv(os.path.join(RESULTS_DIR, 'metrics-summary.csv'), index=False)

# Save as PNG table
import matplotlib.table as tbl
fig, ax = plt.subplots(figsize=(7,2))
ax.axis('off')
table = ax.table(cellText=metrics_df_rounded.values,
                 colLabels=metrics_df_rounded.columns,
                 rowLabels=metrics_df_rounded['model'],
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
plt.title('Model Metrics Summary', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'metrics-summary-table.png'), bbox_inches='tight', dpi=150)
plt.close()

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