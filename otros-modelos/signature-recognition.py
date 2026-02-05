import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix, precision_recall_fscore_support
from sklearn.manifold import TSNE
import random

# ==========================================
# 1. CARGA Y PREPROCESAMIENTO: PAPEL (CNN)
# ==========================================
def load_paper_data(base_path, img_size=(128, 128)):
    def get_images(subfolder):
        imgs = []
        path = os.path.join(base_path, subfolder)
        # Caminar por todos los subfolders de IDs
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg')):
                    img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, img_size)
                        img = img.astype('float32') / 255.0
                        img = 1.0 - img # Invertir: trazo blanco, fondo negro
                        imgs.append(img)
        return np.expand_dims(np.array(imgs), axis=-1)

    print("Cargando imágenes de papel...")
    x_real = get_images('real')
    x_forged = get_images('forged')
    return x_real, x_forged

# ==========================================
# 2. CARGA Y PREPROCESAMIENTO: DIGITAL (LSTM)
# ==========================================
def load_digital_data(base_path, max_steps=100):
    def get_dat_files():
        real_seqs = []
        forged_seqs = []
        import re
        # Esperamos subfolders nUser1 .. nUser40
        for entry in os.listdir(base_path):
            folder = os.path.join(base_path, entry)
            if not os.path.isdir(folder):
                continue
            m = re.match(r'nUser(\d+)', entry)
            if not m:
                # skip unknown folders
                continue
            uid = int(m.group(1))
            target_list = real_seqs if 1 <= uid <= 20 else forged_seqs
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.endswith('.dat'):
                        filepath = os.path.join(root, file)
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                            lines = fh.readlines()
                        data_start = None
                        for idx, ln in enumerate(lines):
                            if ln.strip().lower().startswith('@data'):
                                data_start = idx + 1
                                break
                        if data_start is None:
                            continue
                        from io import StringIO
                        data_text = ''.join(lines[data_start:])
                        try:
                            data = pd.read_csv(StringIO(data_text), header=None).values
                        except Exception:
                            continue
                        # Tomamos X, Y y Presión (Columnas 0, 1, 6)
                        try:
                            coords = data[:, [0, 1, 6]]
                        except Exception:
                            continue
                        # Normalización simple por muestra
                        coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-7)
                        # Padding/Truncado
                        if len(coords) > max_steps:
                            coords = coords[:max_steps]
                        else:
                            pad = np.zeros((max_steps - len(coords), 3))
                            coords = np.vstack((coords, pad))
                        target_list.append(coords)
        return np.array(real_seqs), np.array(forged_seqs)

    print("Cargando datos digitales (.dat)...")
    return get_dat_files()

# ==========================================
# 3. CONSTRUCCIÓN DE MODELOS (AUTOENCODERS)
# ==========================================
def build_paper_ae(input_shape=(128, 128, 1)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'), # Espacio Latente
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_digital_ae(timesteps=100, features=3):
    model = models.Sequential([
        layers.LSTM(64, activation='relu', input_shape=(timesteps, features), return_sequences=False),
        layers.RepeatVector(timesteps),
        layers.LSTM(64, activation='relu', return_sequences=True),
        layers.TimeDistributed(layers.Dense(features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_digital_gru_ae(timesteps=100, features=3):
    model = models.Sequential([
        layers.GRU(64, activation='tanh', input_shape=(timesteps, features), return_sequences=False),
        layers.RepeatVector(timesteps),
        layers.GRU(64, activation='tanh', return_sequences=True),
        layers.TimeDistributed(layers.Dense(features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_digital_lstm_classifier(timesteps=100, features=3):
    model = models.Sequential([
        layers.LSTM(64, input_shape=(timesteps, features)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

def build_digital_encoder(timesteps=100, features=3, embedding_dim=64):
    """Encoder returning a fixed-size embedding from a sequence."""
    inp = layers.Input(shape=(timesteps, features))
    x = layers.Masking()(inp)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dense(embedding_dim)(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    model = models.Model(inputs=inp, outputs=x, name='digital_encoder')
    return model

def _augment_sequence(seq, noise_scale=0.02, drop_rate=0.05):
    # Gaussian jitter
    aug = seq.copy()
    aug = aug + np.random.normal(scale=noise_scale, size=aug.shape)
    # Random time dropout
    T = aug.shape[0]
    mask = np.random.rand(T) > drop_rate
    aug[~mask] = 0.0
    return aug

def _nt_xent_loss(embeddings, batch_size, temperature=0.1):
    # embeddings: (2*batch_size, dim) where i and i+batch are positives
    z = tf.math.l2_normalize(embeddings, axis=1)
    similarities = tf.matmul(z, z, transpose_b=True) / temperature
    # mask out self-similarities
    large_neg = -1e9 * tf.eye(2 * batch_size)
    logits = similarities + large_neg
    labels = tf.concat([tf.range(batch_size, 2*batch_size), tf.range(0, batch_size)], axis=0)
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return tf.reduce_mean(loss)

def pretrain_contrastive(x_real, encoder, batch_size=32, epochs=3, temperature=0.1):
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    n = len(x_real)
    if n == 0:
        print('No real digital sequences for contrastive pretraining.')
        return
    for ep in range(epochs):
        idx = np.random.permutation(n)
        losses = []
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            batch = x_real[batch_idx]
            b = len(batch)
            if b == 0:
                continue
            aug1 = np.stack([_augment_sequence(s) for s in batch], axis=0)
            aug2 = np.stack([_augment_sequence(s) for s in batch], axis=0)
            inputs = np.concatenate([aug1, aug2], axis=0).astype('float32')
            with tf.GradientTape() as tape:
                emb = encoder(inputs, training=True)
                loss = _nt_xent_loss(emb, batch_size=b, temperature=temperature)
            grads = tape.gradient(loss, encoder.trainable_variables)
            opt.apply_gradients(zip(grads, encoder.trainable_variables))
            losses.append(float(loss))
        print(f'Contrastive pretrain epoch {ep+1}/{epochs} — loss: {np.mean(losses):.4f}')

def build_paper_cnn_classifier(input_shape=(128,128,1)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32,3,activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64,3,activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

def build_paper_cnn_deep(input_shape=(128,128,1)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32,3,activation='relu', padding='same'),
        layers.Conv2D(32,3,activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64,3,activation='relu', padding='same'),
        layers.Conv2D(64,3,activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(128,3,activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['AUC'])
    return model

def build_digital_transformer_classifier(timesteps=100, features=3, head_size=64, num_heads=4, ff_dim=128):
    inp = layers.Input(shape=(timesteps, features))
    x = layers.LayerNormalization()(inp)
    # Simple Transformer encoder block
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    x_ff = layers.Dense(ff_dim, activation='relu')(x)
    x_ff = layers.Dense(features)(x_ff)
    x = layers.Add()([x, x_ff])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

# ==========================================
# 4. EJECUCIÓN PRINCIPAL
# ==========================================

# Rutas (Ajusta según tu PC). Use rutas relativas al archivo para evitar
# problemas al ejecutar desde otro directorio.
BASE_DIR = os.path.dirname(__file__)
PATH_PAPER = os.path.join(BASE_DIR, 'paper-signature-data')
PATH_DIGITAL = os.path.join(BASE_DIR, 'digital-signature-data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- PROCESO PAPEL ---
x_p_real, x_p_forged = load_paper_data(PATH_PAPER)

# Split paper data
if len(x_p_real) == 0:
    raise RuntimeError('No paper real images found in PATH_PAPER: ' + PATH_PAPER)

x_p_train, x_p_test_real = train_test_split(x_p_real, test_size=0.2)

# --- PROCESO DIGITAL ---
# load_digital_data returns (real_seqs, forged_seqs)
x_d_real, x_d_forged = load_digital_data(PATH_DIGITAL)
if x_d_real.size == 0 or x_d_forged.size == 0:
    raise RuntimeError('No digital .dat sequences found or missing forged data in PATH_DIGITAL: ' + PATH_DIGITAL)

# Contrastive pretraining (self-supervised) for digital encoder
print("\nPretraining digital encoder with contrastive learning...")
encoder = build_digital_encoder(timesteps=x_d_real.shape[1], features=x_d_real.shape[2], embedding_dim=64)
pretrain_contrastive(x_d_real, encoder, batch_size=16, epochs=3, temperature=0.1)

# Training configuration (increase for better metrics)
PAPER_AE_EPOCHS = 30
PAPER_CLS_EPOCHS = 30
DIGITAL_AE_EPOCHS = 30
DIGITAL_CLS_EPOCHS = 30

print("\nEvaluando modelos para Papel (3 modelos)...")

# Model 1: Autoencoder (unsupervised anomaly)
model_paper_ae = build_paper_ae()
model_paper_ae.fit(x_p_train, x_p_train, epochs=PAPER_AE_EPOCHS, batch_size=16, verbose=1)
# create an explicit encoder submodel for t-SNE/embeddings (AFTER training so model is built)
try:
    encoder_paper = tf.keras.Model(inputs=model_paper_ae.input, outputs=model_paper_ae.layers[4].output)
except Exception:
    encoder_paper = None

# Model 2: Supervised CNN classifier (train on real vs forged)
model_paper_cnn = build_paper_cnn_classifier(input_shape=x_p_real.shape[1:])
X_p = np.concatenate([x_p_real, x_p_forged], axis=0)
Y_p = np.concatenate([np.zeros(len(x_p_real)), np.ones(len(x_p_forged))], axis=0)
X_p_train, X_p_test, Y_p_train, Y_p_test = train_test_split(X_p, Y_p, test_size=0.2, stratify=Y_p)
model_paper_cnn.fit(X_p_train, Y_p_train, epochs=PAPER_CLS_EPOCHS, batch_size=16, verbose=1)

# Model 3: Another autoencoder variant (same architecture for simplicity)
model_paper_ae2 = build_paper_ae()
model_paper_ae2.fit(x_p_train, x_p_train, epochs=max(1, PAPER_AE_EPOCHS//2), batch_size=16, verbose=1)
# Additional paper deep model (deeper CNN classifier)
model_paper_cnn_deep = build_paper_cnn_deep(input_shape=x_p_real.shape[1:])
model_paper_cnn_deep.fit(X_p_train, Y_p_train, epochs=max(1, PAPER_CLS_EPOCHS//2), batch_size=16, verbose=1)

print("\nEvaluando modelos para Digital (3 modelos)...")

# Digital: load returns (real_seqs, forged_seqs)
x_d_real, x_d_forged = load_digital_data(PATH_DIGITAL)
if x_d_real.size == 0 or x_d_forged.size == 0:
    raise RuntimeError('Digital sequences missing (real or forged). Check PATH_DIGITAL: ' + PATH_DIGITAL)

# Split real/forged into train/test portions
x_d_train, x_d_test_real = train_test_split(x_d_real, test_size=0.2)
x_d_f_train, x_d_f_test = train_test_split(x_d_forged, test_size=0.2)

# Model 1: LSTM Autoencoder trained on REAL sequences (unsupervised anomaly detection)
model_digital_ae = build_digital_ae(timesteps=x_d_real.shape[1], features=x_d_real.shape[2])
model_digital_ae.fit(x_d_train, x_d_train, epochs=DIGITAL_AE_EPOCHS, batch_size=16, verbose=1)
# create an explicit encoder submodel for t-SNE/embeddings (AFTER training so model is built)
try:
    encoder_digital = tf.keras.Model(inputs=model_digital_ae.input, outputs=model_digital_ae.layers[0].output)
except Exception:
    encoder_digital = None

# Model 2: LSTM classifier trained SUPERVISED using true forged samples
X_d = np.concatenate([x_d_train, x_d_f_train], axis=0)
Y_d = np.concatenate([np.zeros(len(x_d_train)), np.ones(len(x_d_f_train))], axis=0)
Xd_train, Xd_test, Yd_train, Yd_test = train_test_split(X_d, Y_d, test_size=0.2, stratify=Y_d)
model_digital_clf = build_digital_lstm_classifier(timesteps=x_d_real.shape[1], features=x_d_real.shape[2])
model_digital_clf.fit(Xd_train, Yd_train, epochs=DIGITAL_CLS_EPOCHS, batch_size=16, verbose=1)

# Model 3: GRU Autoencoder trained on REAL sequences
model_digital_gru = build_digital_gru_ae(timesteps=x_d_real.shape[1], features=x_d_real.shape[2])
model_digital_gru.fit(x_d_train, x_d_train, epochs=max(1, DIGITAL_AE_EPOCHS//2), batch_size=16, verbose=1)
# Additional digital time-series model: Transformer classifier
model_digital_transformer = build_digital_transformer_classifier(timesteps=x_d_real.shape[1], features=x_d_real.shape[2])
model_digital_transformer.fit(Xd_train, Yd_train, epochs=max(1, DIGITAL_CLS_EPOCHS//2), batch_size=16, verbose=1)

# ==========================================
# 5. EVALUACIÓN DE ANOMALÍAS
# ==========================================
def calculate_errors(model, data):
    reconstructions = model.predict(data)
    # MSE por cada muestra
    return np.mean(np.square(data - reconstructions), axis=tuple(range(1, data.ndim)))

# Evaluaciones resumidas
def eval_paper_autoencoder(ae_model, x_real_test, x_forged):
    err_r = calculate_errors(ae_model, x_real_test)
    err_f = calculate_errors(ae_model, x_forged)
    y = np.concatenate([np.zeros_like(err_r), np.ones_like(err_f)])
    scores = np.concatenate([err_r, err_f])
    auc = roc_auc_score(y, scores)
    print(f"AE Paper AUC (error as score): {auc:.4f}")
    return auc

def eval_paper_classifier(cls_model, X_test, Y_test):
    preds = cls_model.predict(X_test).ravel()
    auc = roc_auc_score(Y_test, preds)
    print(f"Paper classifier AUC: {auc:.4f}")
    return auc

print("\n--- Evaluación Final: Papel ---")
eval_paper_autoencoder(model_paper_ae, x_p_test_real, x_p_forged)
eval_paper_classifier(model_paper_cnn, X_p_test, Y_p_test)
eval_paper_autoencoder(model_paper_ae2, x_p_test_real, x_p_forged)

print("\n--- Evaluación Final: Digital ---")
err_d_ae = calculate_errors(model_digital_ae, x_d_test_real)
err_d_gru = calculate_errors(model_digital_gru, x_d_test_real)
print(f"Digital AE avg error (real test): {np.mean(err_d_ae):.6f}")
print(f"Digital GRU-AE avg error (real test): {np.mean(err_d_gru):.6f}")
preds_d = model_digital_clf.predict(Xd_test).ravel()
print(f"Digital LSTM classifier AUC: {roc_auc_score(Yd_test, preds_d):.4f}")

# Optional plots (saved to files)
plt.figure()
plt.hist(err_d_ae, bins=30, alpha=0.6, label='AE real errors')
plt.title('Digital AE errors (real test)')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'digital_ae-error-hist.png'))
plt.close()


# --- Visualization helpers (save required charts) ---
def save_paper_plots():
    # ROC for paper classifier
    y_scores = model_paper_cnn.predict(X_p_test).ravel()
    fpr, tpr, _ = roc_curve(Y_p_test, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC={roc_auc_score(Y_p_test, y_scores):.3f}')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Paper CNN classifier ROC (paper_cnn)')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'paper_cnn-roc.png'))
    plt.close()

    # AE error hist (paper)
    err_r = calculate_errors(model_paper_ae, x_p_test_real)
    err_f = calculate_errors(model_paper_ae, x_p_forged)
    plt.figure()
    plt.hist(err_r, bins=30, alpha=0.6, label='Real')
    plt.hist(err_f, bins=30, alpha=0.6, label='Forged')
    plt.title('Paper AE error distribution (paper_ae)')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'paper_ae-error-hist.png'))
    plt.close()

    # Reconstructions examples (paper)
    n_real = min(3, len(x_p_test_real))
    n_forged = min(3, len(x_p_forged))
    idxs_real = random.sample(range(len(x_p_test_real)), n_real)
    idxs_forged = random.sample(range(len(x_p_forged)), n_forged) if n_forged>0 else []
    total_rows = n_real + n_forged
    if total_rows > 0:
        plt.figure(figsize=(6, 3 * total_rows))
        i = 1
        for idx in idxs_real:
            orig = x_p_test_real[idx]
            recon = model_paper_ae.predict(np.expand_dims(orig, 0))[0]
            plt.subplot(total_rows, 2, i)
            plt.imshow(orig.squeeze(), cmap='gray')
            plt.title('Real Orig')
            plt.axis('off')
            i += 1
            plt.subplot(total_rows, 2, i)
            plt.imshow(recon.squeeze(), cmap='gray')
            plt.title('Real Recon')
            plt.axis('off')
            i += 1
        for idx in idxs_forged:
            orig = x_p_forged[idx]
            recon = model_paper_ae.predict(np.expand_dims(orig, 0))[0]
            plt.subplot(total_rows, 2, i)
            plt.imshow(orig.squeeze(), cmap='gray')
            plt.title('Forged Orig')
            plt.axis('off')
            i += 1
            plt.subplot(total_rows, 2, i)
            plt.imshow(recon.squeeze(), cmap='gray')
            plt.title('Forged Recon')
            plt.axis('off')
            i += 1
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'paper_ae-reconstructions.png'))
        plt.close()
    # Reconstructions for paper_ae2
    try:
        if len(x_p_test_real)>0 or len(x_p_forged)>0:
            plt.figure(figsize=(6, 3 * total_rows))
            i = 1
            for idx in idxs_real:
                orig = x_p_test_real[idx]
                recon = model_paper_ae2.predict(np.expand_dims(orig, 0))[0]
                plt.subplot(total_rows, 2, i)
                plt.imshow(orig.squeeze(), cmap='gray')
                plt.title('Real Orig')
                plt.axis('off')
                i += 1
                plt.subplot(total_rows, 2, i)
                plt.imshow(recon.squeeze(), cmap='gray')
                plt.title('Real Recon (paper_ae2)')
                plt.axis('off')
                i += 1
            for idx in idxs_forged:
                orig = x_p_forged[idx]
                recon = model_paper_ae2.predict(np.expand_dims(orig, 0))[0]
                plt.subplot(total_rows, 2, i)
                plt.imshow(orig.squeeze(), cmap='gray')
                plt.title('Forged Orig')
                plt.axis('off')
                i += 1
                plt.subplot(total_rows, 2, i)
                plt.imshow(recon.squeeze(), cmap='gray')
                plt.title('Forged Recon (paper_ae2)')
                plt.axis('off')
                i += 1
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'paper_ae2-reconstructions.png'))
            plt.close()
    except Exception:
        pass

def save_digital_plots():
    # ROC for digital classifier
    y_scores_d = model_digital_clf.predict(Xd_test).ravel()
    fpr, tpr, _ = roc_curve(Yd_test, y_scores_d)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC={roc_auc_score(Yd_test, y_scores_d):.3f}')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Digital LSTM classifier ROC (digital_clf)')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'digital_clf-roc.png'))
    plt.close()

    # AE error hist (digital)
    err_ae = calculate_errors(model_digital_ae, x_d_test_real)
    err_gru = calculate_errors(model_digital_gru, x_d_test_real)
    plt.figure()
    plt.hist(err_ae, bins=30, alpha=0.6, label='LSTM-AE')
    plt.hist(err_gru, bins=30, alpha=0.6, label='GRU-AE')
    plt.title('Digital AE error distribution (digital_ae vs digital_gru)')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'digital_ae-error-hist-compare.png'))
    plt.close()

    # Reconstructions examples (digital) - plot first channel (x) for a few samples
    n = min(3, len(x_d_test_real))
    if n>0:
        plt.figure(figsize=(8, 3*n))
        for i in range(n):
            orig = x_d_test_real[i]
            recon = model_digital_ae.predict(np.expand_dims(orig,0))[0]
            plt.subplot(n, 2, 2*i+1)
            plt.plot(orig[:,0])
            plt.title('Orig X coord')
            plt.subplot(n, 2, 2*i+2)
            plt.plot(recon[:,0])
            plt.title('Recon X coord')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'digital_ae-reconstructions.png'))
        plt.close()
    # GRU reconstructions
    try:
        if n>0:
            plt.figure(figsize=(8, 3*n))
            for i in range(n):
                orig = x_d_test_real[i]
                recon = model_digital_gru.predict(np.expand_dims(orig,0))[0]
                plt.subplot(n, 2, 2*i+1)
                plt.plot(orig[:,0])
                plt.title('Orig X coord')
                plt.subplot(n, 2, 2*i+2)
                plt.plot(recon[:,0])
                plt.title('Recon X coord (GRU)')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'digital_gru-reconstructions.png'))
            plt.close()
    except Exception:
        pass

# Save charts to files
try:
    save_paper_plots()
except Exception as e:
    print('Warning: could not save paper plots:', e)
try:
    save_digital_plots()
except Exception as e:
    print('Warning: could not save digital plots:', e)


# --- Additional evaluation plots: PR curves, confusion matrices, t-SNE, summary table ---
def choose_threshold_roc(y, scores):
    fpr, tpr, thr = roc_curve(y, scores)
    j = np.argmax(tpr - fpr)
    return thr[j]

def plot_pr_curve(y, scores, filename):
    precision, recall, _ = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores)
    plt.figure()
    plt.plot(recall, precision, label=f'AP={ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

def plot_confusion(y_true, scores, threshold, filename, labels=['Real','Forged']):
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.xticks([0,1], labels)
    plt.yticks([0,1], labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

def compute_and_save_additional_metrics():
    results = []
    # Paper classifier PR + confusion
    y_scores_p = model_paper_cnn.predict(X_p_test).ravel()
    auc_p = roc_auc_score(Y_p_test, y_scores_p)
    ap_p = average_precision_score(Y_p_test, y_scores_p)
    thr_p = choose_threshold_roc(Y_p_test, y_scores_p)
    plot_pr_curve(Y_p_test, y_scores_p, 'paper_cnn-pr.png')
    plot_confusion(Y_p_test, y_scores_p, thr_p, 'paper_cnn-confusion.png')
    tn, fp, fn, tp = confusion_matrix(Y_p_test, (y_scores_p>=thr_p).astype(int)).ravel()
    acc = (tp+tn) / (tp+tn+fp+fn)
    results.append({'model':'paper_cnn','modality':'paper','auc':auc_p,'ap':ap_p,'acc':acc})

    # Paper AE models
    err_r = calculate_errors(model_paper_ae, x_p_test_real)
    err_f = calculate_errors(model_paper_ae, x_p_forged)
    y_ae = np.concatenate([np.zeros_like(err_r), np.ones_like(err_f)])
    scores_ae = np.concatenate([err_r, err_f])
    auc_ae = roc_auc_score(y_ae, scores_ae)
    ap_ae = average_precision_score(y_ae, scores_ae)
    thr_ae = choose_threshold_roc(y_ae, scores_ae)
    plot_pr_curve(y_ae, scores_ae, 'paper_ae-pr.png')
    plot_confusion(y_ae, scores_ae, thr_ae, 'paper_ae-confusion.png')
    cm = confusion_matrix(y_ae, (scores_ae>=thr_ae).astype(int))
    acc = np.trace(cm) / np.sum(cm)
    results.append({'model':'paper_ae','modality':'paper','auc':auc_ae,'ap':ap_ae,'acc':acc})

    err_r2 = calculate_errors(model_paper_ae2, x_p_test_real)
    err_f2 = calculate_errors(model_paper_ae2, x_p_forged)
    y_ae2 = np.concatenate([np.zeros_like(err_r2), np.ones_like(err_f2)])
    scores_ae2 = np.concatenate([err_r2, err_f2])
    auc_ae2 = roc_auc_score(y_ae2, scores_ae2)
    ap_ae2 = average_precision_score(y_ae2, scores_ae2)
    thr_ae2 = choose_threshold_roc(y_ae2, scores_ae2)
    plot_pr_curve(y_ae2, scores_ae2, 'paper_ae2-pr.png')
    plot_confusion(y_ae2, scores_ae2, thr_ae2, 'paper_ae2-confusion.png')
    cm2 = confusion_matrix(y_ae2, (scores_ae2>=thr_ae2).astype(int))
    acc2 = np.trace(cm2) / np.sum(cm2)
    results.append({'model':'paper_ae2','modality':'paper','auc':auc_ae2,'ap':ap_ae2,'acc':acc2})

    # Paper deep CNN metrics
    try:
        y_scores_p_deep = model_paper_cnn_deep.predict(X_p_test).ravel()
        auc_p_deep = roc_auc_score(Y_p_test, y_scores_p_deep)
        ap_p_deep = average_precision_score(Y_p_test, y_scores_p_deep)
        thr_p_deep = choose_threshold_roc(Y_p_test, y_scores_p_deep)
        plot_pr_curve(Y_p_test, y_scores_p_deep, 'paper_cnn_deep-pr.png')
        plot_confusion(Y_p_test, y_scores_p_deep, thr_p_deep, 'paper_cnn_deep-confusion.png')
        cm = confusion_matrix(Y_p_test, (y_scores_p_deep>=thr_p_deep).astype(int))
        acc_deep = np.trace(cm) / np.sum(cm)
        results.append({'model':'paper_cnn_deep','modality':'paper','auc':auc_p_deep,'ap':ap_p_deep,'acc':acc_deep})
    except Exception:
        pass

    # Digital: classifier
    y_scores_d = model_digital_clf.predict(Xd_test).ravel()
    auc_d = roc_auc_score(Yd_test, y_scores_d)
    ap_d = average_precision_score(Yd_test, y_scores_d)
    thr_d = choose_threshold_roc(Yd_test, y_scores_d)
    plot_pr_curve(Yd_test, y_scores_d, 'digital_clf-pr.png')
    plot_confusion(Yd_test, y_scores_d, thr_d, 'digital_clf-confusion.png')
    cm = confusion_matrix(Yd_test, (y_scores_d>=thr_d).astype(int))
    accd = np.trace(cm) / np.sum(cm)
    results.append({'model':'digital_clf','modality':'digital','auc':auc_d,'ap':ap_d,'acc':accd})

    # Digital AE models: compute errors on combined Xd_test (real+neg)
    err_ae_all = calculate_errors(model_digital_ae, Xd_test)
    auc_d_ae = roc_auc_score(Yd_test, err_ae_all)
    ap_d_ae = average_precision_score(Yd_test, err_ae_all)
    thr_d_ae = choose_threshold_roc(Yd_test, err_ae_all)
    plot_pr_curve(Yd_test, err_ae_all, 'digital_ae-pr.png')
    plot_confusion(Yd_test, err_ae_all, thr_d_ae, 'digital_ae-confusion.png')
    cm = confusion_matrix(Yd_test, (err_ae_all>=thr_d_ae).astype(int))
    acc_ae = np.trace(cm) / np.sum(cm)
    results.append({'model':'digital_ae','modality':'digital','auc':auc_d_ae,'ap':ap_d_ae,'acc':acc_ae})

    err_gru_all = calculate_errors(model_digital_gru, Xd_test)
    auc_d_gru = roc_auc_score(Yd_test, err_gru_all)
    ap_d_gru = average_precision_score(Yd_test, err_gru_all)
    thr_d_gru = choose_threshold_roc(Yd_test, err_gru_all)
    plot_pr_curve(Yd_test, err_gru_all, 'digital_gru-pr.png')
    plot_confusion(Yd_test, err_gru_all, thr_d_gru, 'digital_gru-confusion.png')
    cm = confusion_matrix(Yd_test, (err_gru_all>=thr_d_gru).astype(int))
    acc_gru = np.trace(cm) / np.sum(cm)
    results.append({'model':'digital_gru_ae','modality':'digital','auc':auc_d_gru,'ap':ap_d_gru,'acc':acc_gru})

    # Digital Transformer classifier metrics
    try:
        y_scores_dt = model_digital_transformer.predict(Xd_test).ravel()
        auc_dt = roc_auc_score(Yd_test, y_scores_dt)
        ap_dt = average_precision_score(Yd_test, y_scores_dt)
        thr_dt = choose_threshold_roc(Yd_test, y_scores_dt)
        plot_pr_curve(Yd_test, y_scores_dt, 'digital_transformer-pr.png')
        plot_confusion(Yd_test, y_scores_dt, thr_dt, 'digital_transformer-confusion.png')
        cm = confusion_matrix(Yd_test, (y_scores_dt>=thr_dt).astype(int))
        acc_dt = np.trace(cm) / np.sum(cm)
        results.append({'model':'digital_transformer','modality':'digital','auc':auc_dt,'ap':ap_dt,'acc':acc_dt})
    except Exception:
        pass
    # UMAP / dimensionality reduction plots (preferred: UMAP; fallback: TSNE)
    try:
        try:
            import umap as _umap
            reducer = _umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
            reducer_name = 'umap'
        except Exception:
            from sklearn.manifold import TSNE as _TSNE
            reducer = _TSNE(n_components=2, random_state=42)
            reducer_name = 'tsne'

        # Paper embedding
        try:
            if 'encoder_paper' in globals() and encoder_paper is not None:
                enc_p = encoder_paper
            else:
                enc_p = tf.keras.Model(inputs=model_paper_ae.input, outputs=model_paper_ae.layers[4].output)
            sample_n = min(300, len(X_p))
            idxs = np.random.choice(len(X_p), sample_n, replace=False)
            _ = model_paper_ae.predict(X_p[idxs[:min(5,len(idxs))]])
            latent = enc_p.predict(X_p[idxs])
            latent_flat = latent.reshape((latent.shape[0], -1))
            emb = reducer.fit_transform(latent_flat)
            labels = Y_p[idxs]
            plt.figure(figsize=(6,6))
            plt.scatter(emb[:,0], emb[:,1], c=labels, cmap='coolwarm', s=8)
            plt.title(f'Paper latent ({reducer_name.upper()})')
            plt.savefig(os.path.join(RESULTS_DIR, f'paper-{reducer_name}.png'))
            plt.close()
        except Exception as e:
            print('Warning: paper embedding failed:', e)

        # Digital embedding (use AE encoder)
        try:
            if 'encoder_digital' in globals() and encoder_digital is not None:
                enc_d = encoder_digital
            else:
                enc_d = tf.keras.Model(inputs=model_digital_ae.input, outputs=model_digital_ae.layers[0].output)
            sample_n = min(300, len(Xd_test))
            idxs = np.random.choice(len(Xd_test), sample_n, replace=False)
            _ = model_digital_ae.predict(Xd_test[idxs[:min(5,len(idxs))]])
            latent = enc_d.predict(Xd_test[idxs])
            latent_flat = latent.reshape((latent.shape[0], -1)) if latent.ndim>2 else latent
            emb = reducer.fit_transform(latent_flat)
            labels = Yd_test[idxs]
            plt.figure(figsize=(6,6))
            plt.scatter(emb[:,0], emb[:,1], c=labels, cmap='coolwarm', s=8)
            plt.title(f'Digital latent ({reducer_name.upper()})')
            plt.savefig(os.path.join(RESULTS_DIR, f'digital-{reducer_name}.png'))
            plt.close()
        except Exception as e:
            print('Warning: digital embedding failed:', e)

    except Exception as e:
        print('Warning: embedding reducer setup failed:', e)

    # Summary table and heatmap separated by modality
    df = pd.DataFrame(results)
    df = df[['model','modality','auc','ap','acc']]
    df_sorted = df.sort_values(['modality','auc'], ascending=[True, False])
    
    # === SUMMARY TABLES (separated by paper and digital) ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Paper models summary
    paper_data = df_sorted[df_sorted['modality'] == 'paper']
    axes[0].axis('off')
    tbl_p = axes[0].table(cellText=np.round(paper_data[['auc','ap','acc']].values, 3),
                          rowLabels=paper_data['model'].values,
                          colLabels=['AUC','AP','ACC'],
                          loc='center')
    tbl_p.auto_set_font_size(False)
    tbl_p.set_fontsize(10)
    tbl_p.scale(1, 2)
    axes[0].set_title('Paper Models: Summary', fontsize=12, fontweight='bold', pad=10)
    
    # Digital models summary
    digital_data = df_sorted[df_sorted['modality'] == 'digital']
    axes[1].axis('off')
    tbl_d = axes[1].table(cellText=np.round(digital_data[['auc','ap','acc']].values, 3),
                          rowLabels=digital_data['model'].values,
                          colLabels=['AUC','AP','ACC'],
                          loc='center')
    tbl_d.auto_set_font_size(False)
    tbl_d.set_fontsize(10)
    tbl_d.scale(1, 2)
    axes[1].set_title('Digital Models: Summary', fontsize=12, fontweight='bold', pad=10)
    
    fig.suptitle('Model Performance Summary (Paper vs Digital)', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'summary-results.png'), bbox_inches='tight', dpi=150)
    plt.close()
    
    # === HEATMAPS (separated by paper and digital) ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Paper models heatmap
    paper_heatmap_data = paper_data[['auc', 'ap', 'acc']].values
    im_p = axes[0].imshow(paper_heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[0].set_xticks(np.arange(3))
    axes[0].set_yticks(np.arange(len(paper_data)))
    axes[0].set_xticklabels(['AUC', 'AP', 'ACC'], fontsize=11)
    axes[0].set_yticklabels(paper_data['model'], fontsize=11)
    for i in range(len(paper_data)):
        for j in range(3):
            axes[0].text(j, i, f'{paper_heatmap_data[i, j]:.3f}',
                        ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    axes[0].set_title('Paper Models: Performance Heatmap', fontsize=12, fontweight='bold', pad=10)
    cbar_p = plt.colorbar(im_p, ax=axes[0])
    cbar_p.set_label('Score', fontsize=11)
    
    # Digital models heatmap
    digital_heatmap_data = digital_data[['auc', 'ap', 'acc']].values
    im_d = axes[1].imshow(digital_heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[1].set_xticks(np.arange(3))
    axes[1].set_yticks(np.arange(len(digital_data)))
    axes[1].set_xticklabels(['AUC', 'AP', 'ACC'], fontsize=11)
    axes[1].set_yticklabels(digital_data['model'], fontsize=11)
    for i in range(len(digital_data)):
        for j in range(3):
            axes[1].text(j, i, f'{digital_heatmap_data[i, j]:.3f}',
                        ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    axes[1].set_title('Digital Models: Performance Heatmap', fontsize=12, fontweight='bold', pad=10)
    cbar_d = plt.colorbar(im_d, ax=axes[1])
    cbar_d.set_label('Score', fontsize=11)
    
    fig.suptitle('Model Performance Heatmap (Paper vs Digital)', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'comparison-metrics-heatmap.png'), bbox_inches='tight', dpi=150)
    plt.close()

    # === NEW COMPARISON PLOTS ===
    
    # 1. Multi-model ROC overlays (paper and digital separated)
    try:
        # Paper ROC overlay (4 models)
        fig, ax = plt.subplots(figsize=(8, 6))
        # paper_cnn ROC
        y_scores_p = model_paper_cnn.predict(X_p_test).ravel()
        fpr, tpr, _ = roc_curve(Y_p_test, y_scores_p)
        auc_p = roc_auc_score(Y_p_test, y_scores_p)
        ax.plot(fpr, tpr, label=f'paper_cnn (AUC={auc_p:.3f})', linewidth=2)
        # paper_ae ROC
        err_r = calculate_errors(model_paper_ae, x_p_test_real)
        err_f = calculate_errors(model_paper_ae, x_p_forged)
        y_ae = np.concatenate([np.zeros_like(err_r), np.ones_like(err_f)])
        scores_ae = np.concatenate([err_r, err_f])
        fpr, tpr, _ = roc_curve(y_ae, scores_ae)
        auc_ae = roc_auc_score(y_ae, scores_ae)
        ax.plot(fpr, tpr, label=f'paper_ae (AUC={auc_ae:.3f})', linewidth=2)
        # paper_ae2 ROC
        err_r2 = calculate_errors(model_paper_ae2, x_p_test_real)
        err_f2 = calculate_errors(model_paper_ae2, x_p_forged)
        y_ae2 = np.concatenate([np.zeros_like(err_r2), np.ones_like(err_f2)])
        scores_ae2 = np.concatenate([err_r2, err_f2])
        fpr, tpr, _ = roc_curve(y_ae2, scores_ae2)
        auc_ae2 = roc_auc_score(y_ae2, scores_ae2)
        ax.plot(fpr, tpr, label=f'paper_ae2 (AUC={auc_ae2:.3f})', linewidth=2)
        # paper_cnn_deep ROC
        try:
            y_scores_p_deep = model_paper_cnn_deep.predict(X_p_test).ravel()
            fpr, tpr, _ = roc_curve(Y_p_test, y_scores_p_deep)
            auc_p_deep = roc_auc_score(Y_p_test, y_scores_p_deep)
            ax.plot(fpr, tpr, label=f'paper_cnn_deep (AUC={auc_p_deep:.3f})', linewidth=2)
        except Exception as e:
            print(f'Warning: paper_cnn_deep ROC failed: {e}')
        ax.plot([0,1],[0,1],'--', color='gray', label='Random')
        ax.set_xlabel('FPR', fontsize=11)
        ax.set_ylabel('TPR', fontsize=11)
        ax.set_title('Paper Models: ROC Comparison', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'comparison-paper-roc.png'), dpi=150)
        plt.close()
        # Digital ROC overlay (4 models)
        fig, ax = plt.subplots(figsize=(8, 6))
        # digital_clf ROC
        y_scores_d = model_digital_clf.predict(Xd_test).ravel()
        fpr, tpr, _ = roc_curve(Yd_test, y_scores_d)
        auc_d = roc_auc_score(Yd_test, y_scores_d)
        ax.plot(fpr, tpr, label=f'digital_clf (AUC={auc_d:.3f})', linewidth=2)
        # digital_ae ROC
        err_ae_all = calculate_errors(model_digital_ae, Xd_test)
        fpr, tpr, _ = roc_curve(Yd_test, err_ae_all)
        auc_d_ae = roc_auc_score(Yd_test, err_ae_all)
        ax.plot(fpr, tpr, label=f'digital_ae (AUC={auc_d_ae:.3f})', linewidth=2)
        # digital_gru ROC
        err_gru_all = calculate_errors(model_digital_gru, Xd_test)
        fpr, tpr, _ = roc_curve(Yd_test, err_gru_all)
        auc_d_gru = roc_auc_score(Yd_test, err_gru_all)
        ax.plot(fpr, tpr, label=f'digital_gru_ae (AUC={auc_d_gru:.3f})', linewidth=2)
        # digital_transformer ROC
        try:
            y_scores_dt = model_digital_transformer.predict(Xd_test).ravel()
            fpr, tpr, _ = roc_curve(Yd_test, y_scores_dt)
            auc_dt = roc_auc_score(Yd_test, y_scores_dt)
            ax.plot(fpr, tpr, label=f'digital_transformer (AUC={auc_dt:.3f})', linewidth=2)
        except Exception as e:
            print(f'Warning: digital_transformer ROC failed: {e}')
        ax.plot([0,1],[0,1],'--', color='gray', label='Random')
        ax.set_xlabel('FPR', fontsize=11)
        ax.set_ylabel('TPR', fontsize=11)
        ax.set_title('Digital Models: ROC Comparison', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'comparison-digital-roc.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f'Warning: ROC overlay plots failed: {e}')
    # 2. Multi-model PR overlays (paper and digital separated)
    try:
        # Paper PR overlay (4 models)
        fig, ax = plt.subplots(figsize=(8, 6))
        # paper_cnn PR
        y_scores_p = model_paper_cnn.predict(X_p_test).ravel()
        precision, recall, _ = precision_recall_curve(Y_p_test, y_scores_p)
        ap_p = average_precision_score(Y_p_test, y_scores_p)
        ax.plot(recall, precision, label=f'paper_cnn (AP={ap_p:.3f})', linewidth=2)
        # paper_ae PR
        precision, recall, _ = precision_recall_curve(y_ae, scores_ae)
        ap_ae = average_precision_score(y_ae, scores_ae)
        ax.plot(recall, precision, label=f'paper_ae (AP={ap_ae:.3f})', linewidth=2)
        # paper_ae2 PR
        precision, recall, _ = precision_recall_curve(y_ae2, scores_ae2)
        ap_ae2 = average_precision_score(y_ae2, scores_ae2)
        ax.plot(recall, precision, label=f'paper_ae2 (AP={ap_ae2:.3f})', linewidth=2)
        # paper_cnn_deep PR
        try:
            y_scores_p_deep = model_paper_cnn_deep.predict(X_p_test).ravel()
            precision, recall, _ = precision_recall_curve(Y_p_test, y_scores_p_deep)
            ap_p_deep = average_precision_score(Y_p_test, y_scores_p_deep)
            ax.plot(recall, precision, label=f'paper_cnn_deep (AP={ap_p_deep:.3f})', linewidth=2)
        except Exception as e:
            print(f'Warning: paper_cnn_deep PR failed: {e}')
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title('Paper Models: Precision-Recall Comparison', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'comparison-paper-pr.png'), dpi=150)
        plt.close()
        # Digital PR overlay (4 models)
        fig, ax = plt.subplots(figsize=(8, 6))
        # digital_clf PR
        precision, recall, _ = precision_recall_curve(Yd_test, y_scores_d)
        ap_d = average_precision_score(Yd_test, y_scores_d)
        ax.plot(recall, precision, label=f'digital_clf (AP={ap_d:.3f})', linewidth=2)
        # digital_ae PR
        precision, recall, _ = precision_recall_curve(Yd_test, err_ae_all)
        ap_d_ae = average_precision_score(Yd_test, err_ae_all)
        ax.plot(recall, precision, label=f'digital_ae (AP={ap_d_ae:.3f})', linewidth=2)
        # digital_gru PR
        precision, recall, _ = precision_recall_curve(Yd_test, err_gru_all)
        ap_d_gru = average_precision_score(Yd_test, err_gru_all)
        ax.plot(recall, precision, label=f'digital_gru_ae (AP={ap_d_gru:.3f})', linewidth=2)
        # digital_transformer PR
        try:
            y_scores_dt = model_digital_transformer.predict(Xd_test).ravel()
            precision, recall, _ = precision_recall_curve(Yd_test, y_scores_dt)
            ap_dt = average_precision_score(Yd_test, y_scores_dt)
            ax.plot(recall, precision, label=f'digital_transformer (AP={ap_dt:.3f})', linewidth=2)
        except Exception as e:
            print(f'Warning: digital_transformer PR failed: {e}')
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title('Digital Models: Precision-Recall Comparison', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'comparison-digital-pr.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f'Warning: PR overlay plots failed: {e}')
    
    # 3. Bar chart: AUC/AP/ACC per model (separated by paper and digital)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Paper models
        paper_models = ['paper_cnn', 'paper_ae', 'paper_ae2']
        paper_data = df_sorted[df_sorted['model'].isin(paper_models)]
        x_pos = np.arange(len(paper_data))
        width = 0.25
        axes[0].bar(x_pos - width, paper_data['auc'], width, label='AUC', alpha=0.8)
        axes[0].bar(x_pos, paper_data['ap'], width, label='AP', alpha=0.8)
        axes[0].bar(x_pos + width, paper_data['acc'], width, label='ACC', alpha=0.8)
        axes[0].set_ylabel('Score', fontsize=11)
        axes[0].set_title('Paper Models: Metrics Comparison', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(paper_data['model'], rotation=15)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim([0, 1.1])
        
        # Digital models
        digital_models = ['digital_clf', 'digital_ae', 'digital_gru_ae']
        digital_data = df_sorted[df_sorted['model'].isin(digital_models)]
        x_pos = np.arange(len(digital_data))
        axes[1].bar(x_pos - width, digital_data['auc'], width, label='AUC', alpha=0.8)
        axes[1].bar(x_pos, digital_data['ap'], width, label='AP', alpha=0.8)
        axes[1].bar(x_pos + width, digital_data['acc'], width, label='ACC', alpha=0.8)
        axes[1].set_ylabel('Score', fontsize=11)
        axes[1].set_title('Digital Models: Metrics Comparison', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(digital_data['model'], rotation=15)
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'comparison-metrics-bars.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f'Warning: bar chart failed: {e}')
    
    # 4. Error distribution box plots (AE models only)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Paper AE errors
        err_p_ae = calculate_errors(model_paper_ae, x_p_forged)
        err_p_ae2 = calculate_errors(model_paper_ae2, x_p_forged)
        axes[0].boxplot([err_p_ae, err_p_ae2], labels=['paper_ae', 'paper_ae2'])
        axes[0].set_ylabel('Reconstruction Error', fontsize=11)
        axes[0].set_title('Paper AE: Error Distribution (on forged)', fontsize=12, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Digital AE errors
        err_d_ae = calculate_errors(model_digital_ae, x_d_f_test)
        err_d_gru = calculate_errors(model_digital_gru, x_d_f_test)
        axes[1].boxplot([err_d_ae, err_d_gru], labels=['digital_ae', 'digital_gru'])
        axes[1].set_ylabel('Reconstruction Error', fontsize=11)
        axes[1].set_title('Digital AE: Error Distribution (on forged)', fontsize=12, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'comparison-error-distributions.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f'Warning: error distribution box plots failed: {e}')
    
    # 5. Metrics heatmap (separated by paper and digital, saved as comparison-metrics-heatmap.png)

    print('\nSaved PR, confusion matrices, t-SNE and comparison plots')

try:
    compute_and_save_additional_metrics()
except Exception as e:
    print('Warning: additional metric generation failed:', e)