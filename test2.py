import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# ============ 1. Chargement des données ============
print("Chargement des données...")
DATA_PATH = "MetroPT3(AirCompressor).csv"  # Vérifier le nom du fichier

try:
    df = pd.read_csv(DATA_PATH)
    print("✓ Fichier chargé avec succès")
except FileNotFoundError:
    print(f"✗ Fichier introuvable : {DATA_PATH}")
    print("Vérifiez le chemin ou le nom du fichier")
    exit()

# Supprimer la première colonne si c'est un index
if df.columns[0].lower() in ["unnamed: 0", "index"]:
    df = df.drop(df.columns[0], axis=1)

df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"Nombre de lignes : {len(df)}")
print(f"Colonnes : {df.columns.tolist()}")
print(f"Période des données : {df['timestamp'].min()} à {df['timestamp'].max()}")

# ============ 2. Définition des périodes de panne ============
failure_windows = [
    ("2020-04-11 11:50:00", "2020-04-12 23:30:00"),
    ("2020-04-17 08:00:00", "2020-04-19 01:30:00"),
    ("2020-04-18 23:00:00", "2020-04-20 22:00:00"),
    ("2020-05-12 14:00:00", "2020-05-13 23:59:00"),
    ("2020-05-17 05:00:00", "2020-05-20 20:00:00"),
    ("2020-05-28 23:33:00", "2020-05-30 06:00:00")
]

df["is_anomaly"] = 0
for start, end in failure_windows:
    mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    df.loc[mask, "is_anomaly"] = 1

print(f"\nNombre d'anomalies : {df['is_anomaly'].sum()}")

# ============ 3. Sélection des caractéristiques numériques ============
feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols.remove("is_anomaly")
print(f"Nombre de features : {len(feature_cols)}")

X = df[feature_cols].values
y = df["is_anomaly"].values

# ============ 4. Division des données ============
train_mask = (df["timestamp"] >= "2020-02-01") & (df["timestamp"] < "2020-03-20") & (df["is_anomaly"] == 0)
test_mask = (df["timestamp"] >= "2020-04-01") & (df["timestamp"] < "2020-07-31")

X_train = X[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

print(f"\nDonnées d'entraînement : {X_train.shape}")
print(f"Données de test : {X_test.shape}")

# ============ 5. Normalisation des données ============
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============ 6. Création des séquences temporelles ============
TIMESTEPS = 30

def create_sequences(data, timesteps):
    Xs = []
    for i in range(len(data) - timesteps):
        Xs.append(data[i:i+timesteps])
    return np.array(Xs)

X_train_seq = create_sequences(X_train, TIMESTEPS)
X_test_seq = create_sequences(X_test, TIMESTEPS)
y_test_seq = y_test[TIMESTEPS:]

print(f"\nSéquences d'entraînement : {X_train_seq.shape}")
print(f"Séquences de test : {X_test_seq.shape}")

# ============ 7. Construction du modèle ============
n_features = X_train_seq.shape[2]

inputs = Input(shape=(TIMESTEPS, n_features))
encoded = LSTM(64, activation="tanh", return_sequences=False)(inputs)
encoded = RepeatVector(TIMESTEPS)(encoded)
decoded = LSTM(64, activation="tanh", return_sequences=True)(encoded)
decoded = TimeDistributed(Dense(n_features))(decoded)

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

# ============ 8. Entraînement du modèle ============
print("\nEntraînement du modèle...")
history = autoencoder.fit(
    X_train_seq, X_train_seq,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    shuffle=True,
    verbose=1
)

# ============ 9. Calcul de l'erreur de reconstruction ============
print("\nCalcul des erreurs...")
X_train_pred = autoencoder.predict(X_train_seq, verbose=0)
train_error = np.mean(np.square(X_train_pred - X_train_seq), axis=(1, 2))

X_test_pred = autoencoder.predict(X_test_seq, verbose=0)
reconstruction_error = np.mean(np.square(X_test_pred - X_test_seq), axis=(1, 2))

threshold = np.percentile(train_error, 95)
y_pred = (reconstruction_error > threshold).astype(int)

# ============ 10. Évaluation du modèle ============
precision, recall, f1, _ = precision_recall_fscore_support(y_test_seq, y_pred, average="binary")
auc = roc_auc_score(y_test_seq, reconstruction_error)

print("\n" + "="*40)
print("Résultats")
print("="*40)
print(f"Seuil (Threshold) : {threshold:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC : {auc:.4f}")
print("="*40)

# ============ 11. Visualisation des résultats ============
plt.figure(figsize=(15, 10))

# Historique de l'entraînement
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Loss entraînement')
plt.plot(history.history['val_loss'], label='Loss validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Historique d’entraînement')
plt.grid(True)

# Distribution des erreurs
plt.subplot(2, 2, 2)
plt.hist(train_error, bins=50, alpha=0.7, label='Erreurs train', density=True)
plt.hist(reconstruction_error, bins=50, alpha=0.7, label='Erreurs test', density=True)
plt.axvline(threshold, color='r', linestyle='--', label=f'Seuil (95%)')
plt.xlabel('Erreur de reconstruction')
plt.ylabel('Densité')
plt.legend()
plt.title('Distribution des erreurs')
plt.grid(True)

# Résultats dans le temps
plt.subplot(2, 1, 2)
time_indices = np.arange(len(y_test_seq))
plt.plot(time_indices, y_test_seq, label='Anomalie réelle', alpha=0.7, linewidth=2)
plt.plot(time_indices, y_pred, label='Anomalie prédite', alpha=0.7, linewidth=2)
plt.plot(time_indices, reconstruction_error / reconstruction_error.max(),
         label='Erreur normalisée', alpha=0.5)
plt.xlabel('Time Step')
plt.ylabel('Valeur')
plt.legend()
plt.title('Résultats de détection d’anomalies')
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n✓ Exécution terminée avec succès !")
