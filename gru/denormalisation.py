import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# 1) Lecture des prédictions normalisées
df_pred = pd.read_csv('gru_result_normaliser.csv', parse_dates=['date'])

# 2) Lecture des prix réels pour refit du scaler
df_all = pd.read_csv('all_fuels_data.csv', parse_dates=['date'])

# Fichier de sortie
output_file = 'gru_results.csv'

# Supprimer le fichier s'il existe déjà
if os.path.exists(output_file):
    os.remove(output_file)

for commodity in df_pred['commodity'].unique():
    print(f"Traitement de {commodity}...")

    # Filtrage des sous-ensembles
    sub_pred = df_pred[df_pred['commodity'] == commodity].copy()
    sub_all = df_all[df_all['commodity'] == commodity].sort_values('date').reset_index(drop=True)

    # Création scaler uniquement pour 'close'
    scaler = MinMaxScaler()
    scaler.fit(sub_all[['close']])

    # Recréer un scaler compatible
    close_scaler = MinMaxScaler()
    close_scaler.min_, close_scaler.scale_ = scaler.min_[0], scaler.scale_[0]

    # Dénormalisation
    sub_pred['y_true_real'] = close_scaler.inverse_transform(sub_pred[['y_true']])
    sub_pred['y_pred_real'] = close_scaler.inverse_transform(sub_pred[['y_pred']])

    # Ne garder que les colonnes nécessaires
    sub_selected = sub_pred[['date', 'commodity', 'y_true_real', 'y_pred_real']]

    # Sauvegarde dans le fichier
    sub_selected.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))

    # Tracé
    sub_plot = sub_selected.drop_duplicates(subset='date', keep='first').sort_values('date')
    plt.figure(figsize=(10, 5))
    plt.plot(sub_plot['date'], sub_plot['y_true_real'], label='Réel')
    plt.plot(sub_plot['date'], sub_plot['y_pred_real'], label='Prédit')
    plt.title(f"{commodity} — Prix dénormalisé")
    plt.xlabel("Date")
    plt.ylabel("Prix de clôture")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
