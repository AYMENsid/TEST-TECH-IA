import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import tensorflow as tf
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 
from statsmodels.tsa.seasonal import STL

# --- Configuration ---
df = pd.read_csv('all_fuels_data.csv', parse_dates=['date'])
commodities = df['commodity'].unique()
seq_len     = 180    # fenêtre historique
horizon     = 14     # horizon pour backtest et pour la prédiction STL
features    = ['close', 'high', 'low', 'volume']
total_days  = 365    # jours à prédire dans le futur

# Pour stocker les résultats
results           = {}
all_ytrue_ytest   = []
all_future_preds  = []

callbacks = [
    EarlyStopping('val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau('val_loss', patience=5, factor=0.5, min_lr=1e-5)
]

def create_seq(data, seq_len, horizon):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+horizon, 0])
    return np.array(X), np.array(y)

# --- Boucle sur chaque commodity ---
for commodity in commodities:
    print(f"Processing {commodity}...")
    sub = df[df['commodity']==commodity].sort_values('date').reset_index(drop=True)
    data = sub[features].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    # Création des séquences
    X, y = create_seq(scaled, seq_len, horizon)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # --- Seq2Seq LSTM ---
    enc_inputs    = Input(shape=(seq_len, len(features)))
    encoder, sh, sc = LSTM(128, return_state=True)(enc_inputs)
    dec_inputs    = RepeatVector(horizon)(sh)
    dec_lstm      = LSTM(128, return_sequences=True)
    dec_outputs   = dec_lstm(dec_inputs, initial_state=[sh, sc])
    dec_dense     = TimeDistributed(Dense(1))
    dec_preds     = dec_dense(dec_outputs)

    model = Model(enc_inputs, dec_preds)
    model.compile(optimizer='adam', loss='mse')
    model.fit(
        X_train, y_train.reshape(-1, horizon, 1),
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )

    # Évaluation sur train et test
    y_pred_train = model.predict(X_train)[:, :, 0]
    y_pred_test  = model.predict(X_test)[:, :, 0]
    metrics = {'Train':{}, 'Test':{}}
    for name, yt, yp in [('Train', y_train, y_pred_train), ('Test', y_test, y_pred_test)]:
        y_true = yt[:,0]
        y_pred = yp[:,0]
        metrics[name] = {
            'MAE':  float(mean_absolute_error(y_true, y_pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'R2':   float(r2_score(y_true, y_pred))
        }
        # Collecte pour CSV y_true vs y_pred
        dates = sub['date'].iloc[seq_len: seq_len + len(y_true)]
        all_ytrue_ytest += list(zip(dates, [commodity]*len(y_true), y_true.tolist(), y_pred.tolist()))
    results[commodity] = metrics

    # --- Décomposition STL + prévision future ---
    stl      = STL(sub['close'], period=365).fit()
    trend    = stl.trend
    seasonal = stl.seasonal
    resid    = stl.resid

    # Tendance extrapolée
    coefs   = np.polyfit(np.arange(len(trend)), trend, 1)
    trend_f = np.polyval(coefs, np.arange(len(trend), len(trend) + total_days))
    # Saison répétée
    season_cycle = seasonal[-365:]
    season_f     = np.tile(season_cycle, int(np.ceil(total_days/365)))[:total_days]

    # --- Modèle Seq2Seq sur les résidus ---
    scaled_res = MinMaxScaler().fit_transform(resid.values.reshape(-1,1))
    Xr, yr     = create_seq(scaled_res, seq_len, horizon)
    Xr_train   = Xr[:split]

    ir    = Input(shape=(seq_len,1))
    en_r, r_sh, r_sc = LSTM(128, return_state=True)(ir)
    r_dec = RepeatVector(horizon)(r_sh)
    dr_lstm = LSTM(128, return_sequences=True)
    dr_out  = TimeDistributed(Dense(1))(dr_lstm(r_dec, initial_state=[r_sh, r_sc]))
    rmodel  = Model(ir, dr_out)
    rmodel.compile(optimizer='adam', loss='mse')
    rmodel.fit(
        Xr_train, yr.reshape(-1, horizon, 1),
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )

    # Forecast des résidus en blocs
    res_preds, seq_r = [], scaled_res[-seq_len:]
    for _ in range(int(np.ceil(total_days/horizon))):
        pr = rmodel.predict(seq_r.reshape(1,seq_len,1))[0,:,0]
        res_preds.extend(pr.tolist())
        seq_r = np.roll(seq_r, -horizon, axis=0)
        seq_r[-horizon:,0] = pr
    res_preds = np.array(res_preds[:total_days]).reshape(-1,1)
    inv_scaler = MinMaxScaler().fit(resid.values.reshape(-1,1))
    res_preds  = inv_scaler.inverse_transform(res_preds).flatten()

    # Prédiction finale = tendance + saison + résidus
    final = trend_f + season_f + res_preds
    future_dates = pd.date_range(start=sub['date'].iloc[-1] + pd.Timedelta(days=1), periods=total_days)
    all_future_preds += list(zip(future_dates, [commodity]*total_days, final.tolist()))

# --- Sauvegarde des fichiers ---
with open('lstm_metrics.json', 'w') as f:
    json.dump(results, f, indent=4)

# Global metrics
global_metrics = {'train':{}, 'test':{}}
for phase in ['train','test']:
    for m in ['MAE','RMSE','R2']:
        global_metrics[phase][m] = float(np.mean([results[c][phase][m] for c in commodities]))
with open('lstm_global_metrics.json','w') as f:
    json.dump(global_metrics, f, indent=4)

# CSV y_true vs y_pred
df_y = pd.DataFrame(all_ytrue_ytest, columns=['date','commodity','y_true','y_pred'])
df_y.to_csv('lstm_result_normaliser.csv', index=False)

# CSV futures
df_fut = pd.DataFrame(all_future_preds, columns=['date','commodity','predicted_close'])
df_fut.to_csv('lstm_future.csv', index=False)

print(" Tous les fichiers ont été générés.")

# --- Génération des graphiques combinés ---
for commodity in commodities:
    hist = df[df['commodity']==commodity]
    fut  = df_fut[df_fut['commodity']==commodity]

    plt.figure(figsize=(10,4))
    # Historique (données réelles)
    plt.plot(hist['date'], hist['close'],
             label='Données réelles', alpha=0.7, linestyle='-')
    # Prédiction future (ligne continue)
    plt.plot(fut['date'], fut['predicted_close'],
             label='Prédiction future', linestyle='-')
    plt.title(f'{commodity} – Historique et Prédiction')
    plt.xlabel('Date')
    plt.ylabel('Prix de clôture')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()