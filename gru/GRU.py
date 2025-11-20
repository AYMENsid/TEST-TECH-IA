import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from statsmodels.tsa.seasonal import STL

# --- Configuration ---
df = pd.read_csv('all_fuels_data.csv', parse_dates=['date'])
commodities = df['commodity'].unique()
seq_len = 180      # fenêtre historique
horizon = 30       # horizon réduit pour backtest fiable
features = ['close', 'high', 'low', 'volume']
total_days = 365   # prédiction un an

# Stockage résultats
evaluation = {}
all_ytrue_ytest = []  # pour CSV y_true vs y_pred
all_future_preds = []  # pour CSV futures

for commodity in commodities:
    print(f"Processing {commodity}...")
    sub = df[df['commodity']==commodity].sort_values('date').reset_index(drop=True)
    data = sub[features].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    def create_seq(data, seq_len, horizon):
        X, y = [], []
        for i in range(len(data) - seq_len - horizon + 1):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len:i+seq_len+horizon, 0])
        return np.array(X), np.array(y)

    X, y = create_seq(scaled, seq_len, horizon)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # --- Modèle GRU multi-horizon ---
    inp = Input(shape=(seq_len, len(features)))
    x = GRU(128)(inp)
    x = Dropout(0.2)(x)
    out = Dense(horizon)(x)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    callbacks = [
        EarlyStopping('val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau('val_loss', patience=5, factor=0.5, min_lr=1e-5)
    ]
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )

    # Évaluation train/test (pas 1)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    metrics = {'Train':{}, 'Test':{}}
    for name, yt, yp in [('Train', y_train, y_pred_train), ('Test', y_test, y_pred_test)]:
        y_true = yt[:,0]
        y_pred = yp[:,0]
        metrics[name] = {
            'MAE': float(mean_absolute_error(y_true, y_pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'R2': float(r2_score(y_true, y_pred))
        }
        dates = sub['date'].iloc[seq_len:seq_len+len(y_true)]
        all_ytrue_ytest += list(zip(dates, [commodity]*len(y_true), y_true.tolist(), y_pred.tolist()))
    evaluation[commodity] = metrics

    # --- Backtest rolling 30j ---
    bt_maes = []
    close_scaler = MinMaxScaler(); close_scaler.min_, close_scaler.scale_ = scaler.min_[0], scaler.scale_[0]
    for w in range(5):
        start = len(sub) - (seq_len + horizon) - w*horizon
        Xb, yb = create_seq(scaled[start:], seq_len, horizon)
        y_true_b = sub['close'].iloc[start+seq_len:start+seq_len+horizon].values
        y_pred_b = model.predict(Xb[:1])[0]
        y_pred_b_inv = close_scaler.inverse_transform(y_pred_b.reshape(-1,1)).flatten()
        bt_maes.append(mean_absolute_error(y_true_b, y_pred_b_inv))
    evaluation[commodity]['Backtest_MAE30d'] = float(np.mean(bt_maes))

    # --- Décomposition STL ---
    res = STL(sub['close'], period=365).fit()
    trend, seasonal, resid = res.trend, res.seasonal, res.resid
    coefs = np.polyfit(np.arange(len(trend)), trend, 1)
    trend_f = np.polyval(coefs, np.arange(len(trend), len(trend)+total_days))
    season_cycle = seasonal[-365:]
    season_f = np.tile(season_cycle, int(np.ceil(total_days/365)))[:total_days]

    # GRU sur résidu multi-horizon
    scaled_res = MinMaxScaler().fit_transform(resid.values.reshape(-1,1))
    Xr, yr = create_seq(scaled_res, seq_len, horizon)
    Xr_train, Xr_test = Xr[:split], Xr[split:]
    yr_train = yr[:split]

    ir = Input(shape=(seq_len,1))
    gr = GRU(128)(ir)
    gr = Dropout(0.2)(gr)
    out_r = Dense(horizon)(gr)
    rmodel = Model(ir, out_r)
    rmodel.compile(optimizer='adam', loss='mse')
    rmodel.fit(
        Xr_train, yr_train,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )

    # prédiction future
    resid_preds = []
    seq_r = scaled_res[-seq_len:]
    for _ in range(int(np.ceil(total_days/horizon))):
        pr = rmodel.predict(seq_r.reshape(1,seq_len,1))[0]
        resid_preds.extend(pr.tolist())
        seq_r = np.roll(seq_r, -horizon, axis=0)
        seq_r[-horizon:,0] = pr
    resid_preds = np.array(resid_preds[:total_days]).reshape(-1,1)
    res_scaler = MinMaxScaler().fit(resid.values.reshape(-1,1))
    resid_preds = res_scaler.inverse_transform(resid_preds).flatten()

    final = trend_f + season_f + resid_preds
    future_dates = pd.date_range(start=sub['date'].iloc[-1]+pd.Timedelta(days=1), periods=total_days)
    all_future_preds += list(zip(future_dates, [commodity]*total_days, final.tolist()))

# --- Sauvegarde fichiers ---
with open('gru_metrics.json','w') as f: json.dump(evaluation, f, indent=4)

global_metrics = {'train':{}, 'test':{}}
for phase in ['train','test']:
    for m in ['MAE','RMSE','R2']:
        global_metrics[phase][m] = float(np.mean([evaluation[c][phase][m] for c in commodities]))
with open('gru_global_metrics.json','w') as f: json.dump(global_metrics, f, indent=4)

df_y = pd.DataFrame(all_ytrue_ytest, columns=['date','commodity','y_true','y_pred'])
df_y.to_csv('gru_result_normaliser.csv', index=False)

df_fut = pd.DataFrame(all_future_preds, columns=['date','commodity','predicted_close'])
df_fut.to_csv('gru_future.csv', index=False)

print(" Fichiers générés pour toutes les commodities.")
