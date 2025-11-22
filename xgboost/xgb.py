import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration
H = 365        # Horizon = 1 an
MAX_LAG = 7    # Lags utilisés
TEST_RATIO = 0.2

# Charger le dataset
df = pd.read_csv('all_fuels_data.csv', parse_dates=['date'])

# Listes pour métriques globales
all_y_true_te = []
all_y_pred_te = []
all_y_true_tr = []
all_y_pred_tr = []
results = {}
all_future_preds = []
all_ytrue_ytest = []

# Pour chaque commodity
for commodity in df['commodity'].unique():
    print(f"\n=== Commodity : {commodity} ===")
    serie = (df[df['commodity']==commodity]
             .sort_values('date')
             .set_index('date')['close']
             .asfreq('D')
             .interpolate())

    # Features
    data = pd.DataFrame(index=serie.index)
    data['dow']   = serie.index.dayofweek
    data['month'] = serie.index.month
    data['roll7'] = serie.rolling(7, min_periods=1).mean().shift(1)
    for lag in range(1, MAX_LAG+1):
        data[f'lag{lag}'] = serie.shift(lag)
    data.dropna(inplace=True)
    serie = serie.loc[data.index]

    #  Cible multi-step
    Y = pd.concat([serie.shift(-h) for h in range(1, H+1)], axis=1)
    Y.columns = [f'y+{h}' for h in range(1, H+1)]
    valid = Y.dropna().index
    X = data.loc[valid]
    Y = Y.loc[valid]

    #  Train/Test split
    split = int(len(X)*(1-TEST_RATIO))
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    Y_tr, Y_te = Y.iloc[:split], Y.iloc[split:]

    # Entraînement XGBoost multi-output
    base = XGBRegressor(n_estimators=300, learning_rate=0.05,
                        max_depth=4, random_state=42, verbosity=0)
    model = MultiOutputRegressor(base)
    model.fit(X_tr, Y_tr)

    #  Prédictions
    Y_pred_tr = model.predict(X_tr)
    Y_pred_te = model.predict(X_te)
    future_pred = model.predict(X.iloc[[-1]])[0]

    # Stocker les valeurs pour évaluation globale (pas +1 uniquement)
    all_y_true_te.extend(Y_te.iloc[:, 0].values)
    all_y_pred_te.extend(Y_pred_te[:, 0])
    all_y_true_tr.extend(Y_tr.iloc[:, 0].values)
    all_y_pred_tr.extend(Y_pred_tr[:, 0])

    #  Évaluation pas +1
    def metrics(y_true, y_pred):
        return {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE':  mean_absolute_error(y_true, y_pred),
            'R2':   r2_score(y_true, y_pred)
        }

    m_tr = metrics(Y_tr.iloc[:,0], Y_pred_tr[:,0])
    m_te = metrics(Y_te.iloc[:,0], Y_pred_te[:,0])
    print("Pas +1 - Train :", m_tr)
    print("       Test  :", m_te)

    results[commodity] = {
        'Train': {k: float(v) for k, v in m_tr.items()},
        'Test':  {k: float(v) for k, v in m_te.items()}
    }

    for date, y_t, y_p in zip(Y_te.index, Y_te.iloc[:,0], Y_pred_te[:,0]):
        all_ytrue_ytest.append({
            'date': date.strftime('%Y-%m-%d'),
            'commodity': commodity,
            'y_true': float(y_t),
            'y_pred': float(y_p)
        })

    for i, val in enumerate(future_pred):
        pred_date = (serie.index[-1] + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d')
        all_future_preds.append({
            'date': pred_date,
            'commodity': commodity,
            'predicted_close': float(val)
        })

    # RMSE et MAE moyens multi-horizon
    rmse_list = [np.sqrt(mean_squared_error(Y_te.iloc[:,h], Y_pred_te[:,h])) for h in range(H)]
    mae_list = [mean_absolute_error(Y_te.iloc[:,h], Y_pred_te[:,h]) for h in range(H)]
    print(f"RMSE moyen 1–{H}j : {np.mean(rmse_list):.3f} | MAE moyen  : {np.mean(mae_list):.3f}")

# Évaluation globale après la boucle
global_metrics = {
    'Train': {
        'MAE': mean_absolute_error(all_y_true_tr, all_y_pred_tr),
        'RMSE': np.sqrt(mean_squared_error(all_y_true_tr, all_y_pred_tr)),
        'R2': r2_score(all_y_true_tr, all_y_pred_tr)
    },
    'Test': {
        'MAE': mean_absolute_error(all_y_true_te, all_y_pred_te),
        'RMSE': np.sqrt(mean_squared_error(all_y_true_te, all_y_pred_te)),
        'R2': r2_score(all_y_true_te, all_y_pred_te)
    }
}

print("\n=== Évaluation Globale (pas +1 sur tous les commodities) ===")
for phase in ["Train", "Test"]:
    print(f"{phase} - R2 : {global_metrics[phase]['R2']:.4f}")
    print(f"{phase} - RMSE: {global_metrics[phase]['RMSE']:.4f}")
    print(f"{phase} - MAE : {global_metrics[phase]['MAE']:.4f}")

# --- Sauvegarde des fichiers ---

#  JSON par commodity
with open('xgb_metrics.json', 'w') as f:
    json.dump(results, f, indent=4)

#  JSON global avec train/test
with open('xgb_global_metrics.json', 'w') as f:
    json.dump(global_metrics, f, indent=4)

#  CSV y_true vs y_pred
df_y = pd.DataFrame(all_ytrue_ytest)
df_y.to_csv('xgb_result.csv', index=False)

#  CSV prédictions futures
df_fut = pd.DataFrame(all_future_preds)
df_fut.to_csv('xgb_future.csv', index=False)

print("Tous les fichiers ont été générés.")

