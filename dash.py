import streamlit as st
import pandas as pd
import json
import plotly.express as px
from datetime import datetime

# === CONFIGURATION ===
st.set_page_config(layout="wide")
st.title("Dashboard de prévision du prix des commodités")
st.markdown("Comparaison des modèles : XGBoost,  LSTM , GRU")

# === FICHIERS À CHARGER ===
models = {
    'LSTM':      ('all_fuels_data.csv',     'lstm/lstm_global_metrics.json'),
    'GRU':       ('all_fuels_data.csv',      'gru/gru_global_metrics.json'),
    'XGB':       ('all_fuels_data.csv',      'xgboost/xgb_global_metrics.json'),
}

dfs = {}
metrics = {}

# === CHARGEMENT DES DONNÉES ===
for name, (csv_file, json_file) in models.items():
    try:
        dfs[name] = pd.read_csv(csv_file, parse_dates=['date'])
        with open(json_file, 'r') as f:
            metrics[name] = json.load(f)
    except FileNotFoundError:
        st.warning(f"Fichiers manquants pour le modèle {name}.")

# === AFFICHAGE DU TABLEAU DE MÉTRIQUES ===
if metrics:
    st.subheader("Résultats comparés (MAE, RMSE, R²)")

    results = []
    for model_name, metric_values in metrics.items():
        train_mae = f"{metric_values.get('train', {}).get('MAE', 0.0):.2f}"
        train_rmse = f"{metric_values.get('train', {}).get('RMSE', 0.0):.2f}"
        train_r2 = f"{metric_values.get('train', {}).get('R2', 0.0):.4f}"

        test_mae = f"{metric_values.get('test', {}).get('MAE', 0.0):.2f}"
        test_rmse = f"{metric_values.get('test', {}).get('RMSE', 0.0):.2f}"
        test_r2 = f"{metric_values.get('test', {}).get('R2', 0.0):.4f}"

        results.append({
            'Modèle': model_name,
            'Train MAE': train_mae,
            'Train RMSE': train_rmse,
            'Train R²': train_r2,
            'Test MAE': test_mae,
            'Test RMSE': test_rmse,
            'Test R²': test_r2
        })

    table = pd.DataFrame(results)
    st.dataframe(table, use_container_width=True)
else:
    st.warning("Aucune métrique disponible.")

# === SÉLECTEUR DE MODÈLE ===
if dfs:
    model_choice = st.selectbox("Choisissez un modèle à visualiser :", list(dfs.keys()))
    df = dfs[model_choice]

    # === SÉLECTEUR DE COMMODITY SI DISPONIBLE ===
    selected_commodity = None
    if 'commodity' in df.columns:
        commodities = df['commodity'].unique()
        selected_commodity = st.selectbox("Choisissez la matière :", commodities)
        df = df[df['commodity'] == selected_commodity]

    # === CHARGEMENT DES PRÉDICTIONS FUTURES ===
    future_df = None
    if model_choice == "GRU":
        try:
            future_df = pd.read_csv("gru/gru_future.csv", parse_dates=['date'])
            if selected_commodity:
                future_df = future_df[future_df['commodity'] == selected_commodity]
        except FileNotFoundError:
            st.warning("Fichier de prédictions futures (gru_future.csv) non trouvé.")
    
    elif model_choice == "LSTM":
        try:
            future_df = pd.read_csv("lstm/lstm_future.csv", parse_dates=['date'])
            if selected_commodity:
                future_df = future_df[future_df['commodity'] == selected_commodity]
        except FileNotFoundError:
            st.warning("Fichier de prédictions futures (lstm_future.csv) non trouvé.")
    
    elif model_choice == "XGB":
        try:
            future_df = pd.read_csv("xgboost/xgb_future.csv", parse_dates=['date'])
            if selected_commodity:
                future_df = future_df[future_df['commodity'] == selected_commodity]
        except FileNotFoundError:
            st.warning("Fichier de prédictions futures (xgb_future.csv) non trouvé.")

    # === SÉLECTEUR DE PÉRIODE ===
    min_date = pd.to_datetime(df['date'].min()).to_pydatetime()
    max_date = pd.to_datetime(df['date'].max()).to_pydatetime()

    start, end = st.slider(
        "Filtrer la période de visualisation :",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    df_filtered = df[(df['date'] >= pd.to_datetime(start)) & (df['date'] <= pd.to_datetime(end))]

    # === AFFICHAGE DU GRAPHIQUE INTERACTIF ===
    fig = px.line(
        df_filtered,
        x='date',
        y=['close'],
        labels={'value': 'Prix ($)', 'date': 'Date'},
        title=f"Prédiction vs Réalité — {model_choice}" + (f" ({selected_commodity})" if selected_commodity else "")
    )

    # Couleurs personnalisées
    fig.data[0].line.color = 'blue'  # Données réelles (blue)
    
    # Ajout des prédictions futures en orange
    if future_df is not None and not future_df.empty:
        fig.add_scatter(
            x=future_df['date'],
            y=future_df['predicted'],
            mode='lines',
            name='Prévision future',
            line=dict(color='orange')
        )

    fig.update_layout(legend_title_text='', legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Aucun fichier de données n'est disponible pour afficher les graphiques.")
