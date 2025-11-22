 README — Prédiction des prix des commodités (XGBoost, LSTM, GRU)

 Projet de prédiction des prix du pétrole & des commodités

Évaluation technique – Ingénieur IA – Digitup Company

Ce projet implémente une solution de bout en bout permettant de :

-   Nettoyer et préparer les données
-   Entraîner plusieurs modèles prédictifs (XGBoost, LSTM, GRU)
-   Comparer leurs performances (MAE, RMSE, R²)
-   Visualiser les résultats via un dashboard Streamlit
-   Exposer les prédictions via une API FastAPI

------------------------------------------------------------------------

 Architecture du projet

    
├── all_fuels_data.csv
├── traitement.ipynb
├── dash.py
├── api.py
├── requirements.txt
│
├── xgboost/
│   ├── xgb_future.csv
│   ├── xgb_global_metrics.json
│   └── ...
│
├── lstm/
│   ├── lstm_future.csv
│   ├── lstm_global_metrics.json
│   └── ...
│
├── gru/
│   ├── gru_future.csv
│   ├── gru_global_metrics.json
│   └── ...

------------------------------------------------------------------------
1. Prétraitement des données - Réalisé dans traitement.ipynb.
    Dataset open source collecter depuis le site Kaggle 

   - chargement des données
   - détection des valeurs manquantes
   - conversion des types
   - suppression des doublons

----------------------------------------------------------------------------
2. Modèles de prédiction

Trois modèles ont été construits pour permettre une comparaison complète .


 A. Modèle XGBoost 80% train / 20% test ( simple division ) 

Le modèle XGBoost utilise des features tabulaires :

- 7 lags (lag1 → lag7)

- moyenne mobile 7 jours

- features calendaires (dow, month)

- prédiction multi-horizon (365 jours)

Les sorties :

xgb_future.csv

xgb_global_metrics.json

xgb_result.csv

  ------------------------------------------
 
B. Modèle LSTM (Seq2Seq + STL)

Le modèle LSTM utilise une architecture Encodeur-Décodeur :

 Encodeur :

lit une fenêtre de 180 jours

 Décodeur :

prédit 14 jours en séquence

 Sliding Window :

chaque fenêtre = 180 jours → prédiction 14 jours

 Décomposition STL :

La série close est décomposée en :

tendance

saisonnalité

résidus

Le LSTM prédit ensuite les résidus, permettant une reconstruction :

prévision = tendance_futur + saisonnalité_futur + résidus_prédits


Les sorties :

lstm_future.csv

lstm_global_metrics.json

lstm_result_normaliser.csv

 --------------------------------------------------- 

C. Modèle GRU (Multi-horizon + STL)

Le GRU est utilisé pour des prédictions plus rapides que lstm  et efficaces :



 Sliding Window :

180 jours → prédiction 30 jours

 Backtest rolling 30 jours :

évaluation réaliste sur plusieurs fenêtres futures

 Décomposition STL + GRU :

Comme le LSTM, mais version GRU, plus rapide.

Sorties :

gru_future.csv

gru_global_metrics.json

gru_result_normaliser.csv

-------------------------------------------------------------------------------------------------------------
3. Évaluation & Comparaison

Les métriques utilisées :  MAE,RMSE,R²

----------------------------------------------------------------------------------

4. Dashboard Streamlit

Le fichier dash.py permet de :

afficher les historiques de prix

comparer XGB / LSTM / GRU

visualiser les prédictions futures

filtrer par commodity

filtrer par période

afficher toutes les métriques dans un tableau

Pour Lancer le dashboard lancez dans le terminal : streamlit run dash.py

-----------------------------------------------------------------------------

Installation rapide

1. Cloner le projet
git clone https://github.com/AYMENsid/TEST-TECH-IA.git
cd TEST-TECH-IA


2. Installer les dépendances
pip install -r requirements.txt

3. Lancer le dashboard
streamlit run dash.py

4. Lancer l’API
uvicorn api : app --reload

 ------------------------------------------------ 

Limitations actuelles :

Pas de données exogènes (macro, géopolitique, météo…)

Pas d’optimisation poussée des hyperparamètres

Pas de cross-validation temporelle complexe

------------------------------------------------------

Pistes d’amélioration :

Ajouter un Transformer 

Ajouter des données externes

Déploiement cloud 

--------------------------------------------------------
