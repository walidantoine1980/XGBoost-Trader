# XGBoost Stock Predictor Pro 📈

Un terminal quantitatif complet construit avec Python et Streamlit. L'application fusionne le Machine Learning (XGBoost), l'analyse technique mathématique avancée et le Traitement du Langage Naturel (NLP) pour générer des signaux de trading et des stratégies de couverture via les Options (Black-Scholes).

## Fonctionnalités Principales

*   **Terminal de Trading (Machine Learning)** : Utilise l'algorithme XGBoost avec une approche de Walk-Forward Validation. Auto-optimisation probabiliste des hyperparamètres via Optuna.
*   **Indicateurs Institutionnels (Features)** : ADX, VWAP, Bandes de Bollinger (Width), MACD, RSI, Stochastique, OBV et Moyennes Mobiles.
*   **Filtre Macro & NLP** : Intègre l'indice de peur (VIX) en temps réel comme coupe-circuit, et analyse le sentiment des news financières récentes via TextBlob pour filtrer les faux signaux.
*   **Module de Pricing d'Options (Black-Scholes)** : Calcule le juste prix, les Greeks (Delta, Gamma, Theta, Vega) et génère un graphique de Payoff dynamique.
*   **Paper Trading Intégré** : Portefeuille virtuel de 100 000 $ pour tester les signaux en direct avec gestion d'historique.
*   **Risk Parity Portfolio** : Construction de portefeuilles multi-actions pondérés par l'inverse de la volatilité.
*   **Académie Interactive** : Modules d'apprentissage intégrés (Scrollytelling) pour comprendre les indicateurs, les mathématiques de XGBoost et les produits dérivés.

## Déploiement via Docker (PaaS / Dokploy / GitHub)

L'application est entièrement dockérisée et prête pour le cloud.

### Pré-requis
*   Docker installé sur votre machine.

### Installation Locale
1. Clonez ce dépôt.
2. Construisez l'image :
   ```bash
   docker build -t xgboost-trader .
   ```
3. Lancez le conteneur :
   ```bash
   docker run -p 8501:8501 -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models xgboost-trader
   ```
   > Note : Les volumes montés (`-v`) permettent de conserver les modèles entraînés (`.pkl`) et l'historique du paper trading (`portfolio.json`) d'un redémarrage à l'autre.

L'application sera accessible sur `http://localhost:8501`.

## Déploiement Sans Docker (Local)

1. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
2. Téléchargez les corpus NLP requis :
   ```bash
   python -m textblob.download_corpora
   ```
3. Lancez Streamlit :
   ```bash
   streamlit run xgboost_trader.py
   ```

## Avertissement de Risque
Ce projet est à but **éducatif**. Les modèles de Machine Learning prédisent le futur en se basant sur le passé, ce qui n'est jamais une garantie sur les marchés financiers. N'utilisez pas ces signaux avec de l'argent réel sans comprendre parfaitement le risque de perte en capital.
