import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
import time
import pickle
import os
import json
import optuna
import scipy.stats as si
import shap
from tickers_db import MAJOR_STOCKS, PREDEFINED_PORTFOLIOS
from textblob import TextBlob

@st.cache_data(ttl=3600)
def get_macro_data(period, interval):
    """Télécharge l'historique des indicateurs macroéconomiques"""
    try:
        spy_raw = yf.download("SPY", period=period, interval=interval, progress=False)
        vix_raw = yf.download("^VIX", period=period, interval=interval, progress=False)
        tnx_raw = yf.download("^TNX", period=period, interval=interval, progress=False)
        
        spy = spy_raw['Close'].iloc[:, 0] if isinstance(spy_raw.columns, pd.MultiIndex) else spy_raw['Close']
        vix = vix_raw['Close'].iloc[:, 0] if isinstance(vix_raw.columns, pd.MultiIndex) else vix_raw['Close']
        tnx = tnx_raw['Close'].iloc[:, 0] if isinstance(tnx_raw.columns, pd.MultiIndex) else tnx_raw['Close']
        
        macro = pd.DataFrame(index=spy.index)
        macro['SPY_Return'] = spy.pct_change()
        macro['VIX'] = vix
        macro['TNX'] = tnx
        return macro.ffill()
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_news_sentiment(ticker):
    """Analyse le sentiment des dernières actualités via NLP (TextBlob)"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return 0, []
        
        polarities = []
        articles = []
        for n in news[:5]:
            content = n.get('content', {})
            title = content.get('title', '')
            blob = TextBlob(title)
            pol = blob.sentiment.polarity
            polarities.append(pol)
            articles.append({'title': title, 'link': content.get('clickThroughUrl', ''), 'polarity': pol})
            
        avg_polarity = np.mean(polarities) if polarities else 0
        return avg_polarity, articles
    except Exception:
        return 0, []


# Création du dossier pour les modèles
os.makedirs("models", exist_ok=True)

# Configuration de la page
st.set_page_config(page_title="XGBoost Stock Predictor Pro", layout="wide", page_icon="📈")

# --- STYLE CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4259;
    }
    .signal-buy {
        color: #00ff00;
        font-weight: bold;
        font-size: 24px;
    }
    .signal-sell {
        color: #ff4b4b;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTIONS DE CALCUL D'INDICATEURS ---
def add_features(df):
    """Calcule les features techniques pour le modèle XGBoost"""
    df = df.copy()
    
    # Rendements
    df['Returns'] = df['Close'].pct_change()
    
    # Moyennes Mobiles
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    
    # Volatilité
    df['Vol_20'] = df['Returns'].rolling(window=20).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bandes de Bollinger
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
    
    # ATR & ADX
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    # ADX (Average Directional Index)
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].shift() - df['Low']
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm_actual = np.where(plus_dm > minus_dm, plus_dm, 0)
    minus_dm_actual = np.where(minus_dm > plus_dm, minus_dm, 0)
    
    tr14 = true_range.rolling(14).sum()
    plus_di14 = 100 * (pd.Series(plus_dm_actual, index=df.index).rolling(14).sum() / tr14)
    minus_di14 = 100 * (pd.Series(minus_dm_actual, index=df.index).rolling(14).sum() / tr14)
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    df['ADX'] = dx.rolling(14).mean()

    # NOUVELLES FEATURES:
    # 1. Volume Ratio
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    # 2. OBV (On-Balance Volume)
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV'] = obv
    
    # 3. Oscillateur Stochastique (%K)
    lowest_low = df['Low'].rolling(window=14).min()
    highest_high = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
    
    # 4. ROC (Rate of Change) sur 10 jours
    df['ROC'] = df['Close'].pct_change(periods=10) * 100
    
    # 5. Largeur des Bandes de Bollinger (BB_Width)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']

    # 6. Approximation VWAP sur 14 jours
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).rolling(window=14).sum() / df['Volume'].rolling(window=14).sum()

    # Lags
    for i in range(1, 4):
        df[f'Lag_{i}'] = df['Close'].shift(i)
        
    # Cible
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return df.dropna()

# --- MOTEUR MACHINE LEARNING ---
class MLTrader:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            objective='binary:logistic',
            random_state=42
        )
        self.is_trained = False
        self.accuracy = 0.0
        self.feature_importances = None
        self.backtest_results = None
        self.advanced_metrics = {}

    def calculate_advanced_metrics(self, test_data):
        metrics = {}
        strat_returns = test_data['Strategy_Return'].fillna(0)
        
        winning_days = len(strat_returns[strat_returns > 0])
        total_trades_days = len(strat_returns[strat_returns != 0])
        win_rate = winning_days / total_trades_days if total_trades_days > 0 else 0
        metrics['Win Rate'] = win_rate
        
        # --- NEW: Kelly Criterion ---
        avg_win = strat_returns[strat_returns > 0].mean() if winning_days > 0 else 0
        avg_loss = abs(strat_returns[strat_returns < 0].mean()) if (total_trades_days - winning_days) > 0 else 1
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio) if win_loss_ratio > 0 else 0
        metrics['Kelly %'] = kelly_pct if kelly_pct > 0 else 0
        
        cum_returns = (1 + strat_returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max
        metrics['Max Drawdown'] = drawdown.min() if not drawdown.empty else 0
        
        mean_ret = strat_returns.mean()
        std_ret = strat_returns.std()
        metrics['Sharpe Ratio'] = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0
        
        downside_returns = strat_returns[strat_returns < 0]
        downside_std = downside_returns.std()
        metrics['Sortino Ratio'] = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0
        
        self.advanced_metrics = metrics

    def train(self, data, optimize=False):
        base_features = ['Returns', 'SMA_20', 'SMA_50', 'SMA_200', 'EMA_9', 'Vol_20', 'RSI', 'MACD', 'Signal_Line', 
                         'BB_Upper', 'BB_Lower', 'BB_Width', 'ATR', 'ADX', 'Volume_Ratio', 'OBV', 'Stoch_K', 'ROC', 
                         'VWAP', 'Lag_1', 'Lag_2', 'Lag_3']
        macro_features = ['SPY_Return', 'VIX', 'TNX']
        
        features = [f for f in base_features + macro_features if f in data.columns]
        X = data[features]
        y = data['Target']
        
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # 4. Walk-Forward Validation
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        
        if optimize:
            # 1. Hyper-Optimisation avec Optuna (Bayésienne)
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 9)
                }
                model = xgb.XGBClassifier(**param, objective='binary:logistic', random_state=42)
                
                # Walk-Forward Validation pour le score
                from sklearn.model_selection import cross_val_score
                score = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy').mean()
                return score

            # Création de l'étude Optuna
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=10) # 10 essais pour un bon compromis vitesse/précision
            
            best_params = study.best_params
            self.model = xgb.XGBClassifier(**best_params, objective='binary:logistic', random_state=42)
            self.model.fit(X_train, y_train)
            st.toast(f"Optimisation Optuna terminée : {best_params}", icon="🧬")
        else:
            self.model.fit(X_train, y_train)
        
        # Récupération de l'importance des variables
        self.feature_importances = pd.DataFrame(
            {'Feature': features, 'Importance': self.model.feature_importances_}
        ).sort_values(by='Importance', ascending=True)
        
        # Backtest
        test_data = data.iloc[split:].copy()
        test_data['Prob'] = self.model.predict_proba(X_test)[:, 1]
        
        # Signal de base
        test_data['Signal'] = np.where(test_data['Prob'] > 0.55, 1, 0)
        
        # 3. Filtre Macro-économique (Régime de Krach)
        if 'VIX' in test_data.columns:
            # Si le VIX dépasse 30, on force le passage en cash (0)
            test_data.loc[test_data['VIX'] > 30, 'Signal'] = 0
            
        # NOUVEAU : Frais de transaction & Slippage
        # On suppose un coût total de 0.1% par transaction (Achat ou Vente)
        transaction_cost = 0.001 
        signal_changes = (test_data['Signal'] != test_data['Signal'].shift(1)).astype(int)
        
        # Le rendement de la stratégie est le rendement du marché moins les frais si on change de position
        test_data['Strategy_Return'] = (test_data['Signal'].shift(1) * test_data['Returns']) - (signal_changes * transaction_cost)
        test_data['Cum_Strategy_Return'] = (1 + test_data['Strategy_Return'].fillna(0)).cumprod()
        test_data['Cum_Market_Return'] = (1 + test_data['Returns'].fillna(0)).cumprod()
        
        self.backtest_results = test_data
        self.accuracy = accuracy_score(y_test, self.model.predict(X_test))
        self.calculate_advanced_metrics(test_data)
        self.is_trained = True
        return features

    def predict(self, last_row, features):
        if not self.is_trained:
            return None
        X_input = last_row[features].values.reshape(1, -1)
        prob = self.model.predict_proba(X_input)[0][1]
        # 3. Coupe-circuit Macro en direct
        if 'VIX' in last_row.columns and last_row['VIX'].values[0] > 30:
            return 0.0 # Force la vente/cash
        return prob
        
    def save(self, filepath):
        state = {
            'model': self.model,
            'accuracy': self.accuracy,
            'feature_importances': self.feature_importances,
            'backtest_results': self.backtest_results,
            'is_trained': self.is_trained,
            'advanced_metrics': getattr(self, 'advanced_metrics', {})
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
            
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        trader = MLTrader()
        trader.model = state['model']
        trader.accuracy = state['accuracy']
        trader.feature_importances = state['feature_importances']
        trader.backtest_results = state['backtest_results']
        trader.is_trained = state['is_trained']
        trader.advanced_metrics = state.get('advanced_metrics', {})
        return trader

def convert_google_to_yahoo_ticker(ticker):
    """Convertit un ticker Google Finance en format Yahoo Finance"""
    if ":" not in ticker:
        return ticker.strip().upper()
        
    exchange, symbol = ticker.split(":", 1)
    exchange = exchange.strip().upper()
    symbol = symbol.strip().upper()
    
    exchange_mapping = {
        'EPA': '.PA', 'LON': '.L', 'TSE': '.TO', 'FRA': '.F', 'BIT': '.MI',
        'BME': '.MC', 'AMS': '.AS', 'EBR': '.BR', 'LIS': '.LS', 'VTX': '.SW',
        'TYO': '.T', 'HKG': '.HK', 'ASX': '.AX', 'NASDAQ': '', 'NYSE': '', 'AMEX': ''
    }
    
    suffix = exchange_mapping.get(exchange, "")
    return f"{symbol}{suffix}"


# --- MODES D'AFFICHAGE ---

def run_single_mode(ticker, period, interval, initial_capital, optimize_model):
    st.subheader(f"Analyse Individuelle : {ticker}")
    model_path = f"models/{ticker}_model.pkl"
    
    if f"trader_{ticker}" not in st.session_state:
        st.session_state[f"trader_{ticker}"] = MLTrader()
        
    trader = st.session_state[f"trader_{ticker}"]

    with st.spinner(f"Téléchargement des données pour {ticker}..."):
        df_raw = yf.download(ticker, period=period, interval=interval, progress=False)
        if df_raw.empty:
            st.error(f"Erreur lors du téléchargement. Vérifiez le ticker (recherché: {ticker}).")
            return
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.droplevel(1)
            
        macro_df = get_macro_data(period, interval)

    df = add_features(df_raw)
    if macro_df is not None:
        df = df.join(macro_df, how='left').ffill().dropna()
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        if st.button("🧠 Entraîner l'IA sur cette action", use_container_width=True):
            with st.status("Entraînement en cours...") as status:
                features = trader.train(df, optimize=optimize_model)
                st.session_state[f"features_{ticker}"] = features
                status.update(label="✅ Modèle entraîné !", state="complete")
                
    with col2:
        if trader.is_trained:
            if st.button("💾 Sauvegarder", use_container_width=True):
                trader.save(model_path)
                st.toast("Modèle sauvegardé sur le disque !", icon="💾")
                
    with col3:
        if os.path.exists(model_path):
            if st.button("📂 Charger", use_container_width=True):
                st.session_state[f"trader_{ticker}"] = MLTrader.load(model_path)
                st.toast("Modèle chargé depuis le disque !", icon="📂")
                st.rerun()
                
    with col4:
        if trader.is_trained:
            st.metric("Précision du Modèle", f"{trader.accuracy:.2%}")

    if trader.is_trained:
        last_row = df.iloc[-1:]
        # Default features if loaded from disk without re-training session
        base_features = ['Returns', 'SMA_20', 'SMA_50', 'EMA_9', 'Vol_20', 'RSI', 'MACD', 'Signal_Line', 
                         'BB_Upper', 'BB_Lower', 'BB_Width', 'ATR', 'ADX', 'Volume_Ratio', 'OBV', 'Stoch_K', 'ROC', 
                         'VWAP', 'Lag_1', 'Lag_2', 'Lag_3']
        macro_features = ['SPY_Return', 'VIX', 'TNX']
        features = [f for f in base_features + macro_features if f in df.columns]
        prob = trader.predict(last_row, features)
        st.progress(float(prob))
        
        st.divider()
        # --- NLP Sentiment Logic ---
        st.subheader("📰 Filtre de Sentiment Actuel (NLP)")
        try:
            news = yf.Ticker(ticker).news
            if news:
                sentiments = []
                for n in news[:10]: # Top 10 news
                    blob = TextBlob(n['title'])
                    sentiments.append(blob.sentiment.polarity)
                avg_sentiment = np.mean(sentiments)
                
                sentiment_text = "Neutre"
                color = "gray"
                if avg_sentiment > 0.15:
                    sentiment_text = "Positif 🟢"
                    color = "green"
                elif avg_sentiment < -0.15:
                    sentiment_text = "Négatif 🔴"
                    color = "red"
                    
                st.markdown(f"Score de Sentiment TextBlob : **<span style='color:{color}'>{avg_sentiment:+.2f} ({sentiment_text})</span>**", unsafe_allow_html=True)
                
                # Alerte de contradiction
                if prob > 0.55 and avg_sentiment < -0.2:
                    st.warning("⚠️ **Contradiction Fondamentale** : L'IA mathématique recommande l'ACHAT, mais l'IA linguistique détecte des nouvelles très NÉGATIVES. Risque de fausse cassure élevé.")
                elif prob < 0.45 and avg_sentiment > 0.2:
                    st.warning("⚠️ **Contradiction Fondamentale** : L'IA mathématique recommande la VENTE, mais l'IA linguistique détecte des nouvelles très POSITIVES.")
            else:
                st.write("Aucune actualité récente disponible pour le NLP.")
        except:
            st.write("Impossible de récupérer les actualités.")
            
        st.divider()
        st.header("📝 Plan de Trading du Jour (Money Management)")
        
        c1, c2, c3, c4 = st.columns([1, 1, 1.5, 1])
        with c1:
            st.subheader("🔮 Signal")
            current_price = last_row['Close'].values[0]
            sma_200 = last_row['SMA_200'].values[0] if 'SMA_200' in last_row.columns else current_price
            
            # --- FILTRE DE RÉGIME (SMA 200) ---
            regime_bull = current_price > sma_200
            
            if prob == 0.0:
                st.markdown("<span class='signal-sell'>⚠️ KRACH (VIX>30) - CASH</span>", unsafe_allow_html=True)
            elif prob > 0.55:
                if regime_bull:
                    st.markdown("<span class='signal-buy'>🟢 ACHAT FORT</span>", unsafe_allow_html=True)
                    st.write("Contexte : Régime Haussier (>SMA 200)")
                else:
                    st.markdown("<span style='color:orange; font-weight:bold; font-size:20px;'>🟡 ACHAT RISQUÉ</span>", unsafe_allow_html=True)
                    st.write("⚠️ Contre-Tendance (<SMA 200)")
                st.write(f"Confiance IA : {prob:.1%}")
            elif prob < 0.45:
                if not regime_bull:
                    st.markdown("<span class='signal-sell'>🔴 VENTE / SHORT</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='signal-sell'>🔴 VENTE / CASH</span>", unsafe_allow_html=True)
                st.write(f"Confiance Baisse : {1-prob:.1%}")
            else:
                st.write("⚪ **NEUTRE** / Attente")
                
        with c2:
            st.subheader("📰 Sentiment News")
            avg_pol, articles = get_news_sentiment(ticker)
            if avg_pol > 0.1:
                st.success(f"Positif (+{avg_pol:.2f})")
            elif avg_pol < -0.1:
                st.error(f"Négatif ({avg_pol:.2f})")
            else:
                st.info(f"Neutre ({avg_pol:.2f})")
                
        with c3:
            st.subheader("🛡️ Gestion du Risque (Kelly)")
            current_atr = last_row['ATR'].values[0] if 'ATR' in last_row.columns else current_price * 0.02
            
            stop_loss = current_price - (1.5 * current_atr) if prob > 0.5 else current_price + (1.5 * current_atr)
            take_profit = current_price + (3 * current_atr) if prob > 0.5 else current_price - (3 * current_atr)
            risk_per_share = abs(current_price - stop_loss)
            
            # --- CRITÈRE DE KELLY (Half-Kelly) ---
            kelly_pct = getattr(trader, 'advanced_metrics', {}).get('Kelly %', 0)
            if kelly_pct > 0:
                risk_pct = min(kelly_pct / 2, 0.05) # Max 5% du compte
                st.info(f"Half-Kelly Actif : {risk_pct*100:.2f}% (Plein Kelly: {kelly_pct*100:.2f}%)")
            else:
                risk_pct = 0.01 # 1% par défaut si Kelly=0
                st.warning("Kelly incalculable. Risque défaut: 1%")
                
            max_loss = initial_capital * risk_pct
            position_size = int(max_loss / risk_per_share) if risk_per_share > 0 else 0
            total_investment = position_size * current_price
            
            if total_investment > initial_capital:
                position_size = int(initial_capital / current_price)
                total_investment = position_size * current_price
                
            st.markdown(f"""
            - **Prix Actuel** : {current_price:.2f} $
            - **Taille de position** : {position_size} actions ({total_investment:,.2f} $)
            - **Stop-Loss (1.5 ATR)** : <span style='color:red'>{stop_loss:.2f} $</span>
            - **Take-Profit (3.0 ATR)** : <span style='color:green'>{take_profit:.2f} $</span>
            """, unsafe_allow_html=True)
            
        with c4:
            st.subheader("📑 Exécution & Export")
            export_text = f"DATE: {datetime.now().strftime('%Y-%m-%d')}\n"
            export_text += f"ACTION: {ticker}\n"
            export_text += f"SIGNAL: {'ACHAT' if prob > 0.55 else 'VENTE/CASH'}\n"
            export_text += f"PRIX ENTREE: {current_price:.2f} $\n"
            export_text += f"STOP-LOSS: {stop_loss:.2f} $\n"
            export_text += f"TAKE-PROFIT: {take_profit:.2f} $\n"
            export_text += f"TAILLE POSITION: {position_size} actions\n"
            export_text += f"RISQUE MAX: {max_loss:.2f} $\n"
            
            st.download_button(
                label="📥 Télécharger le Plan",
                data=export_text,
                file_name=f"Trading_Plan_{ticker}.txt",
                mime="text/plain"
            )
            
            # --- PAPER TRADING BUTTON ---
            if prob > 0.55 and position_size > 0:
                if st.button("🚀 Exécuter sur le Paper Trading"):
                    pf = load_portfolio()
                    cost = position_size * current_price
                    if pf['cash'] >= cost:
                        pf['cash'] -= cost
                        # Update positions
                        if ticker in pf['positions']:
                            old_qty = pf['positions'][ticker]['qty']
                            old_price = pf['positions'][ticker]['avg_price']
                            new_qty = old_qty + position_size
                            new_price = ((old_qty * old_price) + cost) / new_qty
                            pf['positions'][ticker] = {'qty': new_qty, 'avg_price': new_price}
                        else:
                            pf['positions'][ticker] = {'qty': position_size, 'avg_price': current_price}
                        
                        pf['history'].append({
                            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'action': 'BUY',
                            'ticker': ticker,
                            'qty': position_size,
                            'price': current_price,
                            'total': cost
                        })
                        save_portfolio(pf)
                        st.success(f"Ordre exécuté virtuellement ! {position_size} actions {ticker} achetées.")
                    else:
                        st.error("Fonds insuffisants dans le Paper Trading.")
            elif prob < 0.45:
                if st.button("🔴 Liquider Position (Paper Trading)"):
                    pf = load_portfolio()
                    if ticker in pf['positions'] and pf['positions'][ticker]['qty'] > 0:
                        qty = pf['positions'][ticker]['qty']
                        revenue = qty * current_price
                        pf['cash'] += revenue
                        pf['history'].append({
                            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'action': 'SELL',
                            'ticker': ticker,
                            'qty': qty,
                            'price': current_price,
                            'total': revenue,
                            'pnl': revenue - (qty * pf['positions'][ticker]['avg_price'])
                        })
                        del pf['positions'][ticker]
                        save_portfolio(pf)
                        st.success(f"Position liquidée virtuellement ! {qty} actions {ticker} vendues.")
                    else:
                        st.warning("Aucune position ouverte sur ce titre.")

        # --- NOUVEAU: STRATÉGIE OPTIONS RECOMMANDÉE ---
        st.divider()
        st.subheader("🛡️ Stratégie Dérivée Recommandée (Options)")
        current_price_opt = last_row['Close'].values[0]
        
        # Approximation de la volatilité et du taux sans risque
        vol_opt = df['Returns'].std() * np.sqrt(252)
        r_opt = 0.05
        
        if prob > 0.55:
            # Recommande un Call ATM à 30 jours
            strike_opt = round(current_price_opt, 2)
            t_opt_days = 30
            price_opt, delta_opt, gamma_opt, theta_opt, vega_opt = black_scholes(current_price_opt, strike_opt, t_opt_days/365, r_opt, vol_opt, "call")
            st.success(f"**Recommandation : ACHAT de CALL (Pari Haussier avec Levier)**")
            c_o1, c_o2, c_o3 = st.columns(3)
            c_o1.markdown(f"""
            - **Type** : Call
            - **Strike** : {strike_opt} $
            - **Expiration** : Dans {t_opt_days} jours
            """)
            c_o2.markdown(f"""
            - **Prime Estimée (Coût)** : {price_opt:.2f} $ par action
            - **Coût total du contrat (x100)** : {price_opt*100:.2f} $
            """)
            c_o3.markdown(f"""
            - **Delta** : {delta_opt:.3f} (Équivaut à détenir {delta_opt*100:.0f} actions)
            - **Levier Estimé** : {(current_price_opt * delta_opt) / price_opt:.1f}x
            """)
        elif prob < 0.45:
            # Recommande un Put ATM à 30 jours
            strike_opt = round(current_price_opt, 2)
            t_opt_days = 30
            price_opt, delta_opt, gamma_opt, theta_opt, vega_opt = black_scholes(current_price_opt, strike_opt, t_opt_days/365, r_opt, vol_opt, "put")
            st.error(f"**Recommandation : ACHAT de PUT (Pari Baissier ou Couverture)**")
            c_o1, c_o2, c_o3 = st.columns(3)
            c_o1.markdown(f"""
            - **Type** : Put
            - **Strike** : {strike_opt} $
            - **Expiration** : Dans {t_opt_days} jours
            """)
            c_o2.markdown(f"""
            - **Prime Estimée (Coût)** : {price_opt:.2f} $ par action
            - **Coût total du contrat (x100)** : {price_opt*100:.2f} $
            """)
            c_o3.markdown(f"""
            - **Delta** : {delta_opt:.3f} (Équivaut à short {-delta_opt*100:.0f} actions)
            - **Levier Estimé** : {(current_price_opt * abs(delta_opt)) / price_opt:.1f}x
            """)
        else:
            st.info("Aucune stratégie d'options recommandée en régime neutre ou très incertain.")

    st.divider()
    tab1, tab2, tab3 = st.tabs(["📊 Graphique Technique", "📉 Métriques de Backtest", "🧠 Importance des Variables"])
    
    with tab1:
        st.subheader(f"Analyse Technique : {ticker}")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, subplot_titles=('Prix & Moyennes', 'RSI'),
                           row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Prix'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='rgba(173, 216, 230, 0.5)', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='rgba(173, 216, 230, 0.5)', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if trader.is_trained:
            bt_df = trader.backtest_results
            bt_df['Port_Strategy'] = initial_capital * bt_df['Cum_Strategy_Return']
            bt_df['Port_Market'] = initial_capital * bt_df['Cum_Market_Return']
            
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Port_Strategy'], name='Portefeuille XGBoost ($)', line=dict(color='#00ff00', width=2)))
            fig_perf.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Port_Market'], name='Portefeuille Buy & Hold ($)', line=dict(color='gray', dash='dash')))
            fig_perf.update_layout(height=500, template="plotly_dark", title="Évolution du Portefeuille (Données de Test)", yaxis_title="Valeur en $")
            st.plotly_chart(fig_perf, use_container_width=True)
            
            st.subheader("📉 Larmes Quantitatives (Tearsheet)")
            adv = getattr(trader, 'advanced_metrics', {})
            
            final_strategy = bt_df['Port_Strategy'].iloc[-1]
            final_market = bt_df['Port_Market'].iloc[-1]
            profit_strategy = final_strategy - initial_capital
            profit_market = final_market - initial_capital
            diff_profit = profit_strategy - profit_market
            
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Capital Final", f"{final_strategy:,.2f} $", delta=f"{profit_strategy:+,.2f} $")
            m2.metric("Win Rate", f"{adv.get('Win Rate', 0)*100:.1f}%")
            m3.metric("Max Drawdown", f"{adv.get('Max Drawdown', 0)*100:.2f}%")
            m4.metric("Sharpe", f"{adv.get('Sharpe Ratio', 0):.2f}")
            if diff_profit > 0:
                m5.metric("Alpha", f"+{diff_profit:,.2f} $", delta="IA bat le marché", delta_color="normal")
            else:
                m5.metric("Alpha", f"{diff_profit:,.2f} $", delta="Marché bat l'IA", delta_color="inverse")
                
            st.divider()
            st.subheader("🎲 Simulation de Monte Carlo (Value at Risk)")
            st.markdown("Projection probabiliste sur 30 jours (1000 scénarios) pour mesurer le risque extrême.")
            
            # Paramètres
            days_to_simulate = 30
            num_simulations = 1000
            
            # Calcul de la dérive (drift) et de la volatilité de la stratégie
            strat_returns = bt_df['Strategy_Return'].dropna()
            mu = strat_returns.mean()
            sigma = strat_returns.std()
            
            # Génération des chemins aléatoires
            last_price = bt_df['Port_Strategy'].iloc[-1]
            simulation_df = pd.DataFrame()
            
            # Numpy vectorization pour plus de rapidité (au lieu d'une boucle lente)
            daily_returns = np.random.normal(mu, sigma, (days_to_simulate, num_simulations))
            price_paths = np.zeros_like(daily_returns)
            price_paths[0] = last_price
            for t in range(1, days_to_simulate):
                price_paths[t] = price_paths[t-1] * (1 + daily_returns[t])
                
            simulation_df = pd.DataFrame(price_paths)
            
            # Affichage graphique (on prend un échantillon de 100 courbes pour ne pas surcharger)
            fig_mc = go.Figure()
            for x in range(100):
                fig_mc.add_trace(go.Scatter(y=simulation_df[x], mode='lines', line=dict(width=1, color='rgba(0, 255, 0, 0.05)')))
            fig_mc.update_layout(height=400, template="plotly_dark", title="100 Scénarios de Monte Carlo", showlegend=False)
            st.plotly_chart(fig_mc, use_container_width=True)
            
            # Calcul de la VaR (Value at Risk à 95%)
            final_prices = simulation_df.iloc[-1]
            var_95 = np.percentile(final_prices, 5) # Le 5ème centile pire
            perte_potentielle = last_price - var_95
            
            st.error(f"⚠️ **Value at Risk (VaR 95%) sur 1 mois** : Dans 95% des cas, la valeur du portefeuille ne chutera pas en dessous de **{var_95:,.2f} $**.")
            st.info(f"Pire perte estimée sur 30 jours : **-{perte_potentielle:,.2f} $**")
        else:
            st.info("Entraînez le modèle pour voir les performances de backtest.")

    with tab3:
        st.subheader("🤖 Dans le cerveau de XGBoost")
        if trader.is_trained:
            # --- NOUVEAU : SHAP WATERFALL ---
            st.markdown("### 🔬 Explicabilité de la décision du jour (SHAP)")
            try:
                explainer = shap.TreeExplainer(trader.model)
                shap_values = explainer.shap_values(last_row[features])
                
                if isinstance(shap_values, list):
                    sv = shap_values[1][0]
                elif len(shap_values.shape) > 1:
                    sv = shap_values[0]
                else:
                    sv = shap_values
                
                shap_df = pd.DataFrame({'Feature': features, 'Contribution': sv})
                shap_df = shap_df.sort_values(by='Contribution', key=abs, ascending=True).tail(10)
                
                fig_shap = go.Figure(go.Waterfall(
                    name="SHAP", orientation="h",
                    measure=["relative"] * len(shap_df),
                    y=shap_df['Feature'],
                    x=shap_df['Contribution'],
                    text=[f"{x:+.3f}" for x in shap_df['Contribution']],
                    textposition="outside",
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    decreasing={"marker": {"color": "red"}},
                    increasing={"marker": {"color": "green"}}
                ))
                fig_shap.update_layout(
                    title="Impact des variables sur le dernier signal",
                    template="plotly_dark", height=400, margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur lors du calcul SHAP : {e}")

            st.divider()
            st.subheader("Importance Globale des Variables")
            df_imp = trader.feature_importances
            fig_imp = go.Figure(go.Bar(
                x=df_imp['Importance'], y=df_imp['Feature'], orientation='h',
                marker=dict(color=df_imp['Importance'], colorscale='viridis')
            ))
            fig_imp.update_layout(title="Importance des Variables", xaxis_title="Poids", yaxis_title="Indicateur", template="plotly_dark", height=500)
            st.plotly_chart(fig_imp, use_container_width=True)
            st.info("💡 Pour comprendre comment XGBoost fonctionne, consultez '🤖 Académie: XGBoost' dans le menu de gauche.")
        else:
            st.warning("Entraînez le modèle pour débloquer l'analyse du cerveau de l'IA.")

def run_portfolio_mode(tickers, period, interval, initial_capital, optimize_model):
    st.subheader("🌐 Mode Portefeuille Multi-Actions (Risk Parity)")
    st.markdown("""
    Ce mode utilise une approche inspirée de la **Théorie Moderne du Portefeuille de Markowitz**. 
    Le capital n'est pas divisé équitablement, mais alloué de manière **inversement proportionnelle à la volatilité** de chaque actif (Risk Parity). 
    *Un actif très volatil recevra moins de capital qu'un actif très stable, pour lisser le risque global.*
    """)
    
    if st.button("🧠 Entraîner et Simuler le Portefeuille", use_container_width=True):
        with st.status("Analyse des Risques & Entraînement...") as status:
            data_dict = {}
            volatility_dict = {}
            for t in tickers:
                st.write(f"Récupération pour {t}...")
                df_raw = yf.download(t, period=period, interval=interval, progress=False)
                if not df_raw.empty:
                    if isinstance(df_raw.columns, pd.MultiIndex):
                        df_raw.columns = df_raw.columns.droplevel(1)
                    data_dict[t] = df_raw
                    # Volatilité historique
                    ret = df_raw['Close'].pct_change().std()
                    volatility_dict[t] = ret if ret > 0 else 0.01

            # Calcul des poids (Inverse Volatility)
            sum_inv_vol = sum(1/v for v in volatility_dict.values())
            weights = {t: (1/v)/sum_inv_vol for t, v in volatility_dict.items()}
            
            st.write("📈 **Pondérations Calculées (Risk Parity)** :")
            for t, w in weights.items():
                st.write(f"- **{t}** : {w*100:.1f}% ({initial_capital * w:,.2f} $)")
                st.session_state[f"port_capital_{t}"] = initial_capital * w
            
            for t, df_raw in data_dict.items():
                st.write(f"Entraînement de l'IA pour {t}...")
                macro_df = get_macro_data(period, interval)
                df = add_features(df_raw)
                if macro_df is not None:
                    df = df.join(macro_df, how='left').ffill().dropna()
                trader = MLTrader()
                trader.train(df, optimize=optimize_model)
                st.session_state[f"port_trader_{t}"] = trader
            status.update(label="✅ Portefeuille Institutionnel généré !", state="complete")
            
    is_ready = all(f"port_trader_{t}" in st.session_state for t in tickers)
    
    if is_ready:
        all_port_strategy = pd.DataFrame()
        all_port_market = pd.DataFrame()
        
        for t in tickers:
            trader = st.session_state[f"port_trader_{t}"]
            # Récupère le capital alloué ou un capital équitable par défaut
            capital_for_t = st.session_state.get(f"port_capital_{t}", initial_capital/len(tickers))
            bt = trader.backtest_results
            
            if all_port_strategy.empty:
                all_port_strategy['Total'] = bt['Cum_Strategy_Return'] * capital_for_t
                all_port_market['Total'] = bt['Cum_Market_Return'] * capital_for_t
            else:
                all_port_strategy['Total'] = all_port_strategy['Total'].add(bt['Cum_Strategy_Return'] * capital_for_t, fill_value=0)
                all_port_market['Total'] = all_port_market['Total'].add(bt['Cum_Market_Return'] * capital_for_t, fill_value=0)
                
        all_port_strategy = all_port_strategy.dropna()
        all_port_market = all_port_market.dropna()
        
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(x=all_port_strategy.index, y=all_port_strategy['Total'], name='Portefeuille Quantitatif ($)', line=dict(color='#00ff00', width=2)))
        fig_perf.add_trace(go.Scatter(x=all_port_market.index, y=all_port_market['Total'], name='Portefeuille Buy & Hold ($)', line=dict(color='gray', dash='dash')))
        fig_perf.update_layout(height=500, template="plotly_dark", title="Évolution Globale du Portefeuille (Risk Parity)", yaxis_title="Valeur en $")
        st.plotly_chart(fig_perf, use_container_width=True)
        
        final_strategy = all_port_strategy['Total'].iloc[-1]
        final_market = all_port_market['Total'].iloc[-1]
        profit_strategy = final_strategy - initial_capital
        profit_market = final_market - initial_capital
        diff_profit = profit_strategy - profit_market
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Capital Final (IA)", f"{final_strategy:,.2f} $", delta=f"{profit_strategy:+,.2f} $")
        m2.metric("Capital Final (Marché)", f"{final_market:,.2f} $", delta=f"{profit_market:+,.2f} $", delta_color="off")
        
        if diff_profit > 0:
            m3.metric("Surperformance Globale", f"+{diff_profit:,.2f} $", delta="L'IA bat le marché", delta_color="normal")
        else:
            m3.metric("Sous-performance Globale", f"{diff_profit:,.2f} $", delta="Le marché bat l'IA", delta_color="inverse")
    else:
        st.warning("Veuillez cliquer sur Entraîner pour générer la synthèse globale.")


# --- PAGES ACADÉMIE ---

def page_indicators():
    st.title("📚 Académie Quantitative : Les Indicateurs Techniques")
    st.markdown("""
    Bienvenue dans la masterclass sur l'Analyse Quantitative. Contrairement à l'analyse fondamentale (qui lit les bilans comptables), l'analyse quantitative postule que **toute l'information connue est déjà dans le prix et le volume**.
    Les indicateurs sont les "lunettes" qui permettent à notre IA de voir la structure invisible du marché.
    """)
    
    st.header("1. Les Indicateurs de Tendance (Suivre le courant)")
    st.info("La tendance est votre amie. Ces indicateurs aident l'IA à savoir si elle nage avec ou contre le courant.")
    
    with st.expander("📈 Les Moyennes Mobiles (SMA 20 & 50)", expanded=False):
        st.latex(r"SMA_n = \frac{1}{n} \sum_{i=1}^{n} P_i")
        st.markdown("""
        **Théorie Mathématique :** La Simple Moving Average (SMA) calcule la moyenne des prix de clôture sur *n* périodes. Elle agit comme un filtre passe-bas en éliminant le "bruit" quotidien.
        - **SMA 20** : Tendance à court terme (environ 1 mois de trading). Très réactive.
        - **SMA 50** : Tendance à moyen terme (environ 2,5 mois). Plus lente, elle sert de support majeur.
        
        **Exemple Concret :** Imaginez l'action Apple (AAPL). Si son prix actuel est de 150$ mais que sa SMA 50 est à 130$, l'action est dans une forte tendance haussière. 
        
        **Le Signal du "Golden Cross" :** Lorsque la SMA 20 croise la SMA 50 *vers le haut*, c'est un signal d'achat institutionnel. L'IA XGBoost apprend à détecter cet écartement entre les deux moyennes pour comprendre la direction du momentum.
        """)
    
    with st.expander("📉 Le MACD (Convergence / Divergence)"):
        st.latex(r"MACD = EMA_{12}(P) - EMA_{26}(P)")
        st.latex(r"Signal = EMA_{9}(MACD)")
        st.markdown("""
        **Théorie Mathématique :** L'Exponential Moving Average (EMA) donne plus de poids mathématique aux jours récents (contrairement à la SMA qui donne le même poids à tous les jours). Le MACD soustrait l'EMA lente (26 jours) de l'EMA rapide (12 jours).
        
        **Ce que l'IA y voit :** Le MACD ne mesure pas le prix, il mesure la **vitesse** du prix (la dérivée première). 
        - Si MACD > 0 : La tendance court terme est plus rapide que la tendance long terme (Accélération haussière).
        - **Exemple de Trading :** Si le MACD est positif mais commence à s'aplatir, l'IA sait que la hausse "s'essouffle" avant même que le prix ne baisse !
        """)
        
    st.header("2. Les Oscillateurs (Trouver le point de rupture)")
    st.warning("Les oscillateurs sont limités entre deux bornes (souvent 0 et 100). Ils sont cruciaux pour détecter les bulles et les krachs.")
    
    with st.expander("📊 Le RSI (Relative Strength Index)"):
        st.latex(r"RSI = 100 - \frac{100}{1 + RS} \quad \text{où} \quad RS = \frac{\text{Moyenne des Hausses}_{14}}{\text{Moyenne des Baisses}_{14}}")
        st.markdown("""
        **Théorie Mathématique :** Le RSI compare l'ampleur des gains récents à l'ampleur des pertes récentes sur 14 jours.
        
        **Les Bornes Classiques :**
        - **> 70 (Surachat)** : L'action est montée trop vite, trop fort. Les acheteurs sont épuisés.
        - **< 30 (Survente)** : L'action a été massacrée. C'est souvent là que l'argent "intelligent" (Smart Money) commence à racheter discrètement.
        
        **💡 Pro-Tip (La Divergence) :** C'est le signal le plus puissant pour une IA. Si le prix d'Apple atteint un nouveau sommet historique à 180$, mais que le RSI est *plus bas* qu'au sommet précédent (ex: 65 au lieu de 80), c'est une **Divergence Baissière**. Le prix monte par inertie, mais la force d'achat réelle s'est effondrée. L'IA vendra.
        """)
        
    with st.expander("🔄 L'Oscillateur Stochastique (%K)"):
        st.latex(r"\%K = 100 \times \frac{C - L_{14}}{H_{14} - L_{14}}")
        st.markdown("""
        **Théorie Mathématique :** Où se situe le prix de clôture ($C$) par rapport à la fourchette absolue du prix sur 14 jours (Le plus haut $H$ et le plus bas $L$) ?
        
        **Exemple Concret :** Si Tesla oscille entre 200$ et 250$ depuis deux semaines, et clôture ce soir à 245$. Son Stochastique sera de 90% (très proche du plafond). 
        - Contrairement au RSI (qui marche bien en tendance), le Stochastique est le roi des marchés "en range" (qui font du sur-place). L'IA XGBoost l'utilise pour trader les rebonds entre supports et résistances.
        """)
        
    st.header("3. La Volatilité (L'énergie du marché)")
    st.success("La volatilité est cyclique : un marché très calme précède toujours une tempête explosive.")
    
    with st.expander("🌪️ Les Bandes de Bollinger & Le 'Squeeze'"):
        st.latex(r"Upper = SMA_{20} + 2\sigma \quad | \quad Lower = SMA_{20} - 2\sigma")
        st.markdown("""
        **Théorie Mathématique :** On prend la moyenne des prix (SMA 20), et on ajoute/soustrait 2 écarts-types statistiques ($\sigma$). En statistiques, cela signifie que **95% des prix futurs seront contenus dans ces bandes**.
        
        **Le Concept du "Squeeze" (BB Width) :** 
        L'indicateur `BB_Width` mesure l'écartement entre la bande haute et basse. 
        - Quand le marché est ennuyeux, les bandes se resserrent (Squeeze). L'énergie s'accumule comme un ressort.
        - L'IA adore le BB_Width très bas, car cela indique qu'un mouvement violent est imminent. Si l'IA voit un Squeeze + une explosion du volume, elle donne un signal d'Achat Fort immédiat.
        """)
        
    st.header("4. Le Volume & L'Argent Intelligent (Smart Money)")
    with st.expander("🌊 L'OBV (On-Balance Volume) & Volume Ratio"):
        st.latex(r"OBV = OBV_{t-1} + \begin{cases} Volume & \text{si prix } \nearrow \\ -Volume & \text{si prix } \searrow \end{cases}")
        st.markdown("""
        **Théorie Mathématique :** L'OBV est un totaliseur. Si l'action finit dans le vert, on additionne tout le volume du jour. Si elle finit rouge, on le soustrait.
        
        **Détecter les "Baleines" (Institutions) :**
        Les petits traders regardent le prix. Les algorithmes bancaires regardent le volume. 
        - **Phase d'Accumulation :** Le prix de l'action fait du sur-place pendant 3 mois. Les particuliers s'ennuient et vendent. Mais l'OBV monte en flèche ! Pourquoi ? Parce que les fonds d'investissement achètent massivement mais lentement pour ne pas faire monter le prix. 
        - Quand l'IA XGBoost voit l'OBV monter alors que le prix stagne, elle anticipe la future explosion à la hausse.
        """)
        
    st.header("5. L'Économie Mondiale (Macroéconomie)")
    with st.expander("🌍 VIX, SPY et Taux d'intérêts (TNX)"):
        st.markdown("""
        Une action ne vit pas dans une bulle. Elle est soumise à la gravité de l'économie mondiale.
        - **Le VIX (Indice de la Peur)** : Calculé à partir du prix des options sur le S&P 500. S'il dépasse 30, le marché est en mode panique absolue (Krach Covid, Subprimes). L'IA est programmée pour reconnaître ces régimes de crise et couper les positions longues (Achat).
        - **TNX (Obligations à 10 ans US)** : C'est le prix de l'argent. Si les taux obligataires montent (ex: 5%), les investisseurs retirent leur argent des actions risquées (Tech, Crypto) pour le placer sans risque. L'IA utilise le TNX pour ajuster son exposition au risque.
        """)

def page_xgboost():
    st.title("🤖 Académie Quantitative : Plongée dans XGBoost")
    st.markdown("""
    Oubliez les indicateurs de base. Les fonds quantitatifs comme Renaissance Technologies utilisent le Machine Learning. 
    L'algorithme star de ces dernières années sur les données tabulaires (comme la finance) est **XGBoost** (*eXtreme Gradient Boosting*).
    Voici comment nous avons transformé un modèle mathématique en trader professionnel.
    """)
    
    st.header("1. La brique de base : L'Arbre de Décision")
    st.info("XGBoost n'est pas un réseau de neurones. C'est une forêt d'arbres de décision très intelligents.")
    st.markdown("""
    Imaginez un organigramme qui pose des questions "Oui/Non" à la vitesse de la lumière.
    1. À la base (la racine), l'arbre pose la question mathématiquement la plus discriminante : *« Le VIX est-il supérieur à 25 ? »*.
    2. Si **OUI** (Panique de marché), il part à gauche : *« Le RSI est-il < 30 (Survente) ? »* -> Si Oui, prédiction = ACHAT (Rebond).
    3. Si **NON** (Marché calme), il part à droite : *« La SMA 20 est-elle au-dessus de la SMA 50 ? »* -> Si Oui, prédiction = ACHAT (Suivi de tendance).
    
    **Le Problème :** Un seul arbre est stupide. S'il est trop simple, il se trompe. S'il est trop complexe, il apprend le passé par cœur.
    """)
    
    st.header("2. Le Cœur du Réacteur : Le Gradient Boosting")
    st.markdown("""
    XGBoost signifie **Boosting par Gradient**. C'est une méthode d'apprentissage dite *Ensembliste* (La sagesse des foules).
    
    L'algorithme entraîne **100 arbres de décision différents** (`n_estimators = 100`), mais pas de façon aléatoire. Il les entraîne *séquentiellement* :
    - **Arbre n°1** fait une prédiction. Il a raison à 60%, mais il se trompe sur les krachs boursiers.
    - **Arbre n°2** regarde *uniquement* les erreurs de l'Arbre 1. Son seul but dans la vie est d'apprendre à prédire les krachs boursiers.
    - **Arbre n°3** corrige les erreurs de l'Arbre 2, etc.
    
    À la fin, la prédiction finale de l'application (le "Signal XGBoost") est la **somme pondérée** des prédictions de ces 100 arbres experts. Le terme "Gradient" vient du fait que chaque arbre minimise la fonction d'erreur (la descente de gradient) de l'arbre précédent.
    """)
    
    st.header("3. La Salle des Machines : Les Hyperparamètres")
    st.warning("Un mauvais réglage, et l'IA perdra tout votre capital. Voici ce que fait l'option 'Auto-Optimisation' de l'application.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🌳 Max Depth (Profondeur)")
        st.markdown("""
        Détermine le nombre de questions successives qu'un arbre peut poser. 
        - **Dans notre App :** Limité à `5`.
        - **Pourquoi ?** Si on met 20, l'arbre pose tellement de questions spécifiques qu'il finit par dire "J'achète uniquement s'il fait beau, un mardi, et que l'action vaut 124.5$". C'est mortel en finance.
        """)
    with c2:
        st.subheader("🐢 Learning Rate (Vitesse)")
        st.markdown("""
        Définit l'impact de chaque nouvel arbre.
        - **Dans notre App :** Réglé très bas (`0.05`).
        - **Pourquoi ?** L'IA est obligée d'avancer à tout petits pas. Cela l'empêche de tirer des conclusions définitives basées sur une seule aberration de marché (comme un tweet d'Elon Musk).
        """)
    with c3:
        st.subheader("🌲 N Estimators (Quantité)")
        st.markdown("""
        Le nombre total d'arbres dans la forêt.
        - **Dans notre App :** `100` à `200`.
        - **Pourquoi ?** Plus le Learning Rate est bas, plus il faut d'arbres pour converger vers la solution optimale.
        """)
        
    st.header("4. Le Boss Final du Quant : L'Overfitting")
    st.error("**Définition :** L'Overfitting (Sur-apprentissage) survient quand l'IA confond le 'Bruit' (le hasard) avec le 'Signal' (la vraie loi mathématique).")
    st.markdown("""
    Les marchés financiers sont composés à 80% de bruit (hasard) et 20% de signal. 
    
    Si vous montrez à une IA naïve 5 ans d'historique de l'action Apple, elle va mémoriser l'historique parfait. Lors de son backtest sur ces mêmes 5 ans, elle aura 99% de réussite. Vous vous croirez riche. 
    Mais demain matin, sur des données qu'elle n'a jamais vu (le monde réel), elle perdra tout car elle appliquera aveuglément les lois du passé.
    
    **Comment notre application se protège :**
    Nous utilisons la technique du *Train/Test Split*. L'IA s'entraîne uniquement sur les 80% premiers jours (Train). Ensuite, l'application lui cache les réponses et la force à trader sur les 20% derniers jours (Test). La "Précision" affichée dans le Terminal est sa performance sur ces 20% inconnus. C'est la seule métrique qui compte !
    """)

def page_news():
    st.title("📰 Académie Quantitative : NLP & Sentiment de Marché")
    st.markdown("L'analyse technique mathématique ne peut pas tout prévoir. Parfois, un simple tweet ou un scandale médiatique fait s'effondrer une action en 5 minutes. C'est là qu'intervient le NLP.")
    
    st.header("Qu'est-ce que le NLP ?")
    st.markdown("""
    Le **Natural Language Processing (NLP)**, ou Traitement du Langage Naturel, permet à un ordinateur de "lire" et de "comprendre" le texte humain.
    
    Dans notre application, nous utilisons la librairie d'Intelligence Artificielle **TextBlob**. Son rôle est de lire les actualités et de leur attribuer une *Polarité* mathématique.
    """)
    
    st.header("La Mécanique de la Polarité")
    st.info("La Polarité est un score allant de **-1.0 (Désespoir absolu)** à **+1.0 (Euphorie totale)**.")
    st.markdown("""
    TextBlob possède un dictionnaire massif de mots pré-évalués. 
    - Le mot *"effondrement"* vaut -0.8. 
    - Le mot *"record"* vaut +0.7. 
    - Le mot *"bénéfice"* vaut +0.5.
    
    Lorsque l'application télécharge les articles du jour sur Yahoo Finance, elle calcule la moyenne de tous les mots contenus dans les gros titres pour vous donner la "Météo Psychologique" de l'action.
    """)
    
    st.header("L'Approche Hybride du Trader Pro")
    st.warning("**XGBoost (Maths)** + **TextBlob (Émotions)** = **Edge (Avantage)**")
    st.markdown("""
    1. Si XGBoost dit ACHAT et que les News disent POSITIF -> **Achat Fort (Haute conviction)**.
    2. Si XGBoost dit ACHAT mais que les News disent NÉGATIF -> **Danger !** Le modèle mathématique ne sait peut-être pas que le PDG vient de démissionner. Il vaut mieux ignorer le signal.
    """)

def format_large_number(num):
    if num is None or pd.isna(num): return "N/A"
    try:
        num = float(num)
    except:
        return str(num)
    if num >= 1e12: return f"{num/1e12:.2f} T"
    if num >= 1e9: return f"{num/1e9:.2f} B"
    if num >= 1e6: return f"{num/1e6:.2f} M"
    return f"{num:,.2f}"

def page_fundamentals(tickers):
    st.title("🏢 Analyse Fondamentale & Screener")
    st.markdown("Plongez dans les bilans financiers et comparez la santé réelle des entreprises sélectionnées.")
    
    with st.expander("📖 Guide Détaillé des Indicateurs Fondamentaux (Lexique)"):
        st.markdown("""
        **📈 Valorisation & Prix**
        *   **Capitalisation Boursière :** La valeur totale de l'entreprise sur les marchés financiers. (Prix d'une action × Nombre total d'actions).
        *   **P/E Ratio (Price-to-Earnings ou PER) :** Le "Prix sur Bénéfices". Il indique combien d'années de bénéfices il faut pour rembourser le prix de l'action. Un PER élevé indique de fortes attentes de croissance.
        *   **PEG Ratio :** Le P/E Ratio divisé par la croissance attendue des bénéfices. Un PEG < 1 signale souvent que l'action est sous-évaluée par rapport à sa croissance future.
        *   **Price / Sales :** Capitalisation divisée par le Chiffre d'Affaires. Très utile pour les entreprises en forte croissance qui ne dégagent pas encore de bénéfices nets.
        *   **EV / EBITDA :** Valeur d'Entreprise sur EBITDA (bénéfices bruts). C'est le ratio préféré des fonds d'investissement (M&A, Private Equity) car il prend en compte la dette.

        **💵 Trésorerie & Dette**
        *   **Debt / Equity (Dette sur Capitaux Propres) :** Mesure le niveau d'endettement. Un ratio > 100 signifie que l'entreprise a plus de dettes que d'argent apporté par les actionnaires (Risque élevé en cas de crise).
        *   **Current Ratio :** Liquidité à court terme (Actifs / Dettes à moins d'un an). S'il est < 1, l'entreprise pourrait avoir du mal à payer ses factures immédiates.
        *   **Free Cash Flow (Flux de Trésorerie Disponible) :** L'argent "réel" qu'il reste en caisse après avoir payé le fonctionnement et les investissements. C'est le cash qui sert à payer les dividendes ou racheter des actions.
        *   **Rendement Dividende :** Le pourcentage du prix de l'action qui est payé chaque année en cash aux actionnaires.

        **🚀 Rentabilité & Risque**
        *   **Marge Nette :** Le pourcentage de profit pur réalisé sur chaque vente. Une entreprise avec une marge élevée possède un fort pouvoir de fixation des prix (Pricing Power).
        *   **ROE (Return On Equity) :** La rentabilité financière. Si vous donnez 100$ à la direction, combien de profit génère-t-elle ? Un ROE > 15% est généralement excellent.
        *   **Beta :** L'indicateur de volatilité systémique. Beta = 1 équivaut au marché. Beta > 1 signifie que l'action amplifie les mouvements du marché (Plus risquée). Beta < 1 indique une valeur défensive.
        """)

    if not tickers:
        st.warning("Veuillez sélectionner au moins une action dans le menu latéral (Configuration Globale).")
        return
        
    comparative_data = []
    detailed_views = []
    
    with st.spinner("Téléchargement des données fondamentales en cours..."):
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                if 'shortName' not in info and 'longName' not in info:
                    continue
                    
                name = info.get('longName', info.get('shortName', ticker))
                
                # Extraction des métriques pour la table comparative
                comparative_data.append({
                    "Entreprise": name,
                    "Ticker": ticker,
                    "Secteur": info.get('sector', 'N/A'),
                    "Cap. Boursière": info.get('marketCap'),
                    "P/E (Actuel)": info.get('trailingPE'),
                    "P/E (Futur)": info.get('forwardPE'),
                    "PEG Ratio": info.get('pegRatio'),
                    "Marge Nette": info.get('profitMargins'),
                    "ROE": info.get('returnOnEquity'),
                    "Debt/Equity": info.get('debtToEquity'),
                    "Dividende": info.get('dividendYield'),
                    "Beta": info.get('beta')
                })
                
                # Sauvegarde l'info complète pour l'affichage détaillé
                detailed_views.append({'ticker': ticker, 'name': name, 'info': info})
                
            except Exception as e:
                st.error(f"Erreur lors de la récupération pour {ticker}: {e}")

    # --- 1. TABLEAU COMPARATIF ---
    if len(comparative_data) > 1:
        st.header("📊 Synthèse Comparative (Screener)")
        df_comp = pd.DataFrame(comparative_data)
        
        # Formatage du DataFrame pour l'affichage
        df_display = df_comp.copy()
        df_display['Cap. Boursière'] = df_display['Cap. Boursière'].apply(lambda x: format_large_number(x))
        df_display['Marge Nette'] = df_display['Marge Nette'].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
        df_display['ROE'] = df_display['ROE'].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
        df_display['Dividende'] = df_display['Dividende'].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
        
        # Arrondir les colonnes numériques
        for col in ['P/E (Actuel)', 'P/E (Futur)', 'PEG Ratio', 'Debt/Equity', 'Beta']:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        st.divider()

    # --- 2. FICHES DÉTAILLÉES ---
    st.header("🔍 Fiches Détaillées (Deep Dive)")
    for item in detailed_views:
        info = item['info']
        name = item['name']
        ticker = item['ticker']
        
        st.subheader(f"{name} ({ticker})")
        
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        summary = info.get('longBusinessSummary', 'Aucune description disponible.')
        
        with st.expander(f"📝 Profil de l'entreprise : {sector} - {industry}"):
            st.write(summary)
            
        # Colonnes principales
        st.markdown("##### 📈 Valorisation & Prix")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capitalisation", format_large_number(info.get('marketCap')), help="Valeur totale de l'entreprise sur le marché (Prix × Nombre d'actions).")
        c2.metric("P/E Ratio", f"{info.get('trailingPE'):.2f}" if info.get('trailingPE') else "N/A", help="Price-to-Earnings (PER). Indique combien d'années de bénéfices il faut pour rembourser le prix de l'action. Mesure la cherté de l'action.")
        c3.metric("Price / Sales", f"{info.get('priceToSalesTrailing12Months'):.2f}" if info.get('priceToSalesTrailing12Months') else "N/A", help="Capitalisation divisée par le chiffre d'affaires. Utile pour les entreprises non rentables.")
        c4.metric("EV / EBITDA", f"{info.get('enterpriseToEbitda'):.2f}" if info.get('enterpriseToEbitda') else "N/A", help="Valeur d'Entreprise / EBITDA. Évalue la rentabilité brute indépendamment de la dette ou des impôts.")

        st.markdown("##### 💵 Trésorerie & Dette")
        c5, c6, c7, c8 = st.columns(4)
        de = info.get('debtToEquity')
        c5.metric("Debt / Equity", f"{de:.2f}" if de else "N/A", help="Ratio Dette sur Capitaux Propres. > 100 signifie que l'entreprise a plus de dettes que de fonds propres.")
        c6.metric("Current Ratio", f"{info.get('currentRatio'):.2f}" if info.get('currentRatio') else "N/A", help="Liquidité à court terme (>1 = entreprise saine capable de payer ses factures imminentes).")
        c7.metric("Free Cash Flow", format_large_number(info.get('freeCashflow')), help="Le cash réel qu'il reste à l'entreprise après tout paiement. L'oxygène de l'entreprise.")
        c8.metric("Rendement Dividende", f"{info.get('dividendYield')*100:.2f}%" if info.get('dividendYield') else "N/A", help="Le rendement annuel versé en cash directement aux actionnaires.")

        st.markdown("##### 🚀 Rentabilité & Risque")
        c9, c10, c11, c12 = st.columns(4)
        c9.metric("Marge Nette", f"{info.get('profitMargins')*100:.2f}%" if info.get('profitMargins') else "N/A", help="Sur 100$ de ventes, combien reste-t-il de profit net ? Mesure l'avantage concurrentiel.")
        c10.metric("ROE", f"{info.get('returnOnEquity')*100:.2f}%" if info.get('returnOnEquity') else "N/A", help="Return on Equity : Efficacité de la direction à générer du profit avec l'argent des actionnaires.")
        c11.metric("Beta", f"{info.get('beta'):.2f}" if info.get('beta') else "N/A", help="Volatilité systémique. >1 = Plus risqué et volatil que le marché global.")
        c12.metric("PEG Ratio", f"{info.get('pegRatio'):.2f}" if info.get('pegRatio') else "N/A", help="P/E Ratio divisé par la croissance. < 1 signale une action potentiellement sous-évaluée.")
        
        st.markdown("##### 🕵️ Données Alternatives & Marché")
        c13, c14, c15, c16 = st.columns(4)
        c13.metric("Détention Initiés", f"{info.get('heldPercentInsiders')*100:.2f}%" if info.get('heldPercentInsiders') else "N/A", help="Pourcentage d'actions détenues par les dirigeants et fondateurs. Un chiffre élevé montre qu'ils croient en leur entreprise (Skin in the game).")
        c14.metric("Détention Institutionnelle", f"{info.get('heldPercentInstitutions')*100:.2f}%" if info.get('heldPercentInstitutions') else "N/A", help="Pourcentage d'actions détenues par les Hedge Funds et Banques. Un chiffre élevé = confiance de la 'Smart Money'.")
        c15.metric("52-Week High", f"{info.get('fiftyTwoWeekHigh'):.2f} $" if info.get('fiftyTwoWeekHigh') else "N/A", help="Prix le plus haut atteint sur les 52 dernières semaines.")
        c16.empty()
        
        st.write("---")

def page_strategy_academy():
    st.title("🎓 Académie : Stratégie & Risk Management")
    st.markdown("Bienvenue dans le centre de formation avancé pour les Quants.")
    
    st.header("1. L'Alpha (α) et le Beta (β)")
    st.markdown("""
    Dans le monde institutionnel, la performance brute ne veut rien dire. 
    - **Le Beta (β)** : C'est la performance due au marché global. Si vous achetez le S&P 500, vous avez un Beta de 1. Vous gagnez quand le marché gagne.
    - **L'Alpha (α)** : C'est la **véritable valeur ajoutée** de votre stratégie. C'est l'argent gagné indépendamment des mouvements du marché (par exemple, gagner de l'argent pendant un krach, ou battre l'indice de référence).
    """)
    st.info("💡 **Objectif du Trading Algorithmique** : Extraire de l'Alpha régulier, peu importe que le marché monte ou baisse.")

    st.header("2. Les Mathématiques de la Ruine (Drawdown)")
    st.markdown("""
    Le **Maximum Drawdown (MDD)** est votre pire ennemi. C'est la perte maximale historique de votre portefeuille.
    
    Voici pourquoi la gestion du risque est la seule chose qui compte en trading :
    *   Si vous perdez **10%**, il vous faut **11%** de gain pour revenir à zéro.
    *   Si vous perdez **20%**, il vous faut **25%** de gain pour revenir à zéro.
    *   Si vous perdez **50%**, il vous faut **100%** de gain pour revenir à zéro.
    *   Si vous perdez **90%**, il vous faut **900%** de gain !
    """)
    st.warning("🚨 Règle d'Or : Coupez vos pertes rapidement. Ne laissez jamais un trade détruire votre capital, car les mathématiques jouent contre vous.")

    st.header("3. La Règle des 2% et la Taille de Position")
    st.markdown("""
    Comment ne jamais se ruiner ? En utilisant la **Règle des 2%** de risque maximum.
    
    Sur n'importe quel trade, si votre Stop-Loss est touché, vous ne devez perdre que 2% de votre capital total. 
    
    **Calcul de la taille de position** :
    $$ Taille = \\frac{Capital \\times 2\\%}{Prix_{Entrée} - Prix_{StopLoss}} $$
    
    L'application calcule cela automatiquement pour vous dans le "Plan de Trading du Jour".
    """)

    st.header("4. Le Ratio Risque/Récompense (Risk/Reward)")
    st.markdown("""
    Même si votre IA (XGBoost) a un "Win Rate" de seulement 40% (elle a tort 60% du temps), vous pouvez être massivement rentable.
    
    Comment ? Avec un **Risk/Reward de 1:3**.
    *   Vous risquez 100$ pour gagner 300$.
    *   Sur 10 trades : 6 pertes de 100$ (-600$), 4 gains de 300$ (+1200$).
    *   Bénéfice total : **+600$** alors que vous avez eu tort la majorité du temps !
    """)
    st.success("C'est pourquoi notre Plan de Trading définit systématiquement un Stop-Loss à 1.5 ATR et un Take-Profit à 3.0 ATR (Ratio de 1:2 strict).")

def get_portfolio_path():
    # Si le dossier /app/data existe (cas d'un volume monté sur Dokploy), on l'utilise.
    # Sinon, on sauvegarde localement dans le dossier courant.
    data_dir = "/app/data"
    if os.path.exists(data_dir):
        return os.path.join(data_dir, "portfolio.json")
    return "portfolio.json"

def load_portfolio():
    path = get_portfolio_path()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass # Fichier vide ou corrompu, on recrée par défaut
    return {"cash": 100000.0, "positions": {}, "history": []}

def save_portfolio(pf):
    path = get_portfolio_path()
    with open(path, "w") as f:
        json.dump(pf, f, indent=4)

def page_paper_trading():
    st.title("🕹️ Simulateur Paper Trading")
    st.markdown("Exécutez vos stratégies en conditions réelles sans risquer votre argent. Solde de départ : **100 000 $**.")
    
    pf = load_portfolio()
    
    st.header("💼 Mon Portefeuille Virtuel")
    c1, c2, c3 = st.columns(3)
    c1.metric("Liquidités (Cash)", f"{pf['cash']:,.2f} $")
    
    total_positions_value = 0
    if pf['positions']:
        # Fetch current prices for open positions
        tickers_list = list(pf['positions'].keys())
        st.subheader("📈 Positions Ouvertes")
        pos_data = []
        for t in tickers_list:
            try:
                current_price = yf.download(t, period="1d", progress=False)['Close'].iloc[-1]
                if isinstance(current_price, pd.Series):
                    current_price = current_price.iloc[0]
            except:
                current_price = pf['positions'][t]['avg_price'] # fallback
            
            qty = pf['positions'][t]['qty']
            avg_price = pf['positions'][t]['avg_price']
            value = qty * current_price
            pnl = value - (qty * avg_price)
            pnl_pct = (pnl / (qty * avg_price)) * 100
            
            total_positions_value += value
            
            with st.expander(f"📦 {t} | Valeur: {value:.2f} $ | P&L: {pnl:+.2f} $ ({pnl_pct:+.2f}%)"):
                col1, col2 = st.columns(2)
                col1.write(f"**Quantité:** {qty}")
                col1.write(f"**Prix d'Achat Moyen:** {avg_price:.2f} $")
                col2.write(f"**Prix Actuel:** {current_price:.2f} $")
                
                if st.button(f"🔴 Liquider {t} au marché", key=f"liq_{t}"):
                    revenue = qty * current_price
                    pf['cash'] += revenue
                    pf['history'].append({
                        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'action': 'SELL (Manual)',
                        'ticker': t,
                        'qty': qty,
                        'price': current_price,
                        'total': revenue,
                        'pnl': pnl
                    })
                    del pf['positions'][t]
                    save_portfolio(pf)
                    st.success(f"Position {t} liquidée avec succès !")
                    st.rerun()
    else:
        st.info("Aucune position ouverte actuellement. Utilisez le Terminal de Trading pour acheter.")
        
    c2.metric("Valeur des Positions", f"{total_positions_value:,.2f} $")
    c3.metric("Valeur Totale du Portefeuille", f"{(pf['cash'] + total_positions_value):,.2f} $", delta=f"{((pf['cash'] + total_positions_value) - 100000):+,.2f} $")
    
    st.divider()
    st.header("📜 Historique des Transactions")
    if pf['history']:
        hist_df = pd.DataFrame(pf['history'])
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.write("Aucun historique pour le moment.")
        
    if st.button("⚠️ Réinitialiser le compte (Remise à 100 000 $)"):
        save_portfolio({"cash": 100000.0, "positions": {}, "history": []})
        st.success("Compte réinitialisé.")
        st.rerun()

def page_advanced_academy():
    st.title("🎓 Académie : Modélisation Avancée")
    st.markdown("Plongez dans les mathématiques utilisées par les Hedge Funds.")
    
    st.header("1. Optimisation Bayésienne (Optuna)")
    st.markdown("""
    Plutôt que d'essayer des milliers de paramètres au hasard (Grid Search) ou aléatoirement (Randomized Search), les professionnels utilisent **Optuna**.
    *   C'est une IA qui entraîne l'IA. 
    *   Elle utilise des algorithmes probabilistes (TPE - Tree-structured Parzen Estimator) pour deviner quels paramètres vont améliorer le modèle.
    *   Si un essai est mauvais, Optuna "apprend" la zone de l'échec et ne la teste plus.
    """)
    st.info("Dans notre application, cochez **Auto-Optimisation** pour activer Optuna. C'est plus lent, mais le Sharpe Ratio généré est souvent drastiquement supérieur.")

    st.header("2. La Simulation de Monte Carlo")
    st.markdown("""
    Créée lors du projet Manhattan pour la bombe atomique, la simulation de Monte Carlo utilise l'aléatoire pour résoudre des problèmes déterministes.
    
    **En finance :** On calcule la volatilité historique et on génère des milliers de scénarios futurs probables (chemins aléatoires ou Mouvement Brownien).
    Cela permet de calculer la **Value at Risk (VaR)** : "Quel est le scénario catastrophe qui a 5% de chance de se produire ?"
    """)

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
        delta = si.norm.cdf(d1, 0.0, 1.0)
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0)
        delta = -si.norm.cdf(-d1, 0.0, 1.0)
        
    gamma = si.norm.pdf(d1, 0.0, 1.0) / (S * sigma * np.sqrt(T))
    vega = S * si.norm.pdf(d1, 0.0, 1.0) * np.sqrt(T)
    if option_type == "call":
        theta = (- (S * sigma * si.norm.pdf(d1, 0.0, 1.0)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)) / 365
    else:
        theta = (- (S * sigma * si.norm.pdf(d1, 0.0, 1.0)) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)) / 365
    return price, delta, gamma, theta, vega

def page_options_pricing(tickers):
    st.title("🧮 Pricing d'Options (Black-Scholes)")
    st.markdown("Calculez la valeur théorique (Juste Prix) d'une option d'achat (Call) ou de vente (Put) ainsi que ses paramètres de risque (Les Greeks).")
    
    if not tickers:
        st.warning("Veuillez sélectionner au moins une action dans la barre latérale.")
        return
        
    ticker = tickers[0]
    st.header(f"Contrat d'Option sur {ticker}")
    
    try:
        df = yf.download(ticker, period="1y", progress=False)
        S = df['Close'].iloc[-1].item() if isinstance(df['Close'].iloc[-1], pd.Series) else df['Close'].iloc[-1]
        returns = df['Close'].pct_change().dropna()
        sigma_hist = returns.std() * np.sqrt(252) # Volatilité annualisée
        if isinstance(sigma_hist, pd.Series):
            sigma_hist = sigma_hist.iloc[0].item()
        elif hasattr(sigma_hist, 'item'):
            sigma_hist = sigma_hist.item()
    except:
        S = 100.0
        sigma_hist = 0.20
        
    col1, col2 = st.columns(2)
    with col1:
        S_input = st.number_input("Prix de l'action (Spot - S)", value=float(S))
        K = st.number_input("Prix d'exercice (Strike - K)", value=float(S))
        T_days = st.slider("Jours avant expiration", 1, 365, 30)
        T = T_days / 365.0
    with col2:
        sigma = st.slider("Volatilité Implicite (σ)", 0.01, 1.0, float(sigma_hist))
        r = st.slider("Taux d'intérêt sans risque (r)", 0.0, 0.10, 0.05)
        opt_type = st.radio("Type d'Option", ["Call", "Put"])
        
    price, delta, gamma, theta, vega = black_scholes(S_input, K, T, r, sigma, opt_type.lower())
    
    st.divider()
    st.subheader(f"Valeur Théorique de l'Option ({opt_type}) : **{price:.2f} $**")
    
    st.write("### Les Greeks (Paramètres de Risque)")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Delta (Δ)", f"{delta:.3f}", help="Sensibilité au prix. Si l'action monte de 1$, l'option prendra Delta $. (0 à 1 pour Call, -1 à 0 pour Put).")
    g2.metric("Gamma (Γ)", f"{gamma:.4f}", help="Vitesse du Delta. De combien le Delta change si l'action monte de 1$.")
    g3.metric("Theta (Θ)", f"{theta:.3f} $/jour", help="Érosion du temps. Combien l'option perd de valeur chaque jour qui passe (Time Decay).")
    g4.metric("Vega (ν)", f"{vega/100:.3f}", help="Sensibilité à la volatilité. Si la volatilité augmente de 1%, l'option prend Vega $.")

    # --- GRAPHIQUE DE PAYOFF ---
    st.divider()
    st.subheader("📊 Graphique de Payoff à l'expiration")
    
    # Générer des prix possibles à l'expiration (±30% autour du strike)
    prices_range = np.linspace(K * 0.7, K * 1.3, 100)
    
    if opt_type.lower() == "call":
        payoff = np.maximum(prices_range - K, 0) - price
        breakeven = K + price
    else:
        payoff = np.maximum(K - prices_range, 0) - price
        breakeven = K - price
        
    fig_payoff = go.Figure()
    # Zone de profit (vert) et zone de perte (rouge)
    fig_payoff.add_trace(go.Scatter(
        x=prices_range, y=payoff,
        mode='lines',
        name='Profit/Perte',
        line=dict(color='white', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 0, 0.2)' if opt_type.lower() == "call" else 'rgba(255, 0, 0, 0.2)'
    ))
    
    # Ligne 0
    fig_payoff.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Point de Break-even
    fig_payoff.add_vline(x=breakeven, line_dash="dash", line_color="yellow", annotation_text=f"Break-even: {breakeven:.2f} $")
    
    # Prix actuel (Spot)
    fig_payoff.add_vline(x=S_input, line_dash="dot", line_color="blue", annotation_text=f"Spot Actuel: {S_input:.2f} $")

    # Mettre en rouge la partie négative
    payoff_negative = np.where(payoff < 0, payoff, 0)
    fig_payoff.add_trace(go.Scatter(
        x=prices_range, y=payoff_negative,
        mode='none',
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.5)',
        showlegend=False
    ))
    
    fig_payoff.update_layout(
        title=f"Profil de Profit & Perte (Achat de {opt_type})",
        xaxis_title="Prix du sous-jacent à l'expiration ($)",
        yaxis_title="Profit / Perte ($)",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig_payoff, use_container_width=True)

def page_options_academy():
    st.title("🎓 Académie : Options & Black-Scholes")
    st.markdown("Comprendre les produits dérivés, l'arme absolue des Quants institutionnels.")
    
    st.header("1. Qu'est-ce qu'une Option ?")
    st.markdown("""
    Une option est un contrat (un produit dérivé) qui vous donne le **droit** (mais non l'obligation) d'acheter (CALL) ou de vendre (PUT) une action à un prix fixé à l'avance (Strike), jusqu'à une date précise (Expiration).
    
    *   **CALL (Option d'Achat)** : Vous pariez que l'action va monter.
    *   **PUT (Option de Vente)** : Vous pariez que l'action va baisser (ou vous l'utilisez comme assurance pour protéger votre portefeuille d'un krach).
    """)
    
    st.header("2. Le Modèle de Black-Scholes (1973)")
    st.markdown("""
    Avant 1973, personne ne savait comment donner un "juste prix" à une option. Fischer Black et Myron Scholes ont inventé une équation différentielle stochastique qui a révolutionné la finance (et leur a valu le Prix Nobel d'Économie).
    
    La formule prend 5 ingrédients :
    1.  Le prix actuel de l'action.
    2.  Le prix d'exercice (Strike).
    3.  Le temps restant avant l'expiration.
    4.  Le taux d'intérêt sans risque (ex: les bons du Trésor).
    5.  **La Volatilité Implicite** : C'est le paramètre le plus crucial. Plus l'action est nerveuse, plus l'option coûte cher.
    """)
    
    st.header("3. Les 'Greeks' (La gestion du risque)")
    st.info("Les professionnels ne regardent pas le prix de l'option, ils regardent ses Greeks.")
    st.markdown("""
    *   **Delta (Δ)** : L'équivalent en actions. Un Call avec un Delta de 0.5 agit comme si vous possédiez 50 actions réelles.
    *   **Gamma (Γ)** : L'accélération. C'est la convexité de l'option.
    *   **Theta (Θ)** : L'hémorragie. Chaque jour qui passe sans que l'action ne bouge vous fait perdre de l'argent. L'usure du temps.
    *   **Vega (ν)** : L'exposition à la panique. Si le marché panique (le VIX monte), la valeur de votre option augmente, même si l'action ne bouge pas !
    """)

# --- FONCTION PRINCIPALE ---
def main():
    st.sidebar.title("🧭 Menu Principal")
    menu = st.sidebar.radio("Sélectionnez un module :", [
        "📈 Terminal de Trading",
        "🕹️ Paper Trading (Virtuel)",
        "🏢 Fondamentaux Financiers",
        "🧮 Options & Dérivés (Pricing)",
        "🎓 Académie: Stratégie & Risques",
        "🎓 Académie: Modélisation Avancée",
        "🎓 Académie: Options & Dérivés",
        "📚 Académie: Indicateurs",
        "🤖 Académie: XGBoost",
        "📰 Académie: News & Sentiment"
    ])
    
    st.sidebar.divider()
    st.sidebar.header("⚙️ Configuration Globale")
    
    # --- NOUVEAU : Sélection rapide de portefeuilles ---
    portfolio_choice = st.sidebar.selectbox(
        "💡 Sélection rapide de portefeuille",
        options=list(PREDEFINED_PORTFOLIOS.keys()),
        index=0
    )
    
    # Si un portefeuille est choisi, on utilise sa liste. Sinon par défaut Apple.
    default_selection = PREDEFINED_PORTFOLIOS[portfolio_choice] if portfolio_choice != "Sélection Manuelle" else ["Apple Inc. (US)"]
    
    # Filtrer les valeurs par défaut qui sont dans MAJOR_STOCKS pour le multiselect
    valid_defaults = [s for s in default_selection if s in MAJOR_STOCKS.keys()]
    
    # Remplacement par Multiselect
    stock_choices = st.sidebar.multiselect(
        "Rechercher une ou plusieurs actions", 
        options=list(MAJOR_STOCKS.keys()), 
        default=valid_defaults
    )
    
    tickers = []
    
    # Ajout silencieux des tickers qui ne sont pas dans MAJOR_STOCKS (ex: Chine)
    for s in default_selection:
        if s not in MAJOR_STOCKS.keys():
            tickers.append(s)

    if "--- Saisir manuellement ---" in stock_choices:
        custom_ticker = st.sidebar.text_input("Ticker personnalisé (ex: EPA:MC)", "AAPL")
        tickers.append(convert_google_to_yahoo_ticker(custom_ticker))
        
    for choice in stock_choices:
        if choice != "--- Saisir manuellement ---":
            tickers.append(convert_google_to_yahoo_ticker(MAJOR_STOCKS[choice]))
            
    # Déduplication
    tickers = list(set(tickers))
    
    if menu == "📈 Terminal de Trading":
        st.title("🚀 XGBoost Stock Trader Pro")
        st.sidebar.header("Paramètres IA")
        
        period = st.sidebar.selectbox("Période d'historique", ["2y", "5y", "10y", "max"], index=0)
        interval = st.sidebar.selectbox("Intervalle", ["1d", "1wk"], index=0)
        initial_capital = st.sidebar.number_input("Capital Initial Total ($)", min_value=100, max_value=1000000, value=10000, step=100)
        optimize_model = st.sidebar.checkbox("🧠 Auto-Optimisation (Optuna Bayésien)")
        
        if len(tickers) == 0:
            st.warning("👈 Veuillez sélectionner au moins une action dans le menu latéral.")
        elif len(tickers) == 1:
            run_single_mode(tickers[0], period, interval, initial_capital, optimize_model)
        else:
            run_portfolio_mode(tickers, period, interval, initial_capital, optimize_model)
            
    elif menu == "🕹️ Paper Trading (Virtuel)":
        page_paper_trading()
    elif menu == "🏢 Fondamentaux Financiers":
        page_fundamentals(tickers)
    elif menu == "🧮 Options & Dérivés (Pricing)":
        page_options_pricing(tickers)
    elif menu == "🎓 Académie: Stratégie & Risques":
        page_strategy_academy()
    elif menu == "🎓 Académie: Modélisation Avancée":
        page_advanced_academy()
    elif menu == "🎓 Académie: Options & Dérivés":
        page_options_academy()
    elif menu == "📚 Académie: Indicateurs":
        page_indicators()
    elif menu == "🤖 Académie: XGBoost":
        page_xgboost()
    elif menu == "📰 Académie: News & Sentiment":
        page_news()

if __name__ == "__main__":
    main()
