import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    from google import genai
except ImportError:
    genai = None
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
import time
import pickle
import os
import json
import subprocess
import optuna
import scipy.stats as si
from scipy.optimize import brentq
import shap
from tickers_db import MAJOR_STOCKS, PREDEFINED_PORTFOLIOS
import torch
from transformers import pipeline
import glob

# Chargement dynamique des watchlists CSV depuis le dossier WatchList/
def load_custom_watchlists():
    custom_portfolios = {}
    if os.path.exists("WatchList"):
        for filepath in glob.glob("WatchList/*.csv"):
            try:
                df = pd.read_csv(filepath, sep=';', encoding='utf-8', on_bad_lines='skip')
                if len(df.columns) < 2:
                    df = pd.read_csv(filepath, sep=',', encoding='utf-8', on_bad_lines='skip')
                
                filename = os.path.basename(filepath)
                portfolio_name = filename.split('_Watchlist')[0] if '_Watchlist' in filename else filename.replace('.csv', '')
                
                portfolio_items = []
                for _, row in df.iterrows():
                    name = str(row.get('Name', 'Unknown')).strip()
                    symbol = str(row.get('Symbol', '')).strip()
                    if not symbol or symbol.lower() == 'nan':
                        continue
                        
                    if symbol.endswith('.O') or symbol.endswith('.K') or symbol.endswith('.N') or symbol.endswith('.OQ'):
                        clean_symbol = symbol.rsplit('.', 1)[0]
                    else:
                        clean_symbol = symbol
                        
                    display_name = f"{name} ({clean_symbol})"
                    MAJOR_STOCKS[display_name] = clean_symbol
                    portfolio_items.append(display_name)
                    
                if portfolio_items:
                    custom_portfolios[f"📂 {portfolio_name}"] = portfolio_items
            except Exception:
                pass
    return custom_portfolios

PREDEFINED_PORTFOLIOS.update(load_custom_watchlists())

@st.cache_resource(show_spinner="Chargement du modèle Deep Learning NLP (FinBERT)...")
def load_finbert():
    try:
        # Configuration sur CPU ou GPU selon la disponibilité
        device = 0 if torch.cuda.is_available() else -1
        return pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_macro_data(period, interval):
    """Télécharge l'historique des indicateurs macroéconomiques"""
    try:
        spy_raw = yf.download("SPY", period=period, interval=interval, progress=False)
        vix_raw = yf.download("^VIX", period=period, interval=interval, progress=False)
        tnx_raw = yf.download("^TNX", period=period, interval=interval, progress=False)
        dxy_raw = yf.download("DX-Y.NYB", period=period, interval=interval, progress=False)
        
        spy = spy_raw['Close'].iloc[:, 0] if isinstance(spy_raw.columns, pd.MultiIndex) else spy_raw['Close']
        vix = vix_raw['Close'].iloc[:, 0] if isinstance(vix_raw.columns, pd.MultiIndex) else vix_raw['Close']
        tnx = tnx_raw['Close'].iloc[:, 0] if isinstance(tnx_raw.columns, pd.MultiIndex) else tnx_raw['Close']
        dxy = dxy_raw['Close'].iloc[:, 0] if isinstance(dxy_raw.columns, pd.MultiIndex) else dxy_raw['Close']
        
        macro = pd.DataFrame(index=spy.index)
        macro['SPY_Return'] = spy.pct_change()
        macro['VIX'] = vix
        macro['TNX'] = tnx
        macro['DXY'] = dxy
        return macro.ffill()
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_news_sentiment(ticker):
    """Analyse le sentiment des dernières actualités via Deep Learning (FinBERT)"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return 0, []
            
        finbert = load_finbert()
        if finbert is None:
            return 0, [] # Fallback si le modèle crash
            
        polarities = []
        articles = []
        for n in news[:5]:
            content = n.get('content', {})
            title = content.get('title', '')
            if title:
                result = finbert(title)[0]
                label = result['label']
                score = result['score']
                
                # Traduire le label FinBERT en polarité
                if label == 'positive':
                    pol = score
                elif label == 'negative':
                    pol = -score
                else:
                    pol = 0.0
                    
                polarities.append(pol)
                articles.append({'title': title, 'link': content.get('clickThroughUrl', ''), 'polarity': pol})
                
        avg_polarity = np.mean(polarities) if polarities else 0
        return avg_polarity, articles
    except Exception:
        return 0, []

@st.cache_data(ttl=3600)
def get_implied_volatility(ticker, spot_price=None):
    """Récupère la volatilité implicite moyenne (IV) des options ATM les plus proches."""
    try:
        stock = yf.Ticker(ticker)
        dates = stock.options
        if not dates:
            return 0.20 # Fallback
        
        # Prendre la première échéance (front month)
        chain = stock.option_chain(dates[0])
        
        if spot_price:
            # Filtrer les calls autour de la monnaie (ATM)
            calls = chain.calls
            calls['dist'] = abs(calls['strike'] - spot_price)
            atm_calls = calls.sort_values('dist').head(3)
            iv = atm_calls['impliedVolatility'].mean()
        else:
            iv = chain.calls['impliedVolatility'].median()
            
        if pd.isna(iv) or iv == 0:
            return 0.20
        return float(iv)
    except Exception:
        return 0.20



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
        self.xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, objective='binary:logistic', random_state=42)
        self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
        
        self.model = VotingClassifier(
            estimators=[('xgb', self.xgb_model), ('rf', self.rf_model), ('lgb', self.lgb_model)],
            voting='soft'
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
        
        # --- NOUVEAU: VaR & CVaR ---
        if not strat_returns.empty:
            var_95 = np.percentile(strat_returns.dropna(), 5)
            metrics['VaR (95%)'] = var_95
            metrics['CVaR'] = strat_returns[strat_returns <= var_95].mean()
        else:
            metrics['VaR (95%)'] = 0
            metrics['CVaR'] = 0
            
        # --- NOUVEAU: Probabilité de Ruine (Monte Carlo) ---
        metrics['Probabilité de Ruine'] = self.run_monte_carlo(strat_returns.dropna())
        
        self.advanced_metrics = metrics

    def run_monte_carlo(self, returns, num_simulations=1000, ruin_threshold=0.8):
        """Simule 1000 chemins de rendements aléatoires pour évaluer le risque de perdre 20% du capital"""
        if len(returns) == 0:
            return 0.0
        n_days = len(returns)
        # Tirage au sort avec remise
        simulations = np.random.choice(returns, size=(num_simulations, n_days), replace=True)
        cum_simulations = np.cumprod(1 + simulations, axis=1)
        ruin_cases = np.any(cum_simulations < ruin_threshold, axis=1)
        return float(np.mean(ruin_cases))

    def run_event_driven_backtest(self, test_data, initial_capital=100000.0):
        """
        Moteur de Backtest Événementiel (Event-Driven)
        Simule l'exécution de marché réaliste avec Latence, Slippage et Commissions.
        """
        cash = initial_capital
        shares = 0.0
        
        # Paramètres Institutionnels
        commission_rate = 0.0005 # 0.05% broker fee (ex: Interactive Brokers)
        slippage_rate = 0.0010   # 0.10% market impact / slippage (Bid-Ask spread)
        
        portfolio_values = []
        daily_returns = []
        last_val = initial_capital
        
        has_open = 'Open' in test_data.columns
        
        for i in range(len(test_data)):
            row = test_data.iloc[i]
            
            # 1. Latence : On lit le signal de la VEILLE pour agir AUJOURD'HUI à l'Ouverture
            if i == 0:
                signal = 0
            else:
                signal = test_data['Signal'].iloc[i-1]
            
            current_open = row['Open'] if has_open else row['Close']
            current_close = row['Close']
            
            # 2. Exécution de l'ordre à l'Ouverture (Open)
            if signal == 1 and shares == 0:
                # ACHAT
                exec_price = current_open * (1 + slippage_rate) # On paye plus cher (Slippage)
                investable_cash = cash * (1 - commission_rate)
                shares = investable_cash / exec_price
                cash -= (shares * exec_price) + (investable_cash * commission_rate)
                
            elif signal == 0 and shares > 0:
                # VENTE
                exec_price = current_open * (1 - slippage_rate) # On vend moins cher (Slippage)
                proceeds = shares * exec_price
                commission = proceeds * commission_rate
                cash += proceeds - commission
                shares = 0.0
                
            # 3. Mark-to-Market (Évaluation au prix de clôture du jour)
            current_val = cash + (shares * current_close)
            portfolio_values.append(current_val)
            
            ret = (current_val - last_val) / last_val if last_val > 0 else 0
            daily_returns.append(ret)
            last_val = current_val
            
        test_data['Event_Portfolio_Value'] = portfolio_values
        test_data['Strategy_Return'] = daily_returns
        test_data['Cum_Strategy_Return'] = test_data['Event_Portfolio_Value'] / initial_capital
        
        # Le rendement marché classique (Buy & Hold) pour comparer
        test_data['Cum_Market_Return'] = (1 + test_data['Returns'].fillna(0)).cumprod()
        
        return test_data

    def train(self, data, optimize=False, use_wfa=False, wfa_train_window="5Y", wfa_step="6M", wfa_start_date=None, wfa_end_date=None):
        base_features = ['Returns', 'SMA_20', 'SMA_50', 'SMA_200', 'EMA_9', 'Vol_20', 'RSI', 'MACD', 'Signal_Line', 
                         'BB_Upper', 'BB_Lower', 'BB_Width', 'ATR', 'ADX', 'Volume_Ratio', 'OBV', 'Stoch_K', 'ROC', 
                         'VWAP', 'Lag_1', 'Lag_2', 'Lag_3']
        macro_features = ['SPY_Return', 'VIX', 'TNX', 'DXY']
        
        features = [f for f in base_features + macro_features if f in data.columns]
        X = data[features]
        y = data['Target']
        
        if not use_wfa:
            split = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            
            # 4. Walk-Forward Validation (basique)
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
                self.xgb_model = xgb.XGBClassifier(**best_params, objective='binary:logistic', random_state=42)
                self.model = VotingClassifier(
                    estimators=[('xgb', self.xgb_model), ('rf', self.rf_model), ('lgb', self.lgb_model)],
                    voting='soft'
                )
                self.model.fit(X_train, y_train)
                st.toast(f"Optimisation Optuna (XGBoost) terminée : {best_params}", icon="🧬")
            else:
                self.model.fit(X_train, y_train)
            
            # Récupération de l'importance des variables (Moyenne des 3 modèles)
            importances = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
            self.feature_importances = pd.DataFrame(
                {'Feature': features, 'Importance': importances}
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
                
            # 4. Moteur de Backtest Événementiel (Event-Driven)
            test_data = self.run_event_driven_backtest(test_data)
            
            self.backtest_results = test_data
            self.accuracy = accuracy_score(y_test, self.model.predict(X_test))
            self.calculate_advanced_metrics(test_data)
            self.is_trained = True
            return features
            
        else:
            # --- WALK FORWARD ANALYSIS DYNAMIQUE ---
            st.info(f"🔄 Exécution Walk-Forward Analysis (WFA) : {wfa_start_date.strftime('%Y-%m-%d')} à {wfa_end_date.strftime('%Y-%m-%d')}")
            
            window_map = {"2Y": pd.DateOffset(years=2), "5Y": pd.DateOffset(years=5), "7Y": pd.DateOffset(years=7)}
            step_map = {"1M": pd.DateOffset(months=1), "2M": pd.DateOffset(months=2), "6M": pd.DateOffset(months=6), "1Y": pd.DateOffset(years=1)}
            
            train_window = window_map.get(wfa_train_window, pd.DateOffset(years=5))
            step_size = step_map.get(wfa_step, pd.DateOffset(months=6))
            
            current_date = pd.to_datetime(wfa_start_date)
            if current_date.tzinfo is None and data.index.tzinfo is not None:
                current_date = current_date.tz_localize(data.index.tzinfo)
            end_date = pd.to_datetime(wfa_end_date)
            if end_date.tzinfo is None and data.index.tzinfo is not None:
                end_date = end_date.tz_localize(data.index.tzinfo)
                
            all_test_data = []
            all_importances = []
            y_test_all = []
            y_pred_all = []
            
            # Progress bar for WFA
            progress_bar = st.progress(0)
            total_days = (end_date - current_date).days
            start_days = total_days
            
            while current_date < end_date:
                train_start = current_date - train_window
                test_end = current_date + step_size
                
                mask_train = (data.index >= train_start) & (data.index < current_date)
                mask_test = (data.index >= current_date) & (data.index < test_end)
                
                X_train, y_train = X[mask_train], y[mask_train]
                X_test, y_test_slice = X[mask_test], y[mask_test]
                
                if len(X_train) < 50 or len(X_test) == 0:
                    st.write(f"⏩ *Passage* : Données insuffisantes pour la période du {current_date.strftime('%Y-%m-%d')}")
                    current_date += step_size
                    continue
                    
                st.write(f"⏳ **Entraînement WFA** : Historique {train_start.strftime('%Y-%m-%d')} à {current_date.strftime('%Y-%m-%d')} | **Test (Prédiction)** : {current_date.strftime('%Y-%m-%d')} à {test_end.strftime('%Y-%m-%d')}...")
                
                self.xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, objective='binary:logistic', random_state=42)
                self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                self.lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
                
                self.model = VotingClassifier(
                    estimators=[('xgb', self.xgb_model), ('rf', self.rf_model), ('lgb', self.lgb_model)],
                    voting='soft'
                )
                self.model.fit(X_train, y_train)
                
                prob = self.model.predict_proba(X_test)[:, 1]
                pred = self.model.predict(X_test)
                
                step_acc = accuracy_score(y_test_slice, pred)
                st.write(f"✅ **Pas terminé** -> Précision sur ce sous-test : **{step_acc:.2%}**")
                
                test_slice = data[mask_test].copy()
                test_slice['Prob'] = prob
                test_slice['Signal'] = np.where(test_slice['Prob'] > 0.55, 1, 0)
                if 'VIX' in test_slice.columns:
                    test_slice.loc[test_slice['VIX'] > 30, 'Signal'] = 0
                    
                all_test_data.append(test_slice)
                y_test_all.extend(y_test_slice.values)
                y_pred_all.extend(pred)
                
                imp = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
                all_importances.append(imp)
                
                current_date += step_size
                
                if start_days > 0:
                    prog = min(1.0, 1.0 - ((end_date - current_date).days / start_days))
                    progress_bar.progress(prog)
            
            progress_bar.empty()
                
            if len(all_test_data) == 0:
                st.error("WFA : Pas assez de données pour générer un backtest. Élargissez la période téléchargée ou réduisez la fenêtre d'entraînement.")
                return features
                
            final_test_data = pd.concat(all_test_data)
            
            avg_importances = np.mean(all_importances, axis=0)
            self.feature_importances = pd.DataFrame(
                {'Feature': features, 'Importance': avg_importances}
            ).sort_values(by='Importance', ascending=True)
            
            final_test_data = self.run_event_driven_backtest(final_test_data)
            self.backtest_results = final_test_data
            self.accuracy = accuracy_score(y_test_all, y_pred_all)
            self.calculate_advanced_metrics(final_test_data)
            self.is_trained = True
            
            # Le dernier modèle de la boucle WFA est sauvegardé comme demandé
            import joblib
            joblib.dump(self.model, "ia_model.joblib")
                
            return features

    def predict(self, last_row, features):
        if not self.is_trained:
            return None
            
        # Utilisation de ia_model.joblib si disponible (dernier pas WFA)
        import os
        import joblib
        if os.path.exists("ia_model.joblib"):
            model_to_use = joblib.load("ia_model.joblib")
        else:
            model_to_use = self.model
            
        X_input = last_row[features].values.reshape(1, -1)
        prob = model_to_use.predict_proba(X_input)[0][1]
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

def get_company_name_from_yahoo(yahoo_ticker):
    from tickers_db import MAJOR_STOCKS
    for name, google_ticker in MAJOR_STOCKS.items():
        if google_ticker != "CUSTOM":
            if convert_google_to_yahoo_ticker(google_ticker) == yahoo_ticker:
                clean_name = name.split(" (")[0]
                return f"{clean_name} ({yahoo_ticker})"
    return yahoo_ticker

def get_option_multiplier_and_legislation(ticker):
    """
    Retourne le multiplicateur et la législation applicable selon le ticker.
    US: 100 actions/contrat
    EU: 10 actions/contrat (règle simplifiée basée sur le suffixe)
    """
    eu_suffixes = ['.PA', '.L', '.TO', '.F', '.MI', '.MC', '.AS', '.BR', '.LS', '.SW', '.DE']
    if any(ticker.endswith(suffix) for suffix in eu_suffixes):
        return 10, "Législation Européenne (10 actions/contrat - Exercice à l'échéance uniquement)"
    return 100, "Législation Américaine (100 actions/contrat - Exercice et revente libres)"

# --- MODES D'AFFICHAGE ---

def run_single_mode(ticker, period, interval, initial_capital, optimize_model, use_wfa=False, wfa_train_window="5Y", wfa_step="6M", wfa_start_date=None, wfa_end_date=None):
    st.subheader(f"Analyse Individuelle : {get_company_name_from_yahoo(ticker)}")
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
        # S'assurer qu'il n'y a pas de colonnes dupliquées (arrive si yfinance télécharge plusieurs tickers)
        df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()].copy()
            
        macro_df = get_macro_data(period, interval)

    df = add_features(df_raw)
    if macro_df is not None:
        df = df.join(macro_df, how='left').ffill().dropna()
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        if st.button("🧠 Entraîner l'IA sur cette action", use_container_width=True):
            with st.status("Entraînement en cours...") as status:
                features = trader.train(df, optimize=optimize_model, use_wfa=use_wfa, wfa_train_window=wfa_train_window, wfa_step=wfa_step, wfa_start_date=wfa_start_date, wfa_end_date=wfa_end_date)
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
        base_features = ['Returns', 'SMA_20', 'SMA_50', 'SMA_200', 'EMA_9', 'Vol_20', 'RSI', 'MACD', 'Signal_Line', 
                         'BB_Upper', 'BB_Lower', 'BB_Width', 'ATR', 'ADX', 'Volume_Ratio', 'OBV', 'Stoch_K', 'ROC', 
                         'VWAP', 'Lag_1', 'Lag_2', 'Lag_3']
        macro_features = ['SPY_Return', 'VIX', 'TNX', 'DXY']
        features = [f for f in base_features + macro_features if f in df.columns]
        prob = trader.predict(last_row, features)
        st.progress(float(prob))
        
        st.divider()
        # --- NLP Sentiment Logic ---
        st.subheader("📰 Filtre de Sentiment Actuel (FinBERT)")
        try:
            avg_sentiment, articles = get_news_sentiment(ticker)
            if articles:
                sentiment_text = "Neutre"
                color = "gray"
                if avg_sentiment > 0.15:
                    sentiment_text = "Positif 🟢"
                    color = "green"
                elif avg_sentiment < -0.15:
                    sentiment_text = "Négatif 🔴"
                    color = "red"
                    
                st.markdown(f"Score de Sentiment FinBERT : **<span style='color:{color}'>{avg_sentiment:+.2f} ({sentiment_text})</span>**", unsafe_allow_html=True)
                
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
                    # Simulation Event-Driven (Slippage + Commissions)
                    exec_price = current_price * 1.0010  # +0.10% Slippage
                    cost = position_size * exec_price
                    commission = cost * 0.0005           # +0.05% Frais
                    total_cost = cost + commission
                    
                    if pf['cash'] >= total_cost:
                        pf['cash'] -= total_cost
                        # Update positions
                        if ticker in pf['positions']:
                            old_qty = pf['positions'][ticker]['qty']
                            old_price = pf['positions'][ticker]['avg_price']
                            new_qty = old_qty + position_size
                            new_price = ((old_qty * old_price) + total_cost) / new_qty
                            pf['positions'][ticker] = {'qty': new_qty, 'avg_price': new_price}
                        else:
                            pf['positions'][ticker] = {'qty': position_size, 'avg_price': exec_price}
                        
                        pf['history'].append({
                            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'action': 'BUY',
                            'ticker': ticker,
                            'qty': position_size,
                            'price': exec_price,
                            'total': total_cost
                        })
                        save_portfolio(pf)
                        st.success(f"Ordre exécuté virtuellement ! {position_size} actions {ticker} achetées (Slippage et Frais inclus).")
                    else:
                        st.error("Fonds insuffisants dans le Paper Trading.")
            elif prob < 0.45:
                if st.button("🔴 Liquider Position (Paper Trading)"):
                    pf = load_portfolio()
                    if ticker in pf['positions'] and pf['positions'][ticker]['qty'] > 0:
                        qty = pf['positions'][ticker]['qty']
                        
                        # Simulation Event-Driven (Slippage + Commissions)
                        exec_price = current_price * 0.9990 # -0.10% Slippage
                        gross_revenue = qty * exec_price
                        commission = gross_revenue * 0.0005 # -0.05% Frais
                        net_revenue = gross_revenue - commission
                        
                        pf['cash'] += net_revenue
                        pf['history'].append({
                            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'action': 'SELL',
                            'ticker': ticker,
                            'qty': qty,
                            'price': exec_price,
                            'total': net_revenue,
                            'pnl': net_revenue - (qty * pf['positions'][ticker]['avg_price'])
                        })
                        del pf['positions'][ticker]
                        save_portfolio(pf)
                        st.success(f"Position liquidée virtuellement ! {qty} actions {ticker} vendues (Slippage et Frais déduits).")
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
            multiplier, legislation = get_option_multiplier_and_legislation(ticker)
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
            - **Coût total du contrat (x{multiplier})** : {price_opt*multiplier:.2f} $
            - **Norme** : {legislation}
            """)
            c_o3.markdown(f"""
            - **Levier Estimé** : {(current_price_opt * delta_opt) / price_opt:.1f}x
            """)
            
            # --- EXÉCUTION CALL OPTIONS ---
            if price_opt > 0:
                cost_per_contract = price_opt * multiplier
                qty_to_buy = int(max_loss / cost_per_contract) if cost_per_contract > 0 else 0
                qty_to_buy = max(1, qty_to_buy) # Au moins 1 contrat
                total_premium = qty_to_buy * cost_per_contract
                
                if st.button(f"🚀 Acheter {qty_to_buy} Contrats CALL (Paper Trading Options)"):
                    pf_opt = load_options_portfolio()
                    if pf_opt['cash'] >= total_premium:
                        pf_opt['cash'] -= total_premium
                        contract_id = f"{ticker}_CALL_{strike_opt}_{t_opt_days}d_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        pf_opt['positions'][contract_id] = {
                            'ticker': ticker,
                            'type': 'call',
                            'strike': strike_opt,
                            'days_to_expiry': t_opt_days,
                            'premium': price_opt,
                            'qty': qty_to_buy,
                            'multiplier': multiplier,
                            'underlying_price_at_buy': current_price_opt,
                            'buy_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        pf_opt['history'].append({
                            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'action': 'BUY OPTION',
                            'ticker': ticker,
                            'contract': f"CALL Strike {strike_opt}",
                            'qty': qty_to_buy,
                            'premium_paid': price_opt,
                            'total_cost': total_premium
                        })
                        save_options_portfolio(pf_opt)
                        st.success(f"{qty_to_buy} contrats CALL achetés virtuellement ! Retrouvez-les dans 'Paper Trading (Options)'.")
                    else:
                        st.error("Fonds virtuels (Options) insuffisants.")
        elif prob < 0.45:
            multiplier, legislation = get_option_multiplier_and_legislation(ticker)
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
            - **Coût total du contrat (x{multiplier})** : {price_opt*multiplier:.2f} $
            - **Norme** : {legislation}
            """)
            c_o3.markdown(f"""
            - **Levier Estimé** : {(current_price_opt * abs(delta_opt)) / price_opt:.1f}x
            """)
            
            # --- EXÉCUTION PUT OPTIONS ---
            if price_opt > 0:
                cost_per_contract = price_opt * multiplier
                qty_to_buy = int(max_loss / cost_per_contract) if cost_per_contract > 0 else 0
                qty_to_buy = max(1, qty_to_buy) # Au moins 1 contrat
                total_premium = qty_to_buy * cost_per_contract
                
                if st.button(f"🔴 Acheter {qty_to_buy} Contrats PUT (Paper Trading Options)"):
                    pf_opt = load_options_portfolio()
                    if pf_opt['cash'] >= total_premium:
                        pf_opt['cash'] -= total_premium
                        contract_id = f"{ticker}_PUT_{strike_opt}_{t_opt_days}d_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        pf_opt['positions'][contract_id] = {
                            'ticker': ticker,
                            'type': 'put',
                            'strike': strike_opt,
                            'days_to_expiry': t_opt_days,
                            'premium': price_opt,
                            'qty': qty_to_buy,
                            'multiplier': multiplier,
                            'underlying_price_at_buy': current_price_opt,
                            'buy_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        pf_opt['history'].append({
                            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'action': 'BUY OPTION',
                            'ticker': ticker,
                            'contract': f"PUT Strike {strike_opt}",
                            'qty': qty_to_buy,
                            'premium_paid': price_opt,
                            'total_cost': total_premium
                        })
                        save_options_portfolio(pf_opt)
                        st.success(f"{qty_to_buy} contrats PUT achetés virtuellement ! Retrouvez-les dans 'Paper Trading (Options)'.")
                    else:
                        st.error("Fonds virtuels (Options) insuffisants.")
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
            n1, n2, n3 = st.columns(3)
            var_pct = adv.get('VaR (95%)', 0) * 100
            cvar_pct = adv.get('CVaR', 0) * 100
            prob_ruin = adv.get('Probabilité de Ruine', 0) * 100
            n1.metric("VaR Historique (95%)", f"{var_pct:.2f}%", help="Perte maximale journalière dans 95% des cas.")
            n2.metric("CVaR (Expected Shortfall)", f"{cvar_pct:.2f}%", help="Perte moyenne lorsque la VaR est dépassée (Risque extrême).")
            n3.metric("Probabilité de Ruine (MC)", f"{prob_ruin:.2f}%", help="Probabilité de perdre 20% du capital (Bootstrap Monte Carlo).")
                
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
                # Dans un VotingClassifier, les modèles entraînés sont dans estimators_
                fitted_xgb = trader.model.estimators_[0]
                explainer = shap.TreeExplainer(fitted_xgb)
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

def get_black_litterman_weights(returns_df, market_weights, views_dict, tau=0.05, risk_aversion=2.5):
    """
    Calcule les poids optimaux via le modèle de Black-Litterman.
    returns_df: DataFrame des rendements historiques de chaque action
    market_weights: Poids d'équilibre (ex: Risk Parity ou Market Cap)
    views_dict: Dictionnaire {ticker: expected_excess_return} issu de XGBoost
    """
    try:
        tickers = list(returns_df.columns)
        N = len(tickers)
        
        # 1. Matrice de Covariance et Rendements d'équilibre (Prior)
        Sigma = returns_df.cov().values * 252
        W_mkt = np.array([market_weights[t] for t in tickers])
        Pi = risk_aversion * Sigma.dot(W_mkt)
        
        # 2. Matrices des Vues (P, Q, Omega)
        P = np.eye(N)
        Q = np.array([views_dict.get(t, 0.0) for t in tickers])
        
        # Si aucune vue n'est forte, on retourne les poids d'équilibre
        if np.all(Q == 0.0):
            return market_weights
            
        # Incertitude des vues (Méthode de He-Litterman proportionnelle à la variance)
        Omega = np.diag(np.diag(tau * P.dot(Sigma).dot(P.T)))
        
        # 3. Posterior (Rendements ajustés)
        tau_Sigma_inv = np.linalg.inv(tau * Sigma)
        Omega_inv = np.linalg.inv(Omega)
        
        left_term = np.linalg.inv(tau_Sigma_inv + P.T.dot(Omega_inv).dot(P))
        right_term = tau_Sigma_inv.dot(Pi) + P.T.dot(Omega_inv).dot(Q)
        
        posterior_expected_returns = left_term.dot(right_term)
        
        # 4. Nouveaux poids optimisés
        W_bl = np.linalg.inv(risk_aversion * Sigma).dot(posterior_expected_returns)
        
        # Pas de vente à découvert pour notre portefeuille long-only, on ramène à 0
        W_bl = np.clip(W_bl, 0.0, None)
        
        # Normalisation pour que la somme = 1 (100% d'allocation)
        if np.sum(W_bl) > 0:
            W_bl = W_bl / np.sum(W_bl)
        else:
            W_bl = W_mkt # Fallback si problème d'inversion
            
        return {tickers[i]: W_bl[i] for i in range(N)}
    except Exception as e:
        return market_weights # Fallback en cas d'erreur de matrice singulière

def run_portfolio_mode(tickers, period, interval, initial_capital, optimize_model, use_wfa=False, wfa_train_window="5Y", wfa_step="6M", wfa_start_date=None, wfa_end_date=None):
    st.subheader("🌐 Mode Portefeuille Institutionnel (Black-Litterman)")
    st.markdown("""
    Ce mode combine **deux piliers de la finance quantitative** :
    1. **Le Prior (L'Équilibre)** : L'allocation initiale est gérée par la Parité des Risques (Inverse Volatilité) pour lisser le risque statistique.
    2. **Les Vues (L'Alpha XGBoost)** : Le modèle de *Black-Litterman* (Goldman Sachs) prend l'équilibre initial et le "tord" mathématiquement en fonction de la conviction de notre IA sur chaque action.
    """)
    
    if st.button("🧠 Entraîner et Optimiser le Portefeuille", use_container_width=True):
        with st.status("Analyse Macro & Entraînement des modèles...") as status:
            data_dict = {}
            returns_dict = {}
            volatility_dict = {}
            
            for t in tickers:
                st.write(f"Récupération pour {t}...")
                df_raw = yf.download(t, period=period, interval=interval, progress=False)
                if not df_raw.empty:
                    if isinstance(df_raw.columns, pd.MultiIndex):
                        df_raw.columns = df_raw.columns.droplevel(1)
                    # S'assurer qu'il n'y a pas de colonnes dupliquées (évite les DataFrame dans df_raw['Close'])
                    df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()].copy()
                    
                    data_dict[t] = df_raw
                    ret = df_raw['Close'].pct_change().dropna()
                    returns_dict[t] = ret
                    vol_hist = ret.std()
                    
                    if pd.isna(vol_hist) or float(vol_hist) <= 0:
                        volatility_dict[t] = 0.01
                    else:
                        volatility_dict[t] = float(vol_hist)

            # --- 1. Poids d'Équilibre (Risk Parity) ---
            sum_inv_vol = sum(1/v for v in volatility_dict.values())
            market_weights = {t: (1/v)/sum_inv_vol for t, v in volatility_dict.items()}
            
            # --- 2. Entraînement IA & Génération des Vues ---
            views_dict = {}
            for t, df_raw in data_dict.items():
                st.write(f"Entraînement de l'IA pour {t}...")
                macro_df = get_macro_data(period, interval)
                df = add_features(df_raw)
                if macro_df is not None:
                    df = df.join(macro_df, how='left').ffill().dropna()
                trader = MLTrader()
                features = trader.train(df, optimize=optimize_model, use_wfa=use_wfa, wfa_train_window=wfa_train_window, wfa_step=wfa_step, wfa_start_date=wfa_start_date, wfa_end_date=wfa_end_date)
                st.session_state[f"port_trader_{t}"] = trader
                
                # Prédiction actuelle pour le Black-Litterman (La Vue)
                last_row = df.iloc[-1:]
                prob = trader.predict(last_row, features)
                # Transformation de la probabilité en rendement excédentaire attendu
                # Ex: prob=0.60 => (0.60 - 0.50) * 0.50 = +0.05 (+5% d'excès)
                expected_excess = (prob - 0.50) * 0.50 if prob is not None else 0.0
                views_dict[t] = expected_excess
                
                if "probs_dict" not in st.session_state:
                    st.session_state["probs_dict"] = {}
                st.session_state["probs_dict"][t] = prob
                
            # --- 3. Optimisation Black-Litterman ---
            st.write("🧮 Optimisation de la Matrice de Covariance (Black-Litterman)...")
            returns_df = pd.DataFrame(returns_dict).dropna()
            
            # Application du modèle
            bl_weights = get_black_litterman_weights(returns_df, market_weights, views_dict)
            
            st.session_state["bl_weights"] = bl_weights
            st.session_state["market_weights"] = market_weights
            
            for t, w in bl_weights.items():
                st.session_state[f"port_capital_{t}"] = initial_capital * w
                
            status.update(label="✅ Portefeuille Institutionnel généré et optimisé !", state="complete")
            
    is_ready = all(f"port_trader_{t}" in st.session_state for t in tickers)
    
    if is_ready:
        if "bl_weights" in st.session_state:
            st.subheader("💼 Allocation Optimisée du Portefeuille (Signal d'Investissement)")
            bl_weights = st.session_state["bl_weights"]
            market_weights = st.session_state["market_weights"]
            probs_dict = st.session_state.get("probs_dict", {})
            
            cols = st.columns(min(len(tickers), 4))
            idx = 0
            for t, w in bl_weights.items():
                if t in tickers:
                    m_w = market_weights.get(t, 0)
                    diff = w - m_w
                    prob = probs_dict.get(t)
                    if prob is None: prob = 0.5
                    amount = initial_capital * w
                    
                    with cols[idx % 4]:
                        st.metric(label=get_company_name_from_yahoo(t), value=f"{w*100:.1f}%", delta=f"{diff*100:+.1f}% (Ajustement IA)")
                        st.write(f"Capital: **{amount:,.2f} $**")
                        st.write(f"Probabilité IA: **{prob*100:.1f}%**")
                        
                        if amount > 10: # Ne proposer l'investissement que si le montant est significatif
                            if st.button(f"🛒 Investir {t}", key=f"inv_{t}"):
                                pf = load_portfolio()
                                if pf['cash'] < amount:
                                    st.error("Liquidités insuffisantes !")
                                else:
                                    try:
                                        current_price = yf.download(t, period="1d", progress=False)['Close'].iloc[-1]
                                        if isinstance(current_price, pd.Series):
                                            current_price = current_price.iloc[0]
                                        exec_price = float(current_price) * 1.001
                                        qty = amount / exec_price
                                        commission = amount * 0.0005
                                        pf['cash'] -= (amount + commission)
                                        if t in pf['positions']:
                                            old_qty = pf['positions'][t]['qty']
                                            old_avg = pf['positions'][t]['avg_price']
                                            pf['positions'][t]['avg_price'] = ((old_qty * old_avg) + (qty * exec_price)) / (old_qty + qty)
                                            pf['positions'][t]['qty'] += qty
                                        else:
                                            pf['positions'][t] = {'qty': qty, 'avg_price': exec_price}
                                        pf['history'].append({
                                            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            'action': 'BUY (Action indiv. BL)',
                                            'ticker': t, 'qty': qty, 'price': exec_price, 'total': amount + commission, 'pnl': 0.0
                                        })
                                        save_portfolio(pf)
                                        st.toast(f"✅ {t} ajouté au portefeuille avec succès !", icon="✅")
                                    except Exception as e:
                                        st.error(f"Erreur d'exécution: {e}")
                    idx += 1
            st.divider()

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
        **Théorie Mathématique :** On prend la moyenne des prix (SMA 20), et on ajoute/soustrait 2 écarts-types statistiques ($\\sigma$). En statistiques, cela signifie que **95% des prix futurs seront contenus dans ces bandes**.
        
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
    
    st.header("5. L'Explicabilité (Valeurs SHAP)")
    st.success("Comment faire confiance à une boîte noire ? En utilisant SHAP (SHapley Additive exPlanations).")
    st.markdown("""
    Dans la finance institutionnelle, une IA qui dit "Achetez" sans expliquer pourquoi est inutile. Les régulateurs et les gérants de fonds exigent de la transparence. 
    
    C'est pourquoi nous avons intégré **SHAP**, une théorie issue de la théorie des jeux coopératifs (Prix Nobel d'économie). SHAP calcule la contribution *exacte* et mathématique de chaque indicateur technique sur la décision finale du jour.
    
    **Dans l'application (Onglet 3 du Terminal) :**
    Le graphique Waterfall (Cascade) vous montre comment l'IA a réfléchi :
    - La base part de 0 (Neutre).
    - Les barres vertes poussent la prédiction vers l'achat (Ex: Le VWAP monte, ça rajoute +0.15 de confiance).
    - Les barres rouges freinent ou inversent la décision (Ex: Le RSI est en surachat, ça enlève -0.10 de confiance).
    - Le total donne la probabilité finale d'achat ou de vente.
    
    *   **Problème initial :** L'IA dit "Achat Fort avec 75% de confiance", mais on ne sait pas pourquoi. Le graphe "Importance Globale" montre l'importance sur l'ensemble de l'historique, pas celle du jour.
    *   **Solution :** Intégrer la librairie shap (SHapley Additive exPlanations), le standard de l'industrie.
    *   **Implémentation :** Pour le "Plan de Trading du Jour", nous générons un graphique SHAP (Waterfall plot) qui explique mathématiquement la prédiction du jour. Par exemple : "Signal d'Achat généré par : VWAP (+0.15) et MACD (+0.10), mais ralenti par RSI (-0.05)". Cela donne une fiabilité et une confiance énormes au trader.
    """)

def page_news():
    st.title("📰 Académie Quantitative : NLP Profond & Sentiment de Marché")
    st.markdown("L'analyse technique mathématique ne peut pas tout prévoir. Parfois, un simple tweet ou un scandale médiatique fait s'effondrer une action en 5 minutes. C'est là qu'intervient le NLP Profond.")
    
    st.header("Qu'est-ce que le NLP Profond ?")
    st.markdown("""
    Le **Natural Language Processing (NLP)** permet à un ordinateur de "lire" le texte. Historiquement, on utilisait des dictionnaires de mots (comme la librairie *TextBlob*).
    
    Aujourd'hui, l'élite financière utilise des réseaux de neurones complexes : les **Transformers**.
    Dans notre application, nous utilisons **FinBERT**, un modèle d'Intelligence Artificielle de pointe (Hugging Face) entraîné *spécifiquement* sur le jargon financier de Wall Street, des rapports de la SEC et des actualités économiques.
    """)
    
    st.header("La Supériorité de FinBERT")
    st.info("La Polarité est un score allant de **-1.0 (Désespoir absolu)** à **+1.0 (Euphorie totale)**.")
    st.markdown("""
    Contrairement aux anciens modèles qui ne comprenaient pas le contexte (le mot "dette" était toujours négatif, même dans "réduction de la dette"), FinBERT utilise un mécanisme d'**Attention** pour lire la phrase dans sa globalité.
    
    Lorsque l'application télécharge les articles du jour sur Yahoo Finance, le réseau de neurones les analyse et retourne une probabilité mathématique (Positif, Négatif, Neutre), que nous convertissons en un score global pour vous donner la "Météo Psychologique" de l'action.
    """)
    
    st.header("L'Approche Hybride du Trader Pro")
    st.warning("**XGBoost (Maths/Technique)** + **FinBERT (Fondamental/Émotions)** = **Edge (Avantage Institutionnel)**")
    st.markdown("""
    1. Si XGBoost dit ACHAT et que FinBERT dit POSITIF -> **Achat Fort (Haute conviction)**.
    2. Si XGBoost dit ACHAT mais que FinBERT dit NÉGATIF -> **Danger !** Le modèle mathématique ne sait peut-être pas que l'entreprise fait l'objet d'une enquête pour fraude. Il vaut mieux ignorer le signal quantitatif.
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
    st.markdown("Bienvenue dans le centre de formation avancé pour les Quants. Les mathématiques pures ne suffisent pas, il faut comprendre les dynamiques comportementales des marchés.")
    
    st.header("1. L'Alpha (α) et le Beta (β)")
    st.markdown("""
    Dans le monde institutionnel, la performance brute ne veut rien dire. 
    - **Le Beta (β)** : C'est la performance due au marché global. Si vous achetez le S&P 500, vous avez un Beta de 1. Vous gagnez quand le marché gagne. C'est le "rendement paresseux".
    - **L'Alpha (α)** : C'est la **véritable valeur ajoutée** de votre stratégie. C'est l'argent gagné indépendamment des mouvements du marché. 
    
    > [!TIP]
    > **L'Observation des Pros :** Lors du marché haussier post-Covid (2020-2021), beaucoup de traders amateurs se prenaient pour des génies car ils gagnaient 50%. Mais le marché gagnait 60%. Leur Alpha était donc négatif (-10%). Le vrai test d'un algorithme (et son Alpha réel) se mesure lors des marchés baissiers ou latéraux.
    """)
    st.info("💡 **Objectif du Trading Algorithmique** : Extraire de l'Alpha régulier, peu importe que le marché monte ou baisse.")

    st.header("2. Les Mathématiques de la Ruine (Drawdown)")
    st.markdown("""
    Le **Maximum Drawdown (MDD)** est votre pire ennemi. C'est la perte maximale historique de votre portefeuille.
    
    L'asymétrie des pertes est cruelle. La gestion du risque est la seule chose qui compte :
    *   Si vous perdez **10%**, il vous faut **11%** de gain pour revenir à zéro.
    *   Si vous perdez **20%**, il vous faut **25%** de gain pour revenir à zéro.
    *   Si vous perdez **50%**, il vous faut **100%** de gain pour revenir à zéro.
    *   Si vous perdez **90%**, il vous faut **900%** de gain !
    
    > [!WARNING]
    > **Cas d'École : Long-Term Capital Management (LTCM) en 1998.**
    > Ce Hedge Fund était dirigé par les Prix Nobel d'Économie Myron Scholes et Robert Merton. Leurs algorithmes étaient parfaits. Mais ils utilisaient un levier de 25:1. Lors du défaut de la dette russe (un événement "Cygne Noir" imprévisible par leurs modèles), leur portefeuille a subi un Drawdown si rapide que les mathématiques n'ont pas pu les sauver. Le fonds a fait faillite, menaçant l'économie mondiale.
    """)

    st.header("3. Le Money Management Dynamique (Critère de Kelly)")
    st.markdown("""
    Les traders débutants utilisent la **Règle des 2%** fixe. Les fonds quantitatifs (comme Renaissance Technologies) utilisent la formule probabiliste du **Critère de Kelly**.
    
    La formule de Kelly calcule le pourcentage mathématiquement optimal de votre capital à risquer pour maximiser la croissance composée, en évitant la ruine.
    
    $$ Kelly\\,\\% = W - \\frac{1 - W}{R} $$
    *(Où **W** = Win Rate, et **R** = Ratio Gain/Perte moyen)*
    
    > [!NOTE]
    > **L'Exemple du Coin Toss (Pile ou Face biaisé) de Claude Shannon :**
    > Imaginez qu'on vous propose un jeu de pile ou face truqué en votre faveur : vous avez 60% de chances de gagner (W=0.6) et vous gagnez 1$ pour chaque 1$ parié (R=1). 
    > Si vous pariez tout votre capital à chaque lancer, malgré votre avantage mathématique, vous finirez inévitablement ruiné à la première série de pertes.
    > Selon la formule, la mise optimale est de `0.6 - (1 - 0.6) / 1 = 20%`. Parier plus de 20% détruit votre croissance à long terme à cause de la volatilité des pertes.
    
    **Dans l'application :**
    Le plein Kelly étant mathématiquement agressif, le moteur algorithmique utilise automatiquement un **Half-Kelly** (Kelly/2) plafonné à **5%** maximum par position. L'IA gère donc la taille de ses positions intelligemment en fonction de sa propre précision historique.
    """)

    st.header("4. Le Ratio Risque/Récompense (Risk/Reward)")
    st.markdown("""
    Une IA n'a pas besoin d'avoir raison tout le temps pour générer des millions. Même si votre XGBoost a un "Win Rate" de seulement 40% (elle a tort 60% du temps), elle peut dominer le marché.
    
    Comment ? Avec un **Risk/Reward asymétrique de 1:3**.
    *   Vous risquez 100$ pour gagner 300$.
    *   Sur 10 trades : 6 pertes de 100$ (-600$), 4 gains de 300$ (+1200$).
    *   Bénéfice total : **+600$** avec un modèle qui se trompe la majorité du temps !
    
    > [!TIP]
    > L'observation du marché montre que les "Trend Followers" (suiveurs de tendance) ont souvent des Win Rates ridicules (35%), mais capturent des tendances massives (Risk/Reward de 1:5 ou 1:10), ce qui les rend massivement rentables. Notre système utilise la volatilité (l'ATR) pour placer ses stops intelligemment.
    """)

    st.header("5. Le Filtre de Régime (SMA 200)")
    st.markdown("""
    Même la meilleure IA du monde produira de faux signaux d'achat en plein Krach boursier ("Rattraper un couteau qui tombe"). C'est pourquoi nous avons intégré un **Filtre de Régime**.
    
    **La SMA 200 (Moyenne Mobile à 200 jours) :**
    Elle représente la tendance institutionnelle à long terme. C'est la ligne de séparation entre le Bull Market (Hausse) et le Bear Market (Baisse).
    - **Prix > SMA 200** : Les acheteurs ont le contrôle structurel. L'IA a le "feu vert" complet.
    - **Prix < SMA 200** : Marché baissier. L'IA dégradera tout signal d'achat en alerte **Risquée (Contre-Tendance)**. 
    
    > [!IMPORTANT]
    > **Le Krach de 2008 :** Les algorithmes qui n'avaient pas de filtre de régime long-terme ont passé l'année entière à détecter des "creux" (oversold RSI) et à acheter la baisse, se faisant anéantir. La SMA 200 a protégé le capital en interdisant les achats lourds durant toute l'année.
    """)

    st.header("6. Modèle de Black-Litterman (Optimisation de Portefeuille)")
    st.markdown("""
    Créé par Fischer Black et Robert Litterman chez Goldman Sachs en 1990, c'est **LE** modèle institutionnel par excellence.
    
    L'approche classique (Markowitz) crée des portefeuilles absurdes dès que les statistiques d'une action changent légèrement. Black-Litterman résout cela avec l'inférence bayésienne.
    
    1. **Le Prior (L'Équilibre) :** Le modèle part du principe que le marché actuel est "juste" (Risk Parity).
    2. **Les Vues (L'Alpha) :** Le gérant exprime ses convictions. Dans notre système, **XGBoost fournit les convictions**.
    3. **Le Posterior :** Le modèle ajuste doucement le portefeuille pour refléter ces convictions, sans détruire la diversification.
    
    > **Exemple de la méthode :**
    > Si l'équilibre suggère 50% Actions, 50% Or. L'IA XGBoost détecte une probabilité de hausse extrême sur l'Or. Le modèle Black-Litterman ne va pas faire 100% Or (ce serait suicidaire). Il va ajuster à 40% Actions et 60% Or, en quantifiant mathématiquement le niveau de confiance de l'IA par rapport au niveau d'incertitude du marché.
    """)

    st.header("7. Value at Risk (VaR) & Expected Shortfall (CVaR)")
    st.markdown("""
    Le Drawdown regarde le passé. Les régulateurs exigent que les Hedge Funds mesurent le risque *futur*.
    
    *   **VaR (95%) :** *"Dans 95% des scénarios normaux, quelle est la pire perte que je subirai demain ?"*
    *   **CVaR (Expected Shortfall) :** *"Si le marché explose (les 5% restants), quelle sera l'ampleur du désastre ?"*
    
    > [!CAUTION]
    > **Le Problème des Fat Tails (Queues Épaisses) :**
    > Selon une distribution Normale (Gaussienne), le Krach de 1987 (-22% en un jour) est un événement qui ne devrait se produire qu'une fois tous les 14 milliards d'années ! Or, les marchés font des "Fat Tails" : les catastrophes arrivent très souvent. La CVaR permet précisément de mesurer le coût de ces cygnes noirs, là où la VaR simple devient aveugle.
    """)

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

def get_options_portfolio_path():
    data_dir = "/app/data"
    if os.path.exists(data_dir):
        return os.path.join(data_dir, "options_portfolio.json")
    return "options_portfolio.json"

def load_options_portfolio():
    path = get_options_portfolio_path()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {"cash": 100000.0, "positions": {}, "history": []}

def save_options_portfolio(pf):
    path = get_options_portfolio_path()
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
            
            with st.expander(f"📦 {get_company_name_from_yahoo(t)} | Valeur: {value:.2f} $ | P&L: {pnl:+.2f} $ ({pnl_pct:+.2f}%)"):
                col1, col2 = st.columns(2)
                col1.write(f"**Quantité:** {qty}")
                col1.write(f"**Prix d'Achat Moyen:** {avg_price:.2f} $")
                col2.write(f"**Prix Actuel:** {current_price:.2f} $")
                
                if st.button(f"🔴 Liquider {t} au marché", key=f"liq_{t}"):
                    # Simulation Event-Driven (Slippage + Commissions)
                    exec_price = current_price * 0.9990 # -0.10% Slippage
                    gross_revenue = qty * exec_price
                    commission = gross_revenue * 0.0005 # -0.05% Frais
                    net_revenue = gross_revenue - commission
                    
                    pf['cash'] += net_revenue
                    pf['history'].append({
                        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'action': 'SELL (Manual)',
                        'ticker': t,
                        'qty': qty,
                        'price': exec_price,
                        'total': net_revenue,
                        'pnl': net_revenue - (qty * avg_price)
                    })
                    del pf['positions'][t]
                    save_portfolio(pf)
                    st.success(f"Position {t} liquidée avec succès (Slippage et Frais déduits) !")
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

def page_options_paper_trading():
    st.title("🕹️ Paper Trading (Options & Dérivés)")
    st.markdown("Exécutez vos stratégies d'options (Call/Put) en conditions réelles sans risquer votre argent. Solde de départ indépendant : **100 000 $**.")
    
    pf = load_options_portfolio()
    
    st.header("💼 Mon Portefeuille Virtuel (Options)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Liquidités (Cash)", f"{pf['cash']:,.2f} $")
    
    total_positions_value = 0
    if pf['positions']:
        st.subheader("📈 Contrats d'Options Ouverts")
        keys_to_remove = []
        for contract_id, data in pf['positions'].items():
            t = data['ticker']
            qty = data['qty']
            avg_price = data['premium']
            K = data['strike']
            T_days = data['days_to_expiry']
            opt_type = data['type']
            
            # Recalculate price today using Black Scholes
            try:
                df = yf.download(t, period="1y", progress=False)
                current_price = df['Close'].iloc[-1].item() if isinstance(df['Close'].iloc[-1], pd.Series) else df['Close'].iloc[-1]
                sigma = get_implied_volatility(t, current_price)
            except:
                current_price = data['underlying_price_at_buy']
                sigma = 0.20
                
            T = T_days / 365.0
            if T <= 0.001: T = 0.001 # prevent division by zero
            r = 0.05
            
            # Nouvelle estimation du contrat
            current_premium, delta, gamma, theta, vega = black_scholes(current_price, K, T, r, sigma, opt_type.lower())
            
            # Utilise le multiplicateur enregistré ou tente de le retrouver
            multiplier = data.get('multiplier', get_option_multiplier_and_legislation(t)[0])
            
            value = qty * current_premium * multiplier # *multiplier selon la norme
            buy_value = qty * avg_price * multiplier
            pnl = value - buy_value
            pnl_pct = (pnl / buy_value) * 100 if buy_value > 0 else 0
            
            total_positions_value += value
            
            is_expanded = (st.session_state.get('show_lifecycle_for') == contract_id)
            with st.expander(f"📦 {qty} Contrat(s) {opt_type.upper()} sur {t} | Valeur: {value:.2f} $ | P&L: {pnl:+.2f} $ ({pnl_pct:+.2f}%)", expanded=is_expanded):
                col1, col2 = st.columns(2)
                col1.write(f"**Strike (K):** {K} $")
                col1.write(f"**Prime d'Achat:** {avg_price:.2f} $")
                col1.write(f"**Prix Sous-jacent actuel:** {current_price:.2f} $")
                col1.write(f"**Jours restants:** {T_days}")
                
                col2.write(f"**Prime Actuelle (BS):** {current_premium:.2f} $")
                col2.write(f"**Delta (Direction):** {delta:.3f}")
                col2.write(f"**Gamma (Accélération):** {gamma:.3f}")
                col2.write(f"**Theta (Usure/Temps):** {theta:.3f}")
                col2.write(f"**Vega (Volatilité):** {vega:.3f}")
                
                is_eu = "Européenne" in get_option_multiplier_and_legislation(t)[1]
                can_sell = not (is_eu and T_days > 0)
                
                if not can_sell:
                    st.warning("⚠️ Législation EU : Exercice et Revente bloqués avant l'échéance.")
                
                c_btn1, c_btn2 = st.columns(2)
                
                with c_btn1:
                    if st.button(f"🔴 Revendre le contrat", key=f"liq_opt_{contract_id}", disabled=not can_sell, use_container_width=True):
                        revenue = value
                        pf['cash'] += revenue
                        pf['history'].append({
                            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'action': 'SELL OPTION',
                            'ticker': t,
                            'contract': f"{opt_type.upper()} Strike {K}",
                            'qty': qty,
                            'premium_sold': current_premium,
                            'total_received': revenue,
                            'pnl': pnl
                        })
                        keys_to_remove.append(contract_id)
                        st.success(f"Contrat sur {t} revendu avec succès !")
                
                with c_btn2:
                    if st.button(f"📈 Analyser le Cycle de Vie", key=f"life_btn_{contract_id}", use_container_width=True):
                        if st.session_state.get('show_lifecycle_for') == contract_id:
                            st.session_state['show_lifecycle_for'] = None
                        else:
                            st.session_state['show_lifecycle_for'] = contract_id
                        st.rerun()

                if st.session_state.get('show_lifecycle_for') == contract_id:
                    with st.spinner("⏳ Reconstruction de la timeline du contrat..."):
                        try:
                            # Extraction de la date d'achat du contract_id (format YYYYMMDDHHMMSS)
                            purchase_date_str = contract_id.split('_')[-1]
                            purchase_dt = datetime.strptime(purchase_date_str, "%Y%m%d%H%M%S")
                            start_date = purchase_dt.strftime("%Y-%m-%d")
                            
                            hist_df = yf.download(t, start=start_date, progress=False)
                            
                            if not hist_df.empty:
                                dates = []
                                prices = []
                                bs_premiums = []
                                deltas = []
                                thetas = []
                                
                                for date, row in hist_df.iterrows():
                                    days_passed = (date.tz_localize(None) - purchase_dt).days
                                    if days_passed < 0: days_passed = 0
                                    current_T_days = T_days - days_passed
                                    if current_T_days <= 0: current_T_days = 0.001
                                    
                                    spot = row['Close'].item() if isinstance(row['Close'], pd.Series) else row['Close']
                                    
                                    p, d, g, th, v = black_scholes(spot, K, current_T_days/365.0, r, sigma, opt_type.lower())
                                    
                                    dates.append(date)
                                    prices.append(spot)
                                    bs_premiums.append(p * multiplier * qty)
                                    deltas.append(d)
                                    thetas.append(th)
                                    
                                fig_lc = make_subplots(
                                    rows=4, cols=1, 
                                    shared_xaxes=True, 
                                    vertical_spacing=0.08,
                                    subplot_titles=(
                                        f"Course de l'Action vs Strike ({t})", 
                                        f"Prime du Contrat (Valorisation en $)", 
                                        "Grec Directionnel (Delta)",
                                        "Grec Temporel (Theta $/jour)"
                                    )
                                )
                                
                                # Row 1: Action vs Strike
                                fig_lc.add_trace(go.Scatter(x=dates, y=prices, name="Prix Action", line=dict(color='#00d2ff')), row=1, col=1)
                                fig_lc.add_hline(y=K, line_dash="dash", line_color="red", annotation_text="Strike", row=1, col=1)
                                
                                # Row 2: Prime BS (Totale)
                                fig_lc.add_trace(go.Scatter(x=dates, y=bs_premiums, name="Valeur du Contrat ($)", line=dict(color='#00ff88'), fill='tozeroy', fillcolor='rgba(0,255,136,0.1)'), row=2, col=1)
                                fig_lc.add_hline(y=buy_value, line_dash="dash", line_color="orange", annotation_text="Coût d'Achat", row=2, col=1)
                                
                                # Row 3: Delta
                                fig_lc.add_trace(go.Scatter(x=dates, y=deltas, name="Delta", line=dict(color='#ff00ff')), row=3, col=1)
                                
                                # Row 4: Theta
                                fig_lc.add_trace(go.Scatter(x=dates, y=thetas, name="Theta", line=dict(color='#ff3333'), fill='tozeroy', fillcolor='rgba(255,51,51,0.2)'), row=4, col=1)
                                
                                fig_lc.update_layout(height=900, template="plotly_dark", title_text="🧬 Dashboard Rétroactif du Cycle de Vie", margin=dict(l=20, r=20, t=60, b=20), hovermode="x unified")
                                st.plotly_chart(fig_lc, use_container_width=True)
                            else:
                                st.warning("Pas assez de données historiques pour générer le graphique (Achat récent ou week-end). Revenez demain !")
                        except Exception as e:
                            st.error(f"Impossible de reconstruire le cycle de vie. Soit le contrat est trop ancien (avant cette mise à jour), soit une erreur est survenue : {e}")
        
        for k in keys_to_remove:
            del pf['positions'][k]
        if keys_to_remove:
            save_options_portfolio(pf)
            st.rerun()
            
    else:
        st.info("Aucun contrat ouvert. Utilisez le Pricing d'Options pour acheter.")
        
    c2.metric("Valeur des Contrats", f"{total_positions_value:,.2f} $")
    c3.metric("Valeur Totale du Portefeuille", f"{(pf['cash'] + total_positions_value):,.2f} $", delta=f"{((pf['cash'] + total_positions_value) - 100000):+,.2f} $")
    
    st.divider()
    st.header("📜 Historique des Transactions")
    if pf['history']:
        hist_df = pd.DataFrame(pf['history'])
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.write("Aucun historique pour le moment.")
        
    if st.button("⚠️ Réinitialiser le compte d'Options"):
        save_options_portfolio({"cash": 100000.0, "positions": {}, "history": []})
        st.success("Compte d'Options réinitialisé.")
        st.rerun()

def page_advanced_academy():
    st.title("🎓 Académie : Modélisation Avancée (Machine Learning)")
    st.markdown("Plongez dans les algorithmes d'Intelligence Artificielle qui dominent aujourd'hui Wall Street, bien loin des simples moyennes mobiles.")
    
    st.header("1. L'Intelligence d'Ensemble (Stacking) & La Sagesse des Foules")
    st.markdown("""
    Pourquoi utiliser un seul algorithme quand on peut avoir un comité d'experts ? C'est le principe du **Stacking** (ou Voting Classifier).
    
    > [!NOTE]
    > **Le Poids du Bœuf (Francis Galton, 1906) :**
    > Lors d'une foire agricole, 800 personnes devaient deviner le poids d'un bœuf. Certains étaient experts (bouchers), d'autres incompétents. Étonnamment, la *moyenne* de toutes les estimations (1197 livres) était à moins d'une livre du poids exact du bœuf (1198 livres) ! La foule était plus précise que le meilleur des experts.
    
    **Application en Trading :** 
    Notre système utilise un comité composé de :
    1.  **XGBoost :** Le spécialiste des relations complexes non-linéaires.
    2.  **LightGBM :** Le sprinteur qui détecte les micro-anomalies très rapidement.
    3.  **Random Forest :** L'ancien très prudent qui ne s'enflamme jamais.
    
    Le signal final d'achat n'est déclenché que par **Consensus** (Soft Voting). Cela élimine presque tous les faux signaux (le bruit du marché).
    """)

    st.header("2. Arbres de Décision (XGBoost vs Random Forest)")
    st.markdown("""
    Ces algorithmes sont des forêts contenant des milliers d'arbres de décision. Mais ils plantent leurs arbres différemment :
    
    *   **Random Forest (Le Parallèle) :** Il crée 1 000 arbres en même temps. Chaque arbre regarde une partie différente des données (Bagging). La moyenne est très robuste, mais le modèle a du mal à apprendre des choses ultra-complexes.
    *   **XGBoost (Le Séquentiel - Gradient Boosting) :** Il crée 1 arbre. Il regarde où cet arbre s'est trompé. L'arbre n°2 a pour **unique mission** de corriger les erreurs de l'arbre n°1. Et ainsi de suite pendant 1 000 arbres. C'est l'un des algorithmes les plus puissants au monde, mais il risque le "Sur-apprentissage" (Overfitting) si on ne le contrôle pas.
    """)

    st.header("3. Optimisation Bayésienne (Optuna)")
    st.markdown("""
    L'entraînement d'un modèle XGBoost nécessite de régler des "Hyperparamètres" (la profondeur des arbres, le taux d'apprentissage). 
    
    Plutôt que d'essayer des milliers de paramètres au hasard (Random Search), les Quant Funds utilisent **Optuna**, une IA qui entraîne l'IA.
    *   Optuna utilise l'inférence bayésienne (TPE - Tree-structured Parzen Estimator).
    *   Elle explore l'espace des paramètres. Si un essai est mauvais, Optuna mathématise cette zone comme "toxique" et ne la teste plus, gagnant un temps infini.
    """)
    st.info("Dans notre application, cochez **Auto-Optimisation (Optuna)** avant d'entraîner le modèle. Le Sharpe Ratio généré est souvent drastiquement supérieur à une configuration manuelle.")

    st.header("4. Explicabilité Mathématique (SHAP)")
    st.markdown("""
    Le problème du Machine Learning est l'effet "Boîte Noire" (Black Box). Les régulateurs financiers interdisent aux banques de trader sans pouvoir expliquer *pourquoi* la machine a acheté.
    
    Nous utilisons **SHAP (SHapley Additive exPlanations)**, basé sur la Théorie des Jeux Coopératifs (Prix Nobel d'Économie en 2012 pour Lloyd Shapley).
    
    > **Comment ça marche ?**
    > Imaginez 3 joueurs (RSI, MACD, Volume) qui gagnent ensemble 100$. Comment répartir les 100$ équitablement ? Shapley a prouvé qu'il fallait calculer la contribution marginale de chaque joueur dans toutes les coalitions possibles. 
    > Dans l'application, l'onglet "Dans le cerveau de XGBoost" utilise cette équation exacte pour vous dire, par exemple : *"Le signal est d'Achat à 65%, et le RSI a contribué à +15% de cette décision car il était survendu"*.
    """)

    st.header("5. La Simulation de Monte Carlo")
    st.markdown("""
    Créée par le mathématicien Stanislaw Ulam lors du projet Manhattan pour la bombe atomique, la simulation de Monte Carlo utilise l'aléatoire massif pour résoudre des problèmes déterministes complexes.
    
    **En finance (Le Bootstrap Resampling) :**
    Le futur ne répétera jamais exactement le passé. Pour stress-tester notre IA :
    1.  Le moteur prend l'historique des gains/pertes de l'IA.
    2.  Il tire au sort avec remise (Bootstrap) pour générer **1 000 univers parallèles** (des futurs alternatifs).
    3.  Cela nous permet de calculer la **Probabilité de Ruine**. "Dans combien de ces univers alternatifs l'algorithme a-t-il fait faillite ?"
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

def find_strike_for_delta(S, T, r, sigma, option_type, target_delta):
    """
    Trouve le Strike K tel que le Delta théorique de BS correspond au target_delta.
    target_delta doit être positif pour un Call (ex: 0.15) et négatif pour un Put (ex: -0.15).
    """
    def obj(K):
        _, d, _, _, _ = black_scholes(S, K, T, r, sigma, option_type)
        return d - target_delta
    try:
        K_opt = brentq(obj, S * 0.1, S * 3.0)
        return K_opt
    except ValueError:
        return S # Fallback au prix actuel si aucune solution trouvée

def page_options_pricing(tickers):
    st.title("🧮 Pricing d'Options (Black-Scholes)")
    st.markdown("Calculez la valeur théorique (Juste Prix) d'une option d'achat (Call) ou de vente (Put) ainsi que ses paramètres de risque (Les Greeks).")
    
    if not tickers:
        st.warning("Veuillez sélectionner au moins une action dans la barre latérale.")
        return
        
    ticker = tickers[0]
    st.header(f"Contrat d'Option sur {ticker}")
    
    # --- IA RECOMMENDATION ---
    st.subheader("🤖 Recommandation Stratégique XGBoost")
    default_opt_index = 0
    if f"trader_{ticker}" in st.session_state:
        trader = st.session_state[f"trader_{ticker}"]
        if trader.is_trained:
            # Re-predict using latest data
            df_raw = yf.download(ticker, period="1y", interval="1d", progress=False)
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = df_raw.columns.droplevel(1)
            macro_df = get_macro_data("1y", "1d")
            df = add_features(df_raw)
            if macro_df is not None:
                df = df.join(macro_df, how='left').ffill().dropna()
            
            last_row = df.iloc[-1:]
            base_features = ['Returns', 'SMA_20', 'SMA_50', 'SMA_200', 'EMA_9', 'Vol_20', 'RSI', 'MACD', 'Signal_Line', 'BB_Upper', 'BB_Lower', 'BB_Width', 'ATR', 'ADX', 'Volume_Ratio', 'OBV', 'Stoch_K', 'ROC', 'VWAP', 'Lag_1', 'Lag_2', 'Lag_3']
            macro_features = ['SPY_Return', 'VIX', 'TNX', 'DXY']
            features = [f for f in base_features + macro_features if f in df.columns]
            prob = trader.predict(last_row, features)
            
            if prob > 0.6:
                st.success(f"📈 L'IA est fortement **haussière** sur {ticker} (Confiance {prob:.0%}). **Stratégie recommandée : Achat de CALL.**")
                default_opt_index = 0
            elif prob < 0.4:
                st.error(f"📉 L'IA est fortement **baissière** sur {ticker} (Confiance {(1-prob):.0%}). **Stratégie recommandée : Achat de PUT.**")
                default_opt_index = 1
            else:
                st.warning(f"⚖️ L'IA est **neutre** sur {ticker}. Marché incertain, évitez les options directionnelles.")
                default_opt_index = 0
        else:
            st.info("Le modèle IA est présent mais non entraîné.")
            default_opt_index = 0
    else:
        st.info("💡 Entraînez d'abord l'IA dans le Terminal de Trading pour obtenir une recommandation quant à la direction de l'option.")
        default_opt_index = 0

    st.divider()
    
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
        opt_type = st.radio("Type d'Option", ["Call", "Put"], index=default_opt_index)
        
    price, delta, gamma, theta, vega = black_scholes(S_input, K, T, r, sigma, opt_type.lower())
    multiplier, legislation = get_option_multiplier_and_legislation(ticker)
    
    st.divider()
    st.subheader(f"Valeur Théorique de l'Option (Black-Scholes) : **{price:.2f} $** par action")
    
    st.info(f"💡 **Prime vs Contrat :** La valeur ci-dessus est la 'Prime' calculée pour **1 action**. Or, la norme de ce titre ({legislation}) impose des lots de **{multiplier} actions**.\n\n👉 **Le coût réel d'un contrat (1 lot) sera donc de : {price:.2f} $ × {multiplier} = {price * multiplier:.2f} $**")
    st.write("### Les Greeks (Paramètres de Risque)")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Delta (Δ)", f"{delta:.3f}", help="Sensibilité au prix. Si l'action monte de 1$, l'option prendra Delta $. (0 à 1 pour Call, -1 à 0 pour Put).")
    g2.metric("Gamma (Γ)", f"{gamma:.4f}", help="Vitesse du Delta. De combien le Delta change si l'action monte de 1$.")
    g3.metric("Theta (Θ)", f"{theta:.3f} $/jour", help="Érosion du temps. Combien l'option perd de valeur chaque jour qui passe (Time Decay).")
    g4.metric("Vega (ν)", f"{vega/100:.3f}", help="Sensibilité à la volatilité. Si la volatilité augmente de 1%, l'option prend Vega $.")

    # --- GRAPHIQUE DE PAYOFF ---
    st.divider()
    st.subheader("📊 Graphique de Payoff à l'expiration")
    st.markdown(f"**Pourquoi le graphique commence-t-il dans le rouge ?** Parce que pour obtenir ce droit d'acheter/vendre, vous avez payé la Prime aujourd'hui ({price:.2f} $ par action). Pour être rentable au moment de l'expiration, l'action devra donc bouger suffisamment pour rembourser cette prime : c'est le **Break-Even (Seuil de Rentabilité)**.")
    
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

    # --- ACHAT VIRTUEL ---
    st.divider()
    st.subheader("🛒 Exécution Virtuelle (Paper Trading)")
    st.write(f"**Norme applicable :** {legislation}")
    qty_options = st.number_input(f"Combien de contrats souhaitez-vous acheter ? (1 contrat = {multiplier} actions)", min_value=1, value=1)
    cout_total = qty_options * price * multiplier
    st.info(f"💰 **Total débité de votre compte virtuel :** {qty_options} contrats × {price*multiplier:.2f} $ = **{cout_total:.2f} $**")
    
    if st.button("✅ Acheter ce contrat (Paper Trading)", use_container_width=True):
        pf_opt = load_options_portfolio()
        if pf_opt['cash'] >= cout_total:
            pf_opt['cash'] -= cout_total
            contract_id = f"{ticker}_{opt_type.upper()}_{K}_{T_days}d_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            pf_opt['positions'][contract_id] = {
                'ticker': ticker,
                'type': opt_type.lower(),
                'strike': K,
                'days_to_expiry': T_days,
                'premium': price,
                'qty': qty_options,
                'multiplier': multiplier,
                'underlying_price_at_buy': S_input,
                'buy_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            pf_opt['history'].append({
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'action': 'BUY OPTION',
                'ticker': ticker,
                'contract': f"{opt_type.upper()} Strike {K}",
                'qty': qty_options,
                'premium_paid': price,
                'total_cost': cout_total
            })
            save_options_portfolio(pf_opt)
            st.success(f"Contrat acheté avec succès ! Retrouvez-le dans l'onglet 'Paper Trading (Options)'.")
            st.balloons()
        else:
            st.error("Fonds insuffisants dans votre portefeuille d'options virtuel.")

    # --- SURFACE DE VOLATILITÉ ET ANOMALIES ---
    st.divider()
    st.header("🌊 Surface de Volatilité & Scanner d'Anomalies (Live Market)")
    st.markdown("Récupération en temps réel des contrats d'options pour tracer le **Volatility Smile** 3D et détecter les options **Surévaluées/Sous-évaluées** par le marché.")
    
    if st.button("🔍 Lancer le Scanner d'Options en Direct", use_container_width=True):
        with st.spinner("Téléchargement de la chaîne d'options depuis le marché (Cela peut prendre quelques secondes)..."):
            try:
                ticker_obj = yf.Ticker(ticker)
                expirations = ticker_obj.options
                
                if not expirations:
                    st.warning("Aucune chaîne d'options disponible pour cette action sur le marché public.")
                else:
                    data_calls = []
                    # On limite aux 5 prochaines échéances pour ne pas figer l'application
                    for exp in expirations[:5]:
                        try:
                            chain = ticker_obj.option_chain(exp)
                            calls = chain.calls
                            
                            days_to_exp = (pd.to_datetime(exp) - datetime.now()).days
                            if days_to_exp <= 0:
                                continue
                                
                            for _, row in calls.iterrows():
                                strike = row['strike']
                                # Filtre pour ne garder que les strikes proches de la monnaie (± 30%)
                                if 0.7 * S_input <= strike <= 1.3 * S_input:
                                    iv = row['impliedVolatility']
                                    bid = row['bid']
                                    ask = row['ask']
                                    mid_price = (bid + ask) / 2.0
                                    
                                    # Calcul du prix théorique avec la Volatilité Historique (sigma_hist) pour trouver l'anomalie
                                    theo_price = black_scholes(S_input, strike, days_to_exp/365.0, r, sigma_hist, 'call')[0]
                                    
                                    if theo_price > 0:
                                        overval_pct = ((mid_price - theo_price) / theo_price) * 100
                                    else:
                                        overval_pct = 0
                                        
                                    data_calls.append({
                                        'Strike': strike,
                                        'Jours_Expiration': days_to_exp,
                                        'IV': iv,
                                        'Prix_Marché': mid_price,
                                        'Prix_Théorique': theo_price,
                                        'Écart_Surévaluation_%': overval_pct
                                    })
                        except:
                            continue
                    
                    if data_calls:
                        df_surf = pd.DataFrame(data_calls)
                        
                        st.subheader("📊 Surface de Volatilité 3D (Implied Volatility)")
                        fig_3d = go.Figure(data=[go.Mesh3d(
                            x=df_surf['Strike'],
                            y=df_surf['Jours_Expiration'],
                            z=df_surf['IV'],
                            intensity=df_surf['IV'],
                            colorscale='Viridis',
                            opacity=0.8
                        )])
                        fig_3d.update_layout(
                            scene=dict(
                                xaxis_title="Strike ($)",
                                yaxis_title="Jours avant Expiration",
                                zaxis_title="Volatilité Implicite (σ)"
                            ),
                            template="plotly_dark",
                            height=600,
                            margin=dict(l=0, r=0, b=0, t=30)
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)
                        
                        st.subheader("🚨 Scanner d'Anomalies (Opportunités d'Arbitrage)")
                        st.markdown("Ce tableau compare le Prix du Marché (Mid-Price) au Prix Théorique de Black-Scholes. Les contrats fortement surévalués sont d'excellents candidats pour des stratégies de vente (Short Call).")
                        
                        # Trier par les contrats les plus surévalués
                        df_anomalies = df_surf.sort_values(by='Écart_Surévaluation_%', ascending=False).head(15)
                        
                        # Formatage visuel avec Pandas Styler
                        st.dataframe(
                            df_anomalies.style.format({
                                'Strike': '{:.2f} $',
                                'IV': '{:.2%}',
                                'Prix_Marché': '{:.2f} $',
                                'Prix_Théorique': '{:.2f} $',
                                'Écart_Surévaluation_%': '{:+.2f} %'
                            }).background_gradient(subset=['Écart_Surévaluation_%'], cmap='RdYlGn_r'),
                            use_container_width=True
                        )
                    else:
                        st.error("Erreur lors de l'extraction des données ou aucun contrat liquide trouvé.")
            except Exception as e:
                st.error(f"Erreur de connexion à l'API de marché : {e}")

def page_options_academy():
    st.title("🎓 Académie : Options & Black-Scholes (Niveau Avancé)")
    st.markdown("Bienvenue dans le module de formation institutionnel sur les produits dérivés. Comprendre les Options, c'est maîtriser la **gestion du risque** et la **création d'Alpha** dans toutes les conditions de marché.")
    
    st.header("📘 Partie 1 : Qu'est-ce qu'une Option ? (Les Fondations)")
    st.markdown("""
    Contrairement à une action qui représente une fraction d'entreprise, une Option est un **contrat (Produit Dérivé)**. Sa valeur *dérive* du prix d'un autre actif (le sous-jacent).
    Une option vous donne **le DROIT, mais pas l'OBLIGATION**, d'acheter ou de vendre une action à un prix fixé à l'avance, pendant une période donnée.
    
    ### 📈 Le CALL (Option d'Achat)
    **Définition :** Un contrat qui vous donne le droit d'ACHETER l'action à un prix défini (le *Strike*), peu importe le prix réel du marché.
    
    > **Exemple Pratique (L'Immobilier) :** 
    > Vous visitez une maison qui vaut 300 000 €. Vous pensez que le quartier va exploser en valeur grâce à l'arrivée d'une gare. Vous signez une "promesse de vente" : vous donnez 5 000 € aujourd'hui (la **Prime / Premium**), et en échange, vous avez le droit d'acheter la maison à 300 000 € (le **Strike**) n'importe quand pendant les 3 prochains mois.
    > - **Scénario Gagnant :** 2 mois plus tard, la gare est annoncée. La maison vaut 400 000 € ! Grâce au contrat, vous l'achetez 300 000 €. Votre profit est de 100 000 € (moins la prime de 5 000 €). Avec seulement 5 000 € risqués, vous gagnez 95 000 €. **C'est la puissance de l'effet de levier asymétrique.**
    > - **Scénario Perdant :** Le quartier est inondé. La maison tombe à 200 000 €. Êtes-vous obligé de l'acheter ? NON ! Vous déchirez simplement le contrat. Votre perte est limitée à vos 5 000 € de Prime.
    
    ### 📉 Le PUT (Option de Vente)
    **Définition :** Un contrat qui vous donne le droit de VENDRE l'action à un prix défini (Strike), même si elle s'est effondrée.
    
    > **Exemple Pratique (L'Assurance Auto) :** 
    > Vous achetez une voiture neuve pour 50 000 €. Vous achetez une assurance (le **PUT**) pour 1 000 €/an (la **Prime**). L'assurance garantit qu'en cas de destruction (le prix tombe à 0 €), ils vous rachèteront la voiture à 50 000 € (le **Strike**).
    """)
    
    st.header("🔬 Partie 2 : L'Héritage Physique (De la Thermodynamique à Wall Street)")
    st.markdown("""
    Avant d'être une formule financière, l'évaluation des options trouve ses racines dans la physique pure.
    
    *   **Louis Bachelier (1900) :** Cinq ans avant qu'Albert Einstein ne modélise le mouvement aléatoire des particules dans un fluide, Bachelier a utilisé ces équations de *Marche Aléatoire* pour décrire la Bourse de Paris.
    *   **Kiyosi Itō (1944) :** Il invente le calcul stochastique. Les mathématiques classiques (Newton) ne fonctionnent pas pour des variables pleines de "bruit" comme la bourse. Itō crée le moteur permettant d'évaluer l'incertitude.
    *   **Black, Scholes & Merton (1973) :** Ils découvrent la formule qui changera la finance. La révélation incroyable est que l'équation différentielle stochastique qu'ils ont trouvée pour le risque d'une option est une variante exacte de **l'Équation de la Chaleur** de Joseph Fourier (1822) en thermodynamique. En finance, le risque (l'incertitude) se dissipe dans le temps exactement comme la chaleur se dissipe dans un barreau de métal !
    """)
    
    st.header("🧠 Partie 3 : Le Modèle de Black-Scholes (La Révolution Quant)")
    st.info("En 1973, Fischer Black, Myron Scholes et Robert Merton publient l'équation qui a valu un Prix Nobel d'Économie.")
    
    st.latex(r"C(S, t) = S \cdot N(d_1) - K \cdot e^{-rt} \cdot N(d_2)")
    
    st.markdown("""
    Cette équation permet de calculer le "Juste Prix" (Fair Value) d'une option.
    
    **Les 5 Ingrédients :**
    1.  **$S$ (Spot) :** Prix actuel.
    2.  **$K$ (Strike) :** Prix cible d'exercice.
    3.  **$t$ (Time) :** Le temps restant.
    4.  **$r$ (Rate) :** Taux d'intérêt sans risque.
    5.  **$\\sigma$ (Volatilité Implicite) :** L'ingrédient secret. C'est l'estimation de l'amplitude des mouvements futurs. Si l'action bouge de 1% par jour, l'option sera peu chère. Si elle bouge de 10% par jour, l'option sera hors de prix, car l'assureur prend d'énormes risques.
    """)
    
    st.header("🛡️ Partie 4 : La Gestion du Risque (Les 'Greeks')")
    st.markdown("""
    Un trader institutionnel ne dit jamais "J'ai acheté 10 Calls". Il dit "Je suis long de 1 000 Delta et j'ai un Theta négatif".
    """)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Δ (Delta) : Le Compteur de Vitesse")
        st.markdown("""
        **Que se passe-t-il si l'action monte de 1$ ?**
        *   Un Call ATM (A la Monnaie) a généralement un Delta de 0.50. Si l'action monte de 1$, l'option prend 0.50$.
        *   **Exemple Numérique :** Tu as acheté un contrat Call AAPL (Delta 0.50) pour 500$ (5$ de prime). Si AAPL passe de 150$ à 151$, la prime de ton contrat monte à 5.50$. Ton contrat vaut maintenant 550$. Tu viens de gagner 50$ !
        *   **Concept :** Le Delta représente aussi la "probabilité" perçue par le marché que l'option finisse gagnante. Un Delta de 0.20 signifie que le marché estime qu'il n'y a que 20% de chances d'être rentable.
        """)
        
        st.subheader("Θ (Theta) : Le Sablier Mortel")
        st.markdown("""
        **Que se passe-t-il si une journée passe sans que le prix ne bouge ?**
        *   Le Theta est le "loyer" quotidien (Time Decay).
        *   **Exemple Numérique :** Tu achètes une option à 300$ avec un Theta de -10$. Lundi, l'action ne bouge pas. Mardi matin, ton option ne vaut plus que 290$. Mercredi, 280$. Le temps joue *contre* l'acheteur et *pour* le vendeur.
        *   C'est pour cela que 80% des acheteurs d'options perdent : le temps détruit lentement la valeur de leur contrat.
        """)
        
    with c2:
        st.subheader("Γ (Gamma) : L'Accélération (Le Danger)")
        st.markdown("""
        **Que se passe-t-il si l'action accélère violemment ?**
        *   Le Gamma mesure à quelle vitesse le Delta change. C'est l'accélération de ton véhicule.
        *   **Exemple Numérique :** Tu as un Call avec un Delta de 0.50 et un Gamma de 0.10. Si l'action monte de 1$, ton Delta devient 0.60 ! Au prochain dollar de hausse de l'action, l'option ne prendra plus 0.50$, mais 0.60$. Plus ça monte, plus ça gagne vite !
        
        > [!WARNING]
        > **Le Gamma Squeeze (L'Affaire GameStop 2021) :**
        > Quand les particuliers ont acheté des millions de *Calls* OTM sur GameStop, les vendeurs se sont retrouvés avec un Gamma explosif. Pour se couvrir, leurs algorithmes ont été forcés d'acheter l'action. Plus l'action montait, plus le Gamma augmentait le Delta, forçant les algorithmes à acheter *encore plus* d'actions. Une véritable réaction en chaîne nucléaire.
        """)
        
        st.subheader("ν (Vega) : Le Détecteur de Panique")
        st.markdown("""
        **Que se passe-t-il si le marché devient nerveux ?**
        *   La valeur de l'option gonfle du montant du Vega pour chaque 1% de volatilité (IV) supplémentaire.
        *   **Exemple Numérique :** Tu détiens un PUT (assurance à la baisse). L'action stagne, mais la Fed annonce une très mauvaise nouvelle. La panique envahit Wall Street (la Volatilité passe de 20% à 30%). Si ton Vega est de 15$, ton contrat s'apprécie soudainement de 150$ (10 x 15$) simplement parce que *la peur* a augmenté !
        """)
        
    st.divider()
    st.header("🌊 Partie 5 : Le Volatility Smile (Le Traumatisme de 1987)")
    st.markdown("""
    Dans l'équation pure de Black-Scholes, la volatilité implicite est censée être plate (la même pour tous les Strikes). C'est ce qu'on croyait avant le **19 Octobre 1987**.
    
    Ce jour-là (Black Monday), le Dow Jones s'est effondré de 22% en un jour. Ce scénario était mathématiquement "impossible" selon les modèles.
    
    > **Le Réveil des Marchés :** 
    > Depuis ce jour, les traders savent que les modèles ont tort sur les événements extrêmes (Fat Tails). Par conséquent, sur le marché réel, les *Puts* (assurances contre la baisse) très loin de la monnaie (OTM) se vendent beaucoup plus chers que les Calls. 
    > Si on trace la Volatilité Implicite en fonction du Strike, on ne voit plus une ligne droite, mais un "Sourire" tordu (Volatility Smile ou Smirk). 
    > L'outil *Surface de Volatilité 3D* de notre application vous permet de visualiser ce traumatisme de 1987 en direct sur n'importe quelle action.
    """)
    
    st.divider()
    st.header("⏳ Partie 6 : La Vente de Contrats (L'IV Crush & Le métier d'Assureur)")
    st.markdown("""
    Si 80% des acheteurs d'options perdent de l'argent à cause du temps qui passe (*Theta*), cela signifie que **80% des vendeurs encaissent cet argent**. 
    
    ### 🎯 L'Écrasement de la Volatilité (L'Erreur du Débutant)
    Le piège le plus meurtrier pour les investisseurs de détail a lieu lors de l'annonce des résultats trimestriels (Earnings) d'entreprises très volatiles comme Nvidia ou Tesla.
    
    1.  **L'Attente :** La semaine précédant l'annonce, tout le monde achète des Calls. La demande explose. Les Market Makers augmentent massivement la Prime (l'Implied Volatility gonfle).
    2.  **L'Achat naïf :** Le débutant achète un Call la veille de l'annonce en payant une prime délirante (Vega géant).
    3.  **L'Événement :** Nvidia annonce d'excellents résultats ! L'action monte de +5%.
    4.  **Le Désastre (IV Crush) :** L'événement est passé. L'incertitude a disparu. L'Implied Volatility s'effondre instantanément de 150% à 40%. La perte de valeur due au *Vega* est tellement violente qu'elle absorbe totalement le gain de l'action. **Le débutant perd 50% de son investissement alors qu'il a eu raison sur la direction !**
    
    > [!TIP]
    > C'est pour cela que les Quants institutionnels **vendent** des options (Short Volatility) avant les earnings. Ils encaissent la prime hors de prix, et le lendemain matin, l'IV Crush détruit le contrat, leur permettant d'empocher l'argent sans rien faire.
    """)

    st.divider()
    st.header("⚖️ Partie 7 : L'Impact Législatif (Norme US vs Norme EU)")
    st.markdown("""
    Vous avez peut-être remarqué dans notre simulateur la mention **"Législation Européenne"** ou **"Législation Américaine"**.
    
    ### 1. La taille du contrat (Multiplicateur)
    - **États-Unis (AAPL, TSLA, etc.) :** 1 contrat standard représente **100 actions**. Si la prime est de 2$, l'achat vous coûtera 200$.
    - **Europe (LVMH, ASML, etc.) :** Souvent, 1 contrat représente **10 actions** (standard Euronext pour les actions très valorisées). Si la prime est de 2$, le contrat coûte 20$.
    
    ### 2. Le droit d'Exercice (Le piège théorique)
    - **Options Américaines : Exercice Libre.** Vous pouvez appeler le vendeur du contrat n'importe quel jour avant l'échéance et dire "J'exerce mon droit, vends-moi les actions au prix convenu aujourd'hui !".
    - **Options Européennes : Exercice à l'Échéance uniquement.** Vous êtes "bloqué" jusqu'à la date d'expiration pour transformer l'option en actions. 
    
    > **Exemple de Revente (Marché Secondaire) :**
    > Rassurez-vous, si vous détenez un contrat Européen qui a fait +300% de gain latent, vous n'êtes pas obligé d'attendre l'échéance pour encaisser ! Vous pouvez simplement **Revendre le contrat (la prime)** à un autre trader sur le marché secondaire. L'interdiction porte uniquement sur la *livraison physique des actions*, pas sur la spéculation de la prime.
    > 
    > *Cependant, dans notre simulateur de Paper Trading, pour des raisons pédagogiques extrêmes, le bouton de revente des options européennes est volontairement bloqué pour vous faire ressentir pleinement cette contrainte temporelle !*
    """)
    
    st.divider()
    st.header("🔄 Partie 8 : Le Mécanisme de Gain (Action vs Prime)")
    st.markdown("""
    C'est une question récurrente qui met en lumière la différence absolue entre acheter une Action et acheter une Option.
    
    Pour y répondre directement : **OUI, lorsque vous revendez le contrat avant son échéance, votre gain vient de l'écart des Primes (Prime de Revente - Prime d'Achat).**
    
    Mais attention, ces deux concepts (Prime et Prix de l'Action) sont intimement liés. Voici pourquoi :
    
    ### 1. Pourquoi le gain vient-il des Primes ?
    Quand vous achetez une option, vous n'achetez pas l'action. Vous achetez un "bout de papier" (le contrat d'assurance). Sur le marché des options, ce que les traders s'échangent toute la journée, c'est ce bout de papier. La "Prime", c'est tout simplement le prix de ce bout de papier. Donc, si vous l'achetez 5$ et que vous le revendez 8$ à un autre trader, vous avez gagné 3$.
    
    ### 2. Mais d'où vient cette hausse de la Prime ?
    C'est là qu'intervient le prix de l'action ! La Prime (calculée par Black-Scholes) n'évolue pas par magie. Sa valeur *dérive* (d'où le nom "produit dérivé") du comportement de l'action. La Prime de votre contrat a augmenté de 5$ à 8$ **parce que** le prix de l'action a bougé dans le bon sens. C'est ce qu'on appelle le **Delta** (l'un des Grecs) :
    
    - Si l'action Apple monte de 1$ et que votre Delta est de 0.50, la Prime de votre contrat va monter de 0.50$.
    - L'écart de prix de l'action génère la hausse, mais c'est bien la revente de la Prime gonflée qui atterrit dans votre poche.
    
    ### 3. Et si j'attends l'échéance (l'Exercice) ?
    Si vous gardez votre contrat jusqu'au tout dernier jour et que vous l'exercez, alors là oui, le calcul change ! Le gain viendra de l'écart entre le **Prix de l'action aujourd'hui** et le **Prix de votre Strike (K)**, auquel on déduit la Prime que vous aviez payée au début.
    
    **En conclusion :**
    - En "Trading d'Options" (ce que font les Quants et la majorité des professionnels), on **revend la Prime** avant la fin. On se fait de l'argent sur l'écart de la Prime.
    - L'action n'est que le "moteur" qui fait gonfler ou dégonfler le prix de votre Prime. Et c'est justement ce moteur complexe (qui inclut aussi le temps et la peur) que Black-Scholes calcule en temps réel !
    """)
    
    st.divider()
    st.header("🧬 Partie 9 : Le Cycle de Vie Rétroactif (L'Analyse Institutionnelle)")
    st.markdown("""
    Analyser les Grecs à l'instant *T* (le jour de l'achat) ne suffit pas. Une option est un produit "vivant" dont le comportement se déforme de jour en jour.
    
    Pour gérer son risque, un Quant institutionnel doit pouvoir observer visuellement comment les Grecs et la Prime évoluent tout au long de la vie du contrat :
    - **Le passage In-The-Money :** Voir le moment exact où la courbe du prix franchit la ligne du Strike et provoque l'explosion du Delta.
    - **L'érosion temporelle (Theta) :** Constater que même si l'action monte légèrement, la prime peut tout de même s'effondrer dans les derniers jours si la hausse ne compense pas le "loyer" quotidien (Theta).
    - **L'accélération directionnelle :** Comprendre visuellement pourquoi un contrat a gagné ou perdu de la valeur hier par rapport à aujourd'hui.
    
    > [!TIP]
    > **L'outil Ultime :** Dans votre portefeuille virtuel (Paper Trading), nous avons intégré le bouton **"📈 Analyser le Cycle de Vie"**. Cet algorithme "On-The-Fly" télécharge l'historique de l'action depuis votre date d'achat exacte et recalcule la formule de Black-Scholes (Prime et Grecs) pour **chaque jour écoulé**. Vous disposez ainsi d'un tableau de bord rétroactif à 4 niveaux pour autopsier le comportement mathématique de votre contrat !
    """)

def page_rag_gemini(tickers, gemini_api_key):
    st.title("🧠 Analyse Fondamentale & RAG (Gemini AI)")
    st.markdown("""
    Ce module utilise l'Intelligence Artificielle Générative (Google Gemini) pour agir comme un **Analyste Quantitatif**. 
    L'IA va lire les actualités récentes, les données financières, et rédiger un rapport d'expertise pour anticiper des événements comme l'**IV Crush**.
    """)
    
    if not genai:
        st.error("❌ La librairie `google-genai` n'est pas installée. Le système fonctionnera si vous l'installez via le terminal (`pip install google-genai`).")
        return
        
    if not gemini_api_key:
        st.warning("⚠️ Veuillez entrer votre Clé API Gemini dans le menu de gauche pour activer ce module.")
        st.info("Vous pouvez en générer une gratuitement sur [Google AI Studio](https://aistudio.google.com/).")
        return
    # Configure inside the loop now
    if not tickers:
        st.warning("Veuillez sélectionner au moins une action dans le menu latéral.")
        return
        
    selected_ticker = st.selectbox("Sélectionnez l'action à analyser", tickers)
    
    if st.button(f"🔍 Lancer l'Analyse Stratégique sur {selected_ticker}", use_container_width=True):
        with st.spinner(f"Récupération des données et réflexion de l'IA sur {selected_ticker}..."):
            try:
                stock = yf.Ticker(selected_ticker)
                
                # 1. Retrieval (Extraction)
                info = stock.info
                news = stock.news
                
                company_name = info.get('longName', selected_ticker)
                sector = info.get('sector', 'Inconnu')
                industry = info.get('industry', 'Inconnu')
                forward_pe = info.get('forwardPE', 'N/A')
                div_yield = info.get('dividendYield', 'N/A')
                
                # Format news
                news_text = ""
                if news:
                    for n in news[:5]:
                        # Handle different formats returned by yfinance news
                        title = n.get('content', {}).get('title', n.get('title', ''))
                        summary = n.get('content', {}).get('summary', n.get('summary', ''))
                        if not summary:
                            summary = n.get('publisher', '')
                        news_text += f"- Titre : {title}\n  Contexte : {summary}\n\n"
                else:
                    news_text = "Aucune actualité récente trouvée."
                    
                # 1.5 Calcul Quantitatif (XGBoost)
                prob_text = "Non calculé"
                try:
                    df_quant = yf.download(selected_ticker, period="2y", progress=False)
                    if not df_quant.empty:
                        df_quant = add_technical_indicators(df_quant)
                        df_quant = add_macro_indicators(df_quant)
                        df_quant.dropna(inplace=True)
                        
                        trader_key = f"port_trader_{selected_ticker}"
                        if trader_key in st.session_state:
                            trader_quant = st.session_state[trader_key]
                            features_quant = trader_quant.feature_names
                        else:
                            trader_quant = XGBoostTrader(ticker=selected_ticker)
                            features_quant = trader_quant.train(df_quant, optimize=False, use_wfa=False)
                            
                        prob_quant = trader_quant.predict(df_quant.iloc[-1:], features_quant)
                        if prob_quant is not None:
                            direction = "HAUSSIÈRE" if prob_quant > 0.5 else "BAISSIÈRE"
                            prob_text = f"{prob_quant:.1%} de probabilité {direction}"
                except Exception:
                    pass

                # 2. Le Prompt
                prompt = f"""
Agis comme un Trader Quantitatif Institutionnel expert en Options et Dérivés.
Ton objectif est de rédiger un rapport stratégique d'analyse fondamentale sur l'action {company_name} (Ticker: {selected_ticker}).

Voici les informations extraites aujourd'hui (Retrieval) :
- Secteur : {sector} ({industry})
- Forward P/E : {forward_pe}
- Dividend Yield : {div_yield}

Dernières actualités de la société (très important) :
{news_text}

Information Cruciale - Prédiction Quantitative (XGBoost) :
Notre algorithme de Machine Learning (XGBoost) vient d'analyser techniquement cette action.
Prédiction mathématique : {prob_text}

Ta mission est de fusionner ces deux visions pour devenir le Portfolio Manager ultime. Rédige un rapport structuré en français (Markdown) contenant :
1. **Le Sentiment Profond :** Quelle est l'ambiance générale autour de l'entreprise au vu des news ? Y a-t-il un momentum caché ?
2. **Confrontation IA vs Marché :** La prédiction mathématique de XGBoost est-elle alignée avec les fondamentaux actuels, ou y a-t-il une divergence dangereuse ?
3. **Risque d'IV Crush & Volatilité :** Au vu des actus, y a-t-il un événement imminent qui pourrait causer un pic d'Implied Volatility (IV) suivi d'un écrasement brutal ?
4. **Stratégie d'Options Recommandée :** En fonction du sentiment, de XGBoost et de la volatilité, quelle stratégie d'options recommanderais-tu (ex: Vendre des Puts, Iron Condor, etc.) et pourquoi ?

Ne mets pas de blabla d'introduction de chatbot, va droit au but avec un ton très professionnel, technique et institutionnel. Utilise du gras et des listes à puces.
"""
                # 3. Generation
                success = False
                all_errors = {}
                # Les anciens modèles 1.0 et 1.5 ont été dépréciés par Google.
                models_to_try = [
                    'gemini-3-pro-preview',
                    'gemini-2.5-pro',
                    'gemini-2.5-flash'
                ]
                
                response = None
                client = genai.Client(api_key=gemini_api_key.strip())
                for m_name in models_to_try:
                    try:
                        response = client.models.generate_content(model=m_name, contents=prompt)
                        st.info(f"Modèle utilisé avec succès : `{m_name}`")
                        success = True
                        break
                    except Exception as e:
                        all_errors[m_name] = str(e)
                        continue
                        
                if not success:
                    st.error("Aucun modèle Gemini n'a pu traiter la requête. Voici les raisons exactes renvoyées par Google pour chaque modèle testé :")
                    for m, err in all_errors.items():
                        st.warning(f"**{m}** : {err}")
                    return
                st.success("Analyse terminée avec succès !")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"Une erreur est survenue lors de l'analyse : {str(e)}")

def page_bot_config():
    st.title("🤖 Paramétrage du Bot (Headless)")
    st.markdown("Configurez ici l'automatisation. Le bot s'exécutera en tâche de fond tous les soirs, s'entraînera, et vous enverra un signal via Telegram ou Discord.")
    
    BOT_CONFIG_FILE = "bot_config.json"
    
    # Charger la conf existante
    if os.path.exists(BOT_CONFIG_FILE):
        with open(BOT_CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {
            "webhook_discord": "",
            "telegram_bot_token": "",
            "telegram_chat_id": "",
            "tickers": ["Apple Inc. (US)"],
            "run_time": "22:00",
            "is_active": False
        }
        
    st.header("1. Cibles d'Analyse Nocturne")
    
    portfolio_choice = st.selectbox(
        "💡 Sélection rapide de portefeuille",
        options=list(PREDEFINED_PORTFOLIOS.keys()),
        index=0
    )
    
    if portfolio_choice != "Sélection Manuelle":
        default_selection = PREDEFINED_PORTFOLIOS[portfolio_choice]
    else:
        default_selection = config.get("tickers", ["Apple Inc. (US)"])
        
    valid_defaults = [s for s in default_selection if s in MAJOR_STOCKS.keys()]
    
    stock_choices = st.multiselect(
        "Rechercher une ou plusieurs actions",
        options=list(MAJOR_STOCKS.keys())[1:],
        default=valid_defaults
    )
    
    selected_tickers = []
    selected_tickers.extend(stock_choices)
    
    # Ajout silencieux des tickers personnalisés du portefeuille (ex: Chine)
    for s in default_selection:
        if s not in MAJOR_STOCKS.keys() and s not in selected_tickers:
            selected_tickers.append(s)
    
    st.header("2. Canaux d'Alertes")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Discord Webhook")
        webhook_discord = st.text_input("URL du Webhook Discord", value=config.get("webhook_discord", ""))
    with c2:
        st.subheader("Telegram Bot API")
        telegram_token = st.text_input("Bot Token", value=config.get("telegram_bot_token", ""))
        telegram_chat = st.text_input("Chat ID", value=config.get("telegram_chat_id", ""))
        
    st.header("3. Planification")
    run_time_val = config.get("run_time", "22:00")
    run_time_obj = datetime.strptime(run_time_val, "%H:%M").time()
    run_time = st.time_input("Heure d'exécution quotidienne", value=run_time_obj)
    
    if st.button("💾 Sauvegarder la Configuration"):
        config["webhook_discord"] = webhook_discord
        config["telegram_bot_token"] = telegram_token
        config["telegram_chat_id"] = telegram_chat
        config["tickers"] = selected_tickers
        config["run_time"] = run_time.strftime("%H:%M")
        with open(BOT_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        st.success("Configuration sauvegardée !")
        
    st.divider()
    st.header("4. Contrôle du Serveur (Daemon)")
    
    if config.get("is_active", False):
        st.success("🟢 Le Bot Headless est actuellement **ACTIF**.")
        if st.button("🛑 Arrêter le Bot"):
            config["is_active"] = False
            with open(BOT_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            st.rerun()
    else:
        st.warning("🔴 Le Bot Headless est actuellement **ARRÊTÉ**.")
        if st.button("🚀 Démarrer le Bot en Arrière-plan"):
            config["is_active"] = True
            with open(BOT_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            
            # Lancer le sous-processus sans bloquer
            if os.name == 'nt': # Windows
                subprocess.Popen(["python", "headless_bot.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)
            else: # Linux/Mac
                subprocess.Popen(["python3", "headless_bot.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
            st.success("Bot démarré en tâche de fond ! Il s'entraînera chaque soir.")
            st.rerun()

def page_options_backtester(tickers):
    st.title("⏪ Backtester d'Options (Simulation Quantitative)")
    st.markdown("""
    Ce module est le **Graal** des Quants. Il permet de simuler des **Stratégies de Rente Systématiques** (Yield Generation).
    Le moteur mathématique reconstruit l'historique des prix des options passées en ciblant un Delta précis, et simule l'ouverture et la fermeture des contrats jour par jour.
    """)
    
    if not tickers:
        st.warning("Veuillez sélectionner au moins une action dans le menu latéral.")
        return
        
    ticker = tickers[0]
    st.header(f"Configuration du Backtest sur {ticker}")
    
    c1, c2, c3, c4 = st.columns(4)
    strategy = c1.selectbox("Stratégie", ["Single Leg", "Credit Spread", "Straddle", "Iron Condor"])
    strat_type = c2.selectbox("Direction (Type)", ["PUT", "CALL"]) if strategy in ["Single Leg", "Credit Spread"] else None
    strat_action = c3.selectbox("Action", ["Vendre (Short)", "Acheter (Long)"]) if strategy == "Single Leg" else None
    target_delta_abs = c4.number_input("Delta Cible (absolu)", min_value=0.01, max_value=0.99, value=0.15, step=0.01, help="Delta ciblé pour les options vendues (ex: 0.15)")
    target_dte = st.number_input("DTE (Jours à l'échéance)", min_value=1, max_value=365, value=45, step=1)
    
    c5, c6, c7, c8 = st.columns(4)
    take_profit_pct = c5.number_input("Take Profit (% de la prime max)", min_value=10, max_value=100, value=50, step=5)
    stop_loss_pct = c6.number_input("Stop Loss (% de la prime max)", min_value=10, max_value=1000, value=200, step=10)
    capital = c7.number_input("Capital Initial ($)", value=100000, step=10000)
    period = c8.selectbox("Période d'historique", ["1y", "2y", "5y", "10y"], index=2)
    
    if st.button("🚀 Lancer le Backtest Systématique", use_container_width=True):
        with st.spinner("Téléchargement de l'historique et simulation des options..."):
            try:
                df = yf.download(ticker, period=period, progress=False)
                if df.empty:
                    st.error("Pas de données trouvées.")
                    return
                
                closes = df['Close']
                if isinstance(closes, pd.DataFrame): closes = closes.iloc[:, 0]
                
                # Volatilité historique glissante (30 jours)
                returns = closes.pct_change()
                hist_vol = returns.rolling(window=30).std() * np.sqrt(252)
                hist_vol = hist_vol.bfill()
                
                r = 0.05
                multiplier, _ = get_option_multiplier_and_legislation(ticker)
                
                cash = capital
                equity_curve = []
                trades = []
                current_position = None
                
                pb = st.progress(0)
                total_days = len(closes)
                
                for i, (date, spot) in enumerate(closes.items()):
                    if i % 50 == 0: pb.progress(i / total_days)
                    sigma = hist_vol.iloc[i]
                    if sigma < 0.05: sigma = 0.05
                    
                    if current_position is None:
                        # OPEN POSITION
                        legs = []
                        if strategy == "Single Leg":
                            td = target_delta_abs if strat_type == "CALL" else -target_delta_abs
                            K = find_strike_for_delta(spot, target_dte/365.0, r, sigma, strat_type.lower(), td)
                            legs.append({'type': strat_type.lower(), 'action': 'short' if 'Vendre' in strat_action else 'long', 'strike': K})
                        elif strategy == "Credit Spread":
                            td_short = target_delta_abs if strat_type == "CALL" else -target_delta_abs
                            td_long = (target_delta_abs/2.0) if strat_type == "CALL" else -(target_delta_abs/2.0)
                            K_short = find_strike_for_delta(spot, target_dte/365.0, r, sigma, strat_type.lower(), td_short)
                            K_long = find_strike_for_delta(spot, target_dte/365.0, r, sigma, strat_type.lower(), td_long)
                            legs.append({'type': strat_type.lower(), 'action': 'short', 'strike': K_short})
                            legs.append({'type': strat_type.lower(), 'action': 'long', 'strike': K_long})
                        elif strategy == "Straddle":
                            legs.append({'type': 'call', 'action': 'long', 'strike': spot})
                            legs.append({'type': 'put', 'action': 'long', 'strike': spot})
                        elif strategy == "Iron Condor":
                            Kc_s = find_strike_for_delta(spot, target_dte/365.0, r, sigma, 'call', target_delta_abs)
                            Kc_l = find_strike_for_delta(spot, target_dte/365.0, r, sigma, 'call', target_delta_abs/2.0)
                            Kp_s = find_strike_for_delta(spot, target_dte/365.0, r, sigma, 'put', -target_delta_abs)
                            Kp_l = find_strike_for_delta(spot, target_dte/365.0, r, sigma, 'put', -target_delta_abs/2.0)
                            
                            legs.append({'type': 'call', 'action': 'short', 'strike': Kc_s})
                            legs.append({'type': 'call', 'action': 'long', 'strike': Kc_l})
                            legs.append({'type': 'put', 'action': 'short', 'strike': Kp_s})
                            legs.append({'type': 'put', 'action': 'long', 'strike': Kp_l})
                            
                        total_cost_in = 0
                        for leg in legs:
                            p, _, _, _, _ = black_scholes(spot, leg['strike'], target_dte/365.0, r, sigma, leg['type'])
                            leg['premium_in'] = p
                            leg_cost = p * multiplier
                            if leg['action'] == 'short':
                                total_cost_in += leg_cost
                            else:
                                total_cost_in -= leg_cost
                                
                        current_position = {
                            'entry_date': date,
                            'entry_spot': spot,
                            'legs': legs,
                            'net_cost_in': total_cost_in,
                            'days_passed': 0,
                        }
                        
                        cash += total_cost_in
                        equity_curve.append(cash)
                    
                    else:
                        # MANAGE POSITION
                        current_position['days_passed'] += 1
                        days_remaining = target_dte - current_position['days_passed']
                        
                        if days_remaining <= 0:
                            # EXPIRATION
                            closing_cashflow = 0
                            for leg in current_position['legs']:
                                intrinsic = max(0, spot - leg['strike']) if leg['type'] == 'call' else max(0, leg['strike'] - spot)
                                val = intrinsic * multiplier
                                closing_cashflow += (-val if leg['action'] == 'short' else val)
                                    
                            cash += closing_cashflow
                            pnl = current_position['net_cost_in'] + closing_cashflow
                            
                            trades.append({
                                'entry_date': current_position['entry_date'].strftime('%Y-%m-%d'),
                                'exit_date': date.strftime('%Y-%m-%d'),
                                'strategy': strategy,
                                'reason': 'Expiration',
                                'pnl': pnl
                            })
                            current_position = None
                            equity_curve.append(cash)
                            
                        else:
                            # CHECK TP / SL
                            closing_cashflow = 0
                            for leg in current_position['legs']:
                                p, _, _, _, _ = black_scholes(spot, leg['strike'], days_remaining/365.0, r, sigma, leg['type'])
                                val = p * multiplier
                                closing_cashflow += (-val if leg['action'] == 'short' else val)
                                    
                            current_pnl = current_position['net_cost_in'] + closing_cashflow
                            
                            close_position = False
                            reason = ""
                            
                            if current_position['net_cost_in'] > 0: # Stratégie Crédit (Vente)
                                max_profit = current_position['net_cost_in']
                                if current_pnl >= max_profit * (take_profit_pct / 100.0):
                                    close_position = True
                                    reason = "Take Profit"
                                elif current_pnl <= -max_profit * (stop_loss_pct / 100.0):
                                    close_position = True
                                    reason = "Stop Loss"
                            else: # Stratégie Débit (Achat)
                                max_loss = abs(current_position['net_cost_in'])
                                if max_loss > 0:
                                    if current_pnl >= max_loss * (take_profit_pct / 100.0):
                                        close_position = True
                                        reason = "Take Profit"
                                    elif current_pnl <= -max_loss * (stop_loss_pct / 100.0):
                                        close_position = True
                                        reason = "Stop Loss"
                                        
                            if close_position:
                                cash += closing_cashflow
                                trades.append({
                                    'entry_date': current_position['entry_date'].strftime('%Y-%m-%d'),
                                    'exit_date': date.strftime('%Y-%m-%d'),
                                    'strategy': strategy,
                                    'reason': reason,
                                    'pnl': current_pnl
                                })
                                current_position = None
                                equity_curve.append(cash)
                            else:
                                mtm_equity = cash + closing_cashflow
                                equity_curve.append(mtm_equity)
                
                pb.empty()
                
                # --- RESULTS ---
                st.divider()
                st.header("📊 Résultats de la Simulation")
                
                if not trades:
                    st.warning("Aucun trade n'a pu être clôturé sur cette période.")
                    return
                    
                trades_df = pd.DataFrame(trades)
                total_trades = len(trades_df)
                winning_trades = len(trades_df[trades_df['pnl'] > 0])
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                total_pnl = trades_df['pnl'].sum()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Capital Final", f"{equity_curve[-1]:,.2f} $", f"{(equity_curve[-1] - capital):+,.2f} $")
                col2.metric("Total Trades", total_trades)
                col3.metric("Win Rate", f"{win_rate:.2%}")
                col4.metric("Profit Net Moyen / Trade", f"{(total_pnl / total_trades):.2f} $")
                
                dates = closes.index
                bnh = (closes / closes.iloc[0]) * capital
                
                fig = make_subplots(rows=1, cols=1)
                fig.add_trace(go.Scatter(x=dates, y=equity_curve, name="Stratégie Options", line=dict(color='#00ff88', width=2)))
                fig.add_trace(go.Scatter(x=dates, y=bnh, name="Buy & Hold (Action)", line=dict(color='#00d2ff', width=1, dash='dot')))
                
                fig.update_layout(title="Évolution du Portefeuille (Equity Curve)", template="plotly_dark", height=500, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("📝 Journal des Transactions (Trades Log)"):
                    st.dataframe(trades_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Erreur durant la simulation : {e}")

# --- FONCTION PRINCIPALE ---
def main():
    st.sidebar.title("🧭 Menu Principal")
    menu = st.sidebar.radio("Sélectionnez un module :", [
        "📈 Terminal de Trading",
        "🕹️ Paper Trading (Virtuel)",
        "🕹️ Paper Trading (Options)",
        "⏪ Backtester Systématique (Options)",
        "🤖 Paramétrage du Bot (Headless)",
        "🧠 Analyse RAG (Gemini)",
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
    
    gemini_api_key = st.sidebar.text_input("🔑 Clé API Gemini (Optionnel)", type="password", help="Nécessaire uniquement pour le module d'Analyse RAG. Ne jamais écrire en dur dans le code !")
    
    
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
        
        period = st.sidebar.selectbox("Période d'historique", ["2y", "5y", "10y", "max"], index=3)
        interval = st.sidebar.selectbox("Intervalle", ["1d", "1wk"], index=0)
        initial_capital = st.sidebar.number_input("Capital Initial Total ($)", min_value=100, max_value=1000000, value=10000, step=100)
        optimize_model = st.sidebar.checkbox("🧠 Auto-Optimisation (Optuna Bayésien)")
        
        st.sidebar.divider()
        st.sidebar.header("Logique Walk-Forward (WFA)")
        use_wfa = st.sidebar.checkbox("Activer Walk-Forward Analysis (WFA)")
        wfa_train_window = st.sidebar.selectbox("Fenêtre Entraînement", ["2Y", "5Y", "7Y"], index=1)
        wfa_step = st.sidebar.selectbox("Pas Avancement", ["1M", "2M", "6M", "1Y"], index=2)
        wfa_start_date = st.sidebar.date_input("WFA Début", value=datetime(2019, 1, 1))
        wfa_end_date = st.sidebar.date_input("WFA Fin", value=datetime(2026, 5, 9))
        
        if len(tickers) == 0:
            st.warning("👈 Veuillez sélectionner au moins une action dans le menu latéral.")
        elif len(tickers) == 1:
            run_single_mode(tickers[0], period, interval, initial_capital, optimize_model, use_wfa, wfa_train_window, wfa_step, wfa_start_date, wfa_end_date)
        else:
            run_portfolio_mode(tickers, period, interval, initial_capital, optimize_model, use_wfa, wfa_train_window, wfa_step, wfa_start_date, wfa_end_date)
            
    elif menu == "🕹️ Paper Trading (Virtuel)":
        page_paper_trading()
    elif menu == "🕹️ Paper Trading (Options)":
        page_options_paper_trading()
    elif menu == "⏪ Backtester Systématique (Options)":
        page_options_backtester(tickers)
    elif menu == "🤖 Paramétrage du Bot (Headless)":
        page_bot_config()
    elif menu == "🧠 Analyse RAG (Gemini)":
        page_rag_gemini(tickers, gemini_api_key)
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
