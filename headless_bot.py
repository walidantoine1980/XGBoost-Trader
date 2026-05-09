import time
import json
import os
import requests
import schedule
from datetime import datetime
import pandas as pd
import yfinance as yf

# Importer la logique principale du trader
from xgboost_trader import MLTrader, PREDEFINED_PORTFOLIOS, MAJOR_STOCKS

BOT_CONFIG_FILE = "bot_config.json"

def load_config():
    if os.path.exists(BOT_CONFIG_FILE):
        with open(BOT_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "webhook_discord": "",
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "tickers": ["Apple Inc. (US)"],
        "run_time": "22:00",
        "is_active": False
    }

def send_alert(config, message):
    """Envoie un message via Discord et/ou Telegram selon la configuration."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ALERT: {message}")
    
    # Discord
    if config.get("webhook_discord"):
        try:
            requests.post(
                config["webhook_discord"],
                json={"content": f"🤖 **XGBoost Quant Bot**\n{message}"},
                timeout=10
            )
        except Exception as e:
            print(f"Erreur Discord: {e}")
            
    # Telegram
    if config.get("telegram_bot_token") and config.get("telegram_chat_id"):
        try:
            url = f"https://api.telegram.org/bot{config['telegram_bot_token']}/sendMessage"
            payload = {
                "chat_id": config["telegram_chat_id"],
                "text": f"🤖 *XGBoost Quant Bot*\n{message}",
                "parse_mode": "Markdown"
            }
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"Erreur Telegram: {e}")

def run_trading_job():
    config = load_config()
    if not config.get("is_active", False):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bot désactivé. Skipping.")
        return

    send_alert(config, "🔄 Début de la session d'entraînement et d'analyse post-clôture...")
    
    tickers = config.get("tickers", [])
    if not tickers:
        send_alert(config, "⚠️ Aucune action sélectionnée pour le bot.")
        return
        
    for ticker in tickers:
        try:
            # Identifier le symbole yfinance à partir du nom
            # Le format est "Nom (Pays) [SYMBOLE]" ou juste via dict lookup si possible
            # Dans xgboost_trader, on utilise split("[")[-1].replace("]", "") si c'est formaté
            # Pour simplifier, on trouve le symbole directement
            
            # Recherche du symbole dans MAJOR_STOCKS
            symbol = None
            for key, val in MAJOR_STOCKS.items():
                if val == ticker:
                    symbol = key
                    break
            
            if not symbol:
                # Fallback
                symbol = ticker.split("[")[-1].replace("]", "").strip() if "[" in ticker else ticker

            print(f"Analyse de {ticker} ({symbol})...")
            
            # Instanciation de l'IA pour cette action
            trader = MLTrader(symbol, start_date="2020-01-01", end_date=datetime.today().strftime('%Y-%m-%d'))
            
            # 1. Fetch data
            df = trader.fetch_data()
            if df.empty:
                continue
                
            # 2. Entraînement quotidien (Recalibration)
            model, accuracy, precision, clf_report = trader.train()
            
            # 3. Prédiction du jour (Clôture actuelle pour ouverture de demain)
            features = [col for col in trader.data.columns if col not in ['Target', 'Future_Return', 'Signal']]
            latest_data = trader.data[features].iloc[-1:]
            
            prediction = model.predict(latest_data)[0]
            probabilities = model.predict_proba(latest_data)[0]
            prob_buy = probabilities[1] * 100
            
            # 4. Calcul de la VaR
            returns = trader.data['Close'].pct_change().dropna()
            var_95 = -returns.quantile(0.05) * 100
            
            # 5. Logique de Signal Institutionnel
            if prediction == 1 and prob_buy > 60:
                signal_type = "🟢 SIGNAL D'ACHAT"
                action = "Achat simulé exécuté sur le Paper Trading."
            elif prediction == 0 and (100 - prob_buy) > 60:
                signal_type = "🔴 SIGNAL DE VENTE / SHORT"
                action = "Vente simulée exécutée sur le Paper Trading."
            else:
                signal_type = "⚪ NEUTRE"
                action = "Pas de transaction. Confiance mathématique trop faible."
                
            # Message
            msg = (
                f"**{ticker}** ({symbol})\n"
                f"📊 {signal_type}\n"
                f"🎯 Probabilité : {max(prob_buy, 100-prob_buy):.1f}%\n"
                f"🛡️ Value at Risk (VaR 95%) : {var_95:.2f}%\n"
                f"⚙️ {action}"
            )
            
            # Envoyer l'alerte uniquement si le signal est fort
            if prediction == 1 and prob_buy > 60 or prediction == 0 and (100-prob_buy) > 60:
                send_alert(config, msg)
                
            time.sleep(2) # Anti-spam rate limit API
            
        except Exception as e:
            error_msg = f"❌ Erreur lors du traitement de {ticker}: {str(e)}"
            print(error_msg)
            # Ne pas spammer l'API avec des erreurs, juste imprimer en local

    send_alert(config, "✅ Session terminée. Le modèle est à jour et en veille.")

def start_scheduler():
    config = load_config()
    run_time = config.get("run_time", "22:00")
    
    print(f"🚀 Bot Headless démarré. Planification configurée à {run_time} chaque jour.")
    schedule.every().day.at(run_time).do(run_trading_job)
    
    # Envoyer une notification de démarrage
    send_alert(config, "🚀 Le Bot Headless XGBoost a été démarré avec succès. En attente de la clôture des marchés...")

    while True:
        # Recharger la config au cas où elle changerait via l'UI
        config = load_config()
        if not config.get("is_active", False):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Arrêt du Bot demandé par l'UI.")
            break
            
        schedule.run_pending()
        time.sleep(60) # Vérifie toutes les minutes

if __name__ == "__main__":
    start_scheduler()
