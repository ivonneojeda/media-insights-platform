# telegram_bot.py
import os
import pandas as pd
from datetime import datetime
from threading import Thread
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from model_utils import predict_risk

CSV_PATH = r"C:\Users\ivonn\Desktop\Alertas\data\telegram_alerts.csv"
BOT_TOKEN = os.getenv("BOT_TOKEN")

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
else:
    df = pd.DataFrame(columns=["timestamp", "canal", "mensaje", "riesgo"])

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global df
    texto = update.message.text
    canal = update.message.chat.title or update.message.chat.username or "Desconocido"
    riesgo_predicho, _ = predict_risk(texto)
    nueva_alerta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "canal": canal,
        "mensaje": texto,
        "riesgo": riesgo_predicho
    }
    df = pd.concat([df, pd.DataFrame([nueva_alerta])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

def start_telegram_bot():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    app.add_handler(handler)
    app.run_polling()

# Ejecutar en segundo plano
Thread(target=start_telegram_bot, daemon=True).start()
