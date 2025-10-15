import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------
# CONFIGURACIÓN DEL CSV
# -----------------------------------
CSV_PATH = r"C:\Users\ivonn\Desktop\dashboard_maestro\data\alertas.csv"

# -----------------------------------
# MODELO SIMPLE DE CLASIFICACIÓN DE RIESGO
# -----------------------------------
class SimpleRiskModel(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=16, num_classes=3):
        super(SimpleRiskModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x).mean(dim=1)
        x = self.fc(x)
        return F.softmax(x, dim=1)

# Simulamos un modelo ya entrenado
model = SimpleRiskModel()
model.eval()

# -----------------------------------
# TOKENIZADOR SIMPLIFICADO
# -----------------------------------
def tokenize_text(text):
    # Convierte texto en índices numéricos simples
    tokens = [ord(c) % 1000 for c in text.lower()[:100]]  # limita longitud
    return torch.tensor(tokens).unsqueeze(0)

# -----------------------------------
# CARGA Y GUARDA DE CSV
# -----------------------------------
@st.cache_data(ttl=600)
def load_data(path=CSV_PATH):
    if not os.path.exists(path):
        st.warning(f"⚠️ No se encontró el archivo: {path}")
        return pd.DataFrame(columns=["texto", "riesgo"])
    try:
        return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except Exception as e:
        st.error(f"Error al leer el CSV: {e}")
        return pd.DataFrame(columns=["texto", "riesgo"])

def save_data(df, path=CSV_PATH):
    try:
        df.to_csv(path, index=False, encoding="utf-8")
        st.success("✅ Nueva alerta guardada correctamente.")
    except Exception as e:
        st.error(f"Error guardando el CSV: {e}")

# -----------------------------------
# DASHBOARD PRINCIPAL
# -----------------------------------
def show_telegram_dashboard():
    st.title("📡 Dashboard de Alertas Telegram")
    st.markdown("Monitoreo de amenazas y clasificación automática con PyTorch")

    df = load_data()

    if df.empty:
        st.warning("⚠️ No hay datos de alertas todavía.")
    else:
        # --- Vista previa y gráfico ---
        st.subheader("Vista previa de alertas cargadas")
        st.dataframe(df.tail(10))

        if "riesgo" in df.columns:
            st.subheader("Distribución de niveles de riesgo")
            conteo = df["riesgo"].value_counts().sort_index()

            fig, ax = plt.subplots()
            conteo.plot(kind="bar", ax=ax)
            ax.set_xlabel("Nivel de riesgo (0=bajo, 1=medio, 2=alto)")
            ax.set_ylabel("Número de alertas")
            ax.set_title("Distribución de alertas por nivel de riesgo")
            st.pyplot(fig)

    # --- Clasificación de nuevos mensajes ---
    st.subheader("🧠 Clasificador de riesgo con PyTorch")
    nuevo_texto = st.text_area("Escribe un nuevo mensaje o alerta detectada:")

    if st.button("Analizar mensaje"):
        if not nuevo_texto.strip():
            st.warning("Por favor, escribe un mensaje antes de analizarlo.")
            return

        # Tokenizar e inferir nivel de riesgo
        tokens = tokenize_text(nuevo_texto)
        with torch.no_grad():
            salida = model(tokens)
            nivel = int(torch.argmax(salida, dim=1).item())

        niveles = {0: "Bajo", 1: "Medio", 2: "Alto"}
        st.write(f"🔎 **Nivel de riesgo detectado:** {nivel} ({niveles[nivel]})")

        # Guardar en CSV
        nuevo_registro = pd.DataFrame([[nuevo_texto, nivel]], columns=["texto", "riesgo"])
        df_actualizado = pd.concat([df, nuevo_registro], ignore_index=True)
        save_data(df_actualizado)

        # Mostrar tabla actualizada
        st.subheader("📄 Alertas actualizadas")
        st.dataframe(df_actualizado.tail(10))

# -----------------------------------
# EJECUCIÓN LOCAL
# -----------------------------------
if __name__ == "__main__":
    show_telegram_dashboard()

