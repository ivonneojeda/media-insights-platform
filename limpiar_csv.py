# limpiar_csv.py
import pandas as pd
import re

# --- Rutas ---
input_csv = "data/Conversación sobre UNAM 5-7oct25 - ISO.csv"
output_csv = "data/Conversacion_UNAM_limpio.csv"

# --- Cargar CSV ---
df = pd.read_csv(input_csv, encoding="utf-8", on_bad_lines="skip")

# --- Funciones de separación ---
def separar_hashtags(texto):
    if pd.isna(texto): return []
    return [t for t in re.split(r"[ ,;]+", texto) if t.startswith("#")]

def separar_keywords(texto):
    if pd.isna(texto): return []
    return [t for t in re.split(r"[ ,;]+", texto) if not t.startswith("#") and not t.startswith("@")]

# --- Aplicar separación ---
df['hashtags'] = df['keywords'].apply(separar_hashtags)
df['pure_keywords'] = df['keywords'].apply(separar_keywords)

# --- Revisar menciones ---
if 'mentions' not in df.columns:
    df['mentions'] = ""

# --- Guardar CSV limpio ---
df.to_csv(output_csv, index=False, encoding="utf-8")

print(f"✅ CSV limpio generado en: {output_csv}")
