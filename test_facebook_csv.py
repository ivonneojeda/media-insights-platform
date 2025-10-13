import os
import pandas as pd

DATA_PATH = r"C:\Users\ivonn\Desktop\dashboard_maestro\data\sentimiento_2025-09-30_22-00-03.csv"

print(f"📂 Probando ruta: {DATA_PATH}")
print("¿Existe el archivo?:", os.path.exists(DATA_PATH))

if os.path.exists(DATA_PATH):
    try:
        df = pd.read_csv(DATA_PATH)
        print("✅ Archivo cargado correctamente")
        print("Filas:", len(df))
        print("Columnas:", list(df.columns))
        print("\nPrimeras filas:")
        print(df.head())
    except Exception as e:
        print("⚠️ Error al leer CSV:", e)
