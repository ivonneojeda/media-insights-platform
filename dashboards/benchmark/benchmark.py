# benchmark.py
import os
import pandas as pd
import dash

from benchmark_layout import create_benchmark_layout

# --- 1. Cargar CSVs ---
folder_path = r"C:\Users\ivonn\Desktop\Proyecto_benchmark\datos"
files = [
    "ipn_social_growth_jul_sep_2025.csv",
    "tec_social_growth_jul_sep_2025.csv",
    "uam_social_growth_jul_sep_2025.csv",
    "udg_social_growth_jul_sep_2025.csv",
    "unam_social_growth_jul_sep_2025.csv"
]

data_dict = {}
for file in files:
    uni_name = os.path.basename(file).split("_")[0].upper()
    df = pd.read_csv(os.path.join(folder_path, file))
    df['date'] = pd.to_datetime(df['date'])
    data_dict[uni_name] = df

# --- 2. Crear Dash app ---
app = dash.Dash(__name__)
app = create_benchmark_layout(app, data_dict)

# --- 3. Ejecutar app ---
if __name__ == "__main__":
    app.run_server(debug=False, use_reloader=False)
