import subprocess
import sys

# Lista de librerías requeridas
required_packages = [
    "streamlit",
    "pandas",
    "plotly",
    "networkx",
    "prophet"
]

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for package in required_packages:
    try:
        __import__(package)
        print(f"✅ {package} ya está instalado")
    except ImportError:
        print(f"⚠️ {package} no está instalado. Instalando...")
        install(package)

print("\nTodas las librerías necesarias están listas.")
