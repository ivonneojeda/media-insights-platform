import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# URL del modelo en HuggingFace
MODEL_NAME = "Ivonne333/alertas_model"

# Inicialización segura del modelo y tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"No se pudo cargar el modelo de HuggingFace: {e}")
    tokenizer = None
    model = None

def predict_risk(mensaje):
    """
    Predice el nivel de riesgo de un mensaje.
    Devuelve: (riesgo_predicho, probabilidades)
    """
    if not mensaje or not model or not tokenizer:
        return "Desconocido", None

    try:
        # Tokenización
        inputs = tokenizer(mensaje, return_tensors="pt", truncation=True, padding=True)

        # Inferencia sin gradientes
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            riesgo_idx = torch.argmax(probs, dim=-1).item()

        # Mapear índice a etiqueta (asume labels: 0='Bajo', 1='Medio', 2='Alto')
        labels = ["Bajo", "Medio", "Alto"]
        riesgo_predicho = labels[riesgo_idx] if riesgo_idx < len(labels) else "Desconocido"

        return riesgo_predicho, probs.tolist()

    except Exception as e:
        print(f"Error en predict_risk: {e}")
        return "Desconocido", None
