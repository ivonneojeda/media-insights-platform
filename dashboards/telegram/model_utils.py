# model_utils.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Cargar modelo directamente desde Hugging Face
MODEL_PATH = "Ivonne333/alertas_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict_risk(text):
    """
    Predice el nivel de riesgo de un texto.
    Retorna la clase con mayor probabilidad y un diccionario con todas las probabilidades.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    classes = ["Bajo", "Medio", "Alto"]
    prob_dict = {classes[i]: float(probs[0][i]) for i in range(len(classes))}
    pred_class = max(prob_dict, key=prob_dict.get)
    
    return pred_class, prob_dict
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser


