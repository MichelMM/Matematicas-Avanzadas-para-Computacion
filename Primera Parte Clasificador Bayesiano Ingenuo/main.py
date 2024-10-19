# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from naiveBayesClassifier.naiveBayesClassifier import get_predictions
import pandas as pd

app = FastAPI()

path = './naiveBayesClassifier/data/languageDetection.csv'
df = pd.read_csv(path)
predictor = get_predictions(df)

# Definir un modelo Pydantic para la entrada
class PalabraInput(BaseModel):
    palabra: str

@app.post("/identificar")
def detectar_idioma(input: PalabraInput):
    idioma = predictor(input.palabra)
    return {
        "idioma": idioma,
        #"probabilidad": f"{probabilidad * 100:.2f}%"  # Convertimos la probabilidad a porcentaje
    }
