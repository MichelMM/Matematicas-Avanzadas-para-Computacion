import re
import pandas as pd

def remove_noise(text):
    reggex_punctuation = '([^\s\w])'
    reggex_numbers = '\d'
    x = re.sub(reggex_punctuation,"",text)
    x = re.sub(reggex_numbers,"",x)
    return x

def text_to_set(currSet,text):
    text = remove_noise(text)
    words = text.split()
    for e in words:
        currSet.add(e)
    return currSet

#Bayes con suavizado de laplace, regresar el resultado mayor


#Matriz de confusión


#Gráfico de la superficie de Desición


#Transformación de csv a diccionario almacenado en Sets
dictionary = {
    "French":set(),
    "English":set(),
    "Spanish":set(),
}

df = pd.read_csv(r'./data/Language Detection.csv')

for i, row in df.iterrows():
    text = row["Text"]
    language = row["Language"]
    for key in dictionary:
        if language == key:
            text_to_set(dictionary[key],text)
#transformar set a lista, Ordenamiento de palabras en cada disccionario


#interfaz de usuario para recibir y procesar datos




