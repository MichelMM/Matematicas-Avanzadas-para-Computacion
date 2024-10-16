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
        
dictionary = {
    "French":set(),
    "English":set(),
    "Spanish":set(),
}

df = pd.read_csv(r'./text/Language Detection.csv')

for i, row in df.iterrows():
    text = row["Text"]
    language = row["Language"]
    for key in dictionary:
        if language == key:
            text_to_set(dictionary[key],text)
