import re
import pandas as pd
maxsize = 5000

class WordNode:
    def __init__(self,word:str)->None:
        self.word = word
        self.next = None

    def __str__(self):
        return self.word

    def __repr__(self):
        return self.__str__()

def getValue(c):
    accentLetters = ['ñ','é','à', 'è', 'ù', 'â', 'ê', 'î', 'ô', 'û', 'ë', 'ï', 'ü', 'ÿ', 'ç', 'œ','æ']
    if ord(c) > 122:
        try :
            valid = accentLetters.index(c)
            return 26 + accentLetters.index(c)
        except ValueError :
            return -1
    return ord(c)-ord('a')



def hashCode(word):
    h = getValue(word[0])
    for i in range(1,len(word)):
        h = (h*43 + getValue(word[i]))%maxsize
    return h

def createHashList(words:list[str]):
    hashList = [None]*maxsize
    for c in words:
        hash = hashCode(c)
        if hash < 0:
            continue
        if not hashList[hash]:
            hashList[hash] = WordNode(c)
        else:
            curr = hashList[hash]
            repeated = 0
            while curr.next:
                if c == curr.word:
                    repeated+=1
                curr = curr.next
            if c == curr.word:
                repeated+=1
            if repeated == 0:
                curr.next = WordNode(c)
    return hashList

def sizeHashList(hashList: list[WordNode]):
    count = 0
    for i in range(maxsize):
        if hashList[i]:
            curr = hashList[i]
            while curr:
                count+=1  
                curr = curr.next
    return count

def hashSearch(hashList: list[WordNode], word):
    hash = hashCode(word)
    if hashList[hash]:
        curr = hashList[hash]
        while curr:
            if word == curr.word:
                return True
            curr = curr.next
    return False



def remove_noise(text):
    reggex_punctuation = '([^\s\w])'
    reggex_numbers = '\d'
    x = re.sub(reggex_punctuation,"",text)
    x = re.sub(reggex_numbers,"",x)
    return x

def text_to_list(text):
    text = remove_noise(text)
    words = text.casefold().split()
    return words


#Bayes con suavizado de laplace, regresar el resultado mayor


#Matriz de confusión


#Gráfico de la superficie de Desición


dictionary = {
    "French":[],
    "English":[],
    "Spanish":[],
}

df = pd.read_csv(r'./data/Language Detection.csv')

for i, row in df.iterrows():
    text = row["Text"]
    language = row["Language"]
    for key in dictionary:
        if language == key:
            words = text_to_list(text)
            for e in words:
                dictionary[key].append(e)
    

for key in dictionary:
    dictionary[key] = createHashList(dictionary[key])

#interfaz de usuario para recibir y procesar datos
