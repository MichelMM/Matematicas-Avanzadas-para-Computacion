import re
import pandas as pd
#Size of the hash list
maxsize = 5000

#Node class for linked list in hash list implementation, contains string and points to next node
class WordNode:
    def __init__(self,word:str)->None:
        self.word = word
        self.next = None

    def __str__(self):
        return self.word

    def __repr__(self):
        return self.__str__()

#Get value for hashing, uses lowercase letters and extra letters are letters with punctuation that only are used on our 3 used languages, other letters added with different letters return a -1
def getValue(c):
    extraLetters = ['ñ','é','à', 'è', 'ù', 'â', 'ê', 'î', 'ô', 'û', 'ë', 'ï', 'ü', 'ÿ', 'ç', 'œ','æ']
    if ord(c) > 122:
        try :
            valid = extraLetters.index(c)
            return 26 + extraLetters.index(c)
        except ValueError :
            return -1
    return ord(c)-ord('a')

#Get hash value for a word, multiplies by the size of our letter vocabulary and does a sum with the iteration of the current letter, 
#use mod operation to keep hash value in range of our hash list size
def hashCode(word):
    h = getValue(word[0])
    for i in range(1,len(word)):
        h = (h*43 + getValue(word[i]))%maxsize
    return h

#Creates a hash list of maxSize (list of linked list) storing based on hashCode, if a value alredy exists in the current value, 
#it checks for any string equal tu the one being stored, if it's not the case, it gets stored at the end of the linked list
#If a hash value returns a negative value, means the searched word would not be stored on our hash list since it would not be a language specific word
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
#Get a count of our sample size by going in every element on the list that contains an element and iterating throught the inked list on it
def sizeHashList(hashList: list[WordNode]):
    count = 0
    for i in range(maxsize):
        if hashList[i]:
            curr = hashList[i]
            while curr:
                count+=1  
                curr = curr.next
    return count

#Search for a word to be contained in the hash list, if it exists, return True, if not False
def hashSearch(hashList: list[WordNode], word):
    hash = hashCode(word)
    if hashList[hash]:
        curr = hashList[hash]
        while curr:
            if word == curr.word:
                return True
            curr = curr.next
    return False


#Remove any punctuation and numbers from a text using regex
def remove_noise(text):
    reggex_punctuation = '([^\s\w])'
    reggex_numbers = '\d'
    x = re.sub(reggex_punctuation,"",text)
    x = re.sub(reggex_numbers,"",x)
    return x

#calls for remove noise, lowercase all words and finally splits it into a list
def text_to_list(text):
    text = remove_noise(text)
    words = text.casefold().split()
    return words


#Naive bayesian classifier with laplace smoothing, returns the posterior probability with max value
def naive_Bayesian_classifier(text,dictionary):
    #text to words
    words = text_to_list(text)

    #Name any relevant data that will be used in the classifier
    relevantData = {}
    size = "size"#Size of the set (hash list)
    class_prior = "class prior" #Probability of current set (hash list)
    laplace_smoothing = "laplace smoothing" #likelihood aplied with laplace smoothing
    posterior_probability = "posterior probability" # our search value to know wich of our sets is the more likely to be correct
    
    N = 0 #total size of our sets combined
    marginal_probability = 0 #total probability of observing a given Bag-of-Words vector across all classes

    #set new values to use as relevant data and set size in it
    for key in dictionary:
        relevantData[key]={
            size:sizeHashList(dictionary[key]),
            class_prior:0,
            laplace_smoothing:1,
            posterior_probability:0
        }
        N += relevantData[key][size]
    
    #set class prior to final values
    for key in relevantData:
        relevantData[key][class_prior] = relevantData[key][size] / N

    
    #Laplace Smoothing verifying if word is contained in our set (hash list) for every word on the text
    for word in words:
        for key in relevantData:
            exists = 0
            if hashSearch(dictionary[key],word):
                exists +=1
            relevantData[key][laplace_smoothing] *= (exists + 1)/(N+relevantData[key][size])
    
    #Marginal Probability
    for key in relevantData:
        marginal_probability += relevantData[key][laplace_smoothing]*relevantData[key][class_prior]
    
    #Posterior probability
    for key in relevantData:
        relevantData[key][posterior_probability] = (relevantData[key][laplace_smoothing]*relevantData[key][class_prior])/marginal_probability
        print(relevantData[key][posterior_probability])
    
    result = 0
    language = ""
    #Verify wich language has more probability to be the correct one
    for key in relevantData:
        if result < relevantData[key][posterior_probability]:
            result = relevantData[key][posterior_probability]
            language = key

    return language



#Matriz de confusión


#Gráfico de la superficie de Desición


dictionary = {
    "French":[],
    "English":[],
    "Spanish":[],
}

#extraxt values from csv file
df = pd.read_csv(r'./data/Language Detection.csv')

#Iterate throught every row on the csv and storing relevant ones (ones wich are english, spanish or french) on a respective list for its language
for i, row in df.iterrows():
    text = row["Text"]
    language = row["Language"]
    for key in dictionary:
        if language == key:
            words = text_to_list(text)
            for e in words:
                dictionary[key].append(e)
    
#Convert list to hash list
for key in dictionary:
    dictionary[key] = createHashList(dictionary[key])

#interfaz de usuario para recibir y procesar datos
phrases = [
    "The sun sets in the west.",
    "Friendship is a treasure.",
    "Dreams can come true.",
    "El sol se pone en el oeste.",
    "La amistad es un tesoro.",
    "Los sueños pueden hacerse realidad.",
    "Le soleil se couche à l'ouest.",
    "L'amitié est un trésor.",
    "Les rêves peuvent devenir réalité."
]

for e in phrases:
    print(naive_Bayesian_classifier(e,dictionary))
