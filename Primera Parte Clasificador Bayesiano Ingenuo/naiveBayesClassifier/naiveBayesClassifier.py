import re
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

maxsize = 5000
test_size = 0.1

path = './data/languageDetection.csv'
df = pd.read_csv(path)

class WordNode:
    """
    Node class for linked list in hash list implementation, 
    contains string and points to next node
    """
    def __init__(self,word:str)->None:
        self.word = word
        self.next = None

    def __str__(self):
        return self.word

    def __repr__(self):
        return self.__str__()

def getValue(c):
    """
    Get value for hashing, uses lowercase letters and 
    extra letters are letters with punctuation 
    that only are used on our 3 used languages,
    other letters added with different letters return a -1
    Args:
        c (str): letter to get hash value

    Returns:
        int: hash value
    """
    accentLetters = ['ñ','é','à', 'è', 'ù', 'â', 'ê', 'î', 'ô', 'û', 'ë', 'ï', 'ü', 'ÿ', 'ç', 'œ','æ']
    if ord(c) > 122:
        try :
            valid = accentLetters.index(c)
            return 26 + accentLetters.index(c)
        except ValueError :
            return -1
    return ord(c)-ord('a')



def hashCode(word):
    """
    Get hash value for a word, multiplies by the size o
    f our letter vocabulary and does a sum with the iteration of the current letter, 
    use mod operation to keep hash value in range of our hash list size

    Args:
        word (str): word to be haashed

    Returns:
        h(int): hashedValue
    """
    h = getValue(word[0])
    for i in range(1,len(word)):
        h = (h*43 + getValue(word[i]))%maxsize
    return h

def createHashList(words:list[str]):
    """
    Creates a hash list (list of linked lists) to store elements based on hash codes. 
    If a value exists, it checks for duplicates and appends if none are found.

    Negative hash values indicate the word won't be stored, as it likely 
    isn't language-specific.
    Args:
        words (list[str]): list of words to be hashed.

    Returns:
        hashList(list): list of hashed values.
    """
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
    """
    Count the number of elements of the hashed list

    Args:
        hashList (list[WordNode]): hashedList to be counted

    Returns:
        count (int): size of the list
    """
    count = 0
    for i in range(maxsize):
        if hashList[i]:
            curr = hashList[i]
            while curr:
                count+=1  
                curr = curr.next
    return count

def hashSearch(hashList: list[WordNode], word):
    """
    Search for a word to be contained in the hash list, 
    if it exists, return True, if not False

    Args:
        hashList (list[WordNode]): hashed list to search a value.
        word (str): word to search on the list.

    Returns:
        (bool) 
    """
    hash = hashCode(word)
    if hashList[hash]:
        curr = hashList[hash]
        while curr:
            if word == curr.word:
                return True
            curr = curr.next
    return False



def remove_noise(text):
    """
    Remove any punctuation and numbers 
    from a text using regex.

    Args:
        text (int): text to be cleaned

    Returns:
        x(str): cleaned text
    """
    reggex_punctuation = '([^\s\w])'
    reggex_numbers = '\d'
    x = re.sub(reggex_punctuation,"",text)
    x = re.sub(reggex_numbers,"",x)
    return x

def text_to_list(text):
    """
    Calls for remove noise, lowercase all words 
    and finally splits it into a list.

    Args:
        text (str): text to be splitted.

    Returns:
        words(list): list of splitted words.
    """
    text = remove_noise(text)
    words = text.casefold().split()
    return words


class naivebayesClassifier:
    """
    Naive bayesian classifier 
    with laplace smoothing, returns t
    he posterior probability with max value
    """
    def __init__(self, dictionary) -> None:
        self.dictionary = dictionary
        self.relevantData = {}
        self.size = "size"
        self.class_prior = "class prior"
        self.laplace_smoothing = "laplace smoothing"
        self.posterior_probability = "posterior probability"
        self.N = 0
        self.marginal_probability = 0
        
    def generateProbabilityDict(self):
        """
        Generate a probability dictionary
        using a given dictionary.
        """

        #set new values to use as relevant data and set size in it
        for key in self.dictionary:
            self.relevantData[key]={
                self.size:sizeHashList(self.dictionary[key]),
                self.class_prior:0,
                self.laplace_smoothing:1,
                self.posterior_probability:0
            }
            self.N += self.relevantData[key][self.size]
            
        for key in self.relevantData:
            self.relevantData[key][self.class_prior] = \
                self.relevantData[key][self.size] / self.N

    def predict(self, text):
        """
        Predict in wich language the text is.

        Args:
            text (str): text to be recognized.

        Returns:
            language(str): Language of the text.
        """

        tmpRelevantData = deepcopy(self.relevantData)

        words = text_to_list(text)

        for word in words:
            for key in tmpRelevantData:
                exists = 0
                if hashSearch(self.dictionary[key],word):
                    exists +=1
                tmpRelevantData[key][self.laplace_smoothing] *= \
                            (exists + 1)/(self.N+tmpRelevantData[key][self.size])
        #Marginal Probability
        for key in tmpRelevantData:
            self.marginal_probability += tmpRelevantData[key][self.laplace_smoothing]*\
                        tmpRelevantData[key][self.class_prior]
        
        #Posterior probability
        for key in tmpRelevantData:
            tmpRelevantData[key][self.posterior_probability] =\
                                    (tmpRelevantData[key][self.laplace_smoothing]* \
                                    tmpRelevantData[key][self.class_prior])/self.marginal_probability
    
        result = 0
        language = ""
        for key in tmpRelevantData:
            if result < tmpRelevantData[key][self.posterior_probability]:
                result = tmpRelevantData[key][self.posterior_probability]
                language = key

        return language
        
        

def split_data(diccionary):
    """
    Split the data in train and test samples.

    Args:
        diccionary (dict): dictionari to be splitted.

    Returns:
        tuple: tuple that contains splitted samples.
    """
    train_dictionary = test_dictionary = {key: [] for key in dictionary.keys()}
     
    for key in dictionary.keys():
        tmp_list = dictionary[key]
        size = len(tmp_list)
        test_samples_size = round(size * test_size)
        test_samples = tmp_list[:test_samples_size]
        train_samples = [text for text in tmp_list if text not in test_samples]
        train_dictionary[key] = train_samples
        test_dictionary[key] = test_samples

    return train_dictionary, test_dictionary


def get_predictions(df):
    """
    Functions to predict on the app.

    Args:
        df (dataframe): dataframe to train model.
    """

    for i, row in df.iterrows():
        text = row["Text"]
        language = row["Language"]
        for key in dictionary:
            if language == key:
                words = text_to_list(text)
                for e in words:
                    dictionary[key].append(e)

    train_dictionary, test_dictionary = split_data(diccionary=dictionary)

    hashed_dictionary = {}

    for key in train_dictionary:
        hashed_dictionary[key] = createHashList(train_dictionary[key])

    nbc = naivebayesClassifier(hashed_dictionary)
    nbc.generateProbabilityDict()
    return nbc
    


#Matriz de confusión
def generateConfusionMatrix(test_dict, model):
    """
    Generates a confusion matrix plot 
    for a given classifier model.

    Args:
        test_dict (dict): test sampled dictionary
        model (object): model to be evaluated.
    """
    new_test_dict = {}

    for language, words in test_dict.items():
        for word in words:
            new_test_dict[word] = language

    true_values = [new_test_dict[text] for text in new_test_dict.keys()]
    predicted_values = [model.predict(text) for text 
                        in new_test_dict.keys()]

    labels = ['French', 'English', 'Spanish']

    confusion_matrix = [[0 for i in range(3)] for i in range(3)]

    for true, pred in zip(true_values, predicted_values):
        true_index = labels.index(true)
        pred_index = labels.index(pred)
        confusion_matrix[true_index][pred_index] += 1

    plt.figure(figsize=(20,15))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('True Languaje')
    plt.xlabel('Predicted Languaje')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.pdf', format='pdf')


#Gráfico de la superficie de Desición

def generateSurfaceDecisionPlot():
    """
    Generate a surface sample plot.
    """
    pass

if __name__ == '__main__':
    
    dictionary = {
        "French":[],
        "English":[],
        "Spanish":[],
    }

    path = './data/languageDetection.csv'

    df = pd.read_csv(path)

    for i, row in df.iterrows():
        text = row["Text"]
        language = row["Language"]
        for key in dictionary:
            if language == key:
                words = text_to_list(text)
                for e in words:
                    dictionary[key].append(e)

    train_dictionary, test_dictionary = split_data(diccionary=dictionary)

    hashed_dictionary = {}

    for key in train_dictionary:
        hashed_dictionary[key] = createHashList(train_dictionary[key])

    nbc = naivebayesClassifier(hashed_dictionary)
    nbc.generateProbabilityDict()

    generateConfusionMatrix(test_dictionary, nbc)
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
        print(e, nbc.predict(e))
