# This is  chatbot which is build using NLP(natural language processing) libraries like NLTK(natural language toolkit) ,
# spaCy ect.

# first step is to install all the necessary packages and modules in the python terminal.
# Install Necessary Libraries:
#
# nltk: For basic NLP tasks.
# spacy: For advanced NLP tasks and named entity recognition.
# scikit-learn: For training machine learning models.
# requests: If you want the chatbot to fetch information from external APIs.
# You can install these libraries using pip:

# pip install nltk spacy scikit-learn requests

# Download NLTK Data:
# NLTK requires some additional data packages. You can download them using the following commands in a Python shell:

# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# to install this simply in the terminal type python then the python shell will open then there simply install
# its packages one by one. by writing nltk.download('punkt')and others.


# Download spaCy Model:
# spaCy requires a language model. You can download it using the following command in a terminal:
# command-:  python -m spacy download en_core_web_sm
# to do this installation simply write this command in the terminal.


# now Step-2 (Data preprocessing)  to define function Tokenization for this firstly import these libraries.

# Tokenization is the process of splitting text into individual words or tokens. This is an essential step in text
# processing as it allows us to analyze the text more effectively.

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string


# word_tokenize is a function from NLTK that splits text into words.
# stopwords is a list of common words (like "and", "the", etc.) that we often want to ignore in text analysis.
# string provides a list of punctuation characters.

def tokenization(text):
    token = word_tokenize(text.lower())
    token = [word for word in token if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    token = [word for word in token if word not in stop_words]
    return token


# text.lower() converts the text to lowercase to ensure uniformity.
# word_tokenize(text.lower()) splits the text into tokens (words).
# [word for word in tokens if word.isalnum()] removes any tokens that are not alphanumeric (i.e.,it removes punctuation)
# stopwords.words('english') provides a list of common English stop words.
# [word for word in tokens if word not in stop_words] removes stop words from the list of tokens.

# example_text = "Hello! there"
# print(tokenization(example_text))

# Step 3-: Building simple response system.
# In this step we are going to predefine some of the responses for specific's inputs which are very common .


def response_system(input_text):
    reponses = {
        "hello": "Hi! there how can i help you today?",
        "how are you": "I am chat-bot I am always good!",
        "good morning": "Good Morning dear how can i help you today?",
        "bye Goodbye": "Bye dear see you soon!",
        "what is your name": "I am a chat-bot created to assits you!"
    }
    tokens = tokenization(input_text)
    recog_message = None
    for response_key in reponses.keys():
        if input_text in response_key:
            recog_message = reponses[response_key]
    return recog_message if recog_message is not None else "I am sorry I dont understand !"


result = response_system("what is your name")
print(result)

# The above function or part was the Simple rule based system,next we will build the Machine learning response system
# below.
