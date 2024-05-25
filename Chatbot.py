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

# all libraries and packages.
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


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
        "what is your name": "I am a chat-bot created to assits you!"
    }
    tokens = tokenization(input_text)
    recog_message = None
    for response_key in reponses.keys():
        if input_text in response_key:
            recog_message = reponses[response_key]
    return recog_message if recog_message is not None else "I am sorry I dont understand !"


# result = response_system("what is your name")
# print(result)


# The above function  was the Simple rule based system,next we will build the Machine learning response system
# below.

# step 4-: Advanced NLP and spaCy function to define NER(named entity recognition) which is technique used to
# identify named entity in text like name of people, places ect. for that import spaCy adn load the spacy model.
def named_entity_recognition(text):
    nlp = spacy.load('en_core_web_sm')  # loads the small english model from spacy
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities


# nlp(text) processes the text using spaCy's model.
# doc.ents contains the recognized entities in the text.
# The function returns a list of tuples, each containing the entity text and its label (e.g., "India",
# "GPE" for geopolitical entity).

# print(named_entity_recognition("Tell me about New York"))

# Step 5-: Machine Learning for Responses
# function to train the model to generate the response , and using ml allows for more flexibility and adaptability.
# firstly import required libraries sklearn.

def get_ml_response(user_input):
    training_data = [
        ("Hello hii", "Hi there! How can I help you?"),
        ("How are you", "I'm a chatbot, so I'm always good. How about you?"),
        ("What is your name", "I'm a chatbot created to assist you."),
        ("movie movies", "sorry i dont like movies so i cant answer regarding that"),
        ("pokemon", "Pokemon is a animated cartoon !"),
    ]
    #     vectorize training data
    training_text, training_response = zip(*training_data)  # separates the user inputs and responses.
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(training_text)  # converts the user inputs to TF-IDF features.

    #     train the model-:
    model = MultinomialNB()
    model.fit(x_train,
              training_response)  # The model is trained using the TF-IDF features and the corresponding responses.

    # to get the response-:
    x_test = vectorizer.transform([user_input])  # converts the user input to TF-IDF features.
    prediction = model.predict(x_test)  # predicts the response based on the input features
    return prediction[0]


# print(get_ml_response("do you like toy story 2 movie"))

# step 6-: Integrate all functions-:
# combine rule based and ml resposnes
def chatbot_response(user_input):
    response = response_system(user_input)
    if response == "I am sorry I dont understand !":
        response = get_ml_response(user_input)
    return response


# The function first tries to get a rule-based response.
# If the rule-based response is not satisfactory, it uses the machine learning model to generate a response.

if __name__ == "__main__":
    while True:
        user_input = input("YOU: ")
        if user_input.lower() in ["bye", "quit", "exit", "goodbye"]:
            print("chatbot:Goodbye! have a great day!")
            break
        print("Chatbot: ", chatbot_response(user_input))
