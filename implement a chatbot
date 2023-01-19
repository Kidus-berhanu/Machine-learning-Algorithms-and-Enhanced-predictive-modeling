#kidus Berhanu
import nltk
from nltk.chat import eliza
from nltk.classify import NaiveBayesClassifier

#  to Define the training data
#  and Incorporate more training data
training_data = [("Apple", "fruit"),
                 ("Banana", "fruit"),
                 ("Mango", "fruit"),
                 ("Watermelon", "fruit"),
                 ("Carrot", "vegetable"),
                 ("Potato", "vegetable"),
                 ("Onion", "vegetable"),
                 ("Tomato", "vegetable"),
                 ("Grapes", "fruit"),
                 ("Broccoli", "vegetable"),
                 ("Celery", "vegetable"),
                 ("Cucumber", "vegetable"),
                 ("Lemon", "fruit"),
                 ("Lime", "fruit"),
                 ("Pineapple", "fruit"),
                 ("Strawberries", "fruit")]

#  to Define the features function
def extract_features(word):
    return {"word": word}

#  for Creating the classifier
classifier = NaiveBayesClassifier.train([(extract_features(n), category) for (n, category) in training_data])

#  to Define the conversation function
def conversation(input_text):
    # Use the Eliza chatbot to respond to the input
    response = eliza.respond(input_text)
    print("Chatbot: " + response)

#  to Define the fruit classification function
def classify_fruit(input_text):
    fruit_name = input_text.split(" ")[-1]
    fruit_class = classifier.classify(extract_features(fruit_name))
    print("Chatbot: The "+fruit_name+" is a "+fruit_class)

#  to Define the vegetable classification function
def classify_vegetable(input_text):
    vegetable_name = input_text.split(" ")[-1]
    vegetable_class = classifier.classify(extract_features(vegetable_name))
    print("Chatbot: The "+vegetable_name+" is a "+vegetable_class)

#  to Define the main function that routes the user's input to the correct function
def chatbot(input_text):
    input_text = input_text.lower()
    if "fruit" in input_text:
        classify_fruit(input_text)
    elif "vegetable" in input_text:
        classify_vegetable(input_text)
    elif "exit" in input_text:
        print("Chatbot: Goodbye!")
        exit()
    else:
        conversation(input_text)

#  for using the chatbot
print("Chatbot: Hello, how can I help you today?")
while True:
    user_input = input("You: ")
    chatbot(user_input)
