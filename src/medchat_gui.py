import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras import models as keras_models
import json
import random
import tkinter as tk
from tkinter import *

# Initialize the WordNet lemmatizer for normalizing words
word_normalizer = WordNetLemmatizer()

# Load the trained chatbot model and related data
chatbot_model = keras_models.load_model('models/model.h5')
data = json.loads(open('data/prompts_n_responses.json').read())
vocab = pickle.load(open('data/tokens.pkl', 'rb'))
categories = pickle.load(open('data/labels.pkl', 'rb'))

def normalize_sentence(sentence):
    """Tokenize and lemmatize the input sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [word_normalizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, vocab, show_details=True):
    """Convert the input sentence into a bag of words array."""
    sentence_words = normalize_sentence(sentence)
    word_bag = [0] * len(vocab)
    for word in sentence_words:
        for i, vocab_word in enumerate(vocab):
            if vocab_word == word:
                word_bag[i] = 1
                if show_details:
                    print(f"Found in bag: {vocab_word}")
    return np.array(word_bag)

def classify_target(sentence, model):
    """Predict the target class of the input sentence."""
    word_bag = bag_of_words(sentence, vocab, show_details=False)
    predictions = model.predict(np.array([word_bag]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, prob] for i, prob in enumerate(predictions) if prob > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for result in results:
        return_list.append({"target": categories[result[0]], "probability": str(result[1])})
    return return_list

def generate_response(predictions, data_json):
    """Generate a response based on the predicted target."""
    target_tag = predictions[0]['target']
    target_list = data_json['targets']
    for target in target_list:
        if target['tag'] == target_tag:
            response = random.choice(target['responses'])
            break
    return response

def chat_reply(message):
    """Generate a response from the chatbot."""
    predictions = classify_target(message, chatbot_model)
    response = generate_response(predictions, data)
    return response

def send_message():
    """Handle sending messages in the GUI."""
    user_message = entry_box.get("1.0", 'end-1c').strip()
    entry_box.delete("0.0", END)

    if user_message != '':
        chat_log.config(state=NORMAL)
        chat_log.insert(END, "You: " + user_message + '\n\n')
        chat_log.config(foreground="#442265", font=("Verdana", 12))

        bot_reply = chat_reply(user_message)
        chat_log.insert(END, "Bot: " + bot_reply + '\n\n')
        
        chat_log.config(state=DISABLED)
        chat_log.yview(END)

# Create GUI window
app_window = Tk()
app_window.title("MedChat")
app_window.geometry("400x500")
app_window.resizable(width=FALSE, height=FALSE)

# Set the window icon
app_window.iconbitmap('MedChat_icon.jpeg')  

# Create chat window
chat_log = Text(app_window, bd=0, bg="white", height="8", width="50", font="Arial")
chat_log.config(state=DISABLED)

# Bind scrollbar to chat window
scrollbar = Scrollbar(app_window, command=chat_log.yview, cursor="heart")
chat_log['yscrollcommand'] = scrollbar.set

# Create button to send message
send_button = Button(app_window, font=("Times", 12, 'bold','italic'), text="Send", width="12", height=5,
                     bd=0, bg="#13162e", activebackground="#262105", fg='#ffffff',
                     command=send_message)

# Create entry box for user input
entry_box = Text(app_window, bd=0, bg="white", width="29", height="5", font="Arial")

# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
chat_log.place(x=6, y=6, height=386, width=370)
entry_box.place(x=128, y=401, height=90, width=265)
send_button.place(x=6, y=401, height=90)

# Start the GUI main loop
app_window.mainloop()
