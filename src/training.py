import nltk
from nltk.stem import WordNetLemmatizer
from keras import models as keras_models
from keras import layers as keras_layers
from keras import optimizers as keras_optimizers
import json
import pickle
import numpy as np
import os

# Disable oneDNN custom operations to avoid numerical discrepancies
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Download required NLTK data packages
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the WordNet lemmatizer for normalizing words
word_normalizer = WordNetLemmatizer()

# Load the data from the JSON file
with open('data/prompts_n_responses.json') as file:
    data = json.load(file)

# Initialize lists for vocabulary, categories, and training documents
vocab = []
categories = []
training_docs = []
ignore_tokens = ['?', '!', '.', ',']

# Process each target and its patterns
for target in data['targets']:
    for phrase in target['patterns']:
        # Tokenize each word in the phrase
        tokens = nltk.word_tokenize(phrase)
        vocab.extend(tokens)
        # Add the tokenized phrase and its tag to the training documents
        training_docs.append((tokens, target['tag']))
        # Add the tag to categories if not already present
        if target['tag'] not in categories:
            categories.append(target['tag'])

# Lemmatize, lowercase, and remove duplicates from the vocabulary
vocab = [word_normalizer.lemmatize(token.lower()) for token in vocab if token not in ignore_tokens]
vocab = sorted(set(vocab))

# Sort the list of categories (classes)
categories = sorted(set(categories))

# Save the processed vocabulary and categories
pickle.dump(vocab, open('data/tokens.pkl', 'wb'))
pickle.dump(categories, open('data/labels.pkl', 'wb'))

# Prepare the training data for the neural network
training_data = []
empty_output = [0] * len(categories)

for doc in training_docs:
    word_bag = []
    tokenized_words = doc[0]
    tokenized_words = [word_normalizer.lemmatize(word.lower()) for word in tokenized_words]
    
    # Create a word bag: 1 if word exists in vocabulary, otherwise 0
    for word in vocab:
        word_bag.append(1) if word in tokenized_words else word_bag.append(0)

    # Create an output row: 1 for the current tag, 0 for others
    output_row = list(empty_output)
    output_row[categories.index(doc[1])] = 1

    # Add the word bag and corresponding output to training data
    training_data.append([word_bag, output_row])

# Convert training data into NumPy arrays for model training
train_x = np.array([np.array(item[0]) for item in training_data])
train_y = np.array([np.array(item[1]) for item in training_data])

# Define the Sequential model structure
chatbot_model = keras_models.Sequential()
chatbot_model.add(keras_layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
chatbot_model.add(keras_layers.Dropout(0.5))
chatbot_model.add(keras_layers.Dense(64, activation='relu'))
chatbot_model.add(keras_layers.Dropout(0.5))
chatbot_model.add(keras_layers.Dense(len(train_y[0]), activation='softmax'))

# Compile the model with Stochastic Gradient Descent optimizer
sgd_optimizer = keras_optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
chatbot_model.compile(loss='categorical_crossentropy', optimizer=sgd_optimizer, metrics=['accuracy'])

# Train the model on the training data
training_history = chatbot_model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the trained model
chatbot_model.save('models/model.h5')

print("Model created and saved as 'models/model.h5'")
