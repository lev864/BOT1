from flask import Flask, request, jsonify
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Define paths
MODEL_PATH = 'model.h5'
TOKENIZER_PATH = 'tokenizer.pkl'
LBL_ENCODER_PATH = 'label_encoder.pkl'

# Load intents data
with open('intents.json', 'r') as f:
    data = json.load(f)

@app.route('/train', methods=['POST'])
def train():
    global data
    # Prepare the dataset
    tags, patterns, responses = [], [], []
    for intent in data['intents']:
        for pattern in intent['patterns']:
            patterns.append(pattern)
            tags.append(intent['tag'])
        responses.append(intent['responses'])

    # Encode the tags
    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(tags)
    labels = lbl_encoder.transform(tags)

    # Tokenize the patterns
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(patterns)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(patterns)
    padded_sequences = pad_sequences(sequences, padding='post')

    # Train-test split
    train_X, val_X, train_y, val_y = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

    # Build the model
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, 64, input_length=padded_sequences.shape[1]))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(set(labels)), activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train the model
    model.fit(train_X, train_y, epochs=10, validation_data=(val_X, val_y))

    # Save the model
    model.save("BOT1\model.h5")

    # Save the tokenizer and label encoder
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(LBL_ENCODER_PATH, "wb") as f:
        pickle.dump(lbl_encoder, f)

    return jsonify({"message": "Training completed and model saved."})

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')

    # Load the trained model, tokenizer, and label encoder
    model = load_model("BOT1\model.h5")
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(LBL_ENCODER_PATH, 'rb') as f:
        lbl_encoder = pickle.load(f)

    sequence = tokenizer.texts_to_sequences([user_message])
    padded_sequence = pad_sequences(sequence, maxlen=model.input_shape[1], padding='post')
    predicted_label = np.argmax(model.predict(padded_sequence), axis=1)
    tag = lbl_encoder.inverse_transform(predicted_label)[0]

    # Find the response
    response = next((resp for intent in data['intents'] if intent['tag'] == tag for resp in intent['responses']), "Sorry, I don't understand that.")
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
