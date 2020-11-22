import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras

df = pd.read_csv('jokes.csv')
data = np.array(df['Question'] + ' ' + df['Answer'])

print("Getting chars...")
char2int = {}
counter = 1
for text in data:
    text = text.encode('unicode-escape').decode('utf-8')
    for char in text:
        if char not in char2int.keys():
            char2int[char] = counter
            counter += 1
unique_chars = len(char2int.keys())+1
int2char = {value: key for key, value in char2int.items()}

def encode_text(text):
    new_text = []
    for char in text:
        new_text.append(char2int[char])
    return np.array(new_text)

def sequence_text(text):
    sequence_len = 5
    current_state = 0
    X = []
    y = []
    while current_state+sequence_len < len(text):
        X.append(text[current_state:current_state+sequence_len])
        y.append(text[current_state+sequence_len])
        current_state += 1
    return (X, y)

data = [encode_text(text.encode('unicode-escape').decode('utf-8')) for text in data]
X = []
y = []
for text in data:
    text_sequence = sequence_text(text)
    for x_one in text_sequence[0]:
        x_one = np.array(x_one)
        x_one = x_one.reshape(x_one.shape[0],1)
        X.append(x_one)
    for y_one in text_sequence[1]:
        y_one = np.array(y_one)
        y.append(y_one)
X = np.array(X)
y = np.array(y)


model = keras.Sequential()
model.add(keras.layers.LSTM(256, input_shape=X[0].shape, return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(256))
model.add(keras.layers.Dense(unique_chars, activation='softmax'))

model.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy')
model.fit(X, y, epochs=4, batch_size=9000)

model.save_weights("joke_model")
model.load_weights("joke_model")


while True:
    input_sent = input("Enter words of joke: ")
    input_amount = int(input("Enter amount of characters here: "))
    encoded_text = list(encode_text(input_sent))
    new_sentence = input_sent
    for i in range(input_amount):
        newText = np.array(encoded_text)
        newText = newText.reshape(1, newText.shape[0], 1)
        thePred = int2char[np.argmax(model.predict(newText))]
        encoded_text.append(char2int[thePred])
        new_sentence += thePred
    print(new_sentence)
        
   
