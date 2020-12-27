import numpy as np
from keras import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Embedding, LeakyReLU, GlobalMaxPooling1D

with open('HungerGames.txt', 'r') as the_file:
    all_lines = the_file.readlines()

data = ''
for line in all_lines:
    line = line.rstrip('\n')
    data += line + ' '

unique_chars = np.unique(' '.join(data).split())
char2int = {char: index+1 for index, char in enumerate(unique_chars)}
char2int[' '] = len(char2int.keys())+1
int2char = {index: char for char, index in char2int.items()}
len_keys = len(char2int.keys())+1

data_vector = [char2int[char] for char in data]


SEQUENCE_LEN = 40
X = []
y = []
for i in range(len(data_vector)-SEQUENCE_LEN):
    X.append(data_vector[i:i+SEQUENCE_LEN])
    y.append(data_vector[i+SEQUENCE_LEN])

X = np.array(X)
y = np.array(y).reshape(-1,1)

model = Sequential()
model.add(Embedding(input_dim=len_keys, output_dim=16,
                    input_length=SEQUENCE_LEN))
for i in range(3):
    model.add(Conv1D(256, kernel_size=5, padding='same'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.25))
model.add(GlobalMaxPooling1D())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(len_keys, activation='softmax'))

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X, y, epochs=16, batch_size=400)

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.ravel(preds)
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(amount_char):
    output_vector = data_vector[:SEQUENCE_LEN]
    for i in range(amount_char):
        pred_vector = output_vector[-SEQUENCE_LEN:]
        prediction = model.predict(np.array(pred_vector).reshape(1, SEQUENCE_LEN, 1))
        output_vector.append(sample(prediction, 0.3))

    output_sentence = ''
    for num in output_vector:
        output_sentence += int2char[num]
    return output_sentence

print(generate_text(350))


