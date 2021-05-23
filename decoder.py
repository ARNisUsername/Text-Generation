#Import neccessary modules
import numpy as np
import string
import requests 
import tensorflow as tf
from keras import Input, Sequential, Model
from keras.layers import Layer, Embedding, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D

#Get the text filled with the alice in a wonderland book
text_ = requests.get("http://www.gutenberg.org/cache/epub/11/pg11.txt").text
#Clean the text
text_ = text_.lower().strip()
text_ = text_.translate(str.maketrans('', '', string.punctuation))
#Turn each char into a seperate index in a list
text = [char for char in text_]

#Make a char2int dictionary which can map each unique character to an integer
unique_chars = np.unique(text)
char2int = {char: index+1 for index, char in enumerate(unique_chars)}
NUM_chars = len(unique_chars)+1

#Transform the text vector to include the indexes
text = [char2int[char] for char in text]

#Create an X and y array where the X has 80 previous characters and the y has the next character for the prediction data
MAX_LEN = 80
X = []
y = []
for i in range(0, len(text)-MAX_LEN, MAX_LEN):
  X.append(np.array(text[i:i+MAX_LEN]).reshape(MAX_LEN,))
  y.append(text[i+MAX_LEN])
X = np.array(X)
y = np.array(y).reshape(-1,1)

#Function which when given a size, creates a binary size x size array, with the 0 values being the values used by the masked attention layer, and the 1 values being ignored
#As an example, when using this function with size 3, it would create the array [[0,1,1],[0,0,1],[0,0,0]]
def mask_attention(size):
  new_mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return new_mask

#Transformers use Positional Embedding because unlike RNNs, they cannot remember the positions of certain indexes. 
class PositionalEmbedding(Layer):
    def __init__(self, input_dim, input_length):
        super(PositionalEmbedding, self).__init__()

        #Using the mathematical operations given by the original paper, create a positional embedding array with size input_length x input_dim(embedding dim)
        self.pos_embeddings = np.zeros((input_length, input_dim))
        for pos in range(input_length):
            for i in range(0, input_dim, 2):
                self.pos_embeddings[pos, i] = np.sin(pos / (10000 ** ((2*i)/input_dim)))
                self.pos_embeddings[pos, i+1] = np.cos(pos / (10000 ** ((2*(i+1))/input_dim)))

    #A call function will simply return the positional vector
    def call(self, x):
        return self.pos_embeddings

#Transformers use regular embeddings, but the positional embeddings are added to the regular embeddings. So, this layer is the full Transformer Embedding layer
class TransformerEmbedding(Layer):
    def __init__(self, input_dim, output_dim, input_length):
        super(TransformerEmbedding, self).__init__()

        #Create the positional embeddings and prepare the regular Embeddings for when the x value is given during the call
        self.total_embeddings = np.zeros((input_length, output_dim))
        self.pos_embeddings = PositionalEmbedding(input_dim=output_dim, input_length=input_length)(self.total_embeddings)
        self.embeddings = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length)
        
    #When this layer is called upon, calculating the regular embeddings with the input and add it with the positonal embeddings before returning
    def call(self, x):
        x = self.embeddings(x)
        self.total_embeddings = x + self.pos_embeddings
        return self.total_embeddings

#Transformers need a Feed Forward layer to vectorize inputs from the attention layer to something processable. This is a simple demonstration of that
class FeedForward(Layer):
    def __init__(self, units, input_dim, dropout_rate):
        super(FeedForward, self).__init__()

        self.ff = Sequential([
            Dense(units, activation='relu'),
            Dropout(dropout_rate),
            Dense(input_dim)
        ])

    def call(self, x):
        return self.ff(x)

#Since this is a text generation task, we will only use a decoder block, due to the masked attention that forces the network to learn
class DecoderBlock(Layer):
  def __init__(self, ff_units, input_length, embedding_dim, num_chars, n_heads, dropout_rate):
    super(DecoderBlock, self).__init__()

    #The main focus on the __init__ function is to prepare all of the layers to be called on the input 
    self.ff = FeedForward(units=ff_units, input_dim=embedding_dim, dropout_rate=dropout_rate)
    #Using the mask_attention from before, we will create a self.mask variable to generate an input_length*input_length array to be used by the masked attention layer
    self.mask = mask_attention(input_length)
    self.masked_attention = MultiHeadAttention(num_heads=n_heads, key_dim=num_chars)
    self.attention = MultiHeadAttention(num_heads=n_heads, key_dim=num_chars)
    self.norm1, self.norm2, self.norm3 = LayerNormalization(), LayerNormalization(), LayerNormalization()
    #The constant dropouts used here will help with reducing overfitting, which can happen easily in decoders
    self.dropout1, self.dropout2, self.dropout3 = Dropout(dropout_rate), Dropout(dropout_rate), Dropout(dropout_rate)

  #When the decoder block is called upon, it will use all the variables defined by __init__ along with the decoder architecture in performance
  def call(self, x):
      x1 = self.masked_attention(x, value=x, attention_mask=self.mask)
      x1 = self.norm1(x1)
      x2 = self.norm1(x)
      x3 = self.dropout1((x1 + x2))

      x4 = self.attention(x3, value=x3)
      x4 = self.norm2(x4)
      x5 = self.norm2(x3)
      x6 = self.dropout2((x4 + x5))

      x7 = self.ff(x6)
      x7 = self.norm3(x7)
      x8 = self.norm3(x6)
      x9 = self.dropout3((x7 + x8))

      #Return the final layer, which will consist of multiple vectors 
      return x9

#The embedding dimension will only be 16, as we are doing character by character prediction and there is not very much information to give per characer
EMBEDDING_DIM = 16
decoder_input = Input(shape=(MAX_LEN))
#Add the transformer embedding layer immediatly to the input as a vectorization tool
decoder = TransformerEmbedding(input_dim=NUM_chars, output_dim=EMBEDDING_DIM, input_length=MAX_LEN)(decoder_input)
#Using feed forward units of 128, n_heads as 4(for the attention vectors to be more spread out/decentralized) and a dropout rate of 25%, add a single decoder block
decoder = DecoderBlock(ff_units=128, input_length=MAX_LEN, embedding_dim=EMBEDDING_DIM, num_chars=NUM_chars, n_heads=4, dropout_rate=0.25)(decoder)
#Given the vectors from the decoder block, global average pool the vectors in order to capture as much information as possible from them
decoder = GlobalAveragePooling1D()(decoder)
#Add a final linear layer as said by the transformer paper(we will use 128 units here as well) before adding the final output layer with all the characters and softmax
decoder = Dense(128, activation='relu')(decoder)
decoder_output = Dense(NUM_chars, activation='softmax')(decoder)

#Create the model, and using the adam optimizer, fit the model on 50 epochs(more epochs than that will lead to too much overfitting)
model = Model(inputs=decoder_input, outputs=decoder_output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X, y, epochs=50, batch_size=64)

#Since the dataset used here was so little in length, the text outputs from this model aren't very english like. However, this does provide similar results to LSTMs
#While still taking much less time than them. So, this is overall a success
