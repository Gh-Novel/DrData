import pandas as pd
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Function to generate misspellings of a word with noise
def generate_misspelling(word, noise_prob, characters):
    new_word = list(word)
    for i in range(len(new_word)):
        if random.random() < noise_prob:
            new_word[i] = random.choice(characters)
    return ''.join(new_word)

# Function to generate noisy training examples
def generate_noisy_examples(data, num_examples, noise_prob, characters):
    noisy_data = []
    for _, row in data.iterrows():
        name = row['Correct Name']
        for _ in range(num_examples):
            noisy_name = generate_misspelling(name, noise_prob, characters)
            noisy_data.append({'Incorrect Name': noisy_name, 'Correct Name': name})
    return pd.DataFrame(noisy_data)

# Load the dataset from CSV file
data = pd.read_csv('C:/Users/my pc/Desktop/project/NPL_model/filtered_dataset.csv', encoding='utf-8')

# Hyperparameters
num_examples = 50
noise_prob = 0.1
characters = 'abcdefghijklmnopqrstuvwxyz'
learning_rate = 0.001
dropout_rate = 0.2

# Generate noisy examples
noisy_data = generate_noisy_examples(data, num_examples, noise_prob, characters)

# Combine original and noisy data
combined_data = pd.concat([data, noisy_data])

# Convert names to lowercase
combined_data['Correct Name'] = combined_data['Correct Name'].str.lower()
combined_data['Incorrect Name'] = combined_data['Incorrect Name'].str.lower()

# Convert names to character-level sequences
input_texts = combined_data['Incorrect Name'].tolist()
target_texts = combined_data['Correct Name'].tolist()
input_characters = sorted(list(set(''.join(input_texts))))
target_characters = sorted(list(set(''.join(target_texts))))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

# Prepare the training data
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(target_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(target_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t-1, target_token_index[char]] = 1.0

# Define the model architecture
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(units=256, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

attention = Attention()
attention_outputs = attention([decoder_outputs, encoder_outputs])

decoder_dropout = Dropout(dropout_rate)
decoder_outputs = decoder_dropout(attention_outputs)

decoder_dense = Dense(units=num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile and train the model
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, epochs=50, batch_size=32)

# Save the model in .keras format
model.save('name_model.keras')
