from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


# Load the trained model
model = load_model('name_model_rnn.h5')  # Replace 'path_to_your_model.h5' with the actual path to your saved model
# Assuming you have a list of sentences to predict on
sentences_to_predict = ['aasa', 'asha']

# Tokenize and pad the sentences
# (Make sure you use the same tokenizer and padding method that was used during training)
sequences = tokenizer.texts_to_sequences(sentences_to_predict)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
# Predict on the input data
predictions = model.predict(padded_sequences)

# Assuming you have a dictionary to map class indices to class labels
class_labels = {0: 'Label_0', 1: 'Label_1', ...}

# Get the predicted class labels for each sentence
predicted_labels = [class_labels[idx] for idx in np.argmax(predictions, axis=1)]

# Print the predictions
for sentence, label in zip(sentences_to_predict, predicted_labels):
    print(f"Sentence: {sentence} - Predicted Label: {label}")
