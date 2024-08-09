"""
PRE-EXPERIMENT 6: IMPLEMENTATION OF RNN MODEL FOR THE GOLF COURSE REVIEW DATA (adapted from MSDS 458 Assignment 3 code)
"""

#%%
# 1. Import the necessary libraries for this experiment
# -------------------------------------------------------------------------------------------------
import datetime
from packaging import version
from collections import Counter
import numpy as np
import pandas as pd
import time
import os
import re
import string

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import nltk
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras
# %pip install tensorflow_datasets
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as k
from sklearn.model_selection import train_test_split

# Set the default precision for numpy
np.set_printoptions(precision=3, suppress=True)

# Enable display of multiple outputs per Jupyter Notebook cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# %%
# 2. Load in the golf course reviews dataset & create a new label column
# -------------------------------------------------------------------------------------------------
file_path = "top_and_non_golf_course_reviews.csv"
df = pd.read_csv(file_path)

# Create a new label column that indicates whether the review is a top100 course or not
df['top100'] = df['label'].apply(lambda x: 1 if x == 'top100' else 0)

# %%
# 3. Split the dataset into training, validation, and testing sets
# -------------------------------------------------------------------------------------------------
train_df, remaining = train_test_split(df, test_size=0.3, stratify=df['top100'], random_state=42)
val_df, test_df = train_test_split(remaining, test_size=0.5, stratify=remaining['top100'], random_state=42)

# Check the shape of the training, validation, and test sets
print(f"Training Dataset Shape: {train_df.shape}")
print(f"Validation Dataset Shape: {val_df.shape}")
print(f"Test Dataset Shape: {test_df.shape}")

# %%
# 4. Convert the split DataFrames into TensorFlow Datasets
# -------------------------------------------------------------------------------------------------
train_ds = tf.data.Dataset.from_tensor_slices(dict(train_df))
val_ds = tf.data.Dataset.from_tensor_slices(dict(val_df))
test_ds = tf.data.Dataset.from_tensor_slices(dict(test_df))

# %%
# 5. Create `custom_stopwords` function and `text_vectorization` layer
# -------------------------------------------------------------------------------------------------

# Define a `custom_stopwords` function to remove stopwords, strip punctuation, and lowercase the text
def custom_stopwords(input_text):
    """
    Removes stopwords, strips punctuation, and lowers the input text.

    Args:
        input_text (tf.Tensor): The input text to be processed.

    Returns:
        tf.Tensor: The processed text with stopwords removed, punctuation stripped, and lowercased.
    """
    lowercase = tf.strings.lower(input_text)
    stripped_punct = tf.strings.regex_replace(lowercase,
                                            '[%s]' % re.escape(string.punctuation),
                                            '')
    return tf.strings.regex_replace(stripped_punct, r'\b(' + r'|'.join(STOPWORDS) + r')\b\s*', "")

# Download stopwords from the NLTK library
nltk.download('stopwords', quiet=True)
STOPWORDS = stopwords.words("english")

# Define the maxium sequence and token length for this experiment
max_length =  3073
max_tokens = 10000

# Create a TextVectorization layer
text_vectorization = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length,
    standardize=custom_stopwords
)

# %%
# 6. Adapt the TextVectorization layer to a text_only_dataset of the training dataset
# -------------------------------------------------------------------------------------------------

# Create a text_only_train_dataset which contains only the review_text column of the training dataset
text_only_train_dataset = train_ds.map(lambda x: x['review_text'])

# Adapt the TextVectorization layer to the text_only_train_dataset
text_vectorization.adapt(text_only_train_dataset)

# Create int_train_ds, int_val_ds, and int_test_ds from the train, val, and test datasets respectively using the TextVectorization layer
int_train_ds = train_ds.map(lambda x: (text_vectorization(x['review_text']), x['top100']))
int_val_ds = val_ds.map(lambda x: (text_vectorization(x['review_text']), x['top100']))
int_test_ds = test_ds.map(lambda x: (text_vectorization(x['review_text']), x['top100']))

# Batch and pad the datasets to have a sequence lenfth dimension
batch_size = 32
max_sequence_length = 3073

int_train_ds = int_train_ds.map(lambda x, y: (x, y)).padded_batch(batch_size, padded_shapes=(max_sequence_length, ()))
int_val_ds = int_val_ds.map(lambda x, y: (x, y)).padded_batch(batch_size, padded_shapes=(max_sequence_length, ()))
int_test_ds = int_test_ds.map(lambda x, y: (x, y)).padded_batch(batch_size, padded_shapes=(max_sequence_length, ()))


# %%
# 7. Create Baseline Unidirectional RNN Model (36 Units | RMSprop = 0.001 | ReLU | 11,000 Vocab)
# -------------------------------------------------------------------------------------------------
"""
NOTE: THIS IS SERVING MORE AS A PROOF OF CONCEPT MORE THAN A MODEL THAT I AM EXPECTING WILL PERFORM WELL.
"""
# Ckear any existing models in memory
tf.keras.backend.clear_session()

# Define the model constants
vocab_size = 10000
embedding_dim = 356
rnn_units = 32

# Build the unidirectional RNN model
inputs = tf.keras.Input(shape=(None,), dtype="int64", name="input")
embedding = layers.Embedding(input_dim=vocab_size,
                             output_dim=embedding_dim,
                             mask_zero=True, name="embedding")(inputs)
x = layers.SimpleRNN(rnn_units, activation="relu", name="Simple_Unidirectional_RNN")(embedding)
outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

# %%
# 8. Compile the model
# -------------------------------------------------------------------------------------------------
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=["accuracy"])

# Display the model summary
model.summary()

# %%
# 9. Train the model
# -------------------------------------------------------------------------------------------------

# Define the callbacks for the model training
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("P-EXP_6_V1_RNN.keras", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
]

# Train the model
history = model.fit(int_train_ds,
                    validation_data=int_val_ds,
                    epochs=20,
                    callbacks=callbacks)

# %%
# 10. Evaluate the model
# -------------------------------------------------------------------------------------------------

# Load the best model
model = tf.keras.models.load_model("P-EXP_6_V1_RNN.keras")

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(int_test_ds)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Get the training loss, validation loss, training accuracy, and validation accuracy from the history object
training_loss = history.history['loss'][-1] # the -1 index gets the last epoch
validation_loss = history.history['val_loss'][-1]
training_accuracy = history.history['accuracy'][-1]
validation_accuracy = history.history['val_accuracy'][-1]

"""
1.2.2 Extract the training history and add all evaluation metrics into a history DataFrame
"""
# Extract the training history into a pandas DataFrame
history_df = pd.DataFrame({
    'EXP': [1],
    'Model': ['Unidirectional RNN'],
    'Training Loss': [training_loss],
    'Training Accuracy': [training_accuracy],
    'Validation Loss': [validation_loss],
    'Validation Accuracy': [validation_accuracy],
    'Test Loss': [test_loss],
    'Test Accuracy': [test_accuracy]
})

# Inspect the history DataFrame
history_df
# %%
