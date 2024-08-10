# %%
"""
PRE-EXPERIMENT 6: IMPLEMENTATION OF RNN MODEL WITH DOC2VEC EMBEDDINGS FOR THE GOLF COURSE REVIEW DATA
"""
# %%
# 1. Import the necessary libraries
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
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import tensorflow.keras.backend as k

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

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
train_df, remaining = train_test_split(df, test_size=0.33, stratify=df['top100'], random_state=42)
val_df, test_df = train_test_split(remaining, test_size=0.5, stratify=remaining['top100'], random_state=42)

# Check the shape of the training, validation, and test sets
print(f"Training Dataset Shape: {train_df.shape}")
print(f"Validation Dataset Shape: {val_df.shape}")
print(f"Test Dataset Shape: {test_df.shape}")

# %%
# 4. Preprocess the text data from the 'review_text' column 
# -------------------------------------------------------------------------------------------------
# Define a function to preprocess the text data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove stopwords (using the NLTK library)
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Apply the preprocessing function to the 'review_text' column of the df DataFrame
train_df['preprocessed_text'] = train_df['review_text'].apply(preprocess_text)
val_df['preprocessed_text'] = val_df['review_text'].apply(preprocess_text)
test_df['preprocessed_text'] = test_df['review_text'].apply(preprocess_text)

# %%
# 5. Convert the preprocessed text data into a list of tagged documents (Doc2Vec format)
# -------------------------------------------------------------------------------------------------
def create_tagged_documents(df):
    """
    Create a list of tagged documents from the 'preprocessed_text' column of the given DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the 'preprocessed_text' column.

    Returns:
        List[gensim.models.doc2vec.TaggedDocument]: A list of TaggedDocument objects, where each TaggedDocument represents a document with its corresponding tag (index).
    """
    return [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(df['preprocessed_text'])]

tagged_documents = create_tagged_documents(train_df)

# %%
# 6. Train the Doc2Vec model and generate the embeddings
# -------------------------------------------------------------------------------------------------
model = Doc2Vec(vector_size=100, min_count=2, epochs=40)
model.build_vocab(tagged_documents)
model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)

# Define a function to generate the embeddings for the split dataframes
def get_doc2vec_embeddings(model, df):
    return np.array([model.infer_vector(text.split()) for text in df['preprocessed_text']])

# Generate the embeddings for the training, validation, and test sets
train_embeddings = get_doc2vec_embeddings(model, train_df)
val_embeddings = get_doc2vec_embeddings(model, val_df)
test_embeddings = get_doc2vec_embeddings(model, test_df)

# %%
# 7. Create the Simple Unidirectional RNN model
# -------------------------------------------------------------------------------------------------
# Define the Simple Unidirectional RNN model
inputs = tf.keras.Input(shape=(100,))
x = tf.keras.layers.Reshape((100, 1))(inputs)
x = tf.keras.layers.SimpleRNN(32, activation='relu')(x)
ouputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=ouputs, name="Simple_Unidirectional_RNN")

# %%
# 8. Compile the model
# -------------------------------------------------------------------------------------------------
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=["accuracy"])

# Display the model summary
model.summary()

# %%
# 9. Train the model
# -------------------------------------------------------------------------------------------------
# Clear the session 
tf.keras.backend.clear_session()

# Define the callbacks for the model training
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("P-EXP_7_V1_D2V_RNN.keras", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
]

# Train the model
history = model.fit(train_embeddings, train_df['top100'],
                    validation_data=(val_embeddings, val_df['top100']),
                    epochs=20,
                    callbacks=callbacks)

# %%
# 10. Evaluate the model
# -------------------------------------------------------------------------------------------------
# Load the best model
model = tf.keras.models.load_model("P-EXP_7_V1_D2V_RNN.keras")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_embeddings, test_df['top100'])
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Get the training loss, validation loss, training accuracy, and validation accuracy from the history object
training_loss = history.history['loss'][-1] # the -1 index gets the last epoch
validation_loss = history.history['val_loss'][-1]
training_accuracy = history.history['accuracy'][-1]
validation_accuracy = history.history['val_accuracy'][-1]

# Extract the training history into a pandas DataFrame
history_df = pd.DataFrame({
    # 'EXP': [1],
    'Model': ['Unidirectional RNN w/ Doc2Vec Embeddings'],
    'Training Loss': [training_loss],
    'Training Accuracy': [training_accuracy],
    'Validation Loss': [validation_loss],
    'Validation Accuracy': [validation_accuracy],
    'Test Loss': [test_loss],
    'Test Accuracy': [test_accuracy]
})

# Inspect the history DataFrame
history_df

# Generate predictions for the test set
predictions = model.predict(test_embeddings)
predict_labels = (predictions > 0.5).astype(int).flatten()

# Create confusion matrix
cm = confusion_matrix(test_df['top100'], predict_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# %%
