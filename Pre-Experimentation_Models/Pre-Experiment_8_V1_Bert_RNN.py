# %%
"""
PRE-EXPERIMENTATION 8 - CREATING EMBEDDINGS WITH BERT AS INPUT FOR RNN TEXT CLASSIFICATION MODEL
"""
# %%
# A dependency of the preprocessing for BERT inputs
!pip install -U "tensorflow-text==2.13.*"

!pip install "tf-models-official==2.13.*"

# %%
# Import the necessary libraries
import pandas as pd
import numpy as np
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
import tensorflow_text as text  # Required for BERT preprocessing
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# %%
# 1. Load in the data
# ------------------------------------------------------------------------------
file_path = "top_and_non_golf_course_reviews.csv"
df = pd.read_csv(file_path)

# Create a new label column that indicates whether the review is a top100 course or not
df['top100'] = df['label'].apply(lambda x: 1 if x == 'top100' else 0)

# %%
# 2. Split the dataset into training, validation, and testing sets
# ------------------------------------------------------------------------------
train_df, remaining = train_test_split(df, test_size=0.33, stratify=df['top100'], random_state=42)
val_df, test_df = train_test_split(remaining, test_size=0.5, stratify=remaining['top100'], random_state=42)

# Check the shape of the training, validation, and test sets
print(f"Training Dataset Shape: {train_df.shape}")
print(f"Validation Dataset Shape: {val_df.shape}")
print(f"Test Dataset Shape: {test_df.shape}")

# %%
# 3. Convert the split Pandas DataFrames into TensorFlow Datasets
# train_ds = tf.data.Dataset.from_tensor_slices(dict(train_df))
# val_ds = tf.data.Dataset.from_tensor_slices(dict(val_df))
# test_ds = tf.data.Dataset.from_tensor_slices(dict(test_df))

# %%
# 4. Load in the BERT Models
# ------------------------------------------------------------------------------
bert_preprocess_model = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")


# %%
# 5. Create a function to get BERT embeddings
# ------------------------------------------------------------------------------
def get_bert_embeddings(text_input):
    preprocessed = bert_preprocess_model(text_input)
    return bert_encoder(preprocessed)['sequence_output']

# %%
# 6. Define a function that prepares the dataset for training
def prepare_dataset(df, batch_size=32):
    """
    Prepare the dataset for training by creating a TensorFlow dataset from the input DataFrame.
    
    Parameters:
    - df: The input DataFrame containing 'review_text' and 'top100' columns.
    - batch_size: The batch size for the dataset (default value is 32).
    
    Returns:
    - dataset: A TensorFlow dataset with the 'review_text' and 'top100' columns batched and mapped using BERT embeddings.
    - prefetch: Prefetches elements from the dataset for improved performance.
    """
    dataset = tf.data.Dataset.from_tensor_slices((df['review_text'], df['top100']))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x, y: (get_bert_embeddings(x), y))
    return dataset.prefetch(tf.data.AUTOTUNE)

# %%
# 7. Prepare the training, validation, and test datasets for training
train_ds = prepare_dataset(train_df)
val_ds = prepare_dataset(val_df)
test_ds = prepare_dataset(test_df)

# %%
# 8. Define the RNN model
# ------------------------------------------------------------------------------
def create_rnn_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Bidirectional(layers.SimpleRNN(units=64, activation='relu', return_sequences=True))(inputs)
    x = layers.Bidirectional(layers.SimpleRNN(units=32, activation='relu'))(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
    
    model = keras.Model(inptus=[inputs, mask_inputs], outputs=outputs)
    return model

# %%
# 9. Compile the model
# ------------------------------------------------------------------------------
input_shape = (None, 768)

model = create_rnn_model(input_shape)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Inspect the model summary
model.summary()

# %% 
# 10. Train the model
# ------------------------------------------------------------------------------

# Clear any existing models in memory
tf.keras.backend.clear_session()

# Define the callbacks for the model training
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("EXP_1_RNN_TF_TF.keras"),
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
]

# Train the model
start_time = time.time()
history = model.fit(train_ds,
                    validation_data=val_ds,
                    validation_steps=len(val_ds),
                    epochs=20,
                    callbacks=callbacks)
end_time = time.time()
training_time = end_time - start_time

# Print the training time
print(f"Training Time: {training_time} seconds")

# %%
# 11. Evaluate the model
# ------------------------------------------------------------------------------
# Load the best model
model = tf.keras.models.load_model("EXP_1_RNN_TF_TF.keras")

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_ds)
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
    'EXP': [9],
    'Model': ['RNN w/ BERT Embeddings'],
    'Training Loss': [training_loss],
    'Training Accuracy': [training_accuracy],
    'Validation Loss': [validation_loss],
    'Validation Accuracy': [validation_accuracy],
    'Test Loss': [test_loss],
    'Test Accuracy': [test_accuracy],
    'Training Time': [174]
})

# Inspect the history DataFrame
history_df