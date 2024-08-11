import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import time
# Import stopwords from NLTK
import nltk
from nltk.corpus import stopwords
# Import re
import re
import string

# Load and preprocess data
file_path = "top_and_non_golf_course_reviews.csv"
df = pd.read_csv(file_path)
df['top100'] = df['label'].apply(lambda x: 1 if x == 'top100' else 0)

# Define constants
vocab_size = 10000
embedding_dim = 356
max_length = 3073
max_tokens = 10000
batch_size = 32

# Define custom_stopwords function (as in your original code)
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

# Create TextVectorization layer
text_vectorization = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length,
    standardize=custom_stopwords
)

# Adapt TextVectorization layer to all text data
text_vectorization.adapt(df['review_text'])

# Define model creation function
def create_model():
    inputs = keras.Input(shape=(None,), dtype="int64", name="input")
    embedding = layers.Embedding(input_dim=vocab_size,
                                 output_dim=embedding_dim,
                                 mask_zero=True, name="embedding")(inputs)
    x = layers.Bidirectional(layers.SimpleRNN(units=64, activation='relu', return_sequences=True, name="Bidirectional_RNN_1"))(embedding)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.SimpleRNN(units=32, activation='relu', name="Bidirectional_RNN_2"))(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=["accuracy"])
    return model

# Prepare data for k-fold cross-validation
X = df['review_text'].values
y = df['top100'].values

# Implement k-fold cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_scores = []
fold_histories = []

for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
    print(f"Fold {fold}")
    
    # Prepare datasets for this fold
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    
    # Apply TextVectorization and batching
    train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y)).padded_batch(batch_size)
    val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y)).padded_batch(batch_size)
    
    # Create and train the model
    model = create_model()
    
    start_time = time.time()
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=20,
                        callbacks=[
                            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
                        ])
    end_time = time.time()
    
    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(val_ds)
    fold_scores.append(val_accuracy)
    fold_histories.append(history.history)
    
    print(f"Fold {fold} - Validation Accuracy: {val_accuracy:.4f}")
    print(f"Training Time: {end_time - start_time:.2f} seconds")
    print("--------------------")

# Calculate and print the average score
average_score = np.mean(fold_scores)
print(f"Average Validation Accuracy: {average_score:.4f}")

# You can also analyze the training histories if needed
# For example, to get the average training accuracy across all folds:
average_train_acc = np.mean([h['accuracy'][-1] for h in fold_histories])
print(f"Average Training Accuracy: {average_train_acc:.4f}")

# To visualize the learning curves, you could use matplotlib to plot the average training and validation accuracies across folds