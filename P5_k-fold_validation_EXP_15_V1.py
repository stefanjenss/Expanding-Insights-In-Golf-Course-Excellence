import pandas as pd
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.model_selection import KFold

# Load and preprocess data
file_path = "top_and_non_golf_course_reviews.csv"
df = pd.read_csv(file_path)
df['top100'] = df['label'].apply(lambda x: 1 if x == 'top100' else 0)

# Load BERT models
bert_preprocess_model = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

def get_bert_embeddings(text_input):
    preprocessed = bert_preprocess_model(text_input)
    return bert_encoder(preprocessed)['sequence_output']

def prepare_dataset(df, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((df['review_text'], df['top100']))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x, y: (get_bert_embeddings(x), y))
    return dataset.prefetch(tf.data.AUTOTUNE)

def create_1d_cnn_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv1D(filters=64, kernel_size=5, activation='relu', name='1d-cnn_1')(inputs)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool_1')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(filters=32, kernel_size=5, activation='relu', name='1d-cnn_2')(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool_2')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.GlobalMaxPooling1D(name='globalmaxpool')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
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
    
    train_df = pd.DataFrame({'review_text': X_train, 'top100': y_train})
    val_df = pd.DataFrame({'review_text': X_val, 'top100': y_val})
    
    train_ds = prepare_dataset(train_df)
    val_ds = prepare_dataset(val_df)
    
    # Create and compile the model
    input_shape = (None, 768)
    model = create_1d_cnn_model(input_shape)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
        ]
    )
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

# Calculate average training accuracy
average_train_acc = np.mean([h['accuracy'][-1] for h in fold_histories])
print(f"Average Training Accuracy: {average_train_acc:.4f}")

# Optional: Visualize learning curves
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for i, h in enumerate(fold_histories):
    plt.plot(h['accuracy'], label=f'Fold {i+1} Training')
    plt.plot(h['val_accuracy'], label=f'Fold {i+1} Validation')
plt.title('Model Accuracy Across Folds')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()