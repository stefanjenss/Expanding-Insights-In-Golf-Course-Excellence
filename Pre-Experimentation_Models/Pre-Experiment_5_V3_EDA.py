# %%
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

# %%
# Set plots to appear inline
# %matplotlib inline
# Set the default precision for numpy
np.set_printoptions(precision=3, suppress=True)

# Enable display of multiple outputs per Jupyter Notebook cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# %%
# Load in the dataset
file_path = "/Users/stefanjenss/Documents/Expanding-Insights-In-Golf-Course-Excellence/top_and_non_golf_course_reviews.xlsx"
df = pd.read_excel(file_path)

# Convert this datast into a TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices(dict(df))

# Create a text_only_dataset_all which contains only the review_text column
text_only_dataset_all = dataset.map(lambda x: x['review_text'])

# %%
def custom_stopwords(input_text):
    lowercase = tf.strings.lower(input_text)
    stripped_punct = tf.strings.regex_replace(lowercase
                                  ,'[%s]' % re.escape(string.punctuation)
                                  ,'')
    return tf.strings.regex_replace(stripped_punct, r'\b(' + r'|'.join(STOPWORDS) + r')\b\s*',"")

# %%
nltk.download('stopwords',quiet=True)
STOPWORDS = stopwords.words("english")

# %%
"""
3.1 TextVectorization and Adapt() Vocabulary
"""
max_tokens = None
text_vectorization=layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    standardize=custom_stopwords
)
text_vectorization.adapt(text_only_dataset_all)

# %%
doc_sizes = []
corpus = []
for example in dataset.as_numpy_iterator():
  enc_example = text_vectorization(example['review_text'])
  doc_sizes.append(len(enc_example))
  corpus+=list(enc_example.numpy())

# %%
# Analyze the post-preprocessing document sizes & examine distribution of document sizes
doc_sizes = np.array(doc_sizes)
print("Token Size of of Golf Course Reviews in the Dataset:")
print("  Min:", doc_sizes.min())
print("  Max:", doc_sizes.max())
print("  Mean:", doc_sizes.mean())
print("  Median:", np.median(doc_sizes))

# Plot the distribution of document sizes
plt.figure(figsize=(10, 6))
sns.histplot(doc_sizes, bins=50, kde=True)
plt.title("Distribution of Token Size of of Golf Course Reviews in the Dataset", fontsize=15)
plt.xlabel("Tokens per Document", fontsize=12)
plt.ylabel("Number of Courser Reviews", fontsize=12)
plt.tight_layout()
plt.savefig("/Users/stefanjenss/Documents/Expanding-Insights-In-Golf-Course-Excellence/Figures/1_distribution_of_token_size_of_of_golf_course_reviews_in_the_dataset.png", dpi=300)
plt.show()
# %%
print(f"There are {len(text_vectorization.get_vocabulary())} vocabulary words in the corpus.")

# Check the first 50 vocabulary words
print(text_vectorization.get_vocabulary()[:50])