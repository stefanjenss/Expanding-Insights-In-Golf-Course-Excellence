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
# 1. Load the BERT model
bert_preprocess_model = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

# %%
# 2. Create a function to get BERT embeddings
def get_bert_embeddings(text_input):
    preprocessed = bert_preprocess_model(text_input)
    return bert_encoedr(preprocessed)['pooled_output']

# %%

