"""
In this file, we will be starting with the experimentation process for the capstone 
project, which aims to create a machine learning model that can predict whether a golf 
course is top or non-top golf course based on its characteristics.

For this project, we will be using the 'top_and_non_golf_course_reviews.csv' dataset.
This is a novel dataset that I created, and it expands upon the original dataset that
was used in this study's predicesor, which only included 60 reviews from the top 30
ranked golf courses in the United States.

The first phase of the experimentation that will be approach is to use the BERT model
to create embeddings for the golf course reviews, which will subsequently be used as the
input for neural network models that will be built and trained in later phases of this 
research.
"""
# %%
# Import the necessary libraries
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# %%
# 1. Load in the Dataset
# ----------------------------------
file_path = "top_and_non_golf_course_reviews.csv"
df = pd.read_csv(file_path)

# %%
# 2. Data Preprocessing
# ----------------------------------
"""
For data prepprocessing, we will be using the following steps:
    - Tokenization
    - Non-alphabetic token removal
    - Stopword removal
    - Lemmatization
    - Domain-specific stopword removal
"""
import spacy
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# Load necessary NLP models and stopwords that will be used for data preprocessing
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])   # Load the spaCy English language model, disable the parser and named entity recognition for efficiency
standard_stop_words = set(stopwords.words('english'))   # Load the English stopwords from NLTK
tokenizer = RegexpTokenizer(r'\w+')   # Define a regular expression tokenizer to remove and punctuation and perform tokenization using NLTK's RegexpTokenizer
domain_specific_stopwords = set([
    'course', 'play', 'hole', 'green', 'par', 'tee', 'yard', 'golf', 'one', 'bunker', 'fairway', 'leave', 'shot', 'right', 'good', 'club'
])

# Define a function to preprocess the text data
def preprocess_text(text):
    """
    Preprocesses a single document by applying several preprocessing steps:
        - Tokenization
        - Removal of non-alphabetic tokens and tokens with less than 3 characters
        - Conversion to lowercase
        - Removal of stopwords
        - Removal of domain-specific stopwords
        - Lemmatization
    Args:
        text (str): The original text of the golf course review to be preprocessed
    Returns:
        str: The preprocessed text of the golf course review
    """
    # Tokenize the text and convert it to lowercase
    tokens = [token.lower() for token in tokenizer.tokenize(text) if token.isalpha() and len(token) > 2]
    # Remove stopwords and domain-specific stopwords
    tokens = [token for token in tokens if token not in (standard_stop_words | domain_specific_stopwords)]
    # Lemmatize the tokens
    doc = nlp(" ".join(tokens))
    tokens = [token.lemma_ for token in doc if token.lemma_ not in standard_stop_words]
    # Return the preprocessed text as a single string
    return " ".join(tokens)

# Apply the preprocessing function to the 'review_text' column of the df DataFrame
df['cleaned_review_text'] = df['review_text'].apply(preprocess_text)

# Display the first few rows of the DataFrame with the cleaned review text
df[['review_text', 'cleaned_review_text']].head()

# %%
# 3. Create Embeddings for the Golf Course Reviews using BERT
# ----------------------------------
"""
We will be using the BERT model to create embeddings for the golf course reviews.
"""
# Load the pre-trained BERT model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a function to create embeddings for a single golf course review
def create_embeddings(text):
    """
    Creates embeddings for a single golf course review using the pre-trained BERT model and tokenizer.
    Args:
        text (str): The text of the golf course review
    Returns:
        list: The embeddings for the golf course review
    """
    # Tokenize the text and truncate/pad it to fix within the maximum sequence length allowed by the model (512)
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    # Get the embeddings
    outputs = model(**inputs)
    # Get the embeddings
    embeddings = outputs.last_hidden_state.detach().numpy()[0]
    # Return the embeddings
    return embeddings

# Apply the create_embeddings function to the cleaned_review_text column of the df DataFrame
df['embeddings'] = df['cleaned_review_text'].apply(create_embeddings)

# %%
