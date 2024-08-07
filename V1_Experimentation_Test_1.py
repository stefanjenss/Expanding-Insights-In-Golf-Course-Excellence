# PRE-EXPERIMENT #1 - BERT EMBEDDING WITH LSTM MODEL 
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
# -------------------------------------------------------------------------------------------------------
file_path = "top_and_non_golf_course_reviews.csv"
df = pd.read_csv(file_path)

# %%
# 2. Data Preprocessing
# -------------------------------------------------------------------------------------------------------
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
# -------------------------------------------------------------------------------------------------------
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
# 4. Prepare the Dataset for Training
# ----------------------------------
"""
We will be using the embeddings as the features for the training of the model along with the labels 
of the golf course reviews.
"""
# Convert the 'embeddings' column to a list of numpy arrays, this will be used as the features (X)
X = np.array(df['embeddings'].tolist())
y = np.array(df['label'] == 'top100', dtype=int)

# First split: 85% of the data for training, 15% for testing
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Second split: 80% of the remaining data for training, 20% for validation
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

# Split the data into training, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Examine the shapes of the training and testing sets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Examine the shapes of the training, validation, and testing sets
print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)


# %%
# 6. Implement the LSTM Model for Text Classification
# -------------------------------------------------------------------------------------------------------
"""
We will be using the LSTM model for text classification.
"""
# Import the necessary libraries
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.metrics import classification_report

# Define the LSTM model
class LSTMClassifier(nn.Module):
    """
    The `LSTMClassifier` class is a PyTorch model for text classification that uses an LSTM layer and a 
    fully connected lyaer to make predictions about the 'label' of a golf course review--whether or not
    it is a top 100-ranked golf course in the United States.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
# Set the hyperparameters of the model
input_size = X_train.shape[2]
hidden_size = 64
num_layers = 4 # Changed from 2
output_size = 1

# Create the instance of the LSTM model
model = LSTMClassifier(input_size, hidden_size, num_layers, output_size)
        
# %%
# 7. Train the model
# -------------------------------------------------------------------------------------------------------
from torch import FloatTensor
"""
We will train the model with the following features:
    - Optimizer = Adam
    - Loss Function = Binary Cross Entropy with Logits 
"""
# Define the optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Convert the data to PyTorch tensors
X_train_tensor = FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)

# Define the parameters for the training
num_epochs = 40
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y.unsqueeze(1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# %%
# 8. Evaluate the model
# -------------------------------------------------------------------------------------------------------
"""
We will evaluate the model with the following features:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Confusion Matrix
"""
model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test)
    y_pred = model(X_test_tensor)
    y_pred = (torch.sigmoid(y_pred.squeeze()) > 0.5).numpy()

# Print the classification report
print(classification_report(y_test, y_pred))

# %%
# Print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['Non-Top-100', 'Top-100']).plot()
# %%
# Clear the session memory to reinstantiate the model
# import gc
# gc.collect()
# torch.cuda.empty_cache()