# Import Google Drive to access dataset
from google.colab import drive
drive.mount('/content/drive')

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer   # using TF-IDF for feature extraction
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# -----------------------------------------------------------
# STEP 1 : LOAD THE DATASET
# -----------------------------------------------------------
df = pd.read_csv('/content/drive/MyDrive/Machine Learning/spam.csv', 
                 header=None, names=['label','message'], encoding='latin1')
df.head()  # show first 5 rows

# -----------------------------------------------------------
# STEP 2 : TEXT PREPROCESSING
# -----------------------------------------------------------

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):   # Handle non-string values
        return ''
    text = text.lower()  # convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # remove punctuation
    tokens = nltk.word_tokenize(text)  # tokenize text
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # remove stopwords + lemmatize
    return ' '.join(tokens)

# Drop missing messages if any
df = df.dropna(subset=['message'])
print(df['message'].isnull().sum())  # check null values

# STEP 3: FEATURE EXTRACTION (TF-IDF)
# -----------------------------------------------------------
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df['message'])  # Convert text into numerical features
y = df['label']  # Labels (spam/ham)

# STEP 4 : SPLIT DATA
# -----------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# STEP 5 : TRAIN NAIVE BAYES CLASSIFIER
# -----------------------------------------------------------
model = MultinomialNB()
model.fit(x_train, y_train)

# STEP 6 : MAKE PREDICTIONS
# -----------------------------------------------------------
Y_pred = model.predict(x_test)

# STEP 8 : EVALUATE THE MODEL
# -----------------------------------------------------------
print("Accuracy:", accuracy_score(y_test, Y_pred))  # accuracy score
print("\nClassification Report:\n", classification_report(y_test, Y_pred))  # precision, recall, f1-score

# STEP 9 : COMPARE ACTUAL VS PREDICTED
# -----------------------------------------------------------
result = pd.DataFrame({'Actual': y_test, 'Predicted': Y_pred})
print(result)
