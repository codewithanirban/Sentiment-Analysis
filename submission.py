import pandas as pd

# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



new = pd.read_csv('processed_dataset.csv')
print(new.Sentiment.value_counts())


"""
To use the glove file , download it from the link: 
Link : https://www.kaggle.com/datasets/sawarn69/glove6b100dtxt

Download this to run the below files without any error
"""


# Load GloVe Embeddings
def load_glove_embeddings(glove_file_path, embedding_dim=100):
    embeddings_index = {}
    with open(glove_file_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefficients = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefficients
    return embeddings_index

# Build Embedding Matrix
def create_embedding_matrix(word_index, embeddings_index, embedding_dim):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# Preprocess Text
def preprocess_text_glove(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

df = new
# Load Dataset
df['Comments'] = df['Comments'].fillna('')
df['cleaned_text'] = df['Comments'].apply(preprocess_text_glove)

# Tokenize Text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['cleaned_text'])
X_seq = tokenizer.texts_to_sequences(df['cleaned_text'])

# Pad Sequences
max_len = 200  # Adjust as needed
X_padded = pad_sequences(X_seq, maxlen=max_len)

# Create Embedding Matrix
embedding_dim = 100
glove_file_path = "./glove.6B.100d.txt"
embeddings_index = load_glove_embeddings(glove_file_path, embedding_dim)
embedding_matrix = create_embedding_matrix(tokenizer.word_index, embeddings_index, embedding_dim)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_padded, df['Sentiment'], test_size=0.2, random_state=42)


def two_stage_pipeline(X_train, y_train, X_test, y_test, param_grid_stage1, param_grid_stage2, scoring='accuracy', cv=3):
    """
    Two-stage sentiment analysis pipeline with hyperparameter tuning using GridSearchCV.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report
    from sklearn.model_selection import GridSearchCV

    # Stage 1: Positive vs Non-Positive Classification
    print("\n--- Stage 1: Positive vs Non-Positive Classification ---")
    
    # Transform labels for Stage 1
    y_train_stage1 = (y_train == 'Positive').astype(int)
    y_test_stage1 = (y_test == 'Positive').astype(int)
    
    # Define the pipeline for Stage 1
    pipeline_stage1 = Pipeline([
        ('clf', RandomForestClassifier(random_state=42))
    ])
    
    # Grid Search for Stage 1
    grid_stage1 = GridSearchCV(pipeline_stage1, param_grid_stage1, scoring=scoring, cv=cv, n_jobs=-1)
    grid_stage1.fit(X_train, y_train_stage1)
    
    # Best parameters and model
    print(f"Best Parameters (Stage 1): {grid_stage1.best_params_}")
    model_stage1 = grid_stage1.best_estimator_
    
    # Predictions for Stage 1
    stage1_preds = model_stage1.predict(X_test)
    print(classification_report(y_test_stage1, stage1_preds))
    
    # Identify non-positive predictions for Stage 2
    non_positive_indices = np.where(stage1_preds == 0)[0]
    X_train_stage2 = X_train[y_train_stage1 == 0]
    y_train_stage2 = y_train[y_train_stage1 == 0]
    X_test_stage2 = X_test[non_positive_indices]
    y_test_stage2 = y_test.iloc[non_positive_indices]
    
    # Stage 2: Negative vs Neutral Classification
    print("\n--- Stage 2: Negative vs Neutral Classification ---")
    
    # Transform labels for Stage 2
    y_train_stage2 = y_train_stage2.apply(lambda x: 0 if x == 'Negative' else 2)  # 0: Negative, 2: Neutral
    y_test_stage2 = y_test_stage2.apply(lambda x: 0 if x == 'Negative' else 2)
    
    # Define the pipeline for Stage 2
    pipeline_stage2 = Pipeline([
        ('clf', SVC(random_state=42, probability=True))
    ])
    
    # Grid Search for Stage 2
    grid_stage2 = GridSearchCV(pipeline_stage2, param_grid_stage2, scoring=scoring, cv=cv, n_jobs=-1)
    grid_stage2.fit(X_train_stage2, y_train_stage2)
    
    # Best parameters and model
    print(f"Best Parameters (Stage 2): {grid_stage2.best_params_}")
    model_stage2 = grid_stage2.best_estimator_
    
    # Predictions for Stage 2
    stage2_preds = model_stage2.predict(X_test_stage2)
    print(classification_report(y_test_stage2, stage2_preds))

    # Stacking Classifier for Combined Predictions
    from sklearn.ensemble import StackingClassifier
    stacking_model = StackingClassifier(
        estimators=[
            ('stage1', model_stage1),
            ('stage2', model_stage2)
        ],
        final_estimator=RandomForestClassifier(random_state=42),
        cv=5
    )
    
    # Train the stacked model on the complete dataset
    stacking_model.fit(X_train, y_train)
    print("\n--- Stacked Model Trained ---")
    
    # Save the final model
    import pickle
    with open('stacked_sentiment_model.pkl', 'wb') as f:
        pickle.dump(stacking_model, f)
    print("Final model saved as 'stacked_sentiment_model.pkl'")
    
    return stacking_model

# Example Usage
param_grid_stage1 = {
    'clf__n_estimators': [50, 100],
    'clf__max_depth': [10, None],
    'clf__min_samples_split': [2, 5]
}

param_grid_stage2 = {
    'clf__C': [0.1, 1],
    'clf__kernel': ['linear', 'rbf'],
    'clf__gamma': ['scale', 'auto']
}

# Ensure X_train, X_test, y_train, y_test are preprocessed
model = two_stage_pipeline(X_train, y_train, X_test, y_test, param_grid_stage1, param_grid_stage2)

# Predictions and evaluation
y_pred = model.predict(X_test)
print("\n--- Final Model Evaluation ---")
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))