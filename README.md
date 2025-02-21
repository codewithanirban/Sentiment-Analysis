# Sentiment Analysis with Two-Stage Classification  

## **Overview**  
This repository contains an advanced **Sentiment Analysis** model that classifies text data using a **two-stage classification pipeline**. The approach enhances accuracy and interpretability by breaking the sentiment classification into two distinct phases.  

- **Stage 1:** Classifies text as **Positive** or **Non-Positive**.  
- **Stage 2:** Further classifies the **Non-Positive** texts into **Negative** or **Neutral**.  

This hierarchical classification approach allows for improved precision, especially in differentiating between negative and neutral sentiments.  

---

## **Project Features**  
✅ **Two-Stage Classification Pipeline**: A hierarchical approach improves classification accuracy.  
✅ **Preprocessing with NLP Techniques**: Text cleaning, tokenization, stopword removal, and lemmatization.  
✅ **Word Embeddings with GloVe**: Pre-trained embeddings enhance model performance.  
✅ **Stacking Classifier**: Combines Stage 1 & Stage 2 models for better predictions.  
✅ **Hyperparameter Optimization**: Grid search used to tune models.  
✅ **Confusion Matrix & Visualizations**: Provides insights into model performance.  

---

## **Dataset**  
The model is trained on a dataset (`processed_dataset.csv`) containing:  
- **Comments** (Textual Data)  
- **Sentiment Labels**: `Positive`, `Negative`, `Neutral`  

---

## **Pipeline Workflow**  
### **1️⃣ Data Preprocessing**
- Converts text to lowercase  
- Removes URLs, punctuation, and stopwords  
- Applies **lemmatization** for better word representations  
- Uses **GloVe embeddings** for vector representation  

### **2️⃣ Two-Stage Classification Approach**  
1. **Stage 1:**  
   - Classifies sentiments as **Positive** vs. **Non-Positive**  
   - Uses **Random Forest Classifier** with hyperparameter tuning  

2. **Stage 2:**  
   - Further classifies **Non-Positive** into **Negative** or **Neutral**  
   - Uses **Support Vector Classifier (SVC)**  

3. **Stacking Classifier:**  
   - Combines predictions from both stages using a **Stacking Model**  

---

## **Model Training & Evaluation**  
- **Train-Test Split**: 80% training, 20% testing  
- **Evaluation Metrics**:  
  - Accuracy  
  - Precision, Recall, F1-score  
  - Confusion Matrix  

---

## **Installation & Usage**  
### **🔧 Installation**
Clone the repository and install dependencies:  
```bash
git clone https://github.com/codewithanirban/Sentiment-Analysis
cd Sentiment-Analysis
pip install -r requirements.txt
```

### **🚀 Running the Model**
1. **Train the Model**  
   ```bash
   python submission.py
   ```

2. **Make Predictions**  
   ```python
   import pickle
   import numpy as np
   from tensorflow.keras.preprocessing.sequence import pad_sequences

   # Load the trained model
   with open('stacked_sentiment_model.pkl', 'rb') as f:
       model = pickle.load(f)

   # Sample text for prediction
   text = ["I love this product!"]
   processed_text = preprocess_text_glove(text)
   tokenized = tokenizer.texts_to_sequences([processed_text])
   padded = pad_sequences(tokenized, maxlen=200)

   # Predict sentiment
   prediction = model.predict(padded)
   print("Predicted Sentiment:", prediction)
   ```

---

## **Novelty of the Model**
🚀 **What makes this model unique?**  
✅ **Two-Stage Classification** improves sentiment differentiation.  
✅ **Hybrid Approach** (Random Forest + SVM) enhances robustness.  
✅ **GloVe Embeddings** enrich word representations.  
✅ **Stacking Classifier** intelligently combines models for better predictions.  

---

## **Contributors**
- **Anirban Chakraborty** ([GitHub](https://github.com/codewithanirban))  🚀
- **Krittika Choudhuri** ([Github](https://github.com/Krittiii/)) 🚀

---

## **License**
This project is licensed under the MIT License.  

