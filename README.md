# Fake News Detection Classifier

A machine learning project for detecting fake and legitimate news using the Multinomial Naive Bayes classification algorithm. This project demonstrates comprehensive natural language processing (NLP) and text classification techniques applied to identify misinformation in news content.

## 📋 Project Overview

This project implements an automatic fake news detection system that classifies news articles as either **legitimate (0)** or **fake (1)** using a trained Naive Bayes classifier. The model processes raw text data through a complete NLP pipeline including cleaning, preprocessing, feature extraction, and classification.

### Key Statistics

- **Total Dataset Size:** 44,898 news articles
- **Legitimate News:** 21,417 records (47.7%)
- **Fake News:** 23,481 records (52.3%)
- **Training Set:** 35,918 samples (80%)
- **Testing Set:** 8,980 samples (20%)
- **Features Extracted:** 50,000 TF-IDF features

## 🎯 Model Performance

The Multinomial Naive Bayes classifier achieves strong performance across multiple metrics:

| Metric | Training Accuracy | Testing Accuracy |
|--------|-------------------|------------------|
| **Accuracy** | 95.82% | 95.28% |

### Detailed Classification Metrics (Test Set)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Legitimate (0) | 0.96 | 0.94 | 0.95 | 4,284 |
| Fake (1) | 0.95 | 0.96 | 0.96 | 4,696 |
| **Overall** | **0.95** | **0.95** | **0.95** | **8,980** |

## 📁 Dataset Information

The project uses two separate CSV files containing news articles:

- **True.csv** - Authentic news articles with legitimate content
- **Fake.csv** - Fabricated or misleading news articles

### Dataset Features

Each record contains the following columns:

- **title:** Headline of the news article
- **text:** Full body text of the article
- **subject:** Category of the news (e.g., "politicsNews", "worldnews")
- **date:** Publication date
- **label:** Binary classification (0 = Legitimate, 1 = Fake)

## 🛠️ Technology Stack

### Libraries & Dependencies

- **Data Processing:** NumPy, Pandas
- **Machine Learning:** Scikit-Learn (sklearn)
- **NLP & Text Processing:** NLTK
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Google Colab

### Key Algorithms & Techniques

- **Text Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency) with n-gram support (unigrams and bigrams)
- **Classification Model:** Multinomial Naive Bayes
- **Text Preprocessing:** 
  - Lowercasing
  - Special character and digit removal
  - Stopword removal (English stopwords)
  - Porter stemming
  - Tokenization

## 🔄 Pipeline Overview

### 1. **Data Loading & Exploration**
   - Load True.csv and Fake.csv datasets
   - Assign binary labels (0 for legitimate, 1 for fake)
   - Analyze dataset distribution and structure

### 2. **Data Preprocessing**
   - Shuffle dataset for better training generalization
   - Combine true and fake datasets
   - Create balanced training and testing splits

### 3. **Text Cleaning & NLP Processing**
   - Convert text to lowercase
   - Remove special characters and digits
   - Remove extra whitespace
   - Tokenize text into words
   - Remove English stopwords
   - Apply Porter stemming for word normalization
   - Output cleaned and stemmed text

### 4. **Feature Extraction**
   - Apply TF-IDF vectorization
   - Extract 50,000 most important features
   - Use both unigrams (single words) and bigrams (word pairs)
   - Convert text into numerical features

### 5. **Model Training**
   - Split data into training (80%) and testing (20%) sets
   - Train Multinomial Naive Bayes classifier
   - Store trained model for prediction

### 6. **Model Evaluation**
   - Generate predictions on both training and testing sets
   - Calculate accuracy, precision, recall, and F1-scores
   - Create visualizations for performance metrics
   - Generate detailed classification report

## 📊 Visualizations

The project generates several visualizations:

- **Dataset Distribution:** Bar and pie charts showing the balance between legitimate and fake news
- **Classification Metrics:** Performance visualization including precision, recall, F1-score, and accuracy
- **Confusion Matrix:** Shows true positives, true negatives, false positives, and false negatives

## 📈 Model Insights

### Strengths

- **High Overall Accuracy:** 95.28% on test set demonstrates strong generalization
- **Balanced Performance:** Both classes show similar precision and recall (94-96%)
- **Robust Feature Set:** 50,000 TF-IDF features capture nuanced text patterns
- **Efficient Algorithm:** Naive Bayes is fast and suitable for text classification

### Limitations

- **Assumptions:** Multinomial Naive Bayes assumes feature independence, which may not hold for text
- **Dataset Bias:** Model performance depends heavily on dataset quality and diversity
- **Temporal Dynamics:** Fake news tactics evolve; model may require periodic retraining
- **Context Loss:** Stemming and stopword removal may lose important semantic information

## 🔍 Potential Improvements

1. **Advanced Models:** Experiment with SVM, Random Forest, or Deep Learning approaches
2. **Feature Engineering:** Incorporate additional features like readability scores, sentiment analysis, or source credibility
3. **Ensemble Methods:** Combine multiple classifiers for improved robustness
4. **Cross-Validation:** Implement k-fold cross-validation for more reliable performance estimation
5. **Hyperparameter Tuning:** Optimize TF-IDF parameters and classifier hyperparameters
6. **Domain-Specific Dictionaries:** Include curated lists of deceptive language patterns
7. **Transfer Learning:** Leverage pre-trained language models like BERT or GPT

## 📚 References

- NLTK Documentation: https://www.nltk.org/
- Scikit-Learn Documentation: https://scikit-learn.org/
- TF-IDF Explanation: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- Naive Bayes Algorithm: https://en.wikipedia.org/wiki/Naive_Bayes_classifier