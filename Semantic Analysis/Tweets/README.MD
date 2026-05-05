# Sarcasm & Irony Detection — Tweets

Four-class classification of tweets into **normal**, **sarcastic**, **ironic**, and **figurative** language categories, using the same NLP feature engineering pipeline extended to a neural network baseline.

---

## Dataset

**Source:** [Tweets with Sarcasm and Irony](https://www.kaggle.com/datasets/nikhiljohnk/tweets-with-sarcasm-and-irony) (Kaggle)  
**Fetched via:** `kagglehub`

The dataset provides pre-split `train.csv` and `test.csv` files. Each row contains a tweet and a class label (four categories). The notebook uses only the training split for its own stratified 80/20 train–test evaluation.

---

## Pipeline Overview

### 1. Preprocessing
- Load train and test CSVs from the Kaggle dataset
- Label-encode the four-class target with `LabelEncoder`

### 2. NLP Toolkit (NLTK)
Identical `preprocess_tweets()` function as in the Amazon notebook:
- Strip retweet markers, URLs, hashtags, and @-mentions via regex
- Tokenise with `TweetTokenizer` (case-folded, handle-stripped)
- Remove English stopwords and punctuation
- Lemmatise with `WordNetLemmatizer`

The function was originally designed for tweets, making it a natural fit for this dataset.

### 3. Feature Engineering
Same frequency-map approach: a `(word, class)` co-occurrence dictionary is built over the training set, and each tweet is encoded as `[1, freq_class0, freq_class1, freq_class2, freq_class3]` — a 5-dimensional vector. This is a deliberate design choice to keep features interpretable and the pipeline lightweight.

### 4. Models

| Model | Key hyperparameters |
|-------|-------------------|
| **Naive Bayes** (baseline) | `MultinomialNB(alpha=1)` |
| **Random Forest** | `n_estimators=3000`, `class_weight='balanced'` |
| **XGBoost** | `n_estimators=400`, `max_depth=6`, `lr=0.01`, `subsample=0.8` |
| **Neural Network** | 3 hidden layers (128→64→32, ReLU), BatchNorm + Dropout (0.2), softmax output, Adam `lr=0.001`, early stopping |
| **LSTM** | Embedding (dim=64) → 2×LSTM (128→64) → Dense head, `max_len=50`, vocab=10k, early stopping |

All models are evaluated with a stratified 80/20 train–test split. Metrics reported: `classification_report` (precision, recall, F1 per class), ROC-AUC curves (OVR macro average), and a combined confusion matrix panel.

---

## Requirements

```
numpy pandas seaborn matplotlib scikit-learn xgboost tensorflow nltk kagglehub
```

Install with:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn xgboost tensorflow nltk kagglehub
```

NLTK corpora downloaded at runtime:
```python
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## Usage

1. Open the notebook in Jupyter or any compatible environment.
2. Run all cells in order — `kagglehub` downloads the dataset automatically on first run.
3. Results (classification reports, confusion matrices, training curves) are displayed inline.

---


## Notes

- **Best overall:** LSTM leads on sarcasm, irony, and regular speech classification. Random Forest is the strongest model for the *figurative* class — likely because figurative language relies on specific lexical markers (fixed expressions, idioms) that a frequency-map representation captures well, whereas sequential context modelling (LSTM) adds noise for this category.
- The frequency-map feature representation is intentionally minimal as a baseline; contextual embeddings (e.g. a tweet-pretrained BERT) would be the natural next step.
- GridSearchCV is imported but unused — scaffolding for future hyperparameter tuning.
