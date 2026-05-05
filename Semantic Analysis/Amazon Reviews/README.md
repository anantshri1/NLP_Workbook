# Sentiment Analysis — Amazon Product Reviews

Multi-class sentiment classification on Amazon product reviews using classical NLP feature engineering and three ML classifiers.

---

## Dataset

**Source:** [Amazon Reviews Dataset](https://www.kaggle.com/datasets/tarkkaanko/amazon) (Kaggle)  
**Fetched via:** `kagglehub`

The dataset contains Amazon product reviews with a numerical `overall` rating (1–5 stars) and free-text `reviewText`. Ratings are mapped to three sentiment classes:

| Original Rating | Mapped Class |
|----------------|-------------|
| 1, 2           | negative    |
| 3              | neutral     |
| 4, 5           | positive    |

---

## Pipeline Overview

### 1. Preprocessing
- Parse `reviewTime` into year/month features
- Drop rows with missing `reviewText`
- Drop non-text columns (vote counts, Wilson lower bound, etc.) before modelling

### 2. NLP Toolkit (NLTK)
A custom `preprocess_tweets()` function handles text cleaning (originally designed for tweets; applied here as a general-purpose text cleaner since the regex patterns are domain-agnostic):
- Strip retweet markers, URLs, hashtags, and @-mentions using regex
- Tokenise with NLTK's `TweetTokenizer` (case-folded, handle-stripped)
- Remove English stopwords and punctuation
- Lemmatise surviving tokens with `WordNetLemmatizer`

### 3. Feature Engineering
Rather than a standard TF-IDF bag-of-words, the notebook builds a **frequency map** over the training set: for each `(word, class)` pair, it counts how often that word appears in reviews of that class. Each review is then represented as a fixed-length vector `[1, freq_negative, freq_neutral, freq_positive]` (bias term + one count per class). This is a lightweight, interpretable alternative to TF-IDF.

### 4. Models

| Model | Key hyperparameters |
|-------|-------------------|
| **Naive Bayes** (baseline) | `MultinomialNB(alpha=1)` |
| **Random Forest** | `n_estimators=4000`, `class_weight='balanced'` |
| **XGBoost** | `n_estimators=400`, `max_depth=6`, `lr=0.01`, `subsample=0.8` |
| **Neural Network** | 3 hidden layers (128→64→32, ReLU), BatchNorm + Dropout (0.2), trained with and without class weights |
| **LSTM** | Embedding (dim=64) → 2×LSTM (128→64) → Dense head, `max_len=50`, vocab=10k, trained with and without class weights |
| **DistilBERT** | `distilbert-base-uncased` fine-tuned for 3-class classification via HuggingFace `Trainer`; also trained with a custom `WeightedTrainer` using inverse-frequency class weights |

Neural Network and LSTM variants are each run twice (weighted vs unweighted) to assess the effect of class imbalance correction. All models are evaluated on a stratified 80/20 split with classification reports, ROC-AUC curves (OVR macro), and a 3×3 confusion matrix panel.

---

## Requirements

```
numpy pandas seaborn matplotlib scikit-learn xgboost nltk kagglehub
```

Install with:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn xgboost tensorflow transformers torch nltk kagglehub
```

NLTK corpora are downloaded at runtime:
```python
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## Usage

1. Open the notebook in Jupyter or any compatible environment.
2. Run all cells in order — `kagglehub` will download the dataset automatically on first run.
3. Model outputs (classification reports and confusion matrices) are displayed inline.

---

## Notes

- **Best overall:** Classical BoW models collapse on the neutral class regardless of weighting. Weighted LSTM partially recovers it; Weighted DistilBERT handles it most robustly — illustrating the concrete gain from contextual pretraining on an imbalanced real-world task.
- The frequency-map feature representation used by the classical models is deliberately simple and interpretable; DistilBERT's superiority illustrates the gap that contextual pretraining closes.
- Class imbalance is addressed at three levels: `class_weight='balanced'` in tree models, `compute_sample_weight` for XGBoost, and inverse-frequency `CrossEntropyLoss` weights in the neural models.
