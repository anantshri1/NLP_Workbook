# Emotion Classification in Mental Health Dialogues

An end-to-end NLP pipeline for classifying emotions in patient–therapist conversations, benchmarking classical machine learning against deep learning and a domain-adapted transformer.

**Dataset**: [NLP Mental Health Conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations) — a collection of patient context utterances paired with therapist responses.
**Task**: 7-class emotion classification (anger, disgust, fear, joy, neutral, sadness, surprise) on patient context text.

## Pipeline Overview
```
Raw CSV
  └─► EDA (structure, length distributions, class balance)
        └─► Feature Engineering (VADER sentiment + Hartmann emotion labels)
              └─► Text Cleaning (lowercasing, stopwords, lemmatisation)
                    └─► TF-IDF Vectorisation + Label Encoding
                          ├─► Naïve Bayes
                          ├─► Random Forest
                          ├─► Linear SVM  (GridSearchCV)
                          ├─► RBF SVM     (GridSearchCV)
                          ├─► LSTM        (class-weighted, trained from scratch)
                          └─► MentalBERT  (fine-tuned)
```

## Setup

```
pip install numpy pandas scikit-learn matplotlib seaborn nltk spacy
pip install tensorflow torch transformers vaderSentiment
pip install huggingface_hub
python -m spacy download en_core_web_sm
```

For MentalBERT you will need a Hugging Face account and access to the gated model:
```
from huggingface_hub import login
import os
login(token=os.environ["HF_TOKEN"])
```

## Stage 1 — Exploratory Data Analysis
The dataset is structured as a two-column dialogue table: `Context` (patient utterance) and `Response` (therapist reply). Key observations from EDA:
* The dataset has a one-to-many structure — a single patient context can appear multiple times paired with different therapist responses.
* Context lengths vary substantially; a small number of very long contexts (>512 tokens) caused truncation warnings during transformer inference.
* Class imbalance is present: sadness dominates (~31% of samples), while joy and surprise are minority classes with roughly 17 and 23 test samples respectively.

## Stage 2 — Feature Engineering

Two complementary signals were extracted from the raw text before modelling:
* VADER sentiment (`vaderSentiment`) assigns a coarse positive/negative polarity score to each context. This was used as an auxiliary feature alongside `TF-IDF` for classical models. VADER is a lexicon-based analyser and captures surface-level valence well, though it loses emotional granularity.
* Hartmann emotion labels `(j-hartmann/emotion-english-distilroberta-base)` — a DistilRoBERTa model fine-tuned on six emotion datasets — was applied to each context to produce a predicted emotion label. These 7-class labels (anger, disgust, fear, joy, neutral, sadness, surprise) became the target variable for all downstream models.

```
from transformers import pipeline
emotion_model = pipeline('sentiment-analysis', model='j-hartmann/emotion-english-distilroberta-base')
emotions = df['Context'].apply(lambda x: emotion_model(x)[0]['label'])
df['Detected_Emotion'] = emotions
```

## Stage 3 — Text Preprocessing
Preprocessing was applied to the Context column using `NLTK` and `spaCy`:

*Lowercasing and punctuation removal
*Stopword removal (NLTK English stopword list)
*Lemmatisation (WordNetLemmatizer with POS tagging for accuracy)
*Regex-based tokenisation

Labels were integer-encoded using `sklearn.preprocessing.LabelEncoder` (anger=0, disgust=1, fear=2, joy=3, neutral=4, sadness=5, surprise=6).
`TF-IDF vectorisation` (max_features=10000, unigrams and bigrams) was applied after the train/test split to avoid leakage — the vectoriser was fit only on training data.

## Stage 4 — Classical Models
All classical models were trained on TF-IDF features with class-balanced weighting where supported. Train/test split: 80/20, stratified.

### Naïve Bayes
Multinomial NB (alpha=1). Fast baseline — achieves high precision but poor recall on minority classes. Sadness acts as a catch-all, inflating its recall to 1.00 at the cost of other classes.
|  | Precision | Recall | F1 |
|----------|------|---------------|-------------|
| Anger | 1.00 | 0.63 |0.77 |
| disgust |1.00 |0.55| 0.71 |
| fear |0.93 |0.96 |0.94 |
| joy |1.00| 0.18| 0.30 |
| neutral |1.00 | 0.81| 0.89|
|sadness |0.67| 1.00| 0.80| 
| surprise | 1.00| 0.30| 0.47|
| macro avg| 0.94| 0.63| 0.70

