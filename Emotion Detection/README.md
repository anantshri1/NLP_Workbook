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
`TF-IDF vectorisation` was applied for vectorization.

## Stage 4 — Classical Models
All classical models were trained on TF-IDF features with class-balanced weighting where supported. Train/test split: 80/20, stratified.

### Naïve Bayes
Multinomial NB (`alpha=1`). Fast baseline — achieves high precision but poor recall on minority classes. Sadness acts as a catch-all, inflating its recall to 1.00 at the cost of other classes.
|  | Precision | Recall | F1 |
|----------|------|---------------|-------------|
| anger | 1.00 | 0.63 |0.77 |
| disgust |1.00 |0.55| 0.71 |
| fear |0.93 |0.96 |0.94 |
| joy |1.00| 0.18| 0.30 |
| neutral |1.00 | 0.81| 0.89|
|sadness |0.67| 1.00| 0.80| 
| surprise | 1.00| 0.30| 0.47|
| macro avg| 0.94| 0.63| 0.70|

### Random Forest
`RandomForestClassifier` with class weighting. Best classical model overall — handles high-dimensional sparse TF-IDF well and captures non-linear feature interactions.
|  | Precision | Recall | F1 |
|----------|------|---------------|-------------|
| anger | 0.99 | 0.92 |0.95 |
| disgust |1.00 |0.92| 0.96 |
| fear |0.96 |0.98 |0.97 |
| joy |1.00| 1.00| 1.00 |
| neutral |0.99 | 0.92| 0.95|
|sadness |0.92| 0.99| 0.95| 
| surprise | 0.90| 0.78| 0.84|
| macro avg| 0.97| 0.93| 0.95 |

### Linear SVM
Tuned via `GridSearchCV` over `C ∈ {0.001, 0.01, 0.1, 1, 10, 100}`, 5-fold CV. Best C=10, best CV accuracy 0.928.
|  | Precision | Recall | F1 |
|----------|------|---------------|-------------|
| anger | 0.99 | 0.94 |0.96 |
| disgust |1.00 |0.92| 0.96 |
| fear |0.96 |0.97 |0.97 |
| joy |1.00| 1.00| 1.00 |
| neutral |0.97 | 0.92| 0.94|
|sadness |0.92| 0.99| 0.95| 
| surprise | 0.90| 0.78| 0.84|
| macro avg| 0.96| 0.93| 0.95 |

### SVM with RBF Kernal
Tuned via `GridSearchCV` over `C ∈ {0.001, 0.01, 0.1, 1, 10, 100}`, 5-fold CV. Best C=10, best CV accuracy 0.916 — slightly below Linear SVM, as expected for linearly-separable TF-IDF representations.
|  | Precision | Recall | F1 |
|----------|------|---------------|-------------|
| anger | 0.99 | 0.90 |0.94 |
| disgust |1.00 |0.92| 0.96 |
| fear |0.99 |0.97 |0.98 |
| joy |1.00| 1.00| 1.00 |
| neutral |0.99 | 0.92| 0.95|
|sadness |0.88| 1.00| 0.94| 
| surprise | 0.90| 0.78| 0.84|
| macro avg| 0.96| 0.93| 0.94|

## Stage 5 — LSTM
A two-layer LSTM trained from scratch on integer-encoded, padded sequences. Class imbalance was handled via `compute_class_weight('balanced')` passed to `model.fit`.

Architecture:
```
Embedding(10000, 64, input_length=600)
SpatialDropout1D(0.2)
LSTM(64, return_sequences=True)
Dropout(0.2)
LSTM(32)
Dropout(0.2)
Dense(32, activation='relu')
Dense(7, activation='softmax')
```
Loss: `sparse_categorical_crossentropy`. Optimiser: `Adam` (`lr=0.001`). Early stopping on `val_loss` (patience=5).
**Result**: The LSTM underperforms all tuned classical models. Training embeddings from scratch on a dataset of this size produces insufficient representations — the model lacks the semantic grounding that pretrained embeddings would provide. This is the expected outcome and serves as a useful pedagogical contrast: it demonstrates why pretrained representations matter for NLP tasks with limited data.

|  | Precision | Recall | F1 |
|----------|------|---------------|-------------|
| anger | 0.77 | 0.84 |0.80 |
| disgust |0.86 |0.87| 0.86 |
| fear |0.97 |0.92 |0.95 |
| joy |0.94| 1.00| 0.97 |
| neutral |0.91 | 0.92| 0.91|
|sadness |0.91| 0.93| 0.92| 
| surprise | 0.94| 0.74| 0.83|
| macro avg| 0.90| 0.89| 0.89|

## Stage 6 — MentalBERT
`mental/mental-bert-base-uncased fine-tuned` for sequence classification. MentalBERT is BERT-base pretrained on mental health corpora (Reddit mental health communities, clinical notes), giving it domain-specific language priors not present in standard BERT.
**Model**: `AutoModelForSequenceClassification` with `num_labels=7`.
**Training**: HuggingFace `Trainer` API, 3 epochs, batch size 8, warmup steps 500, weight decay 0.01.
**Input**: Contexts tokenised to max_length=128, truncated and padded.

```
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "mental/mental-bert-base-uncased", num_labels=7
)
```

Result: Best performing model across all classes.
|  | Precision | Recall | F1 |
|----------|------|---------------|-------------|
| anger | 0.99 | 0.98 |0.99 |
| disgust |1.00 |0.98| 0.99 |
| fear |0.97 |0.99 |0.98 |
| joy |1.00| 1.00|1.00 |
| neutral |0.97 | 0.96| 0.97|
|sadness |0.97| 1.00| 0.98| 
| surprise | 0.94| 1.00| 0.98|
| macro avg| 0.98| 0.95| 0.96|

## Results
|  Model | Accuracy F1 | Notes |
|----------|------|---------------|
| MentalBERT | 0.98| Best overall; domain-adapted transformer|
| Random Forest| 0.96 | Best classical |
| Linear SVM | 0.95 | |
| RBF SVM | 0.95| |
|Weighted LSTM | 0.9 | Trained from scratch; limited by data size|
| Naïve Bayes | 0.83 | Strong precision, poor minority-class recall| 

## Key Findings
* Domain pretraining matters more than model size. MentalBERT edges out Random Forest not through architectural complexity alone, but because its pretraining corpus matches the target domain. The language of mental health dialogue — hedged expressions, emotionally ambiguous phrasing, clinical register — is underrepresented in general corpora and over-represented in MentalBERT's pretraining data.
* TF-IDF + Random Forest is a remarkably strong baseline. The near-parity between RF and MentalBERT suggests that emotion in this corpus is largely expressed through distinctive vocabulary rather than subtle syntactic or contextual cues.
* The characteristic confusion is fear/neutral/surprise → sadness. Across all models, residual errors concentrate at this boundary. This is linguistically coherent: in a mental health dialogue setting, fear, neutral affect, and surprise can all manifest with a subdued, heavy tone that overlaps with sadness at the surface level. This is a genuine ambiguity in the data, not a modelling failure.
* LSTM trained from scratch cannot compete with pretrained representations on small data. The LSTM result is not a failure — it is the correct result, and illustrates the core motivation for transfer learning in NLP.

## What's Next

* Word embeddings — replace TF-IDF with GloVe or Word2Vec averaged sentence vectors as features for classical models, and compare. This bridges the gap between bag-of-words and contextual representations.
* LSTM with pretrained embeddings — initialise the embedding layer with GloVe weights rather than training from scratch, to isolate the effect of representation quality from architecture choice.
* Hyperparameter search for MentalBERT — Optuna integration via `trainer.hyperparameter_search` to tune learning rate, batch size, and warmup steps.
* Error analysis — qualitative inspection of the fear/neutral/surprise–sadness confusion cases to understand whether they reflect genuine label ambiguity in the Hartmann model's annotations.
