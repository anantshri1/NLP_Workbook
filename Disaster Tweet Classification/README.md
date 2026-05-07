# Disaster Tweet Classification — End-to-End NLP Pipeline

A full NLP pipeline for binary classification of disaster tweets, benchmarking classical ML,
GloVe-powered deep learning, and a fine-tuned transformer.

**Dataset:** [NLP Getting Started — Disaster Tweets](https://www.kaggle.com/datasets/vbmokin/nlp-with-disaster-tweets-cleaning-data/data)  
**Task:** Binary classification — real disaster (1) vs not (0)  
**Class distribution:** 57% No Disaster / 43% Disaster

---

## Pipeline

```
Raw CSV
  └─► EDA (length distributions, class balance, vocabulary)
        └─► Text Cleaning (lowercase, URLs, mentions, hashtags, contractions, lemmatisation)
              └─► Train/Test Split (80/20, stratified)
                    ├─► TF-IDF (unigrams + bigrams)
                    │     ├─► RandomForest (TF-IDF baseline)
                    │     └─► RandomForest (averaged GloVe vectors)
                    ├─► GloVe sequences (glove-wiki-gigaword-100, padded to 50)
                    │     └─► LSTM (pretrained GloVe, class-weighted)
                    └─► Raw tokenised text
                          └─► DistilBERT (fine-tuned, custom weighted loss)
```

---

## Setup

```bash
pip install numpy pandas scikit-learn matplotlib seaborn nltk spacy
pip install tensorflow torch transformers gensim
python -m spacy download en_core_web_sm
```

---

## Stage 1 — EDA

- Dataset: ~7,600 tweets, two classes
- Class imbalance: 57% No Disaster, 43% Disaster — mild but handled via class weighting throughout
- 95th percentile tweet length: ~26 tokens after cleaning — informed `max_len=50` for LSTM

---

## Stage 2 — Text Cleaning

```python
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'http\\S+|https\\S+|www\\.\\S+|bit\\.ly\\S*', '', text)
    text = re.sub(r'@\\w+', '<MENTION>', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'([a-z])\\1{2,}', r'\\1\\1', text)   # wiiild -> wiild
    text = re.sub(r'\\s+', ' ', text).strip()
    # contraction expansion, stopword removal, lemmatisation
    ...
```

Key decisions: `bit.ly` shortlinks required an explicit pattern (single-slash URLs bypass
standard `https?://` regex). Whitespace collapse applied last. Character deduplication caps
repeated chars at 2.

---

## Stage 3 — GloVe Embeddings

Loaded via gensim: `glove-wiki-gigaword-100` (400,000 vocab, 100-dimensional vectors).

```python
import gensim.downloader as api
glove = api.load("glove-wiki-gigaword-100")
```

Embedding matrix built from training vocabulary only (12,918 tokens), 9,738 non-zero rows.
Sequences padded to `max_len=50` with `padding='pre'`.

Two representations compared:

- **Averaged GloVe** — mean-pool all token vectors per tweet → fixed 100-dim vector → RF input
- **GloVe sequences** — integer sequences with embedding matrix initialisation → LSTM input

---

## Stage 4 — Random Forest Baselines

`RandomForestClassifier(n_estimators=1000, class_weight='balanced')`

### RF + TF-IDF

| Class       | Precision | Recall | F1   |
|-------------|-----------|--------|------|
| No Disaster | 0.79      | 0.90   | 0.84 |
| Disaster    | 0.84      | 0.68   | 0.75 |
| **accuracy**|           |        | **0.81** |

### RF + Averaged GloVe

| Class       | Precision | Recall | F1   |
|-------------|-----------|--------|------|
| No Disaster | 0.79      | 0.90   | 0.84 |
| Disaster    | 0.83      | 0.68   | 0.75 |
| **accuracy**|           |        | **0.80** |

TF-IDF marginally outperforms averaged GloVe. Averaging collapses the tweet to a single vector,
destroying word order and diluting signal from high-discriminative tokens like "earthquake" or
"evacuate". TF-IDF weights these directly.

---

## Stage 5 — LSTM with GloVe Embeddings

Single-layer LSTM with pretrained GloVe initialisation, fine-tuned during training.

```
Embedding(12918, 100, weights=[embedding_matrix], trainable=True)
SpatialDropout1D(0.2)
LSTM(64, dropout=0.3, recurrent_dropout=0.3)
Dense(32, relu)
Dense(1, sigmoid)
```

Loss: `binary_crossentropy`. Optimiser: Adam (lr=0.0001). Early stopping patience=5.
Predictions: `(model.predict(X) > 0.5).astype(int).flatten()` — sigmoid output, not argmax.

| Class       | Precision | Recall | F1   |
|-------------|-----------|--------|------|
| No Disaster | 0.82      | 0.87   | 0.85 |
| Disaster    | 0.81      | 0.75   | 0.78 |
| **accuracy**|           |        | **0.82** |

Outperforms both RF baselines. Sequential GloVe representations capture contextual word
relationships that averaging and TF-IDF miss.

---

## Stage 6 — DistilBERT (fine-tuned, custom weighted loss)

`distilbert-base-uncased` with a custom `WeightedTrainer` subclassing HuggingFace `Trainer`
to apply per-class loss weighting at training time.

```python
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        weights = torch.tensor([
            len(y_train) / (2 * (y_train == 0).sum()),
            len(y_train) / (2 * (y_train == 1).sum()),
        ], dtype=torch.float).to(outputs.logits.device)
        loss = nn.CrossEntropyLoss(weight=weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss
```

Training: 3 epochs, batch size 8, warmup steps 500, weight decay 0.01, max length 128.
Training loss converged to 0.225. Predictions: `np.argmax(preds.predictions, axis=1)`.

| Class       | Precision | Recall | F1   |
|-------------|-----------|--------|------|
| No Disaster | 0.83      | 0.83   | 0.83 |
| Disaster    | 0.77      | 0.77   | 0.77 |
| **accuracy**|           |        | **0.80** |

---

## Results Summary

| Model                        | Accuracy | Disaster F1 | No Disaster F1 |
|------------------------------|----------|-------------|----------------|
| RF + TF-IDF                  | 0.81     | 0.75        | 0.84           |
| LSTM + GloVe                 | 0.82     | 0.78        | 0.85           |
| RF + Averaged GloVe          | 0.80     | 0.75        | 0.84           |
| DistilBERT (weighted)        | 0.80     | 0.77        | 0.83           |

---

## Key Findings

**RF + TF-IDF is a strong and stable baseline.** High-discriminative disaster vocabulary
("earthquake", "wildfire", "evacuate") is weighted directly by TF-IDF, and RF captures
this without needing sequential context.

**Averaged GloVe does not improve over TF-IDF.** Mean-pooling destroys word order and
dilutes signal from individually predictive tokens — the key limitation of any bag-of-vectors
approach on short texts.

**LSTM + GloVe sequences is the best model.** Processing tokens sequentially with pretrained
embeddings recovers context that averaging loses. The improvement over RF (+1 point accuracy,
+3 F1 on Disaster class) is modest but consistent.

**DistilBERT underperforms expectations.** Without further hyperparameter tuning and with
only 3 epochs on CPU/MPS, it does not outperform the LSTM. The custom weighted loss helps
minority class recall but the model needs Optuna tuning to reach its ceiling.

**The Disaster class is consistently harder across all models.** Tweets about disasters are
linguistically diverse — metaphor ("this traffic is a disaster"), sarcasm, and ambiguous
phrasing all increase false negatives. No Disaster tweets tend to have more distinctive
casual vocabulary that all models handle well.

---

## What's Next

- Optuna hyperparameter search for DistilBERT (`learning_rate`, `batch_size`, `num_epochs`)
