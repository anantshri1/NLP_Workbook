# NLP Workbook

A collection of NLP projects built with HuggingFace Transformers, TensorFlow/Keras, 
and classical methods. Each notebook covers a distinct task and architecture.

| Notebook | Task | Key Technique | Best Result |
|----------|------|---------------|-------------|
| Tweets: Sarcasm & Irony | 4-class classification of tweets. Compares LSTM with classical approaches. | BoW + LSTM | LSTM best on sarcasm/irony; RF best on figurative |
| Amazon Reviews | Weighted DistilBERT fine-tuning on a large-scale review dataset with class imbalance correction. Compares transformer vs. classical approaches on 3-class sentiment. |BoW → LSTM → DistilBERT | Weighted DistilBERT > Weighted LSTM > classical |
| Mental Health Responses | Fine-tuned MentalBERT (domain-specific BERT) on a mental health dialogue dataset for multiclass emotion classification. Covers class imbalance handling, tokenisation, and evaluation across 7 emotion categories. | TF-IDF → LSTM → MentalBERT | MentalBERT > Random Forest > classical > Weighted LSTM |
| Tweets: Disaster vs None | Benchmarked DistilBERT fine-tuning against GloVe embeddings + LSTM and classical baselines for binary classification of disaster-related tweets. | TF-IDF vs GloVe on RandomForest → LSTM → Weighted DistilBERT | TF-IDF on RandomForest best on Majority; Weighted DistilBERT best on Minority |


**Stack:** HuggingFace Transformers · TensorFlow/Keras · scikit-learn · NLTK · pandas
