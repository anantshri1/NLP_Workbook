# NLP Workbook

A progression of Natural Language Processing (NLP) notebooks built during a career transition from theoretical 
physics to ML/AI research. Each notebook adds a layer of complexity — from 
lightweight feature engineering to fine-tuned transformers — on progressively 
harder tasks.

| Notebook | Task | Key Technique | Best Result |
|----------|------|---------------|-------------|
| Tweets: Sarcasm & Irony | 4-class classification | BoW + LSTM | LSTM best on sarcasm/irony; RF best on figurative |
| Amazon Reviews | 3-class sentiment | BoW → LSTM → DistilBERT | Weighted DistilBERT > Weighted LSTM > classical |
| Mental Health Responses | 7-class sentiment | TF-IDF → LSTM → MentalBERT | MentalBERT > Random Forest > classical > Weighted LSTM |
