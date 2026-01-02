# Movie-Rating-Classifier

## Project Overview
Developed a binary classifier to distinguish between highly-rated and low-rated movies based on metadata and textual features. The model uses features such as title, release year, runtime, votes, writers, directors, genre, and sentiment from movie overviews and taglines.

---

## Technologies
- Python (pandas, DuckDB, glob)
- ML: scikit-learn, XGBoost, Optuna
- NLP: SentenceTransformers, TextBlob, TF-IDF, PCA

---

## Data & Features
- **Movie Metadata:** Title, release year, runtime, votes, writers, directors
- **Credits:** Writers and directors
- **Custom Features:**
  - Sentiment: Polarity & subjectivity from overview and tagline
  - Genre: Multi-genre encoding, weighted, PCA-reduced
  - Temporal: Movie age, seasonal trends
  - Director/Writer metrics: Expertise, top-10 flags
  - Text embeddings: Title embeddings, length, keyword flags

---

## Modeling & Pipeline
- Unified ingestion and train/validation/test splits
- Robust data cleaning and KNN imputation for sparse features
- Feature scaling with StandardScaler
- XGBoost and RandomForest models, optimized via Optuna and Grid Search
- Hyperparameter tuning and cross-validation for model robustness

---

## Results & Conclusions
- Validation Accuracy: 80%  
- Outperforms baseline models
- Feature engineering and NLP features significantly improved predictive power
- Demonstrates ability to combine structured and unstructured data in a unified ML pipeline


**Outperforms random and baseline models**



