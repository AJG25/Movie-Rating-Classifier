# Movie-Rating-Classifier

## Project Overview
This project develops a binary classifier to distinguish between highly-rated and low-rated movies based on structured metadata and textual features. The dataset includes information such as title, release year, runtime, number of votes, and details about writers and directors. The goal of the analysis is to predict whether a movie is highly rated based on features like title, release year, runtime, votes, writers, and directors.

## Data
The dataset includes:
-Movie Metadata: Title, release year, runtime, number of votes, writers, and directors
-Credits: Writers and directors

## Custom Engineered Features:
-Sentiment Features: Overview and tagline for sentiment analysis
-Genre Information: Multi-genre encoding, popularity-based weighting and PCA reduction
-Temporal Features: Movie age, seasonal effects, and trends over time

These additional features were created to enhance the predictive signal beyond the raw metadata.

## Pipeline Summary
Tools & Technologies
-Data Handling: DuckDB, pandas, glob
-ML Libraries: scikit-learn, XGBoost, Optuna
-NLP & Embeddings: SentenceTransformers, TextBlob, TF-IDF, PCA

## Data Processing
-Unified ingestion across multiple files and splits ((train/validation/test splits)
-Robust data cleaning (handling missing values, type coercion, and normalization)
-KNN imputation for sparse numerical features (e.g., runtime, votes)

## Feature Engineering
-Director/Writer Metrics: Expertise and top-10 flags based on genre history
-Text Features: Title embeddings (MiniLM), length, and keyword flags
-Temporal Features: Movie age, seasonal trends, yearly distribution
-Sentiment: Polarity & subjectivity from overview and tagline
-Genre Encoding: One-hot, weighted, PCA-reduced

## Modeling
-XGBoost and RandomForest models, optimized with Optuna
-Feature scaling using StandardScaler
-Hyperparameter tuning with Grid Search

## Performance
-Validation Accuracy: 80.0%

Outperforms random and baseline models
