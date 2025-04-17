import argparse
import glob
import unicodedata
import string
import duckdb
import pandas as pd
from datetime import date, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression  # For stacking
from sklearn.neural_network import MLPClassifier  # For neural network


from sentence_transformers import SentenceTransformer
from textblob import TextBlob  # For sentiment analysis
from xgboost import XGBClassifier  # Gradient boosting
import optuna  # Bayesian optimization
import re
import numpy as np
import ast

######################################
# STEP 1: DATA INGESTION (Modified to include movies_metadata.csv)
######################################

def load_data_duckdb(path_pattern):
    con = duckdb.connect(database=':memory:')
    sql = f"CREATE TABLE tmp AS SELECT * FROM read_csv_auto('{path_pattern}')"
    con.execute(sql)
    df = con.execute("SELECT * FROM tmp").df()
    print(f"[INFO] Loaded data from DuckDB path pattern: {path_pattern}. Shape={df.shape}")
    return df

def load_data_pandas(path_pattern):
    csv_files = glob.glob(path_pattern)
    if not csv_files:
        raise FileNotFoundError(f"[ERROR] No files found matching pattern: {path_pattern}")
    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    print(f"[INFO] Loaded data from Pandas path pattern: {path_pattern}. Shape={df.shape}")
    return df

def load_movies_metadata(metadata_path):
    """Loads the movies_metadata.csv file."""
    try:
        metadata_df = pd.read_csv(metadata_path)
        print(f"[INFO] Loaded movies metadata: {metadata_path}. Shape={metadata_df.shape}")
        # Data type conversions and cleaning for metadata_df
        metadata_df['budget'] = pd.to_numeric(metadata_df['budget'], errors='coerce').fillna(0)
        metadata_df['popularity'] = pd.to_numeric(metadata_df['popularity'], errors='coerce').fillna(0)
        metadata_df['revenue'] = pd.to_numeric(metadata_df['revenue'], errors='coerce').fillna(0)
        metadata_df['runtime'] = pd.to_numeric(metadata_df['runtime'], errors='coerce').fillna(0)
        metadata_df['vote_average'] = pd.to_numeric(metadata_df['vote_average'], errors='coerce').fillna(0)
        metadata_df['vote_count'] = pd.to_numeric(metadata_df['vote_count'], errors='coerce').fillna(0)
        for col in ['original_title', 'title', 'overview', 'tagline']:
            if col in metadata_df:
                metadata_df[col] = metadata_df[col].astype(str).str.strip().str.lower()
        return metadata_df
    except FileNotFoundError:
        print(f"[ERROR] Movies metadata file not found: {metadata_path}")
        return None
    except Exception as e:
        print(f"[ERROR] Error loading movies metadata: {e}")
        return None

######################################
# STEP 2: CLEANING
######################################

def clean_data(df):
    pattern = {r"\N", "\n", "nan", "none", "null", "na", "-", "?", ""}  # Use raw string for \N

    def unify_missing_values(df):
        def check_weird(cell):
            if isinstance(cell, str):
                cell_str = cell.strip().lower()
                if cell_str in pattern:
                    return pd.NA
            return cell

        return df.applymap(check_weird)
    df = unify_missing_values(df)

    for col in ['startYear', 'endYear', 'runtimeMinutes', 'numVotes']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['primaryTitle', 'originalTitle', 'genres']:
        if col in df.columns:
            df[col] = df[col].astype('string').str.strip().str.lower()

    print(f"[INFO] Cleaning done. Columns={df.columns.tolist()}. Shape={df.shape}")
    return df


def unify_release_year(df):
    if 'startYear' in df.columns and 'endYear' in df.columns:
        df['releaseYear'] = df.apply(
            lambda row: row['startYear'] if pd.notna(row['startYear']) else row['endYear'],
            axis=1
        )
    elif 'release_date' in df.columns:  # Prioritize release_date from metadata
        df['releaseYear'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

    if 'releaseYear' in df:  # Check if it exists now
        df['releaseYear'] = pd.to_numeric(df['releaseYear'], errors='coerce')
        min_year, current_year = 1890, date.today().year
        df.loc[df['releaseYear'] < min_year, 'releaseYear'] = current_year
        df.loc[df['releaseYear'] > current_year, 'releaseYear'] = current_year
        df.drop(columns=['startYear','endYear', 'release_date'], inplace=True, errors='ignore')
        print("[INFO] unify_release_year done: now have 'releaseYear' column.")

    return df

######################################
# IMPUTATIONS
######################################

def imputation_runtimeMinutes(df):
    if 'runtimeMinutes' in df.columns:
        # Use k-NN imputation
        imputer = KNNImputer(n_neighbors=5)  # Choose an appropriate number of neighbors
        df['runtimeMinutes'] = imputer.fit_transform(df[['runtimeMinutes']])
        print("[INFO] Imputed runtimeMinutes with k-NN.")
    return df

def imputation_numVotes(df):
    if 'numVotes' in df.columns:
        # Use k-NN imputation
        imputer = KNNImputer(n_neighbors=5)
        df['numVotes'] = imputer.fit_transform(df[['numVotes']])
        print("[INFO] Imputed numVotes with k-NN.")
    return df

######################################
# DIRECTORS/WRITERS LOGIC
######################################

def aggregate_directors_writers(df,
                                directing_json='imdb/directing.json',
                                writing_json='imdb/writing.json',
                                movies_csv='imdb/movies.csv',  # Added movies_csv parameter
                                training_data=False):
    """
    Aggregates director/writer information.
    """
    try:
        directing_df = pd.read_json(directing_json)
        writing_df   = pd.read_json(writing_json)
        print(f"[INFO] Loaded directing: {directing_json}, writing: {writing_json}")
    except FileNotFoundError:
        print("[WARN] No directing/writing JSON found. Skipping director/writer merges.")
        return df

    if 'movie' in directing_df.columns and 'tconst' not in directing_df.columns:
        directing_df = directing_df.rename(columns={'movie': 'tconst'})
    if 'movie' in writing_df.columns and 'tconst' not in writing_df.columns:
        writing_df   = writing_df.rename(columns={'movie': 'tconst'})

    # Load the movies data
    try:
        movies_df = pd.read_csv(movies_csv)
        print(f"[INFO] Loaded movies data: {movies_csv}")

        # Clean up movie titles by removing the year
        movies_df['title'] = movies_df['title'].apply(lambda x: re.sub(r' \(\d{4}\)$', '', x)) # Remove the year in parenthesis.
        movies_df['title'] = movies_df['title'].str.strip().str.lower()
        movies_df['genres'] = movies_df['genres'].astype('string').str.strip().str.lower()

    except FileNotFoundError:
        print(f"[WARN] No movies CSV found: {movies_csv}.  Skipping genre merge.")
        movies_df = None  # Set to None for later checks
    except Exception as e:
        print(f"[ERROR] Error loading or processing movies CSV: {e}")
        movies_df = None

    # --- Director/Writer Genre Expertise ---
    def calculate_genre_expertise(person_df, role):
        """Calculates genre distribution for directors or writers."""
        df = person_df.copy()
        if 'genres' not in df.columns:
            return pd.DataFrame() # Return empty if no genres

        df = df.dropna(subset=['genres'])
        df['genres_list'] = df['genres'].str.split('|')

        # Explode the list to have one row per genre per movie per person
        df_exploded = df.explode('genres_list')

        # Count the occurrences of each genre for each person
        genre_counts = df_exploded.groupby([role, 'genres_list']).size().reset_index(name='genre_count')

        # Calculate the total number of movies for each person
        total_movies = genre_counts.groupby(role)['genre_count'].sum().reset_index(name='total_count')

        # Merge the counts and totals
        genre_counts = genre_counts.merge(total_movies, on=role)

        # Calculate the genre distribution (expertise)
        genre_counts['genre_expertise'] = genre_counts['genre_count'] / genre_counts['total_count']

        # Pivot the table to have genres as columns
        expertise_df = genre_counts.pivot(index=role, columns='genres_list', values='genre_expertise').reset_index()
        expertise_df = expertise_df.fillna(0)  # Fill missing values with 0
        expertise_df = expertise_df.rename(columns={'genres_list': 'genre'})

        return expertise_df

    # Merge genres into directing_df and writing_df before expertise calculation
    if movies_df is not None:
        directing_df = pd.merge(directing_df, movies_df[['title', 'genres']], left_on='tconst', right_on='title', how='left')
        directing_df.drop('title', axis=1, inplace=True)
        writing_df = pd.merge(writing_df, movies_df[['title', 'genres']], left_on='tconst', right_on='title', how='left')
        writing_df.drop('title', axis=1, inplace=True)


    director_expertise = calculate_genre_expertise(directing_df, 'director')
    writer_expertise = calculate_genre_expertise(writing_df, 'writer')

    # --- Director Stats (example) ---
    # Calculate director stats on the entire directing_df or just training data

    # Initialize director_stats with basic counts
    director_stats = directing_df.groupby('director').agg(
        dir_movie_count=('tconst', 'nunique')
    ).reset_index()

    # Try to add average rating if the column exists
    if 'averageRating' in directing_df.columns:
        # Merge director stats into directing_df for averageRating calculation
        directing_df = directing_df.merge(director_stats, on='director', how='left')

        # Calculate director_stats including dir_avg_rating
        director_stats = directing_df.groupby('director').agg(
            dir_movie_count=('tconst', 'nunique'),
            dir_avg_rating=('averageRating', 'mean')  # Director's historical average rating
        ).reset_index()

        # Merge director stats into the main DataFrame
        df = df.merge(director_stats, on='director', how='left')
        df['dir_avg_rating'] = df['dir_avg_rating'].fillna(df['dir_avg_rating'].median())
    else:
        print("[WARN] 'averageRating' column not found in directing_df. Skipping dir_avg_rating calculation.")


    # Create an in-memory DuckDB connection
    con = duckdb.connect(database=':memory:')

    con.register("directors", directing_df)
    con.register("writers",   writing_df)
    con.register("df", df) #Register original df

    # --- Build features ---
    # a) Count how many distinct directors per tconst
    query_dir_count_by_movie = """
    SELECT
        tconst,
        COUNT(DISTINCT director) as director_count
    FROM directors
    GROUP BY tconst
    """
    df_dir_count_by_movie = con.execute(query_dir_count_by_movie).df()

    # b) Count how many distinct writers per tconst
    query_wri_count_by_movie = """
    SELECT
        tconst,
        COUNT(DISTINCT writer) as writer_count
    FROM writers
    GROUP BY tconst
    """
    df_wri_count_by_movie = con.execute(query_wri_count_by_movie).df()

    # c) "Any director also a writer"?
    query_dir_writer_any = """
    SELECT
       d.tconst,
       MAX(CASE WHEN d.director = w.writer THEN 1 ELSE 0 END) AS is_any_dir_writer
    FROM directors d
    JOIN writers w
      ON d.tconst = w.tconst
    GROUP BY d.tconst
    """
    df_dir_writer_any = con.execute(query_dir_writer_any).df()

    # d) Top 10 Directors by average votes
    query_dir_avg_votes = """
    SELECT
       d.director,
       AVG(m.numVotes) AS dir_avg_votes
    FROM directors d
    JOIN df m
        ON d.tconst = m.tconst
    GROUP BY d.director
    ORDER BY dir_avg_votes DESC
    """
    df_dir_avg_votes = con.execute(query_dir_avg_votes).df()
    top_10_directors = df_dir_avg_votes.head(10)['director'].tolist()

    # e) Top 10 Writers by average votes
    query_wri_avg_votes = """
    SELECT
       w.writer,
       AVG(m.numVotes) AS wri_avg_votes
    FROM writers w
    JOIN df m
        ON w.tconst = m.tconst
    GROUP BY w.writer
    ORDER BY wri_avg_votes DESC
    """
    df_wri_avg_votes = con.execute(query_wri_avg_votes).df()
    top_10_writers = df_wri_avg_votes.head(10)['writer'].tolist()

    # --- Merge these partial DataFrames in Python ---
    df_agg = df[['tconst']].drop_duplicates().copy()

    df_agg = df_agg.merge(df_dir_count_by_movie, on='tconst', how='left')
    df_agg = df_agg.merge(df_wri_count_by_movie, on='tconst', how='left')
    df_agg = df_agg.merge(df_dir_writer_any,   on='tconst', how='left')

    for c in ['director_count','writer_count','is_any_dir_writer']:
        df_agg[c] = df_agg[c].fillna(0).astype(int)


    # --- Create "is_top_10_director", "is_top_10_writer" ---
    directing_top10 = directing_df[ directing_df['director'].isin(top_10_directors) ]
    top10_dir_flags = directing_top10.groupby('tconst', as_index=False)\
                                 .agg(is_top_10_director=('director', 'size'))
    top10_dir_flags['is_top_10_director'] = (top10_dir_flags['is_top_10_director']>0).astype(int)
    top10_dir_flags = top10_dir_flags[['tconst','is_top_10_director']]

    writing_top10 = writing_df[ writing_df['writer'].isin(top_10_writers) ]
    top10_wri_flags = writing_top10.groupby('tconst', as_index=False)\
                               .agg(is_top_10_writer=('writer', 'size'))
    top10_wri_flags['is_top_10_writer'] = (top10_wri_flags['is_top_10_writer']>0).astype(int)
    top10_wri_flags = top10_wri_flags[['tconst','is_top_10_writer']]

    df_agg = df_agg.merge(top10_dir_flags, on='tconst', how='left')
    df_agg['is_top_10_director'] = df_agg['is_top_10_director'].fillna(0).astype(int)

    df_agg = df_agg.merge(top10_wri_flags, on='tconst', how='left')
    df_agg['is_top_10_writer'] = df_agg['is_top_10_writer'].fillna(0).astype(int)


    # --- Merge back to original df ---
    df_final = df.merge(df_agg, on='tconst', how='left')

    # Clean titles in df_final for matching
    df_final['originalTitle'] = df_final['originalTitle'].astype(str).apply(lambda x: re.sub(r' \(\d{4}\)$', '', x)) # Remove the year in parenthesis.
    df_final['originalTitle'] = df_final['originalTitle'].str.strip().str.lower()


    # --- Merge genres and expertise ---
    if movies_df is not None:
        df_final = pd.merge(df_final, movies_df[['title', 'genres']], left_on='originalTitle', right_on='title', how='left')
        df_final.drop('title', axis=1, inplace=True)  # Drop redundant title

        # Merge director expertise
        if not director_expertise.empty:
            df_final = df_final.merge(director_expertise, on='director', how='left', suffixes=('', '_dir_expertise'))
            for col in director_expertise.columns:
                if col != 'director':
                    df_final[col] = df_final[col].fillna(0)

        # Merge writer expertise
        if not writer_expertise.empty:
            df_final = df_final.merge(writer_expertise, on='writer', how='left', suffixes=('', '_wri_expertise'))
            for col in writer_expertise.columns:
                if col != 'writer':
                    df_final[col] = df_final[col].fillna(0)


    print(f"[INFO] Aggregated directors/writers. df_final shape={df_final.shape}")
    return df_final

######################################
# STEP 3: FEATURE ENGINEERING
######################################

def add_title_length(df):
    # Use 'title' if it exists, otherwise 'primaryTitle'
    title_col = 'title' if 'title' in df.columns else 'primaryTitle' if 'primaryTitle' in df.columns else None

    if title_col:
        title_length = df[title_col].apply(
            lambda x: len(x.replace(' ', '')) if isinstance(x, str) else 0
        )
        df = pd.concat([df, pd.DataFrame({'title_length': title_length})], axis=1) #Concat
        print(f"[INFO] Added title_length feature (using '{title_col}').")
    else:
        print("[WARN] Neither 'title' nor 'primaryTitle' found. Skipping title_length.")
    return df

def text_clean_for_tfidf(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.strip()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    return text

def text_clean_for_embedding(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    t = text.strip()
    cleaned = []
    for ch in t:
        if unicodedata.category(ch)[0] != 'C':
            cleaned.append(ch)
    t = ''.join(cleaned).lower()
    t = unicodedata.normalize('NFKD', t).encode('ascii','ignore').decode('utf-8')
    return t

def apply_tfidf(df, text_col='primaryTitle'):
    # Check for 'title' first, fallback to primaryTitle
    if 'title' in df.columns:
        text_col = 'title'
    elif text_col not in df.columns:
        print(f"[WARN] Text column '{text_col}' not found. Skipping TF-IDF.")
        return df

    df = df.copy()
    df['title_for_tfidf'] = df[text_col].fillna('').apply(text_clean_for_tfidf)

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['title_for_tfidf'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                            columns=vectorizer.get_feature_names_out())
    df_out = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    print(f"[INFO] apply_tfidf created {tfidf_df.shape[1]} TF-IDF columns.")
    return df_out

def apply_embeddings(df, text_col='primaryTitle'):
   # Check if 'title' column exists, otherwise use the provided text_col
    if 'title' in df.columns:
        text_col = 'title'  # Use 'title' if it exists
    elif text_col not in df.columns:
        print(f"[WARN] Text column '{text_col}' not found.  Skipping embeddings.")
        return df

    df = df.copy()
    df['title_for_emb'] = df[text_col].fillna('').apply(text_clean_for_embedding)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    emb = model.encode(df['title_for_emb'].tolist())
    emb_df = pd.DataFrame(emb, columns=[f'emb_{i}' for i in range(emb.shape[1])])
    df_out = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
    print(f"[INFO] apply_embeddings created {emb_df.shape[1]} embedding columns.")
    return df_out

def combine_tfidf_embeddings(df, tfidf_prefix='tfidf_', embedding_prefix='emb_', pca_components=50):
    """Combines TF-IDF and embeddings using PCA for dimensionality reduction."""
    tfidf_cols = [col for col in df.columns if tfidf_prefix in col]
    embedding_cols = [col for col in df.columns if embedding_prefix in col]

    tfidf_df = df[tfidf_cols].fillna(0)
    embedding_df = df[embedding_cols].fillna(0)

    # Apply PCA to embeddings
    pca = PCA(n_components=pca_components)
    pca_embeddings = pca.fit_transform(embedding_df)
    pca_df = pd.DataFrame(pca_embeddings, columns=[f'pca_{i}' for i in range(pca_components)])

    # Concatenate TF-IDF and PCA components
    combined_df = pd.concat([df.reset_index(drop=True), tfidf_df, pca_df.reset_index(drop=True)], axis=1) # Added tfidf

    print(f"[INFO] Combined TF-IDF and embeddings using PCA. Added {pca_components} PCA components.")
    return combined_df

def add_sentiment_analysis(df, text_col='primaryTitle'):
    """Adds sentiment analysis features (polarity and subjectivity) to the DataFrame."""
    # Check for 'overview' and 'tagline', create combined text column
    if 'overview' in df.columns and 'tagline' in df.columns:
        df['combined_text'] = df['overview'].fillna('') + ' ' + df['tagline'].fillna('')
        text_col = 'combined_text'
    elif 'overview' in df:
      text_col = 'overview'
    elif 'tagline' in df:
        text_col = 'tagline'
    elif text_col not in df.columns:
        print(f"[WARN] Text column for sentiment analysis not found. Skipping.")
        return df


    def analyze_sentiment(text):
        if pd.isna(text) or not isinstance(text, str):
            return 0, 0  # Return neutral sentiment for missing values
        analysis = TextBlob(text)
        return analysis.sentiment.polarity, analysis.sentiment.subjectivity

    # Use pd.concat for efficiency
    sentiments = df[text_col].apply(analyze_sentiment).tolist()
    sentiment_df = pd.DataFrame(sentiments, columns=['text_sentiment_polarity', 'text_sentiment_subjectivity'])
    df = pd.concat([df, sentiment_df], axis=1)
    #Remove the temporary columns created.
    df = df.drop(columns = ['combined_text'], errors = 'ignore')
    print("[INFO] Added sentiment analysis features.")
    return df

def extract_title_patterns(df, text_col='primaryTitle'):
    """Extracts common title patterns (e.g., "Part III", "The Return of...") as boolean flags."""
    # Check for 'title' and use it if available
    if 'title' in df.columns:
        text_col = 'title'
    elif text_col not in df.columns:
        print(f"[WARN] Text column '{text_col}' not found. Skipping title pattern extraction.")
        return df

    def check_pattern(text, pattern):
        if pd.isna(text) or not isinstance(text, str):
            return 0
        return int(pattern in text)

    df['title_contains_part'] = df[text_col].apply(lambda x: check_pattern(x, 'part'))
    df['title_contains_return'] = df[text_col].apply(lambda x: check_pattern(x, 'return'))
    # Add more patterns as needed
    print("[INFO] Extracted title patterns.")
    return df

def add_temporal_features(df):
    """Adds temporal features like movie age, month of release, seasonality, and trend."""
    current_year = date.today().year
    if 'releaseYear' in df.columns:
        df['movie_age'] = current_year - df['releaseYear']

        # --- Seasonality (using a rolling window) ---
        # Create a 'day_of_year' column (assuming releaseYear is the year of release)
        # If you have a more precise release date, use that instead.
        df['day_of_year'] = df['releaseYear'].apply(lambda year: (date(int(year), 1, 1) - date(int(year), 1, 1)).days + 1)

        # Define seasons (adjust as needed)
        df['season'] = pd.cut(df['day_of_year'],
                              bins=[0, 90, 180, 270, 366],  # Example: Spring, Summer, Autumn, Winter
                              labels=['winter', 'spring', 'summer', 'autumn'])

        # One-hot encode the season
        df = pd.get_dummies(df, columns=['season'], prefix='season', dummy_na=False)


        # --- Trend (number of movies in production that year) ---
        movie_count_per_year = df.groupby('releaseYear').size().reset_index(name='movies_in_year')
        df = df.merge(movie_count_per_year, on='releaseYear', how='left')

    print("[INFO] Added movie_age, seasonality, and trend features.")
    return df

def one_hot_encode_genres(df):
    if 'genres' not in df.columns:
        print("[WARN] 'genres' column not found, trying 'genre_names'.")
        if 'genre_names' in df.columns:
            df.rename(columns={'genre_names': 'genres'}, inplace=True)
        else:
            print("[WARN] No 'genres' or 'genre_names' column. Skipping one-hot encoding.")
            return df

    # Handle both string and list representations of genres
    if isinstance(df['genres'].iloc[0], str):  # Check type of the first element
        df['genres_list'] = df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x.split('|') if isinstance(x,str) else [])
    else: # Assuming list of dictionaries
      df['genres_list'] = df['genres'].apply(lambda x: [g['name'].lower() for g in x] if isinstance(x, list) else [])

    # Get unique genres
    unique_genres = []
    for genres in df['genres_list'].dropna():
        if isinstance(genres, list):
            unique_genres.extend(genres)
    unique_genres = list(set(unique_genres))

    # --- Genre Popularity/Relevance ---
    if 'numVotes' in df.columns:  # You could also use 'averageRating' if available
        # Calculate average numVotes per genre
        genre_popularity = {}
        for genre in unique_genres:
            # Filter rows containing the genre
            genre_movies = df[df['genres_list'].apply(lambda x: genre in x if isinstance(x, list) else False)]
            # Calculate mean numVotes (handle empty selections)
            avg_votes = genre_movies['numVotes'].mean() if not genre_movies.empty else 0
            genre_popularity[genre] = avg_votes

    elif 'vote_average' in df.columns: # Use metadata's vote_average if available
        genre_popularity = {}
        for genre in unique_genres:
          genre_movies = df[df['genres_list'].apply(lambda x: genre in x if isinstance(x, list) else False)]
          avg_votes = genre_movies['vote_average'].mean() if not genre_movies.empty else 0
          genre_popularity[genre] = avg_votes

    else:
        # Default popularity if numVotes isn't available
        genre_popularity = {genre: 1.0 for genre in unique_genres}

    # --- Efficient One-Hot Encoding and Popularity Weighting ---
    genre_dummies = pd.get_dummies(df['genres_list'].apply(pd.Series).stack()).groupby(level=0).sum()

    # Apply popularity weights
    for genre in unique_genres:
        if genre in genre_dummies.columns:
            genre_dummies[genre] = genre_dummies[genre] * genre_popularity.get(genre, 1.0)  # Use .get for safety

    genre_dummies = genre_dummies.rename(columns=lambda x: f"genre_{x}")

     # --- Genre Interactions (more efficiently) ---
    genre_cols = [col for col in genre_dummies.columns if col.startswith('genre_')]
    interaction_data = {}  # Store interaction columns in a dictionary
    for i in range(len(genre_cols)):
        for j in range(i + 1, len(genre_cols)):
            genre1 = genre_cols[i]
            genre2 = genre_cols[j]
            interaction_data[f'{genre1}_AND_{genre2}'] = genre_dummies[genre1] * genre_dummies[genre2]

    # Create a DataFrame from the interaction data
    interaction_df = pd.DataFrame(interaction_data)

    # Concatenate everything efficiently
    df = pd.concat([df.drop(columns=['genres', 'genres_list'], errors='ignore'), genre_dummies, interaction_df], axis=1)


    print("[INFO] One-hot encoded 'genres', added interactions and popularity weighting.")
    return df

def apply_pca(df, prefix, n_components=50):
    """Applies PCA to features starting with a given prefix."""
    cols = [col for col in df.columns if col.startswith(prefix)]
    if not cols:
        print(f"[WARN] No columns found with prefix '{prefix}'. Skipping PCA.")
        return df

    data = df[cols].fillna(0)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)
    pca_df = pd.DataFrame(pca_result, columns=[f'{prefix}pca_{i}' for i in range(n_components)])

    df = pd.concat([df.reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)
    df = df.drop(columns=cols) #Drop original columns
    print(f"[INFO] Applied PCA to '{prefix}' features. Added {n_components} PCA components.")
    return df

######################################
# Merging Metadata
######################################
def merge_metadata(df, metadata_df):
    """Merges the main DataFrame with the movies metadata."""
    if metadata_df is None:
        print("[WARN] metadata_df is None. Skipping merge.")
        return df

    # Clean up movie titles for merging
    metadata_df['original_title'] = metadata_df['original_title'].str.strip().str.lower()
    df['originalTitle'] = df['originalTitle'].str.strip().str.lower()

    # Perform the merge, handling potential duplicates in metadata
    df = pd.merge(df, metadata_df, left_on='originalTitle', right_on='original_title', how='left', suffixes=('', '_meta'))

    # Drop duplicate columns and rename _meta columns
    for col in metadata_df.columns:
        if col + '_meta' in df.columns:
            # Prefer the original column if it exists and has non-null values
            if col in df.columns:
                df[col] = df[col].fillna(df[col + '_meta'])
            else:  #If the original df does not contain it
                df.rename(columns={col + '_meta': col}, inplace=True)
            df.drop(columns=[col + '_meta'], inplace=True, errors='ignore')

    print(f"[INFO] Merged metadata. df shape={df.shape}")
    return df



######################################
# STEP 4: PIPELINE
######################################

def process_dataset(path_pattern, use_duckdb=True, feature_method='embedding', training_data=False, metadata_path='movies_metadata.csv'):
    """
    Main pipeline function.
    """
    # 1) Load
    if use_duckdb:
        df = load_data_duckdb(path_pattern)
    else:
        df = load_data_pandas(path_pattern)

    # Load metadata
    metadata_df = load_movies_metadata(metadata_path)

    # 2) Clean & unify releaseYear
    df = clean_data(df)
    df = unify_release_year(df)


    # 3) Optional imputations
    df = imputation_runtimeMinutes(df)
    df = imputation_numVotes(df)

    # 4) Merge directors/writers
    df = aggregate_directors_writers(df, training_data=training_data)


    # 5) Merge Metadata *BEFORE* feature engineering
    df = merge_metadata(df, metadata_df)

    # 6) Additional numeric feature
    df = add_title_length(df)

    # 7) Text feature method
    if feature_method == 'tfidf':
        df = apply_tfidf(df, text_col='primaryTitle')
    elif feature_method == 'embedding':
        df = apply_embeddings(df, text_col='primaryTitle')
    elif feature_method == 'combined':
        df = apply_tfidf(df, text_col='primaryTitle')
        df = apply_embeddings(df, text_col='primaryTitle')
        df = combine_tfidf_embeddings(df)


    # 8) Sentiment Analysis
    df = add_sentiment_analysis(df)

    # 9) Title Patterns
    df = extract_title_patterns(df)

    # 10) Temporal Features
    df = add_temporal_features(df)

    # 11) One-hot encode genres
    df = one_hot_encode_genres(df)

    # 12) Apply PCA
    df = apply_pca(df, prefix='genre_', n_components=20) # Example: Reduce genre features
    if feature_method == 'combined':
        df = apply_pca(df, prefix='pca_', n_components = 10) # Reduce combined features

    print(f"[INFO] process_dataset done. Final shape={df.shape}")
    return df

######################################
# STEP 5: MODEL TRAINING & EVALUATION
######################################
def train_and_evaluate_model(train_df, val_df=None, test_df=None, label_col='label', model_type='xgb'):
    """
    Trains and evaluates models, including Optuna hyperparameter optimization.
    Now supports multiple model types and stacking.
    """

    if label_col not in train_df.columns:
        raise ValueError(f"[ERROR] Label column '{label_col}' not in train data.")
    train_df = train_df.dropna(subset=[label_col])
    y_train = train_df[label_col].astype(bool)

    exclude_cols = [label_col, 'primaryTitle', 'title_for_tfidf', 'title_for_emb', 'tconst', 'director', 'writer', 'genres', 'title', 'overview', 'tagline', 'belongs_to_collection', 'homepage','imdb_id','original_language','poster_path','production_companies','production_countries', 'spoken_languages', 'status', 'video', 'id', 'adult'] #Added to remove string and other useless columns
    feature_cols = [c for c in train_df.columns
                    if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[c])]
    X_train = train_df[feature_cols].fillna(0)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # --- Prepare Validation Data ---
    if val_df is not None and not val_df.empty:
        # Make sure val_df has 'tconst' before processing
        if 'tconst' not in val_df.columns:
            val_df['tconst'] = val_df.index
        val_df_processed = process_and_prepare_for_prediction(val_df, scaler, feature_cols, label_col)
        X_val_scaled = val_df_processed['X']
        y_val = val_df_processed['y']  # This will be None if label_col is missing
    else:
        X_val_scaled = None
        y_val = None


    # --- Optuna Objective Function ---
    def objective(trial):
        if model_type == 'rf':
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 4, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42,
            }
            model = RandomForestClassifier(**param)

        elif model_type == 'xgb':
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),  # L1 regularization
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0), # L2 regularization
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False # Suppress a warning
            }
            model = XGBClassifier(**param)

        elif model_type == 'gb':  # Gradient Boosting
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'random_state': 42,
            }
            model = GradientBoostingClassifier(**param)
        
        elif model_type == 'nn':  # Neural Network
            param = {
                'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(64,), (128,), (64, 32), (128, 64)]),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),  # Regularization term
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
                'max_iter': trial.suggest_int('max_iter', 200, 500), # Increase max_iter
                'random_state': 42,
                'early_stopping': True,  # Enable early stopping
                'validation_fraction': 0.2, # Fraction for early stopping validation
            }
            model = MLPClassifier(**param)
            

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Use Stratified K-Fold within Optuna
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []

        for train_idx, val_idx in skf.split(X_train_scaled, y_train):
            X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_train_fold, y_train_fold)
            if hasattr(model, "predict_proba"): # For models that have predict_proba
                preds = model.predict_proba(X_val_fold)[:, 1]
                preds = (preds > 0.5).astype(bool)  # Threshold probabilities
            else:
                preds = model.predict(X_val_fold)
            
            accuracy = accuracy_score(y_val_fold, preds)
            accuracies.append(accuracy)

        return np.mean(accuracies)

    # --- Run Optuna Optimization ---
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)  # Adjust n_trials as needed

    print("[INFO] Optuna optimization complete.")
    best_params = study.best_params
    print(f"[INFO] Best parameters: {best_params}")
    print(f"[INFO] Best score: {study.best_value}")

    # --- Retrain with Best Parameters ---
    if model_type == 'rf':
        best_model = RandomForestClassifier(**best_params, random_state=42)
    elif model_type == 'xgb':
        best_params['random_state'] = 42
        best_params['eval_metric'] = 'logloss'
        best_params['use_label_encoder'] = False
        best_model = XGBClassifier(**best_params)
    elif model_type == 'gb':
        best_params['random_state'] = 42
        best_model = GradientBoostingClassifier(**best_params)
    elif model_type == 'nn':
        best_params['random_state'] = 42
        best_params['early_stopping'] = True
        best_model = MLPClassifier(**best_params)    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    best_model.fit(X_train_scaled, y_train)


    # ---- VALIDATION PREDICTION ----
    if val_df is not None and not val_df.empty:
      process_and_save_predictions(val_df, best_model, scaler, feature_cols, label_col, "validation_predictions.csv")

    # ---- TEST PREDICTION ----
    if test_df is not None and not test_df.empty:
        process_and_save_predictions(test_df, best_model, scaler, feature_cols, label_col, "test_predictions.csv")

    return best_model, best_params  # Return best model and parameters




def process_and_prepare_for_prediction(df, scaler, feature_cols, label_col):
    """
    Processes the DataFrame for prediction, handling missing columns and scaling.
    Returns a dictionary containing the scaled features (X) and the target (y),
    or None for y if the label column is missing.
    """
    df_copy = df.copy()

    # Ensure tconst is present
    if 'tconst' not in df_copy.columns:
        df_copy['tconst'] = df_copy.index

    df_copy = df_copy.drop_duplicates(subset=['tconst'], keep='first')

    missing_cols = set(feature_cols) - set(df_copy.columns)
    if missing_cols:
        missing_df = pd.DataFrame(0, index=df_copy.index, columns=list(missing_cols))
        df_copy = pd.concat([df_copy, missing_df], axis=1)

    df_copy = df_copy[feature_cols]
    X = df_copy.fillna(0)
    X_scaled = scaler.transform(X)

    # Handle the target variable (y)
    if label_col in df.columns:
        y = df[label_col].astype(bool)
    else:
        y = None  # No target variable available

    return {'X': X_scaled, 'y': y}


def process_and_save_predictions(df, model, scaler, feature_cols, label_col, output_filename):
    """Process input DataFrame, make predictions, and save to CSV."""
    df_copy = df.copy()

    # Ensure tconst is in df_copy
    if 'tconst' not in df_copy.columns:
        print("[WARN] 'tconst' column missing in DataFrame. Adding it using index.")
        df_copy['tconst'] = df_copy.index  # Or use a suitable identifier if available

    # Ensure one prediction per tconst
    df_copy = df_copy.drop_duplicates(subset=['tconst'], keep='first') # Keep the first occurence if there are duplicates


    # --- Efficiently add missing columns ---
    missing_cols = set(feature_cols) - set(df_copy.columns)
    if missing_cols:  # Only create the DataFrame if there are missing columns
        missing_df = pd.DataFrame(0, index=df_copy.index, columns=list(missing_cols))
        df_copy = pd.concat([df_copy, missing_df], axis=1)


    # Ensure correct column order
    df_copy = df_copy[feature_cols]

    X = df_copy.fillna(0)
    X = scaler.transform(X)
    
    if hasattr(model, "predict_proba"): # Models like RF, XGBoost, GB
      preds = model.predict_proba(X)[:, 1]
      preds = (preds > 0.5).astype(bool)
    else: # Models like LogisticRegression, LinearSVC (no proba)
      preds = model.predict(X)

    # Convert 1 -> True and 0 -> False
    df_copy['prediction'] = preds.astype(bool)

    df_copy = df_copy.copy() # Defragment the copy

    output_df = pd.DataFrame({'prediction': df_copy['prediction']})
    output_df.to_csv(output_filename, index=False)

    print(f"[INFO] Predictions saved to {output_filename}")

######################################
# STEP 6: MAIN
######################################
if __name__ == "__main__":
    """
    Example usage:
    python pipeline.py
    """

    # 1) Process train
    print("[INFO] Processing TRAIN data ...")
    df_train = process_dataset('imdb/train-*.csv', use_duckdb=True, feature_method='combined', training_data=True)  # Flag as training data

    # 2) Process val (some might have no label)
    print("[INFO] Processing VALIDATION data ...")
    df_val = process_dataset('imdb/validation_*.csv', use_duckdb=True, feature_method='combined')

    # 3) Process test
    print("[INFO] Processing TEST data ...")
    df_test = process_dataset('imdb/test_*.csv', use_duckdb=True, feature_method='combined')

    # Process val hidden and test hidden
    print("[INFO] Processing VALIDATION HIDDEN data ...")
    df_val_hidden = process_dataset('imdb/validation_hidden.csv', use_duckdb=True, feature_method='combined')

    # Process test hidden and test hidden
    print("[INFO] Processing TEST HIDDEN data ...")
    df_test_hidden = process_dataset('imdb/test_hidden.csv', use_duckdb=True, feature_method='combined')

    # 4) Train & Evaluate (now includes test set and Optuna)
    print("[INFO] Training & evaluating ...")
    model, best_params = train_and_evaluate_model(df_train, df_val, df_test, label_col='label', model_type='xgb') # Changed to Optuna

    # Make prediction on val and test hidden and save them.
    # Get feature columns after training
    exclude_cols = ['label', 'primaryTitle', 'title_for_tfidf', 'title_for_emb', 'tconst',  'director', 'writer', 'genres', 'title', 'overview', 'tagline', 'belongs_to_collection', 'homepage','imdb_id','original_language','poster_path','production_companies','production_countries', 'spoken_languages', 'status', 'video', 'id', 'adult']
    feature_cols = [c for c in df_train.columns
                    if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_train[c])]

    # Scale the features
    X_train = df_train[feature_cols].fillna(0)
    scaler = StandardScaler()
    scaler.fit(X_train)


    # Process validation_hidden
    print("[INFO] Generating predictions for validation_hidden.csv ...")
    process_and_save_predictions(df_val_hidden, model, scaler, feature_cols, 'label', "validation_predictions.csv")

    # Process test_hidden
    print("[INFO] Generating predictions for test_hidden.csv ...")
    process_and_save_predictions(df_test_hidden, model, scaler, feature_cols, 'label', "test_predictions.csv")

    print("[INFO] Pipeline complete.")