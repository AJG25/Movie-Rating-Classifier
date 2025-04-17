import pandas as pd
import glob
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import string
import unicodedata

def load_and_merge_train_data(train_path='imdb/train-*.csv'):
    """Loads all training CSV files and merges them into a single DataFrame."""
    train_files = glob.glob(train_path)
    if not train_files:
        raise FileNotFoundError("No files found matching the given path pattern.")

    df_list = [pd.read_csv(file) for file in train_files]
    train_df = pd.concat(df_list, ignore_index=True)
    
    if 'Unnamed: 0' in train_df.columns:
        train_df = train_df.drop(columns=['Unnamed: 0'])
    
    return train_df

def load_dataset(dataset_path):
    """Loads a dataset from a given path."""
    return pd.read_csv(dataset_path)

def clean_data(df):
    """Cleans the dataset by handling missing values, fixing incorrect formats, and ensuring correct data types."""
    
    pattern = r'(?i)^(?:\s*|\\N|\\n|nan|none|null|na|-|\?)$'
    df = df.replace(to_replace=pattern, value=pd.NA, regex=True)

    # 2. Convert known numeric columns to nullable integer
    df['startYear'] = pd.to_numeric(df['startYear'], errors='coerce').astype('Int64')
    df['endYear'] = pd.to_numeric(df['endYear'], errors='coerce').astype('Int64')
    df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'], errors='coerce').astype('Int64')
    df['numVotes'] = pd.to_numeric(df['numVotes'], errors='coerce').astype('Int64')

    # 3. Convert selected text columns to string dtype and strip whitespace
    #    (keeping <NA> intact for missing)
    df['primaryTitle'] = df['primaryTitle'].astype('string').str.strip()
    df['originalTitle'] = df['originalTitle'].astype('string').str.strip()

    return df

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

def analyze_data_quality(df):
    """
    Analyzes data quality including completeness, uniqueness, and outliers.
    Also generates histograms and boxplots for numeric & boolean columns.
    """

    # Common missing-like values
    missing_values = ['\\n', 'nan', 'null', 'none', '-', '?', '', 'na']

    results = []
    numeric_columns = []  # To store numeric columns for later plotting

    for column in df.columns:
        col_data = df[column]
        col_type = col_data.dtype
        total_values = len(col_data)

        # Replace placeholders in object/string columns with NaN
        if col_type == object:
            col_data_str = col_data.astype("string").str.strip().str.lower()
            col_data_str = col_data_str.replace(missing_values, pd.NA)
            df[column] = col_data_str
            col_data = df[column]
        
        # Count missing values
        missing_values_count = col_data.isna().sum()
        missing_percent = (missing_values_count / total_values) * 100

        # Uniqueness & duplicates
        unique_values = col_data.nunique(dropna=True)
        duplicate_count = col_data[col_data.notna()].duplicated().sum()
        sample_values = col_data.dropna().unique()[:5]  

        # Outlier detection for numeric columns (excluding booleans)
        if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data):
            numeric_columns.append(column)
            col_data_numeric = pd.to_numeric(col_data, errors='coerce')

            # IQR method
            q1 = col_data_numeric.quantile(0.25)
            q3 = col_data_numeric.quantile(0.75)
            iqr = q3 - q1
            lower_bound_iqr = q1 - 1.5 * iqr
            upper_bound_iqr = q3 + 1.5 * iqr
            iqr_mask = (col_data_numeric < lower_bound_iqr) | (col_data_numeric > upper_bound_iqr)
            iqr_outliers = iqr_mask.sum()

            # MAD method
            median = col_data_numeric.median()
            mad = stats.median_abs_deviation(col_data_numeric.dropna())
            mad_threshold = 3
            lower_bound_mad = median - mad_threshold * mad
            upper_bound_mad = median + mad_threshold * mad
            mad_mask = (col_data_numeric < lower_bound_mad) | (col_data_numeric > upper_bound_mad)
            mad_outliers = mad_mask.sum()

            # Trimmed mean (removing 10% from each tail)
            trimmed_mean = stats.trim_mean(col_data_numeric.dropna(), proportiontocut=0.1)
        else:
            iqr_outliers = "N/A"
            mad_outliers = "N/A"
            trimmed_mean = "N/A"

        results.append({
            "Column": column,
            "Data Type": col_type,
            "Total Values": total_values,
            "Missing Values": missing_values_count,
            "Missing %": round(missing_percent, 2),
            "Unique Values": unique_values,
            "Duplicate Count": duplicate_count,
            "IQR Outliers": iqr_outliers,
            "MAD Outliers": mad_outliers,
            "Trimmed Mean": trimmed_mean,
            "Sample Values": sample_values
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Missing %", ascending=False).reset_index(drop=True)

    #  **Generate Charts for Numeric Data**
    num_numeric = len(numeric_columns)

    if num_numeric > 0:
        fig, axes = plt.subplots(nrows=(num_numeric // 2 + num_numeric % 2), ncols=2, figsize=(12, 5 * (num_numeric // 2 + num_numeric % 2)))
        axes = axes.flatten()

        for idx, col in enumerate(numeric_columns):
            # Histogram
            sns.histplot(df[col].dropna(), bins=30, kde=True, ax=axes[idx])
            axes[idx].set_title(f"Distribution of {col}")
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel("Frequency")
        
        # Hide empty subplots if odd number of charts
        for idx in range(num_numeric, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()

        # **Boxplots for Outlier Visualization**
        fig, axes = plt.subplots(nrows=(num_numeric // 2 + num_numeric % 2), ncols=2, figsize=(12, 5 * (num_numeric // 2 + num_numeric % 2)))
        axes = axes.flatten()

        for idx, col in enumerate(numeric_columns):
            sns.boxplot(y=df[col], ax=axes[idx])
            axes[idx].set_title(f"Boxplot of {col}")

        for idx in range(num_numeric, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()

    return results_df



def clean_titles(df, title_column):
    """Standardizes the movie titles by converting to lowercase and removing unnecessary spaces."""
    df[title_column] = df[title_column].str.lower().str.strip()
    return df


def delete_duplicated_rows(df):
    dup = df[df.duplicated()].shape[0]
    df = df.drop_duplicates(subset = None, keep ='first', inplace = False)
    return (df)

def whitespace(df, col):
    df[col] = df[col].str.strip()
    df[col] = df[col].str.replace(r'/^(\s){1,}$/', '')
    return(df)

def split_attached_words(df, col):
    #artist.apply(lambda row: ','.join(x for x in sorted(row.roles)), axis=1)
    df[col] = df[col].apply(lambda row: row if pd.isnull(row) else  ' '.join(re.findall('[A-Z][^A-Z]*', row)))
    return(df)

def character_normalization(df, col):
    from unidecode import unidecode
    df[col] = df[col].apply(lambda row: row if pd.isnull(row) else unidecode(row))
    return (df)


def plot_numerical_distributions(df):
    """
    Function to plot distributions of numerical columns and check for normality using the Shapiro-Wilk test.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    None: Displays plots and prints normality test results.
    """
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    for column in numerical_columns:
        plt.figure(figsize=(10, 5))

        # Histogram
        sns.histplot(df[column].dropna(), bins=30, kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

        # Normality test
        stat, p_value = stats.shapiro(df[column].dropna())
        print(f"Shapiro-Wilk Test for {column}: W={stat:.4f}, p={p_value:.4f}")
        if p_value > 0.05:
            print(f"{column} appears to follow a normal distribution (p > 0.05).\n")
        else:
            print(f"{column} does NOT follow a normal distribution (p ≤ 0.05).\n")


def plot_numeric_counts(df, column_name, bin_size=None):
    """
    Plots a bar chart showing the count of unique numeric values (e.g., years, budgets).
    """

    df_filtered = df.dropna(subset=[column_name])


    if pd.api.types.is_numeric_dtype(df_filtered[column_name]):
        df_filtered[column_name] = df_filtered[column_name].astype(int)

        if bin_size:
            min_val, max_val = df_filtered[column_name].min(), df_filtered[column_name].max()
            bins = list(range(min_val, max_val + bin_size, bin_size))
            labels = [f"{b}-{b + bin_size - 1}" for b in bins[:-1]]
            df_filtered[column_name] = pd.cut(df_filtered[column_name], bins=bins, labels=labels, right=False)

    value_counts = df_filtered[column_name].value_counts().sort_index()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(value_counts.index.astype(str), value_counts.values)
    plt.xlabel(column_name)
    plt.ylabel("Count")
    plt.title(f"Count of '{column_name}' Values")
    plt.xticks(rotation=45, ha='right')

    if len(value_counts) > 20:
        plt.xticks(ticks=range(0, len(value_counts), max(1, len(value_counts) // 10)),
                   labels=value_counts.index[::max(1, len(value_counts) // 10)])
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_categorical_counts(df, column_name):
    """
    Plots a bar chart showing the count of unique categorical or boolean values.
    """

    df_filtered = df.dropna(subset=[column_name])

    if df_filtered[column_name].dtype == 'bool':
        df_filtered[column_name] = df_filtered[column_name].astype(str)

    elif pd.api.types.is_categorical_dtype(df_filtered[column_name]):
        df_filtered[column_name] = df_filtered[column_name].astype(str)

    value_counts = df_filtered[column_name].value_counts()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(value_counts.index.astype(str), value_counts.values) 
    plt.xlabel(column_name)
    plt.ylabel("Count")
    plt.title(f"Count of Unique Values in '{column_name}'")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def fingerprint_keying(text):
    """
    Applies "Fingerprint" Keying to normalize text by:
    1. Removing surrounding whitespace.
    2. Converting to lowercase.
    3. Removing punctuation and control characters.
    4. Converting characters to their ASCII equivalents.
    5. Tokenizing (splitting by whitespace).
    6. Sorting and deduplicating tokens.
    
    Args:
        text (str): The input text string.

    Returns:
        str: The fingerprinted version of the text.
    """
    if pd.isna(text) or not isinstance(text, str):
        return None  # Return None for missing or invalid data

    # Step 1: Trim whitespace
    text = text.strip()

    # Step 2: Convert to lowercase
    text = text.lower()

    # Step 3: Remove punctuation and control characters
    text = ''.join(char for char in text if char not in string.punctuation and unicodedata.category(char)[0] != 'C')

    # Step 4: Convert to ASCII equivalent (normalize Unicode)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    # Step 5: Tokenize (split by whitespace)
    tokens = text.split()

    # Step 6: Sort and deduplicate tokens
    unique_sorted_tokens = sorted(set(tokens))

    # Join tokens back into a single string
    return ' '.join(unique_sorted_tokens)

