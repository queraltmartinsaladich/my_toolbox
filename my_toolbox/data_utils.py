import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def inspect_structure(df, label_col=None):
    """Check for missing values and types. Label distribution is optional."""
    print("\n" + "="*30)
    print("      DATASET STRUCTURE")
    print("="*30)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\n[!] Missing Values Detected:")
        print(missing[missing > 0])
    else:
        print("\n[+] No missing values detected.")

    if label_col and label_col in df.columns:
        print(f"\n=== LABEL DISTRIBUTION ({label_col}) ===")
        counts = df[label_col].value_counts()
        percent = df[label_col].value_counts(normalize=True) * 100
        dist_df = pd.DataFrame({'Count': counts, 'Percentage': percent.map("{:.2f}%".format)})
        print(dist_df)
    elif label_col:
        print(f"\n[!] Warning: Label column '{label_col}' not found in DataFrame.")
    
    return missing

def inspect_text_quality(df, text_col="text"):
    """Deep dive into text lengths. Safe-guards against non-string types."""
    if text_col not in df.columns:
        print(f"[!] Error: Text column '{text_col}' not found.")
        return None

    texts = df[text_col].fillna("").astype(str)
    lengths = texts.apply(len)
    words = texts.apply(lambda x: len(x.split()) if x.strip() else 0)

    print("\n" + "="*30)
    print(f"   TEXT QUALITY: {text_col}")
    print("="*30)
    stats = pd.DataFrame({
        'Chars': lengths.describe(),
        'Words': words.describe()
    })
    print(stats.round(2))

    print("\n--- SAMPLES ---")
    print(f"Shortest ({lengths.min()} chars): {texts.loc[lengths.idxmin()][:80]}...")
    print(f"Longest ({lengths.max()} chars):  {texts.loc[lengths.idxmax()][:80]}...")
    
    return {"char_lengths": lengths, "word_counts": words}

def plot_data_health(df, text_col="text", label_col=None, output_dir="results", analysis="test"):
    """Visual summary. Dynamically adjusts if label_col is missing."""
    os.makedirs(output_dir, exist_ok=True)
    has_labels = label_col and label_col in df.columns
    fig_cols = 2 if has_labels else 1
    fig, axes = plt.subplots(1, fig_cols, figsize=(7 * fig_cols, 5))
    
    if fig_cols == 1:
        axes = [axes]

    words = df[text_col].fillna("").astype(str).apply(lambda x: len(x.split()))
    sns.histplot(words, bins=30, ax=axes[0], color='skyblue', kde=True)
    axes[0].set_title(f'Word Count Distribution: {text_col}')
    axes[0].set_xlabel('Word Count')

    if has_labels:
        sns.countplot(x=label_col, data=df, ax=axes[1], palette='Blues_d')
        axes[1].set_title(f'Class Balance: {label_col}')
        axes[1].set_xlabel('Class')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"data_health_check_{analysis}.png"))
    plt.close()

def full_data_inspection(df, text_col="text", label_col=None, output_dir="results", analysis="text"):
    """
    The one-stop-shop for data checking. 
    Works for both training (labeled) and inference (unlabeled) data.
    """
    print(f"Starting inspection for column: '{text_col}'...")
    
    inspect_structure(df, label_col=label_col)
    inspect_text_quality(df, text_col=text_col)
    plot_data_health(df, text_col=text_col, label_col=label_col, output_dir=output_dir, analysis=analysis)
    
    print(f"\nArtifacts saved to {output_dir}/")

def get_length_stats_by_label(df, text_col, label_col='label'):
    """Quickly check if document length correlates with labels."""
    if label_col not in df.columns:
        print(f"Label column '{label_col}' not found. Returning global stats.")
        return df[text_col].apply(lambda x: len(str(x).split())).describe()

    temp_df = df.copy()
    temp_df['char_count'] = temp_df[text_col].apply(lambda x: len(str(x)))
    temp_df['word_count'] = temp_df[text_col].apply(lambda x: len(str(x).split()))
    
    stats = temp_df.groupby(label_col)[['char_count', 'word_count']].mean()
    print("\n=== AVG LENGTH PER LABEL ===")
    print(stats)
    return stats