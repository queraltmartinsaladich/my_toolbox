import logging
import pandas as pd
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_text_duplicates(df, text_col):
    """
    Identifies exact duplicate text entries.
    """
    total = len(df)
    exact_dupes = df.duplicated(subset=[text_col]).sum()
    
    if exact_dupes > 0:
        logger.warning(f"DUPLICATE ALERT: Found {exact_dupes} exact duplicates ({exact_dupes/total:.1%}).")
    else:
        logger.info("[+] No exact duplicates found.")
    return exact_dupes

def debug_label_consistency(df, label_col):
    """
    Checks for white-space issues or casing inconsistencies in labels.
    """
    if not label_col or label_col not in df.columns:
        return
        
    unique_labels = df[label_col].unique().tolist()
    # Check if cleaning would reduce the number of unique labels
    cleaned_labels = df[label_col].astype(str).str.strip().str.lower().unique()
    
    if len(unique_labels) != len(cleaned_labels):
        logger.warning(f"LABEL INCONSISTENCY: {len(unique_labels)} raw labels vs {len(cleaned_labels)} cleaned.")
        logger.warning(f"Potential typos/spacing in: {unique_labels}")

def check_tokenization_viability(df, text_col):
    """
    Ensures text contains actual alphanumeric characters (prevents empty vectors).
    """
    # Regex for at least one alphanumeric character
    invalid_mask = df[text_col].apply(lambda x: not bool(re.search(r'\w', str(x))))
    invalid_count = invalid_mask.sum()
    
    if invalid_count > 0:
        logger.warning(f"TOKENIZATION ALERT: {invalid_count} rows contain no words/numbers and will fail vectorization.")
        sample = df[invalid_mask][text_col].head(2).values
        logger.info(f"Invalid samples: {list(sample)}")

def debug_dataframe(data, text_col, label_col=None):
    """
    The Master Debug Pipeline. 
    Protects against crashes, data leakage, and silent failures.
    """
    logger.info("--- STARTING FULL DEBUG PIPELINE ---")
    
    # 1. Structural Checks
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    
    if text_col not in data.columns:
        raise KeyError(f"Text column '{text_col}' not found.")

    # 2. Nulls & Empty Strings (The "Crasher" Check)
    null_count = data[text_col].isnull().sum()
    empty_strings = (data[text_col].astype(str).str.strip() == "").sum()
    
    if null_count > 0 or empty_strings > 0:
        logger.warning(f"Cleaning {null_count} nulls and {empty_strings} empty strings...")
        data = data.dropna(subset=[text_col])
        data = data[data[text_col].astype(str).str.strip() != ""].copy()

    # 3. Label Integrity
    if label_col and label_col in data.columns:
        debug_label_consistency(data, label_col)
        
        # Check Class Balance
        counts = data[label_col].value_counts()
        balance = counts / len(data) * 100
        logger.info(f"Class Balance:\n{balance.to_string(formatters={'': '{:,.2f}%'.format})}")
        
        if counts.min() < 5:
            logger.warning(f"EXTREME IMBALANCE: Class '{counts.idxmin()}' has only {counts.min()} samples.")
            
        # Data Leakage Check (Simple string match)
        sample_label = str(data[label_col].iloc[0]).lower()
        if len(sample_label) > 3: # Avoid checking tiny labels
            overlap = data[text_col].str.lower().str.contains(sample_label, regex=False).mean()
            if overlap > 0.4:
                logger.warning(f"POTENTIAL LEAKAGE: Label '{sample_label}' appears in {overlap:.1%} of texts.")

    # 4. Text Quality Checks
    check_text_duplicates(data, text_col)
    check_tokenization_viability(data, text_col)

    # 5. Encoding Check
    non_ascii = data[text_col].apply(lambda x: len(str(x).encode('ascii', 'ignore')) != len(str(x))).sum()
    if non_ascii > 0:
        logger.info(f"Note: {non_ascii} rows contain non-ASCII characters (emojis/symbols).")

    # 6. Sample Print
    logger.info(f"Final sample from '{text_col}':")
    for i, s in enumerate(data[text_col].head(2).values):
        logger.info(f"  [{i+1}]: {str(s)[:80]}...")

    logger.info("--- DEBUG PIPELINE COMPLETE ---\n")
    return data