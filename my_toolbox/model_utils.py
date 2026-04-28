import os
import joblib
import logging
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- IO UTILS ---

def save_model(model, model_name: str, folder="models"):
    """Saves any trained object (model, vectorizer, or pipeline)."""
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, model_name)
    try:
        joblib.dump(model, filepath)
        logger.info(f"Successfully saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save: {e}")

def load_model(filepath: str):
    """Loads a saved object from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No file found at {filepath}")
    return joblib.load(filepath)

# --- CORE LOGIC ---

class Vectorization:
    def __init__(self, ngram_range=(1, 2), max_features=10000, stop_words='english'):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words=stop_words,
            min_df=2,
            max_df=0.9)
        self.fitted = False

    def fit_transform(self, raw_documents):
        X = self.vectorizer.fit_transform(raw_documents)
        self.fitted = True
        return X

    def transform(self, raw_documents):
        if not self.fitted:
            raise RuntimeError("Vectorizer must be fitted before transforming.")
        return self.vectorizer.transform(raw_documents)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

class ModelBase:
    """Internal helper to allow functions to accept both raw text and matrices."""
    def _prepare_input(self, X, vectorizer=None):
        # 1. If input is already a matrix/array, return it
        if isinstance(X, (np.ndarray, pd.DataFrame)) or issparse(X):
            return X
        
        # 2. If input is text, we MUST have a vectorizer
        if isinstance(X, (list, pd.Series, np.ndarray)):
            if vectorizer is None:
                raise ValueError("Input is raw text, but no 'vectorizer' was provided.")
            
            # Auto-fit if the vectorizer isn't ready, otherwise just transform
            if hasattr(vectorizer, 'fitted') and not vectorizer.fitted:
                return vectorizer.fit_transform(X)
            return vectorizer.transform(X)
        
        raise TypeError(f"Unsupported input type: {type(X)}")

# --- USER CLASSES ---

class Classification(ModelBase):
    def __init__(self, classifier=None):
        """Initializes with a Logistic Regression default or a custom sklearn-like model."""
        self.model = classifier or LogisticRegression(max_iter=1000, class_weight='balanced')
        self.is_fitted = False

    def train(self, X, y, vectorizer=None, cv=None):
        """Trains the model. y is strictly required."""
        if y is None:
            raise ValueError("Classification requires labels (y) to train.")
        
        X_input = self._prepare_input(X, vectorizer)
        
        if cv:
            scores = cross_val_score(self.model, X_input, y, cv=cv, scoring='f1_macro')
            logger.info(f"CV F1-Macro: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
            
        self.model.fit(X_input, y)
        self.is_fitted = True

    def tune(self, X, y, param_grid, vectorizer=None, cv=3):
        """Runs GridSearchCV to find optimal hyperparameters."""
        X_input = self._prepare_input(X, vectorizer)
        grid = GridSearchCV(self.model, param_grid, cv=cv, scoring='f1_macro')
        grid.fit(X_input, y)
        self.model = grid.best_estimator_
        return grid.best_params_

    def predict(self, X, vectorizer=None):
        X_input = self._prepare_input(X, vectorizer)
        return self.model.predict(X_input)

class Regression(ModelBase):
    def __init__(self, regressor=None):
        self.model = regressor or Ridge()
        self.is_fitted = False

    def train(self, X, y, vectorizer=None, cv=None):
        X_input = self._prepare_input(X, vectorizer)
        if cv:
            scores = cross_val_score(self.model, X_input, y, cv=cv, scoring='neg_mean_squared_error')
            logger.info(f"CV MSE: {-np.mean(scores):.4f}")
        self.model.fit(X_input, y)
        self.is_fitted = True

    def predict(self, X, vectorizer=None):
        X_input = self._prepare_input(X, vectorizer)
        return self.model.predict(X_input)

class Similarity(ModelBase):
    def get_similarities(self, X, Y=None, vectorizer=None):
        """Returns a similarity matrix. If Y is provided, compares X vs Y."""
        matrix_x = self._prepare_input(X, vectorizer)
        if Y is not None:
            matrix_y = self._prepare_input(Y, vectorizer)
            return cosine_similarity(matrix_x, matrix_y)
        return cosine_similarity(matrix_x)

    def find_duplicates(self, texts, vectorizer=None, threshold=0.95):
        """Finds pairs within a single list that exceed the similarity threshold."""
        sim_matrix = self.get_similarities(texts, vectorizer=vectorizer)
        results = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if sim_matrix[i, j] >= threshold:
                    results.append({
                        'item_a': texts[i], 
                        'item_b': texts[j], 
                        'score': round(float(sim_matrix[i, j]), 4)
                    })
        return results

class Clustering:
    def __init__(self, n_clusters=5):
        self.model = KMeans(n_clusters=n_clusters, random_state=3, n_init='auto')
        self.n_clusters = n_clusters

    def train(self, X_vec):
        """Strictly requires vectorized input."""
        self.model.fit(X_vec)
        return self.model.labels_

    def predict(self, X_vec):
        return self.model.predict(X_vec)

    def get_top_terms_per_cluster(self, vectorizer, n_terms=10):
        """Extracts the most significant words for each cluster."""
        feature_names = vectorizer.get_feature_names()
        ordered_centroids = self.model.cluster_centers_.argsort()[:, ::-1]
        
        cluster_dict = {}
        for i in range(self.n_clusters):
            top_words = [feature_names[ind] for ind in ordered_centroids[i, :n_terms]]
            cluster_dict[i] = top_words
        return cluster_dict