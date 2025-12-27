import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.processing import Normalizer, Tokenizer


class LsaAnalyser:
    def __init__(self, documents):
        self.documents = documents
        self.tokenizer = Tokenizer()
        self.normalizer = Normalizer()
        self.vectorizer = TfidfVectorizer()
        self.svd = TruncatedSVD(n_components=200, random_state=42)
        self.doc_vectors = None

    def train_model(self):
        print('Training LSA model (TF-IDF + SVD)')
        tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        self.doc_vectors = self.svd.fit_transform(tfidf_matrix)

    def find_similar_docs(self, user_text: str):
        words = self.tokenizer.tokenize_words(user_text)
        normalized_words = self.normalizer.normalize(words)
        processed_user_text = ' '.join(normalized_words)

        user_tfidf = self.vectorizer.transform([processed_user_text])
        user_vector = self.svd.transform(user_tfidf)

        similarities = cosine_similarity(user_vector, self.doc_vectors)

        top_indices = np.argsort(similarities[0])[::-1][:3]
        return [(idx, similarities[0][idx]) for idx in top_indices]
