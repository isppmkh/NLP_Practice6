import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Embedding, SimpleRNN
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.processing import Normalizer
from src.processing import Tokenizer as WordTokenizer


class RnnAnalyser:
    def __init__(self, documents, labels):
        self.documents = documents
        self.labels = np.array(labels)
        self.max_words = 10000
        self.max_len = 150
        self.embedding_dim = 32

        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.model = None

        self._embedder = None
        self._doc_embeddings = None

    def train_model(self):
        print('Training RNN tokenizer')
        self.tokenizer.fit_on_texts([' '.join(doc) for doc in self.documents])
        sequences = self.tokenizer.texts_to_sequences(
            [' '.join(doc) for doc in self.documents]
        )
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len)
        num_classes = len(np.unique(self.labels))

        model = Sequential(
            [
                Embedding(self.max_words, self.embedding_dim),
                Dropout(0.5),
                SimpleRNN(32, return_sequences=True),
                Dropout(0.5),
                SimpleRNN(32, return_sequences=False),
                Dense(num_classes, activation='softmax'),
            ]
        )

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        self.model = model

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=2, restore_best_weights=True
        )

        print('Training RNN model')
        self.model.fit(
            padded_sequences,
            self.labels,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            verbose=2,
            callbacks=[early_stopping],
        )

        print('Calculating document embeddings')
        self._embedder = Model(
            inputs=self.model.inputs,
            outputs=self.model.layers[4].output,
        )
        self._doc_embeddings = self._embedder.predict(padded_sequences, verbose=0)

    def find_similar_docs(self, user_text):
        word_tokenizer = WordTokenizer()
        normalizer = Normalizer()

        words = word_tokenizer.tokenize_words(user_text)
        normalized_words = normalizer.normalize(words)
        user_sequence = self.tokenizer.texts_to_sequences([' '.join(normalized_words)])
        user_padded = pad_sequences(user_sequence, maxlen=self.max_len)

        user_embedding = self._embedder.predict(user_padded, verbose=0)

        similarities = cosine_similarity(user_embedding, self._doc_embeddings)[0]
        most_similar_indices = np.argsort(similarities)[::-1][:3]
        return [(i, similarities[i]) for i in most_similar_indices]
