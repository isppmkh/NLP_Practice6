import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer

from src.processing import Normalizer, Tokenizer


class TextGenerator:
    def __init__(self):
        self.seq_len = 10
        self.epochs = 5

        self.tokenizer = Tokenizer()
        self.normalizer = Normalizer()
        self.keras_tokenizer = KerasTokenizer(num_words=3000)
        self.model = None
        self.total_words = None

    def _preprocess(self, text: str) -> str:
        tokens = self.tokenizer.tokenize_words(text)
        normalized = [token.lower() for token in tokens if token.isalpha()]
        return ' '.join(normalized)

    def train_model(self, documents: list[list[str]]) -> None:
        corpus = [' '.join(doc) for doc in documents]
        preprocessed_corpus = [self._preprocess(text) for text in corpus]

        self.keras_tokenizer.fit_on_texts(preprocessed_corpus)
        self.total_words = len(self.keras_tokenizer.word_index) + 1

        print('Creating n-gram sequences')
        sequences = []
        for line in preprocessed_corpus:
            token_list = self.keras_tokenizer.texts_to_sequences([line])[0]
            for i in range(2, len(token_list) + 1):
                n_gram_sequence = token_list[:i]
                sequences.append(n_gram_sequence)
        sequences = pad_sequences(sequences, maxlen=self.seq_len, padding='pre')

        X = sequences[:, :-1]
        y = sequences[:, -1]

        print('Building LSTM model')
        self.model = Sequential(
            [
                Embedding(
                    self.total_words,
                    50,
                    input_length=self.seq_len - 1,
                ),
                LSTM(64),
                Dense(self.total_words, activation='softmax'),
            ]
        )

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'],
        )

        print(f'Training LSTM model for {self.epochs} epochs')
        self.model.fit(X, y, epochs=self.epochs, verbose=1, batch_size=256)
        print('Model training completed')

    def generate(self, seed: str, num_words: int = 20) -> str:
        result = seed
        print(f'Generating text from seed: {seed}')
        seed_proc = self._preprocess(seed)


        for _ in range(num_words):
            token_list = self.keras_tokenizer.texts_to_sequences([seed_proc])[0]
            token_list = pad_sequences(
                [token_list], maxlen=self.seq_len - 1, padding='pre'
            )

            predictions = self.model.predict(token_list, verbose=0)[0]

            # температура
            top_indices = np.argsort(predictions)[-5:]
            top_probs = predictions[top_indices]
            top_probs = top_probs / np.sum(top_probs)
            predicted = np.random.choice(top_indices, p=top_probs)

            output_word = None
            for word, index in self.keras_tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break

            if output_word is None:
                break

            result += ' ' + output_word
            seed_proc += ' ' + output_word

        return result
