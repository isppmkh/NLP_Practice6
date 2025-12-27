import pickle
from pathlib import Path

import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


class DialogBot:
    def __init__(self):
        self.seq_len = 20
        self.embedding_dim = 100
        self.tokenizer = Tokenizer()
        self.model = None
        self.vocab_size = None

    def train(self, text_corpus: str):
        n_epochs = 30
        batch_size = 128
        validation_split = 0.1

        text_corpus = text_corpus[: len(text_corpus) // 2]

        sentences = text_corpus.split('.')

        self.tokenizer.fit_on_texts(sentences)
        self.vocab_size = len(self.tokenizer.word_index) + 1

        print(f'Vocab size: {self.vocab_size}')

        input_sequences = []

        for sentence in sentences:
            if len(sentence.strip()) == 0:
                continue

            token_list = self.tokenizer.texts_to_sequences([sentence])[0]

            for i in range(1, len(token_list)):
                n_gram_seq = token_list[: i + 1]
                input_sequences.append(n_gram_seq)

        input_sequences = input_sequences[:50000]

        print(f'Total sequences: {len(input_sequences)}')

        max_seq_len = max([len(x) for x in input_sequences])

        sequences = np.array(
            pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')
        )

        X = sequences[:, :-1]
        y = sequences[:, -1]
        y = to_categorical(y, num_classes=self.vocab_size)
        self.model = Sequential(
            [
                Embedding(
                    self.vocab_size, self.embedding_dim, input_length=X.shape[1]
                ),
                LSTM(150, return_sequences=True),
                Dropout(0.2),
                LSTM(100),
                Dense(self.vocab_size, activation='softmax'),
            ]
        )

        self.model.compile(
            loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']
        )

        self.model.summary()

        self.model.fit(
            X,
            y,
            epochs=n_epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
        )

    def generate_response(self, user_input: str) -> str:
        max_words = 20
        temperature = 0.8

        result = user_input.lower()

        for _ in range(max_words):
            token_list = self.tokenizer.texts_to_sequences([result])[0]
            token_list = pad_sequences(
                [token_list], maxlen=self.seq_len - 1, padding='pre'
            )

            predictions = self.model.predict(token_list, verbose=0)[0]


            # Температура
            predictions = np.asarray(predictions).astype('float64')
            predictions = np.log(predictions + 1e-8) / temperature
            exp_preds = np.exp(predictions)
            predictions = exp_preds / np.sum(exp_preds)

            predicted_id = np.random.choice(len(predictions), p=predictions)

            output_word = None
            for word, idx in self.tokenizer.word_index.items():
                if idx == predicted_id:
                    output_word = word
                    break

            if output_word is None:
                break

            result += ' ' + output_word

        response = result[len(user_input) :].strip()
        return response if response else 'I dont understand.'

    def save(self, model_dir: str = 'model'):
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save(path / 'model.h5')

        with open(path / 'tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)

        with open(path / 'config.pkl', 'wb') as f:
            pickle.dump(
                {
                    'seq_len': self.seq_len,
                    'vocab_size': self.vocab_size,
                    'embedding_dim': self.embedding_dim,
                },
                f,
            )

    def load(self, model_dir: str = 'model'):
        path = Path(model_dir)

        with open(path / 'config.pkl', 'rb') as f:
            config = pickle.load(f)
            self.seq_len = config['seq_len']
            self.vocab_size = config['vocab_size']
            self.embedding_dim = config['embedding_dim']

        with open(path / 'tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)

        self.model = load_model(path / 'model.h5')

