import nltk
from nltk.tokenize import word_tokenize


class Tokenizer:
    def __init__(self):
        self._check_nltk()

    def _check_nltk(self) -> None:
        try:
            nltk.data.find('tokenizers/punkt')
        except Exception:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except Exception:
            nltk.download('punkt_tab', quiet=True)
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except Exception:
            nltk.download('averaged_perceptron_tagger', quiet=True)
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except Exception:
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)

    def tokenize_words(self, text: str) -> list[str]:
        return word_tokenize(text)
