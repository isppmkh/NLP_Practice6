import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


class Normalizer:
    def __init__(self):
        self._check_nltk()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def _check_nltk(self) -> None:
        print('Checking NLTK data')
        try:
            nltk.data.find('corpora/stopwords')
        except Exception:
            nltk.download('stopwords', quiet=True)
        try:
            nltk.data.find('corpora/wordnet')
        except Exception:
            nltk.download('wordnet', quiet=True)
        try:
            nltk.data.find('corpora/omw-1.4')
        except Exception:
            nltk.download('omw-1.4', quiet=True)

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        return [token for token in tokens if token.lower() not in self.stop_words]

    def lemmatize_tokens(self, tokens: list[str]) -> list[str]:
        return [self.lemmatizer.lemmatize(token.lower()) for token in tokens]

    def replace_with_synonyms(self, tokens: list[str]) -> list[str]:
        synonym_tokens = []
        for token in tokens:
            synsets = wordnet.synsets(token)
            if synsets:
                synonym_tokens.append(synsets[0].lemmas()[0].name())
            else:
                synonym_tokens.append(token)
        return synonym_tokens

    def normalize(self, tokens: list[str]) -> list[str]:
        tokens = [token for token in tokens if token.isalpha()]
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize_tokens(tokens)
        tokens = self.replace_with_synonyms(tokens)
        return tokens
