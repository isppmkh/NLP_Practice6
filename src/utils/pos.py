import nltk

from src.processing import Tokenizer


def analyze_pos(text: str) -> dict[str, list[str]]:
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize_words(text)
    pos_tags = nltk.pos_tag(tokens)

    nouns = [word for word, tag in pos_tags if tag.startswith('NN')]
    verbs = [word for word, tag in pos_tags if tag.startswith('VB')]
    adjectives = [word for word, tag in pos_tags if tag.startswith('JJ')]
    pos_sequence = [tag for _, tag in pos_tags]

    return {
        'nouns': nouns,
        'verbs': verbs,
        'adjectives': adjectives,
        'pos_sequence': pos_sequence,
    }

