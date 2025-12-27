from pathlib import Path

from sklearn.datasets import fetch_20newsgroups

from src.processing import Normalizer, Tokenizer


def load_and_preprocess_data(return_labels=False):
    cache_path = Path('train')

    categories = [
#        'comp.graphics',
        'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware'
    ]

    if not cache_path.exists():
        print('Dataset not found in cache. Downloading... (may take a few minutes)')
    else:
        print('Loading dataset from cache at train/')

    dataset = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        data_home=str(cache_path)
    )

    raw_docs = dataset.data
    labels = dataset.target
    print(f'Loaded {len(raw_docs)} documents')

    print('Preprocessing documents')
    tokenizer = Tokenizer()
    normalizer = Normalizer()

    processed_docs = []
    for doc in raw_docs:
        words = tokenizer.tokenize_words(doc)
        normalized_words = normalizer.normalize(words)
        if normalized_words:
            processed_docs.append(normalized_words)

    if return_labels:
        return processed_docs, labels
    return processed_docs
