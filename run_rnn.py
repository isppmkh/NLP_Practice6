import argparse
import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.analysers.rnn import RnnAnalyser
from src.utils.loader import load_and_preprocess_data


def main():
    parser = argparse.ArgumentParser(description='Find similar documents using an RNN')
    parser.add_argument('filepath', type=str, help='Path to the text file')
    args = parser.parse_args()

    user_file = Path(args.filepath)
    if not user_file.exists():
        print(f'Error: File not found at "{user_file}"')
        return
    user_text = user_file.read_text(encoding='utf-8')

    documents, labels = load_and_preprocess_data(return_labels=True)
    analyser = RnnAnalyser(documents, labels)
    analyser.train_model()

    print(f'Finding documents similar to content of: "{user_file}"')
    similar_docs = analyser.find_similar_docs(user_text)

    print('\nTop 3 most similar documents:\n')
    for doc_id, score in similar_docs:
        print(f'Doc {doc_id}, similarity: {score:.2f}')
        preview_docs = ' '.join(documents[doc_id][:15])
        print(f'Preview: {preview_docs}')
        print('-' * 20)


if __name__ == '__main__':
    main()
