import argparse
from pathlib import Path

from src.analysers import LsaAnalyser
from src.utils import load_and_preprocess_data


def main():
    parser = argparse.ArgumentParser(description='Find similar documents using LSA')
    parser.add_argument('filepath', type=str, help='Path to the text file for analysis')
    args = parser.parse_args()

    user_file = Path(args.filepath)
    if not user_file.exists():
        print(f'Error: File not found at "{user_file}"')
        return
    user_text = user_file.read_text(encoding='utf-8')

    documents = load_and_preprocess_data()
    analyser = LsaAnalyser([' '.join(doc) for doc in documents])
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
