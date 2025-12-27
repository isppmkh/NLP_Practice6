from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from src.processing import Normalizer, Tokenizer


class Doc2VecAnalyser:
    def __init__(self, documents):
        self.documents = documents
        self.tagged_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]
        self.model = None
        self.tokenizer = Tokenizer()
        self.normalizer = Normalizer()

    def train_model(self):
        print('Training Doc2Vec model')
        self.model = Doc2Vec(
            self.tagged_documents,
            vector_size=200,
            window=5,
            min_count=2,
            workers=4,
            epochs=20
        )

    def find_similar_docs(self, user_text: str):
        words = self.tokenizer.tokenize_words(user_text)
        normalized_words = self.normalizer.normalize(words)
        inferred_vector = self.model.infer_vector(normalized_words)
        similar_docs = self.model.dv.most_similar([inferred_vector], topn=3)
        return similar_docs
