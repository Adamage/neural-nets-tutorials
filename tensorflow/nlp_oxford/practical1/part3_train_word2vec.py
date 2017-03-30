from gensim.models import Word2Vec
from utils import ensure_tokenized_sentences, load_gensim_model

if __name__ == "__main__":
    model = load_gensim_model(model_path='model',
                              sentences_path='sentences.json')
    # Switching to  just KeyedVectors instance of the whole model to free memory.
    word_vectors = model.wv
    del model
    print("Printing some example vectors:")
    print("computer: " + str(word_vectors['computer']))
    print("Printing out most similar words:")
    print("woman + king - man = " + str(word_vectors.most_similar(positive=['woman', 'king'],
                                                                  negative=['man'])))
    print("italy + rome - france = " + str(word_vectors.most_similar(positive=['italy', 'rome'],
                                                                     negative=['france'])))
