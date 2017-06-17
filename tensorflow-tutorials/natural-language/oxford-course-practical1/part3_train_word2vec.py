import os

from utils import gensim_utils, preprocessing_utils

if __name__ == "__main__":
    dump_path = 'sentences.json'
    ppu = preprocessing_utils()
    sentences_ted = ppu.ensure_tokenized_sentences(dump_path)
    model = gensim_utils.ensure_gensim_model(sentences_ted, 'model')

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
