import os
from utils import gensim_utils, preprocessing_utils

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

if __name__ == "__main__":
    # Define paths and directories.
    ppu = preprocessing_utils()

    metadata_file = 'metadata.tsv'
    word2vec_model_path = os.path.join(ppu.outputs_dir, 'ted_word2vec_model')
    checkpoint = 'ted_show.ckpt'
    metadata_path = os.path.join(ppu.outputs_dir, metadata_file)
    checkpoint_path = os.path.join(ppu.outputs_dir, checkpoint)
    dump_path = os.path.join(ppu.outputs_dir, 'sentences_ted.json')

    # Ensure we have the data.
    sentences_ted = ppu.ensure_tokenized_sentences(dump_path)

    # Load/train model and inspect its elements.
    gensim_model = gensim_utils.ensure_gensim_model(sentences_ted, word2vec_model_path)
    weights = gensim_model.wv.syn0
    idx2words = gensim_model.wv.index2word
    vocab_size = weights.shape[0]
    embedding_dim = weights.shape[1]

    # Create a file with words themselves, to 'label' points on the t-SNE visualisation.
    with open(metadata_path, 'w') as f:
        f.writelines("\n".join(idx2words))

    # Create graph, variables.
    tf.reset_default_graph()
    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=False, name="W")
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
    embedding_init = W.assign(embedding_placeholder)

    # Define the checkpoint saver and projector configuration.
    writer = tf.summary.FileWriter(ppu.outputs_dir, graph=tf.get_default_graph())
    saver = tf.train.Saver()

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = W.name
    embedding.metadata_path = metadata_path
    projector.visualize_embeddings(writer, config)

    # Now, we run the operation, which is to assign words from gensim model to the tensorflow variables.
    with tf.Session() as sess:
        sess.run(embedding_init, feed_dict={embedding_placeholder: weights})
        save_path = saver.save(sess, checkpoint_path)