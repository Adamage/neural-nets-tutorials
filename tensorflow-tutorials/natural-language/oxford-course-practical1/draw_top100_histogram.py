from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from utils import preprocessing_utils

if __name__ == "__main__":
    dump_path = 'sentences.json'
    sentences_ted = preprocessing_utils.ensure_tokenized_sentences(dump_path)

    #  Count frequencies
    slownik = preprocessing_utils.create_frequency_dict(sentences_ted)
    counter = Counter(slownik)
    top_100 = counter.most_common(100)

    # Histogram
    labels = [x[0] for x in top_100]
    values = [x[1] for x in top_100]
    num = len(labels)
    fig, ax = plt.subplots()
    barwidth = 0.3
    ax.bar(np.arange(num), values, barwidth)
    ax.set_xticks(np.arange(num) + barwidth / 2)
    ax.set_xticklabels(labels)
    plt.savefig('histogram_top100.png')
