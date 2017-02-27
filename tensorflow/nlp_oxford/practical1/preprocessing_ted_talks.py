import numpy as np
import os
from random import shuffle
import re
import pandas
import urllib.request
import zipfile
import lxml.etree
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import json

def get_data():
    if not os.path.isfile('ted_en-20160408.zip'):
        urllib.request.urlretrieve(
            "https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip",
            filename="ted_en-20160408.zip")

def extract_subtitles():
    with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
        doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
    input_text = '\n'.join(doc.xpath('//content/text()'))
    del doc
    return input_text

def get_sentences_strings(input_text_noparens):
    sentences_strings_ted = []
    for line in input_text_noparens.split('\n'):
        m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
        sentences_strings_ted.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)
    return sentences_strings_ted

def create_frequency_dict(sentences):
    slownik = dict()
    for sentence in sentences:
        for word in sentence:
            if word not in slownik.keys():
                slownik[word] = 1
            else:
                slownik[word] += 1
    return slownik

def serialize_to_json(sentences, save_to_path):
    with open(save_to_path, 'w') as f:
        json.dump(sentences, f)

def deserialize_from_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    dump_path = 'sentences.json'
    if os.path.isfile(dump_path):
        sentences_ted = deserialize_from_json(dump_path)
    else:
        get_data()
        input_subtitles = extract_subtitles()
        input_text_noparens = re.sub(r'\([^)]*\)', '', input_subtitles)
        del input_subtitles
        sentences_strings_ted = get_sentences_strings(input_text_noparens)
        del input_text_noparens

        # Tokenize by white space
        sentences_ted = []
        for sent_str in sentences_strings_ted:
            tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
            sentences_ted.append(tokens)
        del sentences_strings_ted
        serialize_to_json(sentences_ted, dump_path)

    #  Count frequencies
    slownik = create_frequency_dict(sentences_ted)
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
    plt.show()
    plt.savefig('histogram_top100.png')
