import json
import os
import re
import urllib.request
import zipfile
from gensim.models import Word2Vec
from lxml import etree


def get_data():
    if not os.path.isfile('ted_en-20160408.zip'):
        urllib.request.urlretrieve(
            "https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip",
            filename="ted_en-20160408.zip")


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


def extract_subtitles():
    with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
        doc = etree.parse(z.open('ted_en-20160408.xml', 'r'))
    input_text = '\n'.join(doc.xpath('//content/text()'))
    del doc
    return input_text


def serialize_to_json(sentences, save_to_path):
    with open(save_to_path, 'w') as f:
        json.dump(sentences, f)


def deserialize_from_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def ensure_tokenized_sentences(dump_path):
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

    return sentences_ted


def train_gensim_model(dump_path):
    sentences_ted = ensure_tokenized_sentences(dump_path)
    model = Word2Vec(sentences_ted,
                     size=100,
                     window=5,
                     min_count=5,
                     workers=4)
    return model


def load_gensim_model(model_path, sentences_path):
    if os.path.isfile(model_path):
        return Word2Vec.load(model_path)
    else:
        return train_gensim_model(sentences_path)
