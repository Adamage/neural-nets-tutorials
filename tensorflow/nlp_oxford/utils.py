import json
import os
import re
import urllib.request
import zipfile
from random import shuffle

from gensim.models import Word2Vec
from lxml import etree


class preprocessing_utils:
    @staticmethod
    def get_ted_data():
        if not os.path.isfile('ted_en-20160408.zip'):
            urllib.request.urlretrieve(
                "https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip",
                filename="ted_en-20160408.zip")

    @staticmethod
    def get_wiki_data():
        if not os.path.isfile('wikitext-103-raw-v1.zip'):
            urllib.request.urlretrieve("https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip",
                                       filename="wikitext-103-raw-v1.zip")

    @staticmethod
    def load_wiki_raw_data():
        preprocessing_utils.get_wiki_data()

        with zipfile.ZipFile('wikitext-103-raw-v1.zip') as zf:
            try:
                text_loaded = zf.open('wikitext-103/wiki.train.tokens', 'r').read()
                return str(text_loaded, encoding='utf-8')
            except Exception as e:
                print("Failed to load wiki raw data from zipped file. Details: " + str(e))

    @staticmethod
    def load_wiki_sentences():
        sentences = []
        loaded_data = preprocessing_utils.load_wiki_raw_data()
        for line in loaded_data.split('\n'):
            sentence = [i for i in line.split('.') if (i and len(i.split()) >= 5)]
            sentences.extend(sentence)

        return sentences

    @staticmethod
    def preprocess_wiki_sentences(sentences):
        pattern1 = "[^a-z]"
        pattern2 = r'\([^)]*\)'
        for i in range(len(sentences)):
            sentences[i] = re.sub(pattern1, " ", sentences[i].lower())
            sentences[i] = re.sub(pattern2, "", sentences[i])

        return sentences

    @staticmethod
    def get_a_fraction_of_collection(fraction_int, collection):
        shuffle(collection)
        cutoff_index = int(len(collection) / fraction_int)
        collection = collection[:cutoff_index]
        return collection

    @staticmethod
    def get_sentences_strings(input_text_noparens):
        sentences_strings_ted = []
        for line in input_text_noparens.split('\n'):
            m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
            sentences_strings_ted.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)
        return sentences_strings_ted

    @staticmethod
    def create_frequency_dict(sentences):
        slownik = dict()
        for sentence in sentences:
            for word in sentence:
                if word not in slownik.keys():
                    slownik[word] = 1
                else:
                    slownik[word] += 1
        return slownik

    @staticmethod
    def extract_subtitles():
        with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
            doc = etree.parse(z.open('ted_en-20160408.xml', 'r'))
        input_text = '\n'.join(doc.xpath('//content/text()'))
        del doc
        return input_text

    @staticmethod
    def serialize_to_json(sentences, save_to_path):
        with open(save_to_path, 'w') as f:
            json.dump(sentences, f)

    @staticmethod
    def deserialize_from_json(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

    @staticmethod
    def ensure_tokenized_sentences(dump_path):
        if os.path.isfile(dump_path):
            sentences_ted = preprocessing_utils.deserialize_from_json(dump_path)
        else:
            preprocessing_utils.get_ted_data()
            input_subtitles = preprocessing_utils.extract_subtitles()
            input_text_noparens = re.sub(r'\([^)]*\)', '', input_subtitles)
            del input_subtitles
            sentences_strings_ted = preprocessing_utils.get_sentences_strings(input_text_noparens)
            del input_text_noparens

            # Tokenize by white space
            sentences_ted = []
            for sent_str in sentences_strings_ted:
                tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
                sentences_ted.append(tokens)
            del sentences_strings_ted
            preprocessing_utils.serialize_to_json(sentences_ted, dump_path)

        return sentences_ted


class gensim_utils:
    @staticmethod
    def train_gensim_model_and_load(sentences):
        model = Word2Vec(sentences,
                         size=100,
                         window=5,
                         min_count=5,
                         workers=4)
        model.save()
        return model

    @staticmethod
    def load_gensim_model(model_path):
        if os.path.isfile(model_path):
            return Word2Vec.load(model_path)
