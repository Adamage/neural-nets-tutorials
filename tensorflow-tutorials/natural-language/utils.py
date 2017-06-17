import json
import os
import re
import urllib.request
import zipfile
from random import shuffle

from gensim.models import Word2Vec
from lxml import etree


class preprocessing_utils:
    def __init__(self):
        script_dir = os.path.dirname(__file__)
        self.raw_data_dir = os.path.join(script_dir, 'raw_data')
        self.outputs_dir = os.path.join(script_dir, 'outputs')
        self.pretrained_dir = os.path.join(script_dir, 'pretrained')

        for directory in [self.raw_data_dir, self.outputs_dir, self.pretrained_dir]:
            if not os.path.isdir(directory):
                os.mkdir(directory)

    def get_ted_data(self):
        if not os.path.isfile(os.path.join(self.raw_data_dir, 'ted_en-20160408.zip')):
            urllib.request.urlretrieve(
                "https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip",
                filename="ted_en-20160408.zip")

    def get_wiki_data(self):
        if not os.path.isfile(os.path.join(self.raw_data_dir, 'wikitext-103-raw-v1.zip')):
            urllib.request.urlretrieve("https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip",
                                       filename="wikitext-103-raw-v1.zip")

    def load_wiki_raw_data(self):
        self.get_wiki_data()

        with zipfile.ZipFile(os.path.join(self.outputs_dir, 'wikitext-103-raw-v1.zip')) as zf:
            try:
                text_loaded = zf.open('wikitext-103/wiki.train.tokens', 'r').read()
                return str(text_loaded, encoding='utf-8')
            except Exception as e:
                print("Failed to load wiki raw data from zipped file. Details: " + str(e))

    def load_wiki_sentences(self):
        sentences = []
        loaded_data = self.load_wiki_raw_data()
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

    def extract_subtitles(self):
        with zipfile.ZipFile(os.path.join(self.raw_data_dir, 'ted_en-20160408.zip'), 'r') as z:
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

    def ensure_tokenized_sentences(self, dump_path):
        if os.path.isfile(dump_path):
            sentences_ted = preprocessing_utils.deserialize_from_json(dump_path)
        else:
            self.get_ted_data()
            input_subtitles = self.extract_subtitles()
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
    def ensure_gensim_model(path_to_preprocessed_sentences, path_to_trained_model):
        if os.path.isfile(path_to_trained_model):
            model = gensim_utils.load_gensim_model(model_path=path_to_trained_model)
        else:
            model = gensim_utils.train_gensim_model_and_load(path_to_preprocessed_sentences, path_to_trained_model)

        return model

    @staticmethod
    def train_gensim_model_and_load(sentences, path_to_trained_model):
        model = Word2Vec(sentences,
                         size=300,
                         window=7,
                         min_count=5,
                         workers=4)
        model.save(path_to_trained_model)
        return model

    @staticmethod
    def load_gensim_model(model_path):
        if os.path.isfile(model_path):
            return Word2Vec.load(model_path)
