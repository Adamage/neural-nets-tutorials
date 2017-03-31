from utils import preprocessing_utils, gensim_utils
import os
from collections import Counter
from sklearn.manifold import TSNE
from bokeh.plotting import figure, ColumnDataSource, show
from bokeh.models.annotations import LabelSet

if __name__ == "__main__":
    serialized_wiki = 'wiki_sentences.json'
    wiki_sentences = None

    if not os.path.isfile(serialized_wiki):
        wiki_sentences = preprocessing_utils.load_wiki_sentences()
        wiki_sentences = preprocessing_utils.preprocess_wiki_sentences(wiki_sentences)
        wiki_sentences = preprocessing_utils.get_a_fraction_of_collection(5, wiki_sentences)
        preprocessing_utils.serialize_to_json(sentences=wiki_sentences, save_to_path=serialized_wiki)
    else:
        wiki_sentences = preprocessing_utils.deserialize_from_json(serialized_wiki)

    if os.path.isfile('wiki_model'):
        model = gensim_utils.load_gensim_model(model_path='wiki_model')
    else:
        model = gensim_utils.train_gensim_model_and_load(wiki_sentences)

    # Count frequencies.
    frequency_dict = preprocessing_utils.create_frequency_dict(wiki_sentences) # TODO it works badly
    counter = Counter(frequency_dict)
    top_1000 = [x[0] for x in counter.most_common(1000)]

    # Visualise in 2D.
    tsne_obj = TSNE(n_components=2, random_state=0)
    words_top_1000_vectors = model[top_1000]
    words_top_1000_tsne = tsne_obj.fit_transform(words_top_1000_vectors)

    fig = figure(tools="pan,wheel_zoom,reset,save",
                 toolbar_location="above",
                 title="word2vec T-SNE visualisation for top 1000 words - wikipedia")

    source = ColumnDataSource(data=
    dict(
        x1=words_top_1000_tsne[:, 0],
        x2=words_top_1000_tsne[:, 1],
        names=top_1000))

    fig.scatter(x="x1", y="x2", size=9, source=source)

    labels = LabelSet(x="x1", y="x2", text="names", y_offset=6, text_font_size="9pt", text_color="#555555",
                      source=source, text_align='center')

    fig.add_layout(labels)
    show(fig)


