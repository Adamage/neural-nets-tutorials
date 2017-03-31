from utils import preprocessing_utils, gensim_utils
from collections import Counter
from sklearn.manifold import TSNE
from bokeh.plotting import figure, ColumnDataSource, show
from bokeh.models.annotations import LabelSet
import os

if __name__ == "__main__":
    dump_path = 'sentences.json'
    model = None
    sentences_ted = preprocessing_utils.ensure_tokenized_sentences(dump_path)
    if os.path.isfile('model'):
        model = gensim_utils.load_gensim_model(model_path='model')
    else:
        model = gensim_utils.train_gensim_model_and_load(sentences_ted)

    # Count frequencies.
    slownik = preprocessing_utils.create_frequency_dict(sentences_ted)
    counter = Counter(slownik)
    top_1000 = [x[0] for x in counter.most_common(1000)]

    # Visualise in 2D.
    tsne_obj = TSNE(n_components=2, random_state=0)
    words_top_1000_vectors = model[top_1000]
    words_top_1000_tsne = tsne_obj.fit_transform(words_top_1000_vectors)

    fig = figure(tools="pan,wheel_zoom,reset,save",
                 toolbar_location="above",
                 title="word2vec T-SNE visualisation for top 1000 words")

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